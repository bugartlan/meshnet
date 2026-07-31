from pathlib import Path

import meshio
import numpy as np
import numpy.typing as npt
import pyvista as pv
import torch
import trimesh
from torch_geometric.data import Data

from meshnet.mgn.geodesics import SurfaceGeodesics

pv.OFF_SCREEN = True


class GraphBuilderBase:
    """Base class for constructing geometric graphs from finite element meshes and simulation data.

    Converts mesh geometry and nodal outputs into PyTorch Geometric Data objects suitable for
    graph neural network training. Subclasses implement different node feature augmentation strategies.

    Node features: [x, y, z, fx, fy, fz, is_boundary]
        - Spatial coordinates (x, y, z)
        - Nodal forces computed from gaussian-weighted load distribution (fx, fy, fz)
        - Binary boundary flag indicating fixed support nodes (is_boundary)

    Edge features: [dx, dy, dz, distance]
        - Displacement vector between connected nodes (dx, dy, dz)
        - Euclidean distance metric (distance)
    """

    def __init__(self, std: float = 0.001):
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")

        self.std = std
        self.boundary_tol = 1e-6
        self.num_categorical = 1  # For boundary flags
        self._geodesic_mesh: meshio.Mesh | None = None
        self._surface_geodesics: SurfaceGeodesics | None = None

    def build(
        self,
        mesh: meshio.Mesh,
        y: npt.ArrayLike,
        contacts: list[tuple] | None = None,
    ) -> Data:
        if y.shape[0] != mesh.points.shape[0]:
            raise ValueError(
                f"Output array y must have shape [num_nodes, num_output_features], but got {y.shape} and {mesh.points.shape[0]} nodes."
            )

        # Node feature matrix with shape [num_nodes, num_node_features]
        x = self._make_nodes(mesh, contacts)
        y = torch.tensor(y, dtype=torch.float32)
        edge_index, edge_attr = self._make_edges(mesh)

        num_physical_nodes = mesh.points.shape[0]

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_physical_nodes=num_physical_nodes,
            contacts=contacts,
        )

    def gaussian_loads(
        self,
        mesh: meshio.Mesh,
        contacts: list[tuple],
    ) -> torch.Tensor:
        """Distribute contact forces over the surface using geodesic Gaussians."""
        coords = np.asarray(mesh.points, dtype=np.float64)
        n = coords.shape[0]
        if not contacts:
            return torch.zeros((n, 3), dtype=torch.float32)

        geodesics = self._get_surface_geodesics(mesh)

        frc = np.asarray([f for _, f in contacts], dtype=np.float64)
        if frc.shape != (len(contacts), 3) or not np.all(np.isfinite(frc)):
            raise ValueError("Contact forces must be finite 3D vectors")

        w = np.zeros((n, len(contacts)), dtype=np.float64)
        for contact_index, (point, _) in enumerate(contacts):
            distances = geodesics.distance_from(point)
            w[geodesics.vertex_indices, contact_index] = np.exp(
                -(distances**2) / (2 * self.std**2)
            )

        w_sum = w.sum(axis=0, keepdims=True)
        if not np.all(np.isfinite(w_sum)) or np.any(w_sum <= np.finfo(np.float64).tiny):
            raise ValueError(
                "Contact kernel normalization failed; std may be too small "
                "for the surface mesh"
            )
        w /= w_sum

        return torch.from_numpy((w @ frc).astype(np.float32, copy=False))

    def _get_surface_geodesics(self, mesh: meshio.Mesh) -> SurfaceGeodesics:
        """Return the cached solver for ``mesh``, rebuilding on identity change."""
        if self._geodesic_mesh is not mesh or self._surface_geodesics is None:
            self._geodesic_mesh = mesh
            self._surface_geodesics = SurfaceGeodesics.from_mesh(
                mesh,
                tolerance=self.boundary_tol,
            )
        return self._surface_geodesics

    def _make_nodes(
        self,
        mesh: meshio.Mesh,
        loads: list[tuple[npt.ArrayLike, npt.ArrayLike]],
    ) -> torch.Tensor:
        vertices = mesh.points.astype(np.float32, copy=False)

        # Position Coordinates
        coords = torch.from_numpy(vertices)

        # Force Vectors
        forces = self.gaussian_loads(mesh, loads)

        # Boundary Mask
        mask_np = np.isclose(vertices[:, 2], 0.0, atol=self.boundary_tol).astype(
            np.float32
        )[:, None]
        mask = torch.from_numpy(mask_np)

        return torch.hstack([coords, forces, mask])

    def _make_edges(self, mesh: meshio.Mesh) -> torch.Tensor:
        edge_index = []
        edge_attr = []
        edge_sets = []

        v = mesh.points
        for cell in mesh.cells:
            data = cell.data
            if "triangle" in cell.type:
                edge_sets.append(
                    np.vstack(
                        [
                            data[:, [0, 1]],
                            data[:, [1, 2]],
                            data[:, [2, 0]],
                        ]
                    )
                )
            elif "tetra" in cell.type:
                edge_sets.append(
                    np.vstack(
                        [
                            data[:, [0, 1]],
                            data[:, [0, 2]],
                            data[:, [0, 3]],
                            data[:, [1, 2]],
                            data[:, [1, 3]],
                            data[:, [2, 3]],
                        ]
                    )
                )
        if not edge_sets:
            raise ValueError("No supported cell types (tetra, triangle) found in mesh.")

        edges = np.vstack(edge_sets)
        edges.sort(axis=1)
        unique_edges = np.unique(edges, axis=0)

        src, dst = unique_edges[:, 0], unique_edges[:, 1]
        disp = v[dst] - v[src]  # shape (E, 3)
        dist = np.linalg.norm(disp, axis=1, keepdims=True)  # shape (E, 1)

        edge_index = np.hstack(
            [np.stack([src, dst], axis=0), np.stack([dst, src], axis=0)]
        )  # shape (2, 2E)
        edge_attr = np.vstack(
            [np.hstack([disp, dist]), np.hstack([-disp, dist])]
        )  # shape (2E, 4)

        return (
            torch.tensor(edge_index, dtype=torch.long),
            torch.tensor(edge_attr, dtype=torch.float32),
        )


class GraphBuilderAugment(GraphBuilderBase):
    """Graph builder that augments node features with global contact information.

    Each node is enriched with features from all contact points and their forces,
    enabling the model to learn from global context.

    Node features: [x, y, z, fx, fy, fz, cp1_rel_x, cp1_rel_y, cp1_rel_z, cf1_x, cf1_y, cf1_z, ..., is_boundary]
        - Position (x, y, z)
        - Nodal forces from gaussian load distribution (fx, fy, fz)
        - For each contact: relative displacement to contact point (cp_rel_x/y/z) and contact force (cf_x/y/z)
        - Boundary flag (1 if z ≈ 0, else 0)

    Edge features: [dx, dy, dz, distance]
        - Displacement vector and euclidean distance between nodes
    """

    def _make_nodes(
        self,
        mesh: meshio.Mesh,
        loads: list[tuple[np.ndarray, np.ndarray]],
    ) -> torch.Tensor:
        vertices = mesh.points
        num_nodes = vertices.shape[0]

        # Position Coordinates
        coords = torch.tensor(vertices, dtype=torch.float32)

        # Global Attributes
        if loads:
            loads.sort(key=lambda x: tuple(x[0]))

            Ps = []
            Fs = []
            for p, f in loads:
                Ps.append(torch.tensor(vertices - p, dtype=torch.float32))
                Fs.append(torch.tensor(np.tile(f, (num_nodes, 1)), dtype=torch.float32))

            inter = torch.stack([torch.stack(Ps), torch.stack(Fs)], dim=1).reshape(
                -1, num_nodes, 3
            )

            attrs = inter.permute(1, 0, 2).reshape(num_nodes, -1)
        else:
            attrs = torch.zeros((num_nodes, 0), dtype=torch.float32)

        # Force Vectors
        forces = self.gaussian_loads(mesh, loads)

        # Boundary Mask
        mask = torch.zeros((num_nodes, 1), dtype=torch.float32)
        mask[np.isclose(vertices[:, 2], 0.0, atol=self.boundary_tol)] = 1

        return torch.hstack([coords, forces, attrs, mask])


class GraphBuilderVirtual(GraphBuilderBase):
    """Graph builder with virtual nodes at contact points.

    Virtual nodes represent contact locations and connect to all physical nodes,
    enabling the model to tap into contact information.

    Node features: [x, y, z, fx, fy, fz, is_boundary, is_virtual]
        - Position (x, y, z)
        - Nodal forces from gaussian load distribution (fx, fy, fz)
        - Boundary flag (1 if z ≈ 0, else 0)
        - Virtual flag (1 for contact nodes, 0 for mesh nodes)

    Edge features: [dx, dy, dz, distance]
        - Displacement vector and euclidean distance between nodes
    """

    def __init__(self, std: float = 0.001):
        super().__init__(std)
        self.num_categorical = 2  # For boundary and virtual flags

    def build(
        self,
        mesh: meshio.Mesh,
        y: np.ndarray | None = None,
        contacts: list[tuple] | None = None,
    ) -> Data:
        contacts = contacts or []

        if y is None:
            y = np.zeros((mesh.points.shape[0], 4), dtype=np.float32)
        elif y.shape[0] != mesh.points.shape[0]:
            raise ValueError(
                f"Output array y must have shape [num_nodes, num_output_features], but got {y.shape} and {mesh.points.shape[0]} nodes."
            )

        # Node feature matrix with shape [num_nodes, num_node_features]
        x = self._make_nodes(mesh, contacts)

        # Pad for virtual nodes
        if contacts:
            y = np.vstack([y, np.zeros((len(contacts), y.shape[1]), dtype=np.float32)])
        y = torch.tensor(y, dtype=torch.float32)

        # Edges
        edge_index, edge_attr = self._make_edges(mesh)
        if contacts:
            v_idx, v_attr = self._make_virtual_edges(mesh, contacts)
            edge_index = torch.hstack([edge_index, v_idx])
            edge_attr = torch.vstack([edge_attr, v_attr])

        num_physical_nodes = mesh.points.shape[0]

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_physical_nodes=num_physical_nodes,
            contacts=contacts,
        )

    def _make_nodes(
        self,
        mesh: meshio.Mesh,
        loads: list[tuple[np.ndarray, np.ndarray]],
    ) -> torch.Tensor:
        vertices = mesh.points
        num_nodes = vertices.shape[0]

        # Position Coordinates
        coords = torch.tensor(vertices, dtype=torch.float32)

        # Force Vectors
        forces = self.gaussian_loads(mesh, loads)

        # Boundary Mask
        mask = torch.zeros((num_nodes, 1), dtype=torch.float32)
        mask[np.isclose(vertices[:, 2], 0.0, atol=self.boundary_tol)] = 1

        physical_nodes = torch.hstack(
            [coords, forces, mask, torch.zeros((num_nodes, 1))]
        )
        virtual_nodes = self._make_virtual_nodes(loads)
        return torch.vstack([physical_nodes, virtual_nodes])

    def _make_virtual_nodes(
        self, loads: list[tuple[np.ndarray, np.ndarray]]
    ) -> torch.Tensor:
        # Virtual nodes for contacts; # (n_v, 3)
        ps = torch.tensor(np.stack([p for p, _ in loads]), dtype=torch.float32)
        fs = torch.tensor(np.stack([f for _, f in loads]), dtype=torch.float32)
        is_boundary = (
            torch.isclose(ps[:, 2], torch.zeros(len(loads)), atol=self.boundary_tol)
            .float()
            .unsqueeze(1)
        )
        virtual_flag = torch.ones(len(loads), 1)
        return torch.cat([ps, fs, is_boundary, virtual_flag], dim=1)

    def _make_virtual_edges(
        self, mesh: meshio.Mesh, contacts: list[tuple[np.ndarray, np.ndarray]]
    ) -> torch.Tensor:
        # Create virtual edges from virtual nodes to their corresponding physical nodes
        n_phys = mesh.points.shape[0]
        n_virtual = len(contacts)
        p = torch.arange(n_phys, dtype=torch.long)
        v = torch.arange(n_phys, n_phys + n_virtual, dtype=torch.long)

        # Each virtual node connects to every physical node: (n_virtual * n_phys,)
        v_rep = v.repeat_interleave(n_phys)
        p_tiled = p.repeat(n_virtual)

        edge_index = torch.stack([v_rep, p_tiled], dim=0)  # (2, n_virtual * n_phys)

        # Match base edge features: [dx, dy, dz, distance] for directed edges.
        phys_coords = torch.as_tensor(mesh.points, dtype=torch.float32)
        virt_coords = torch.from_numpy(np.stack([p for p, _ in contacts])).float()
        all_coords = torch.vstack([phys_coords, virt_coords])

        src, dst = edge_index
        disp = all_coords[dst] - all_coords[src]
        dist = torch.norm(disp, dim=1, keepdim=True)
        edge_attr = torch.hstack([disp, dist])

        return edge_index, edge_attr


class GraphVisualizer:
    def __init__(self, mesh: trimesh.Trimesh, jupyter_backend: bool = True):
        self.mesh = mesh
        self.pv_mesh = pv.wrap(mesh)
        self.jupyter_backend = jupyter_backend

    @staticmethod
    def _default_scalar_bar_args() -> dict:
        return {
            "vertical": True,
            "position_x": 0.84,
            "position_y": 0.1,
            "width": 0.08,
            "height": 0.8,
        }

    @staticmethod
    def _graph_contacts(graph: Data) -> list[tuple[np.ndarray, np.ndarray]]:
        contacts = getattr(graph, "contacts", None)
        if contacts is None:
            return []
        return list(contacts)

    def _mesh_scale(self, ratio: float = 0.1) -> float:
        x_min, x_max, y_min, y_max, z_min, z_max = self.pv_mesh.bounds
        return max(x_max - x_min, y_max - y_min, z_max - z_min) * ratio

    def _add_contact_vectors(
        self,
        plotter: pv.Plotter,
        contacts: list[tuple[np.ndarray, np.ndarray]],
        arrow_scale: float,
        sphere_radius: float | None = None,
        debug: bool = False,
    ) -> None:
        for point, force in contacts:
            if debug:
                print(f"Contact point: {point}, Force: {force}")

            if sphere_radius is not None:
                sphere = pv.Sphere(radius=sphere_radius)
                sph = sphere.translate(point, inplace=False)
                plotter.add_mesh(sph, color="red", opacity=1)

            arrow = pv.Arrow(
                start=np.asarray(point), direction=np.asarray(force), scale=arrow_scale
            )
            plotter.add_mesh(arrow, color="red")

    @staticmethod
    def _compute_von_mises(tensor: torch.Tensor) -> torch.Tensor:
        s_xx, s_yy, s_zz = tensor[:, 0], tensor[:, 1], tensor[:, 2]
        t_xy, t_yz, t_zx = tensor[:, 3], tensor[:, 4], tensor[:, 5]

        return torch.sqrt(
            0.5
            * (
                (s_xx - s_yy) ** 2
                + (s_yy - s_zz) ** 2
                + (s_zz - s_xx) ** 2
                + 6 * (t_xy**2 + t_yz**2 + t_zx**2)
            )
        )

    def compute_von_mises(self, graph: Data) -> torch.Tensor:
        """Compute von Mises stress from the stress tensor in graph.y."""
        n_phys = graph.num_physical_nodes
        stress = graph.y[:n_phys, 3:9]
        return self._compute_von_mises(stress)

    @staticmethod
    def _save_html_or_show(plotter: pv.Plotter, save_path: str | Path | None) -> None:
        if save_path is not None:
            plotter.export_html(str(save_path))
        else:
            plotter.show()

    def plot_field(
        self,
        graph: Data,
        field_data: torch.Tensor,
        field_name: str,
        save_path: str | Path | None = None,
        cmap: str = "Oranges",
        clim: tuple | None = None,
        scalar_bar_args: dict | None = None,
        show_contacts: bool = True,
        clip_bottom: bool = False,
        debug: bool = False,
    ):
        """Unified plotting pipeline for scalar or vector fields on graph nodes."""
        pv_mesh = self.pv_mesh.copy()
        pv_mesh.point_data[field_name] = field_data

        if clip_bottom:
            pv_mesh = self.pv_mesh.clip(normal=(0, 0, 1), origin=(0, 0, 1e-6))

        # Configure PyVista plotter
        plotter = pv.Plotter(notebook=self.jupyter_backend, off_screen=True)
        plotter.add_mesh(
            pv_mesh,
            scalars=field_name,
            point_size=1,
            render_points_as_spheres=True,
            show_edges=True,
            clim=clim,
            scalar_bar_args=scalar_bar_args or self._default_scalar_bar_args(),
            cmap=cmap,
        )

        # Optional contacts rendering
        if show_contacts:
            contacts = self._graph_contacts(graph)
            if len(contacts) > 0:
                scale = self._mesh_scale(ratio=0.1)
                self._add_contact_vectors(
                    plotter,
                    contacts=contacts,
                    arrow_scale=scale,
                    sphere_radius=scale * 0.1,
                    debug=debug,
                )

        plotter.show_axes()
        self._save_html_or_show(plotter, save_path=save_path)
        return plotter

    def von_mises(
        self,
        graph: Data,
        save_path: str | None = None,
        cmap: str = "Oranges",
        clim: tuple | None = None,
        scalar_bar_args: dict | None = None,
        show_contacts: bool = False,
        clip_bottom: bool = False,
        debug: bool = False,
    ):
        n_phys = graph.num_physical_nodes
        stress = graph.y[:n_phys, 3:9]
        vm = self._compute_von_mises(stress).detach().cpu().numpy()
        return self.plot_field(
            graph=graph,
            field_data=vm,
            field_name="von Mises [Pa]",
            save_path=save_path,
            cmap=cmap,
            clim=clim,
            scalar_bar_args=scalar_bar_args,
            show_contacts=show_contacts,
            clip_bottom=clip_bottom,
            debug=debug,
        )

    def displacement(
        self,
        graph: Data,
        cmap: str = "Oranges",
        clim: tuple | None = None,
        show_contacts: bool = False,
        save_path: str | Path | None = None,
        scalar_bar_args: dict | None = None,
        scale: float = 1e9,
        debug: bool = False,
    ):
        n_phys = graph.num_physical_nodes
        clim = (clim[0] * scale, clim[1] * scale) if clim is not None else None
        disp = graph.y.detach().cpu().numpy()[:n_phys, :3] * scale
        return self.plot_field(
            graph=graph,
            field_data=disp,
            field_name=f"displacement [{1 / scale:.1e} m]",
            save_path=save_path,
            cmap=cmap,
            clim=clim,
            scalar_bar_args=scalar_bar_args,
            show_contacts=show_contacts,
            debug=debug,
        )

    def force(
        self,
        graph: Data,
        cmap: str = "Oranges",
        save_path: str | Path | None = None,
        scalar_bar_args: dict | None = None,
        debug: bool = False,
    ):
        n_phys = graph.num_physical_nodes
        force = graph.x[:n_phys, 3:6].norm(dim=1).detach().cpu().numpy()[:n_phys]
        return self.plot_field(
            graph=graph,
            field_data=force,
            field_name="force",
            save_path=save_path,
            cmap=cmap,
            scalar_bar_args=scalar_bar_args,
            show_contacts=True,
            debug=debug,
        )
