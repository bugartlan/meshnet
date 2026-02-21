import meshio
import numpy as np
import pyvista as pv
import torch
import trimesh
from torch_geometric.data import Data


class GraphBuilder:
    def __init__(self, std: float = 0.001):
        self.std = std
        self.boundary_tol = 1e-6

    def build(
        self,
        mesh: meshio.Mesh,
        y: np.ndarray,
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

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def gaussian_loads(self, coords: np.ndarray, contacts: list[tuple]) -> float:
        num_nodes = coords.shape[0]
        nodal_forces = np.zeros((num_nodes, 3), dtype=np.float32)

        if not contacts:
            return nodal_forces

        for point, force in contacts:
            point = np.asarray(point)  # (3,)
            force = np.asarray(force)  # (3, )

            dist_sq = np.sum((coords - point) ** 2, axis=1)
            weights = np.exp(-dist_sq / (2 * self.std**2))
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                weights /= weights_sum

            nodal_forces += weights[:, None] * force

        return torch.tensor(nodal_forces, dtype=torch.float32)

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
        forces = self.gaussian_loads(vertices, loads)

        # Boundary Mask
        mask = torch.zeros((num_nodes, 1), dtype=torch.float32)
        mask[np.isclose(vertices[:, 2], 0.0, atol=self.boundary_tol)] = 1

        return torch.hstack([coords, forces, attrs, mask])

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
                raise ValueError(
                    "No supported cell types (tetra, triangle) found in mesh."
                )
        edges = np.vstack(edge_sets)
        edges.sort(axis=1)
        unique_edges = np.unique(edges, axis=0)

        src, dst = unique_edges[:, 0], unique_edges[:, 1]
        disp = v[dst] - v[src]  # shape (E, 3)
        dist = np.linalg.norm(disp, axis=1, keepdims=True)  # shape (E, 1)

        # Indices
        edge_index_fwd = np.stack([src, dst], axis=0)
        edge_index_bwd = np.stack([dst, src], axis=0)
        edge_index = np.hstack([edge_index_fwd, edge_index_bwd])  # shape (2, 2E)

        # Attributes
        attr_fwd = np.hstack([disp, dist])  # shape (E, 4)
        attr_bwd = np.hstack([-disp, dist])  # shape (E, 4)
        edge_attr = np.vstack([attr_fwd, attr_bwd])  # shape (2E, 4)

        return (
            torch.tensor(edge_index, dtype=torch.long),
            torch.tensor(edge_attr, dtype=torch.float32),
        )


class GraphBuilderLocal(GraphBuilder):
    def build(
        self,
        mesh: meshio.Mesh,
        y: np.ndarray,
        contacts: list[tuple] | None = None,
    ) -> Data:
        if y.shape[0] != mesh.points.shape[0]:
            raise ValueError(
                f"Output array y must have shape [num_nodes, num_output_features], but got {y.shape} and {mesh.points.shape[0]} nodes."
            )

        # Node feature matrix with shape [num_nodes, num_node_features]
        x = self._make_nodes(mesh, contacts)
        y = torch.tensor(y, dtype=torch.float32)
        edge_index, edge_attr = self._make_edges(mesh, x)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def _make_nodes(
        self,
        mesh: meshio.Mesh,
        loads: list[tuple[np.ndarray, np.ndarray]],
    ) -> torch.Tensor:
        vertices = mesh.points
        num_nodes = vertices.shape[0]

        loads.sort(key=lambda x: tuple(x[0]))

        # Position Coordinates
        coords = torch.tensor(vertices, dtype=torch.float32)

        # Force Vectors
        forces = self.gaussian_loads(vertices, loads)

        # Boundary Mask
        mask = torch.zeros((num_nodes, 1), dtype=torch.float32)
        mask[np.isclose(vertices[:, 2], 0.0, atol=self.boundary_tol)] = 1

        nodes = torch.hstack(
            [coords, forces, mask, torch.zeros((num_nodes, 1), dtype=torch.float32)]
        )

        # Virtual Nodes for Contacts
        ps = torch.tensor([p for p, _ in loads], dtype=torch.float32)  # (n_v, 3)
        fs = torch.tensor([f for _, f in loads], dtype=torch.float32)  # (n_v, 3)
        is_boundary = (
            torch.isclose(ps[:, 2], torch.zeros(len(loads)), atol=self.boundary_tol)
            .float()
            .unsqueeze(1)
        )
        virtual_flag = torch.ones(len(loads), 1)
        virtual_nodes = torch.cat([ps, fs, is_boundary, virtual_flag], dim=1)

        return torch.vstack([nodes, virtual_nodes])

    def _make_virtual_edges(self, nodes: torch.Tensor) -> torch.Tensor:
        # Create virtual edges from virtual nodes to their corresponding physical nodes
        virtual_node_indices = torch.where(nodes[:, -1] == 1)[0]
        physical_node_indices = torch.where(nodes[:, -1] == 0)[0]

        n_v = len(virtual_node_indices)
        n_p = len(physical_node_indices)

        # Each virtual node connects to every physical node
        v_repeated = virtual_node_indices.repeat_interleave(n_p)  # (n_v * n_p,)
        p_tiled = physical_node_indices.repeat(n_v)  # (n_v * n_p,)

        edge_index_fwd = torch.stack([v_repeated, p_tiled], dim=0)
        edge_index_bwd = torch.stack([p_tiled, v_repeated], dim=0)
        edge_index = torch.cat(
            [edge_index_fwd, edge_index_bwd], dim=1
        )  # (2, 2*n_v*n_p)

        return edge_index, torch.zeros((edge_index.shape[1], 4), dtype=torch.float32)

    def _make_edges(self, mesh: meshio.Mesh, nodes: torch.Tensor) -> torch.Tensor:
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
                raise ValueError(
                    "No supported cell types (tetra, triangle) found in mesh."
                )
        edges = np.vstack(edge_sets)
        edges.sort(axis=1)
        unique_edges = np.unique(edges, axis=0)

        src, dst = unique_edges[:, 0], unique_edges[:, 1]
        disp = v[dst] - v[src]  # shape (E, 3)
        dist = np.linalg.norm(disp, axis=1, keepdims=True)  # shape (E, 1)

        # Indices
        edge_index_fwd = np.stack([src, dst], axis=0)
        edge_index_bwd = np.stack([dst, src], axis=0)
        edge_index = np.hstack([edge_index_fwd, edge_index_bwd])  # shape (2, 2E)

        # Attributes
        attr_fwd = np.hstack([disp, dist])  # shape (E, 4)
        attr_bwd = np.hstack([-disp, dist])  # shape (E, 4)
        edge_attr = np.vstack([attr_fwd, attr_bwd])  # shape (2E, 4)

        # Add virtual edges
        virtual_edge_index, virtual_edge_attr = self._make_virtual_edges(nodes)

        edge_index = np.hstack([edge_index, virtual_edge_index.numpy()])
        edge_attr = np.vstack([edge_attr, virtual_edge_attr.numpy()])

        return (
            torch.tensor(edge_index, dtype=torch.long),
            torch.tensor(edge_attr, dtype=torch.float32),
        )


class GraphVisualizer:
    def __init__(self, mesh: trimesh.Trimesh, jupyter_backend: bool = True):
        self.mesh = mesh
        self.pv_mesh = pv.wrap(mesh)
        self.jupyter_backend = jupyter_backend

    def stress(
        self,
        graph: Data,
        clim: tuple = None,
        save_path: str | None = None,
        debug: bool = False,
    ):
        self.pv_mesh.point_data["von_mises"] = graph.y.numpy()[:, 3]

        plotter = pv.Plotter(notebook=self.jupyter_backend)
        plotter.add_mesh(
            self.pv_mesh,
            scalars="von_mises",
            point_size=1,
            render_points_as_spheres=True,
            show_edges=True,
            clim=clim,
        )

        x_min, x_max, y_min, y_max, z_min, z_max = self.pv_mesh.bounds
        scale = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.1

        x = graph.x[0].cpu().numpy()
        contacts = x[6:-1].reshape(-1, 6)
        contacts[:, :3] -= x[:3]
        contacts[:, :3] *= -1
        for v in contacts:
            p = v[:3]
            f = v[3:]
            if debug:
                print(f"Contact point: {p}, Force: {f}")
            # Visualize the contact point
            sphere = pv.Sphere(radius=scale * 0.1)
            sph = sphere.translate(p, inplace=False)
            plotter.add_mesh(sph, color="red", opacity=1)

            # Visualize the force arrow
            arrow = pv.Arrow(start=np.asarray(p), direction=f, scale=scale)
            plotter.add_mesh(arrow, color="red")

        plotter.show_axes()
        if save_path is not None:
            plotter.export_html(save_path)
        else:
            plotter.show()

    def displacement(
        self,
        graph: Data,
        clim: tuple = None,
        save_path: str | None = None,
        debug: bool = False,
    ):
        self.pv_mesh.point_data["displacement"] = graph.y.numpy()[:, :3]

        plotter = pv.Plotter(notebook=self.jupyter_backend)
        plotter.add_mesh(
            self.pv_mesh,
            scalars="displacement",
            point_size=1,
            render_points_as_spheres=True,
            show_edges=True,
            clim=clim,
        )

        plotter.show_axes()
        if save_path is not None:
            plotter.export_html(save_path)
        else:
            plotter.show()

    def bottom(self, graph: Data, clim: tuple = None, save_path: str | None = None):
        # First, add von_mises to the full mesh
        self.pv_mesh.point_data["von_mises"] = graph.y.numpy()[:, 3]

        # Then clip the mesh with the data already assigned
        pv_mesh_boundary = self.pv_mesh.clip(normal=(0, 0, 1), origin=(0, 0, 1e-6))

        plotter = pv.Plotter(notebook=self.jupyter_backend)
        plotter.add_mesh(
            pv_mesh_boundary,
            scalars="von_mises",
            point_size=1,
            render_points_as_spheres=True,
            show_edges=True,
            clim=clim,
        )
        plotter.show_axes()
        if save_path is not None:
            plotter.export_html(save_path)
        else:
            plotter.show()

    def force(self, graph: Data):
        self.pv_mesh.point_data["force_magnitude"] = graph.x[:, 3:6].norm(dim=1).numpy()

        plotter = pv.Plotter(notebook=self.jupyter_backend)
        plotter.add_mesh(
            self.pv_mesh,
            scalars="force_magnitude",
            point_size=1,
            render_points_as_spheres=True,
            show_edges=True,
        )

        plotter.show_axes()
        plotter.show()

    def boundary(self, graph: Data):
        self.pv_mesh.point_data["is_boundary"] = graph.x[:, -1].numpy()

        plotter = pv.Plotter(notebook=self.jupyter_backend)
        plotter.add_mesh(
            self.pv_mesh,
            scalars="is_boundary",
            point_size=1,
            render_points_as_spheres=True,
            show_edges=True,
        )

        plotter.show_axes()
        plotter.show()
