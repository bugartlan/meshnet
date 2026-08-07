from pathlib import Path

import numpy as np
import pyvista as pv
import torch
import trimesh
from torch_geometric.data import Data

from meshnet.utils.math import calculate_von_mises


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
        vm = calculate_von_mises(stress).detach().cpu().numpy()
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
            field_name="force [N]",
            save_path=save_path,
            cmap=cmap,
            scalar_bar_args=scalar_bar_args,
            show_contacts=True,
            debug=debug,
        )
