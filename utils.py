import meshio
import numpy as np
import pyvista
import torch
import trimesh
from scipy.spatial import cKDTree
from torch_geometric.data import Data, HeteroData


def info(graph: HeteroData, debug=False):
    graph.validate(raise_on_error=True)

    node_dim = graph.num_node_features
    edge_dim = graph.num_edge_features
    output_dim = graph.y.shape[1]

    if debug:
        print("Node feature dim:", node_dim)
        print("Edge feature dim:", edge_dim)
        print("Output dim:", output_dim)

        print("Keys:", graph.keys())
        print("Number of nodes:", graph.num_nodes)
        print("Number of edges:", graph.num_edges)

    return node_dim, edge_dim, output_dim


def find_contacts(graph: Data) -> dict[np.ndarray, np.ndarray]:
    forces = dict()
    for x in graph.x:
        coord = x[:3].cpu().numpy()
        force = x[3:6].cpu().numpy()
        if np.linalg.norm(force) > 1e-6:
            forces[tuple(coord)] = force

    return forces


def make_pv_mesh(mesh: trimesh.Trimesh, graph: Data, labels: list) -> pyvista.PolyData:
    pv_mesh = pyvista.wrap(mesh)
    tree = cKDTree(mesh.vertices)
    _, idx = tree.query(mesh.vertices)
    for i, label in enumerate(labels):
        values = graph.y[:, i].cpu().numpy().squeeze()
        values_mesh = values[idx]
        pv_mesh[label] = values_mesh
    return pv_mesh


def visualize_graph(
    pv_mesh: pyvista.PolyData,
    graph: Data,
    label: str,
    jupyter_backend: str = None,
    force_arrows: bool = False,
    show: bool = True,
    filename: str = None,
    clim: tuple = None,
):
    x_min, x_max, y_min, y_max, z_min, z_max = pv_mesh.bounds
    scale = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.1

    plotter = pyvista.Plotter(notebook=jupyter_backend is not None)
    if label not in pv_mesh.array_names:
        raise ValueError(f"Label '{label}' not found in mesh array names.")

    plotter.add_mesh(
        pv_mesh,
        scalars=label,
        point_size=1,
        render_points_as_spheres=True,
        show_edges=True,
        clim=clim,
    )

    if force_arrows:
        forces = find_contacts(graph)
        for coord, force in forces.items():
            arrow = pyvista.Arrow(
                start=np.asarray(coord),
                direction=force,
                scale=scale,
            )
            plotter.add_mesh(arrow, color="red")

    plotter.show_axes()
    if show:
        plotter.show(jupyter_backend=jupyter_backend)
    if filename is not None:
        plotter.export_html(filename)


def visualize(
    mesh: trimesh.Trimesh,
    graph: Data,
    v_idx: int = 0,
    label: str = "VonMisesStress",
    jupyter_backend: str = None,
    force_arrows=False,
    show: bool = True,
    filename: str = None,
):
    values = graph.y[:, v_idx].cpu().numpy().squeeze()

    tree = cKDTree(mesh.vertices)
    _, idx = tree.query(mesh.vertices)

    values_mesh = values[idx]
    pv_mesh = pyvista.wrap(mesh)
    pv_mesh[label] = values_mesh

    scale = mesh.extents.max() * 0.1
    plotter = pyvista.Plotter(notebook=jupyter_backend is not None)
    plotter.add_mesh(
        pv_mesh,
        scalars=label,
        point_size=1,
        render_points_as_spheres=True,
        show_edges=True,
    )

    if force_arrows:
        forces = find_contacts(graph)
        for coord, force in forces.items():
            arrow = pyvista.Arrow(
                start=np.asarray(coord),
                direction=force,
                scale=scale,
            )
            plotter.add_mesh(arrow, color="red")

    plotter.show_axes()

    if filename is not None:
        plotter.export_html(filename)

    if show:
        plotter.show(jupyter_backend=jupyter_backend)


def msh_to_trimesh(mesh: meshio.Mesh) -> trimesh.Trimesh:
    """
    Convert a meshio.Mesh to a trimesh.Trimesh object.
    """
    triangles = [c.data for c in mesh.cells if c.type == "triangle"]
    faces = np.vstack(triangles)
    return trimesh.Trimesh(vertices=mesh.points, faces=faces, process=False)


def normalize(graph, stats):
    """Normalize graph features using provided statistics."""

    g = graph.clone()
    # Handle std = 0 case by replacing with 1 (skips normalization for constant features)
    x_std_safe = torch.where(
        stats["x_std"][:-1] > 0,
        stats["x_std"][:-1],
        torch.ones_like(stats["x_std"][:-1]),
    )
    y_std_safe = torch.where(
        stats["y_std"] > 0, stats["y_std"], torch.ones_like(stats["y_std"])
    )
    e_std_safe = torch.where(
        stats["e_std"] > 0, stats["e_std"], torch.ones_like(stats["e_std"])
    )

    g.x[:, :-1] = (g.x[:, :-1] - stats["x_mean"][:-1]) / x_std_safe
    g.y = (g.y - stats["y_mean"]) / y_std_safe
    g.edge_attr = (g.edge_attr - stats["e_mean"]) / e_std_safe
    return g
