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


def visualize(
    mesh: trimesh.Trimesh,
    graph: Data,
    jupyter_backend: str = None,
    force_arrows=False,
    show: bool = True,
    filename: str = None,
):
    stress = graph.y[:, 0].cpu().numpy().squeeze()

    tree = cKDTree(mesh.vertices)
    _, idx = tree.query(mesh.vertices)

    stress_mesh = stress[idx]
    pv_mesh = pyvista.wrap(mesh)
    pv_mesh["VonMisesStress"] = stress_mesh

    plotter = pyvista.Plotter(notebook=jupyter_backend is not None)
    plotter.add_mesh(
        pv_mesh,
        scalars="VonMisesStress",
        point_size=5,
        render_points_as_spheres=True,
        show_edges=True,
    )

    if force_arrows:
        forces = find_contacts(graph)
        for coord, force in forces.items():
            arrow = pyvista.Arrow(
                start=np.asarray(coord),
                direction=force,
                scale=2.0,
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
