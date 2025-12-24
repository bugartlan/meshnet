import numpy as np
import pyvista
import trimesh
from scipy.spatial import cKDTree
from torch_geometric.data import Data, HeteroData


def find_contact_nodes(graph: HeteroData) -> dict[np.ndarray, np.ndarray]:
    forces = dict()
    edge_index = graph["node", "contact", "node"].edge_index.cpu().numpy()

    for i in range(edge_index.shape[1]):
        v1, v2 = edge_index[:, i]
        coord1 = tuple(graph["node"].x[v1][3:6].tolist())
        coord2 = tuple(graph["node"].x[v2][3:6].tolist())
        f1 = graph["node"].x[v1][6:9].cpu().numpy()
        f2 = graph["node"].x[v2][6:9].cpu().numpy()
        if np.linalg.norm(f1) > 0 and coord1 not in forces:
            forces[coord1] = f1
        elif np.linalg.norm(f2) > 0 and coord2 not in forces:
            forces[coord2] = f2

    return forces


# def visualize(
#     mesh: trimesh.Trimesh,
#     graph: HeteroData,
#     scale: float = 1.0,
#     jupyter_backend: str = None,
# ):
#     forces = find_contact_nodes(graph)

#     coords = graph["node"].x[:, 3:6].cpu().numpy() * scale
#     stress = graph["node"].y[:, 0].cpu().numpy().squeeze()

#     tree = cKDTree(coords)
#     _, idx = tree.query(mesh.vertices)

#     stress_mesh = stress[idx]
#     pv_mesh = pyvista.wrap(mesh)
#     pv_mesh["VonMisesStress"] = stress_mesh

#     plotter = pyvista.Plotter(notebook=jupyter_backend is not None)
#     plotter.add_mesh(
#         pv_mesh,
#         scalars="VonMisesStress",
#         point_size=5,
#         render_points_as_spheres=True,
#         show_edges=True,
#     )
#     for coord, force in forces.items():
#         arrow = pyvista.Arrow(
#             start=np.asarray(coord) * scale,
#             direction=force / np.linalg.norm(force),
#             scale=2.0,
#         )
#         plotter.add_mesh(arrow, color="red")
#     plotter.show_axes()
#     plotter.show(jupyter_backend=jupyter_backend)


def info(graph: HeteroData, debug=False):
    graph.validate(raise_on_error=True)

    node_dim = graph.num_node_features["node"]
    mesh_edge_dim = graph.num_edge_features["node", "mesh", "node"]
    contact_edge_dim = graph.num_edge_features["node", "contact", "node"]
    output_dim = graph["node"].y.shape[1]

    if debug:
        print("Node feature dim:", node_dim)
        print("Mesh edge feature dim:", mesh_edge_dim)
        print("Contact edge feature dim:", contact_edge_dim)
        print("Output dim:", output_dim)

        print("Keys:", graph.keys())
        print("Number of nodes:", graph.num_nodes)
        print("Number of edges:", graph.num_edges)
        print(
            "Number of mesh edges:", graph["node", "mesh", "node"].edge_index.shape[1]
        )
        print(
            "Number of contact edges:",
            graph["node", "contact", "node"].edge_index.shape[1],
        )

    return (
        node_dim,
        mesh_edge_dim,
        contact_edge_dim,
        output_dim,
    )


def visualize(mesh: trimesh.Trimesh, graph: Data, jupyter_backend: str = None):
    coords = graph.x[:, 3:6].cpu().numpy()
    stress = graph.y[:, 0].cpu().numpy().squeeze()

    tree = cKDTree(coords)
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
    plotter.show_axes()
    plotter.show(jupyter_backend=jupyter_backend)
