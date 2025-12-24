import time

import numpy as np
import torch
import trimesh
from torch_geometric.data import HeteroData

from fem import eval, fea


def is_root_node(vertex: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    return np.isclose(vertex[:, 2], 0.0, atol=tol)


def project_contacts(
    mesh: trimesh.Trimesh,
    contacts: list[tuple] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Project contact points onto the mesh and distributes forces barycentrically to the vertices.
    """
    contact_forces = {}

    if contacts is None or len(contacts) == 0:
        return np.array([], dtype=np.int64), contact_forces

    points, forces = zip(*contacts)
    points = np.asarray(points).reshape(-1, 3)
    forces = np.asarray(forces).reshape(-1, 3)

    # Find closest surface points and triangles
    closest, distance, triangle_ids = trimesh.proximity.closest_point(mesh, points)

    if np.any(distance > 1e-5):
        raise ValueError("At least one contact point is not on the mesh surface.")

    # Compute Barycentric weights
    # Shape: (N, 3, 3)
    triangle_coords = mesh.vertices[mesh.faces[triangle_ids]]
    barycentric_weights = trimesh.triangles.points_to_barycentric(
        triangle_coords, closest
    )

    # Distribute forces vectors: Force * Weight
    # Shape: (N, 3, 3) -> (Contact, Vertex, Force)
    forces = (forces[:, None, :] * barycentric_weights[:, :, None]).reshape(-1, 3)
    vertices = mesh.faces[triangle_ids].flatten()

    return vertices, {i: f for i, f in zip(vertices, forces)}


def build_graph(
    mesh: trimesh.Trimesh, domain, func, contacts: list[tuple] | None = None
) -> HeteroData:
    graph = HeteroData()

    contact_vertex_indices, contact_forces = project_contacts(mesh, contacts)

    graph["node"].x = build_node_features(mesh, contact_vertex_indices, contact_forces)
    graph["node"].y = build_node_labels(mesh, graph["node"].x, domain, func)

    # Build edges
    mesh_edge_index, mesh_edge_attr, contact_edge_index, contact_edge_attr = (
        build_edges(mesh, contact_vertex_indices, contact_forces)
    )

    graph["node", "mesh", "node"].edge_index = mesh_edge_index
    graph["node", "mesh", "node"].edge_attr = mesh_edge_attr
    graph["node", "contact", "node"].edge_index = contact_edge_index
    graph["node", "contact", "node"].edge_attr = contact_edge_attr

    return graph


def build_node_features(
    mesh: trimesh.Trimesh,
    contact_points: np.ndarray,
    contact_forces: dict[int, np.ndarray],
) -> torch.Tensor:
    N = mesh.vertices.shape[0]

    contact_points = sorted(contact_points.tolist())

    root_mask = is_root_node(mesh.vertices)
    contact_mask = np.zeros(N, dtype=bool)
    contact_mask[contact_points] = True
    mesh_mask = ~(root_mask | contact_mask)

    types = np.zeros((N, 3))
    types[root_mask, 0] = 1.0
    types[mesh_mask, 1] = 1.0
    types[contact_mask, 2] = 1.0

    forces = np.zeros((N, 3))
    for idx, f in contact_forces.items():
        forces[idx] = f

    features = np.hstack([types, mesh.vertices, forces])

    print(
        f"Total nodes: {N}, "
        f"Root nodes: {np.sum(root_mask)}, "
        f"Mesh nodes: {np.sum(mesh_mask)}, "
        f"Contact nodes: {np.sum(contact_mask)}"
    )

    return torch.tensor(features, dtype=torch.float32)


def build_node_labels(
    mesh: trimesh.Trimesh, nodes: torch.tensor, domain, func
) -> torch.Tensor:
    coords = nodes[:, 3:6].cpu().numpy()
    labels = eval(domain, func, coords)
    return torch.tensor(labels, dtype=torch.float32)


def build_edges(
    mesh: trimesh.Trimesh,
    contact_points: np.ndarray,
    contact_forces: dict[int, np.ndarray],
) -> torch.Tensor:
    contact_vertex_set = set(contact_points.tolist())

    contact_edge_index = []
    contact_edge_attr = []
    mesh_edge_index = []
    mesh_edge_attr = []

    for v1, v2 in mesh.edges_unique:
        disp = mesh.vertices[v2] - mesh.vertices[v1]
        dist = np.linalg.norm(disp)
        if v1 in contact_vertex_set:
            contact_edge_index.append((v1, v2))
            force = contact_forces[v1]
            contact_edge_attr.append(np.hstack([disp, dist, force]))
        elif v2 in contact_vertex_set:
            contact_edge_index.append((v2, v1))
            force = contact_forces[v2]
            contact_edge_attr.append(np.hstack([-disp, dist, force]))
        mesh_edge_index.extend([(v1, v2), (v2, v1)])
        mesh_edge_attr.extend([np.hstack([disp, dist]), np.hstack([-disp, dist])])

    def to_edge_tensor(edges):
        if len(edges) == 0:
            return np.empty((2, 0), dtype=np.int64)
        return torch.tensor(np.array(edges).T, dtype=torch.int64)

    def to_attr_tensor(attrs, dim):
        if len(attrs) == 0:
            return torch.empty((0, dim), dtype=torch.float32)
        return torch.tensor(np.array(attrs), dtype=torch.float32)

    return (
        to_edge_tensor(mesh_edge_index),
        to_attr_tensor(mesh_edge_attr, 4),
        to_edge_tensor(contact_edge_index),
        to_attr_tensor(contact_edge_attr, 7),
    )


def generate_data_one(
    mesh: trimesh.Trimesh,
    max_f: float,
    n: int,
    stl="cantilever.stl",
    msh="cantilever.msh",
    output_filename="cantilever_data.pt",
) -> list[HeteroData]:
    """Generate n graphs from random contact points on one given mesh."""
    graphs = []
    for i in range(n):
        print(f"Generating graph {i + 1}/{n}...")
        # Random contact point on the mesh surface
        contact_point, face_id = trimesh.sample.sample_surface(mesh, count=1)
        contact_point = contact_point[0]
        contact_force = np.random.uniform(-max_f, max_f, size=(3,))
        contacts = [(contact_point, contact_force)]

        domain, stresses_vm = fea(
            contacts,
            contact_radius=2.0,
            debug=False,
            filename_stl=stl,
            filename_msh=msh,
        )
        graph = build_graph(mesh, domain, stresses_vm, contacts=contacts)
        graphs.append(graph)

    torch.save(graphs, output_filename)
    print(f"Saved dataset with {len(graphs)} graphs to {output_filename}.")
    return graphs


def generate_data_from_list(
    mesh: trimesh.Trimesh,
    f: np.ndarray,
    points: list[int],
    stl="cantilever.stl",
    msh="cantilever.msh",
    output_filename="cantilever_0.pt",
) -> list[HeteroData]:
    """Generate n graphs from given contact points on one given mesh."""
    graphs = []
    points = np.asarray(points).reshape(-1, 3)
    for i in range(len(points)):
        print(f"Generating graph {i + 1}/{len(points)}...")
        contact_point = points[i]
        contacts = [(contact_point, f)]

        domain, stresses_vm = fea(
            contacts,
            contact_radius=2.0,
            debug=False,
            filename_stl=stl,
            filename_msh=msh,
        )
        graph = build_graph(mesh, domain, stresses_vm, contacts=contacts)
        graphs.append(graph)

    torch.save(graphs, output_filename)
    print(f"Saved dataset with {len(graphs)} graphs to {output_filename}.")
    return graphs


if __name__ == "__main__":
    filename = "cantilever"
    input_dir = "meshes/"
    output_dir = "data/"
    stl = f"{input_dir}{filename}.stl"
    msh = f"{input_dir}{filename}.msh"
    F = 10.0
    N = 2

    start = time.time()

    # Load mesh and generate dataset
    mesh = trimesh.load_mesh(stl)
    # generate_data_one(
    #     mesh,
    #     max_f=F,
    #     n=N,
    #     stl=stl,
    #     msh=msh,
    #     output_filename=f"{output_dir}{filename}_{N}.pt",
    # )

    generate_data_from_list(
        mesh,
        f=np.array([F, 0.0, 0.0]),
        points=[(0.0, 5.0, 10.0), (0.0, 5.0, 20.0)],
        stl=stl,
        msh=msh,
        output_filename=f"{output_dir}{filename}_2.pt",
    )

    end = time.time()
    print(f"Data generation took {end - start:.2f} seconds.")
