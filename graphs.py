"""
Graph representation of a mesh under contact loads.

Node Features (x):
    - Position (p): 3D world coordinates [x, y, z]
    - Force (f): Applied external force vector [fx, fy, fz]
    - Type (m): Node classification with a boolean mask
        * Root node (fixed boundary): 1
        * Mesh node (free to move): 0

Edge Features (e):
    - Displacement (Δp): Relative vector from source to target node [Δx, Δy, Δz]
    - Distance (d): Euclidean distance ||Δp||
"""

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data

tol_dirichlet = 1e-3


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

    return {i: f for i, f in zip(vertices, forces)}


def build_graph(
    mesh: trimesh.Trimesh, y: np.ndarray, contacts: list[tuple] | None = None
) -> Data:
    # Node feature matrix with shape [num_nodes, num_node_features]
    x = make_nodes(mesh, project_contacts(mesh, contacts))
    y = torch.tensor(y)
    edge_index, edge_attr = make_edges(mesh)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def make_nodes(mesh: trimesh.Trimesh, loads: dict[int, np.ndarray]):
    v = mesh.vertices
    n = v.shape[0]

    # Cartesian coords
    coords = torch.tensor(v)

    # Forces
    forces = torch.zeros((n, 3))
    for idx, f in loads.items():
        forces[idx] = torch.tensor(f)

    # Boundary mask
    mask = torch.zeros((n, 1))
    mask[np.isclose(v[:, 2], 0.0, atol=tol_dirichlet)] = 1

    return torch.hstack([coords, forces, mask])


def make_edges(mesh: trimesh.Trimesh) -> torch.Tensor:
    edge_index = []
    edge_attr = []

    v = mesh.vertices
    for i, j in mesh.edges_unique:
        disp = v[j] - v[i]
        dist = np.linalg.norm(disp)
        edge_index.extend([(i, j), (j, i)])
        edge_attr.extend([np.hstack((disp, dist)), np.hstack((-disp, dist))])

    edge_index = np.asarray(edge_index).T
    edge_attr = np.asarray(edge_attr)
    return torch.tensor(edge_index), torch.tensor(edge_attr)
