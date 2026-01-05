"""
Graph representation of a mesh under contact loads.

Node Features (x):
    - Position (p): 3D world coordinates [x, y, z]
    - Force (f): Applied external force vector [fx, fy, fz]
    - Relative Position 1 (rp1): Vector from a reference point [rpx, rpy, rpz]
    - Reference Force 1 (rf1): Reference force vector at the reference point [rfx, rfy, rfz]
    - Relative Position 2 (rp2): Vector from a reference point [rpx, rpy, rpz]
    - Reference Force 2 (rf2): Reference force vector at the reference point [rfx, rfy, rfz]
    - Type (m): Node classification with a boolean mask
        * Root node (fixed boundary): 1
        * Mesh node (free to move): 0

Edge Features (e):
    - Displacement (Δp): Relative vector from source to target node [Δx, Δy, Δz]
    - Distance (d): Euclidean distance ||Δp||
"""

import meshio
import numpy as np
import torch
import trimesh
from torch_geometric.data import Data

tol_dirichlet = 1e-3


def find_contact_patches(
    mesh: meshio.Mesh,
    r: float = 2.0,
    contacts: list[tuple] | None = None,
) -> np.ndarray:
    """
    Find mesh vertices within a given radius of contact points.
    """
    if contacts is None or len(contacts) == 0:
        return {}

    points, forces = zip(*contacts)
    points = np.asarray(points).reshape(-1, 3)
    forces = np.asarray(forces).reshape(-1, 3)

    # Find closest surface points and their normals
    faces = np.vstack([c.data for c in mesh.cells if c.type == "triangle"])
    tm = trimesh.Trimesh(vertices=mesh.points, faces=faces, process=False)

    closest, distance, triangle_ids = trimesh.proximity.closest_point(tm, points)
    within = np.linalg.norm(mesh.points[None, :, :] - points[:, None, :], axis=-1) <= r
    aligned = (tm.face_normals[triangle_ids] @ tm.vertex_normals.T) > 0.5
    mask = within & aligned

    contact_nodes: dict[int, np.ndarray] = {}
    for j in range(points.shape[0]):
        idx = np.where(mask[j])[0]
        f = forces[j] / np.sum(mask[j])
        for i in idx:
            contact_nodes[i] = contact_nodes.get(i, 0.0) + f

    return contact_nodes


def build_graph(
    mesh: meshio.Mesh,
    y: np.ndarray,
    contacts: list[tuple] | None = None,
) -> Data:
    # Node feature matrix with shape [num_nodes, num_node_features]
    x = make_nodes(mesh, find_contact_patches(mesh, r=2.0, contacts=contacts), contacts)
    y = torch.tensor(y, dtype=torch.float32)
    edge_index, edge_attr = make_edges(mesh)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def make_nodes(
    mesh: meshio.Mesh,
    loads: dict[int, np.ndarray],
    reference: list[tuple] | None = None,
) -> torch.Tensor:
    v = mesh.points
    n = v.shape[0]

    # Cartesian coords
    coords = torch.tensor(v, dtype=torch.float32)

    # Forces
    forces = torch.zeros((n, 3), dtype=torch.float32)
    if loads:
        idx = list(loads.keys())
        val = np.array(list(loads.values()))
        forces[idx] = torch.tensor(val, dtype=torch.float32)

    # Global attributes
    # TODO: sorted by coords or forces if multiple contacts
    Ps = []  # (len(reference), n, 3)
    Fs = []  # (len(reference), n, 3)
    for p, f in reference:
        Ps.append(torch.tensor(v - p, dtype=torch.float32))
        Fs.append(torch.tensor(np.tile(f, (n, 1)), dtype=torch.float32))
    inter = torch.stack([torch.stack(Ps), torch.stack(Fs)], dim=1).reshape(-1, n, 3)
    attrs = inter.permute(1, 0, 2).reshape(n, -1)

    # Boundary mask
    mask = torch.zeros((n, 1), dtype=torch.float32)
    mask[np.isclose(v[:, 2], 0.0, atol=tol_dirichlet)] = 1

    return torch.hstack([coords, forces, attrs, mask])


def make_edges(mesh: meshio.Mesh) -> torch.Tensor:
    edge_index = []
    edge_attr = []
    edge_sets = []

    v = mesh.points
    for cell in mesh.cells:
        data = cell.data
        if cell.type == "triangle":
            edge_sets.append(
                np.vstack(
                    [
                        data[:, [0, 1]],
                        data[:, [1, 2]],
                        data[:, [2, 0]],
                    ]
                )
            )
        elif cell.type == "tetra":
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
