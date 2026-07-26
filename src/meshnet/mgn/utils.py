import meshio
import numpy as np
import torch
import trimesh
from torch_geometric.data import Data


def info(graph: Data, debug=False):
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


def msh_to_trimesh(mesh: meshio.Mesh) -> trimesh.Trimesh:
    """
    Convert a meshio.Mesh to a trimesh.Trimesh object.

    Args:
        mesh (meshio.Mesh): Input meshio mesh.

    Returns:
        trimesh.Trimesh: Converted trimesh object.
    """
    triangles = [c.data for c in mesh.cells if "triangle" in c.type]
    faces = np.vstack(triangles)
    return trimesh.Trimesh(vertices=mesh.points, faces=faces, process=False)


def get_weight(
    z: torch.Tensor, dim: int, mode: str = "all", alpha: float = 1.0, tol: float = 1e-4
):
    """
    Compute weights for loss function based on z-coordinate.

    Args:
        z (torch.Tensor): z-coordinates of the nodes.
        dim (int): Dimension of the output.
        mode (str): Weighting mode. Options are 'weighted', 'bottom', 'all'.
        alpha (float): Scaling factor for 'weighted' mode.
        tol (float): Tolerance for 'bottom' mode.

    Returns:
        torch.Tensor: Weights for each node.
    """
    if mode == "weighted":
        weight = torch.exp(-alpha * z)
        weight /= weight.mean()
    elif mode == "bottom":
        weight = torch.isclose(z, torch.zeros_like(z), atol=tol)
    elif mode == "all":
        weight = torch.ones_like(z)
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Supported: 'weighted', 'bottom', 'all'."
        )

    if weight.dim() == 1:
        weight = weight.unsqueeze(-1)

    return weight.repeat(1, dim)


def grad_u(graph: Data):
    i = graph.edge_index[0]  # source
    j = graph.edge_index[1]  # destination

    x = graph.x[:, :3]  # coords
    u = graph.y[:, :3]  # displacements

    dx = x[j] - x[i]  # edge vectors
    du = u[j] - u[i]  # displacement differences

    # weights: w_ik = 1 / (||dx_ij||^2 + epsilon)
    w = 1.0 / (dx.pow(2).sum(dim=1, keepdim=True) + 1e-12)  # [E, 1]

    # Accumulate per node
    # A_i = sum w * dx * dx^T -> [N, 3, 3]
    # B_i = sum w * du * dx^T -> [N, 3, 3]
    N = x.size(0)
    A = torch.zeros((N, 3, 3), dtype=x.dtype, device=x.device)
    B = torch.zeros((N, 3, 3), dtype=x.dtype, device=x.device)

    dx_col = dx.unsqueeze(2)  # [E, 3, 1]
    dx_row = dx.unsqueeze(1)  # [E, 1, 3]
    du_col = du.unsqueeze(2)  # [E, 3, 1]

    A_e = w.unsqueeze(2) * (dx_col @ dx_row)  # [E, 3, 3]
    B_e = w.unsqueeze(2) * (du_col @ dx_row)  # [E, 3, 3]

    A.index_add_(0, i, A_e)
    B.index_add_(0, i, B_e)

    # Regularize + Invert
    A_reg = A + 1e-6 * torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0)

    # G = B * A^{-1}
    G = torch.linalg.solve(A_reg, B.mT).mT  # [N, 3, 3]
    return G


def strain_stress_vm(graph: Data, E: float, nu: float):
    G = grad_u(graph)  # [N, 3, 3]

    # Small strain tensor: eps = 0.5 * (G + G^T)
    eps = 0.5 * (G + G.mT)  # [N, 3, 3]

    # Lame parameters
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    # Stress tensor: sigma = lam * tr(eps) * I + 2 * mu * eps
    tr = torch.einsum("nii->n", eps)  # [N]
    I = torch.eye(3, dtype=eps.dtype, device=eps.device).unsqueeze(0)  # [1, 3, 3]

    sigma = lam * tr.view(-1, 1, 1) * I + 2 * mu * eps  # [N, 3, 3]

    # Von Mises stress
    s = sigma - torch.einsum("nii->n", sigma).view(-1, 1, 1) / 3 * I
    vm = torch.sqrt(1.5 * (s * s).sum(dim=(1, 2)))  # [N]

    return eps, sigma, vm
