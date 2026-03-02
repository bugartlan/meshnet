import meshio
import numpy as np
import pyvista
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


def find_contacts(graph: Data, tol: float = 1e-6) -> dict[np.ndarray, np.ndarray]:
    """
    Find contact points and their associated forces from the graph data.

    Args:
        graph (Data): The input graph containing node features.
        tol (float, optional): Tolerance for force magnitude to consider as contact. Defaults to 1e-6.

    Returns:
        dict[np.ndarray, np.ndarray]: A dictionary mapping contact point coordinates to force vectors.
    """

    x = graph.x.cpu().numpy()
    coords = x[:, :3]
    forces = x[:, 3:6]

    magnitudes = np.linalg.norm(forces, axis=1)
    mask = magnitudes > tol

    # Filter active contacts
    active_coords = coords[mask]
    active_forces = forces[mask]

    # Using tuple as key since numpy arrays are not hashable
    return {tuple(c): f for c, f in zip(active_coords, active_forces)}


def make_pv_mesh(mesh: trimesh.Trimesh, graph: Data, labels: list) -> pyvista.PolyData:
    """
    Generate a PyVista mesh with graph data mapped onto it.

    Args:
        mesh (trimesh.Trimesh): The input mesh.
        graph (Data): The graph containing node data.
        labels (list): List of labels corresponding to graph.y columns.

    Returns:
        pyvista.PolyData: The PyVista mesh with data arrays.
    """
    pv_mesh = pyvista.wrap(mesh)
    for i, label in enumerate(labels):
        # Map graph data to mesh vertices
        pv_mesh[label] = graph.y[:, i].cpu().numpy().squeeze()
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
    """
    Visualize a graph on a PyVista mesh.

    Args:
        pv_mesh (pyvista.PolyData): The PyVista mesh.
        graph (Data): The graph containing node data.
        label (str): The label of the data to visualize.
        jupyter_backend (str, optional): Jupyter notebook backend. Defaults to None.
        force_arrows (bool, optional): Whether to show force arrows. Defaults to False.
        show (bool, optional): Whether to display the plot. Defaults to True.
        filename (str, optional): If provided, saves the plot to this HTML file. Defaults to None.
        clim (tuple, optional): Color limits for the scalar bar. Defaults to None
    """
    if label not in pv_mesh.array_names:
        raise ValueError(f"Label '{label}' not found in mesh array names.")

    plotter = pyvista.Plotter(notebook=jupyter_backend is not None)
    plotter.add_mesh(
        pv_mesh,
        scalars=label,
        point_size=1,
        render_points_as_spheres=True,
        show_edges=True,
        clim=clim,
    )

    if force_arrows:
        # Calculate scale for arrows
        x_min, x_max, y_min, y_max, z_min, z_max = pv_mesh.bounds
        scale = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.1

        x = graph.x[0].cpu().numpy()
        contacts = x[6:-1].reshape(-1, 6)
        contacts[:, :3] -= x[:3]
        contacts[:, :3] *= -1
        for v in contacts:
            p = v[:3]
            f = v[3:]

            # Visualize the contact point
            sphere = pyvista.Sphere(radius=scale * 0.1)
            sph = sphere.translate(p, inplace=False)
            plotter.add_mesh(sph, color="red", opacity=1)

            # Visualize the force arrow
            arrow = pyvista.Arrow(start=np.asarray(p), direction=f, scale=scale)
            plotter.add_mesh(arrow, color="red")

    plotter.show_axes()
    if show:
        plotter.show(jupyter_backend=jupyter_backend)
    if filename is not None:
        plotter.export_html(filename)


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
    elif mode == "bottom":
        weight = (z < tol).to(z.dtype)
    elif mode == "all":
        weight = torch.ones_like(z)
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Supported: 'weighted', 'bottom', 'all'."
        )

    # Normalize weights to have mean 1
    weight = weight / weight.mean()
    if weight.dim() == 1:
        weight = weight.unsqueeze(-1)

    return weight.expand(z.shape[0], dim)


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
