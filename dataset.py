import argparse
import time
from pathlib import Path

import meshio
import numpy as np
import torch
import trimesh

from graphs import build_graph
from solver import interp, interp_pyvista, solve
from utils import info, msh_to_trimesh


def parse_args():
    p = argparse.ArgumentParser(description="Generate dataset from meshes.")

    # --- IO Configuration ---
    p.add_argument(
        "mesh_name",
        type=str,
        help="Name of the mesh family to sample (e.g., 'cantilever'). Matches folder in meshes/.",
    )
    p.add_argument(
        "--file", type=str, default=None, help="Optional specific mesh file to process."
    )
    p.add_argument(
        "--coarse-mesh",
        action="store_true",
        help="If set, use the coarse mesh for sampling instead of the fine mesh.",
    )

    # --- Output Configuration ---
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save the generated .pt file.",
    )
    p.add_argument(
        "--output-filename",
        type=str,
        default=None,
        help="Explicit filename for the output. If None, auto-generated from params.",
    )

    # --- Sampling Configuration ---
    p.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate.",
    )
    p.add_argument(
        "--num-contacts",
        type=int,
        default=1,
        help="Number of random force application points per sample.",
    )
    p.add_argument(
        "-f",
        "--force-max",
        type=float,
        default=15.0,
        help="Maximum magnitude for the random force vector.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    # --- Runtime Flags ---
    p.add_argument("--debug", action="store_true", help="Enable debug mode.")

    return p.parse_args()


def make_graphs(
    mesh: meshio.Mesh,
    f: np.ndarray,
    points: np.ndarray,
    msh_path: Path,
    contact_radius: float,
    debug: bool = False,
):
    """
    Generate a list of graphs by performing FEA for each set of contact points and forces.

    Args:
        mesh (meshio.Mesh): The input mesh.
        f (np.ndarray): Force vectors of shape (num_samples, num_contacts, 3).
        points (np.ndarray): Contact points of shape (num_samples, num_contacts, 3).
        msh_path (Path): Path to the .msh file for FEA solver.
        contact_radius (float): Radius around contact points.
        debug (bool): Whether to enable verbose logging.

    Returns:
        list[Data]: List of PyTorch Geometric Data objects.
    """
    graphs = []

    # Iterate over samples
    for p, f_vec in zip(points, f):
        # Create (point, force) pairs
        contacts = list(zip(p, f_vec))

        # Run FEA
        domain, stresses_vm, uh = solve(
            contacts,
            contact_radius=contact_radius,
            debug=debug,
            filename_msh=str(msh_path),
        )

        # Interpolate and build graph
        y = interp_pyvista(domain, [uh, stresses_vm], mesh.points)
        print(mesh.points.shape, y.shape)
        graphs.append(build_graph(mesh, y, radius=contact_radius, contacts=contacts))

    return graphs


# TODO: maybe use fixed radius instead?
def get_adaptive_radius(mesh: trimesh.Trimesh, multiplier: float = 2.0) -> float:
    """Compute an adaptive radius based on the average edge length of the mesh."""
    avg_edge_length = mesh.edges_unique_length.mean()
    return multiplier * avg_edge_length


def sample(mesh: trimesh.Trimesh, num_points: int, seed: int = 42) -> np.ndarray:
    """Sample points uniformly on the mesh surface."""
    points, _ = trimesh.sample.sample_surface(
        mesh, count=int(num_points * 2), seed=seed
    )
    points = points[points[:, 2] > 0.001]  # Filter points above z=0.001
    return points[:num_points]


def generate1(
    file: Path,
    n_samples: int,
    n_contacts: int,
    force_max: float,
    rng: np.random.Generator,
    file_coarse: Path = None,
    seed: int = 42,
    debug: bool = False,
    output_path: str = None,
):
    # Check file existence
    if not file.exists():
        raise FileNotFoundError(f"Missing MSH: {file}")
    if file_coarse is not None and not file_coarse.exists():
        raise FileNotFoundError(f"Missing coarse MSH: {file_coarse}")

    # Load mesh
    mesh_mio = meshio.read(file_coarse if file_coarse is not None else file)
    mesh_tri = msh_to_trimesh(mesh_mio)

    # TODO: sample points using antipodal grasp sampler
    points = sample(mesh_tri, n_samples * n_contacts, seed=seed)
    points = points.reshape(n_samples, n_contacts, 3)

    forces = rng.uniform(
        low=-force_max,
        high=force_max,
        size=(n_samples, n_contacts, 3),
    )

    contact_radius = get_adaptive_radius(mesh_tri, multiplier=2.0)
    graphs = make_graphs(mesh_mio, forces, points, file, contact_radius, debug=debug)

    node_dim, edge_dim, output_dim = info(graphs[0], debug=False)
    torch.save(
        {
            "params": {
                "node_dim": node_dim,
                "edge_dim": edge_dim,
                "output_dim": output_dim,
            },
            "graphs": graphs,
            "mesh": file.stem,
        },
        output_path,
    )
    print(f"Saved {len(graphs)} graphs to {output_path}.")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    msh_path = Path("meshes") / args.mesh_name / "msh"
    msh_coarse_path = Path("meshes") / args.mesh_name / "msh_coarse"

    rng = np.random.default_rng(args.seed)
    start = time.time()

    files_to_process = (
        [msh_path / args.file] if args.file is not None else msh_path.glob("*.msh")
    )

    for f in files_to_process:
        name = f.stem
        f_coarse = msh_coarse_path / f"{name}.msh" if args.coarse_mesh else None

        if args.output_filename is not None:
            output_path = args.output_dir / args.output_filename
        else:
            filename = f"{name}_{args.num_contacts}_{args.num_samples}_{'c' if args.coarse_mesh else 'f'}.pt"
            output_path = args.output_dir / filename
        generate1(
            f,
            args.num_samples,
            args.num_contacts,
            args.force_max,
            rng,
            file_coarse=f_coarse,
            seed=args.seed,
            debug=args.debug,
            output_path=output_path,
        )

    end = time.time()
    print(f"Data generation took {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()
