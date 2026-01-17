import argparse
import time
from pathlib import Path

import meshio
import numpy as np
import torch
import trimesh

from fem import eval as evalulate
from fem import fea
from graphs import build_graph
from utils import info, msh_to_trimesh


def parse_args():
    p = argparse.ArgumentParser(description="Generate dataset from meshes.")

    # --- IO Configuration ---
    p.add_argument(
        "mesh_name",
        type=str,
        help="Name of the mesh family to sample (e.g., 'cantilever'). Matches folder in meshes/.",
    )

    # --- Output Configuration ---
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save the generated .pt file.",
    )
    p.add_argument(
        "--filename",
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
        default=10.0,
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
        domain, stresses_vm, uh = fea(
            contacts,
            contact_radius=contact_radius,
            debug=debug,
            filename_msh=str(msh_path),
        )

        # Interpolate and build graph
        y = evalulate(domain, [uh, stresses_vm], mesh.points)
        graphs.append(build_graph(mesh, y, radius=contact_radius, contacts=contacts))

    return graphs


def get_adaptive_radius(mesh: trimesh.Trimesh, multiplier: float = 2.0) -> float:
    """Compute an adaptive radius based on the average edge length of the mesh."""
    avg_edge_length = mesh.edges_unique_length.mean()
    return multiplier * avg_edge_length


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    msh_path = Path("meshes") / args.mesh_name / "msh"

    rng = np.random.default_rng(args.seed)
    start = time.time()

    n_samples = args.num_samples * args.num_contacts
    data = []
    meshes = []
    for f in msh_path.glob("*.msh"):
        name = f.stem
        msh = msh_path / f"{name}.msh"
        if not msh.exists():
            raise FileNotFoundError(f"Missing MSH: {msh}")

        # Load mesh and generate dataset
        mesh_mio = meshio.read(msh)
        mesh_tri = msh_to_trimesh(mesh_mio)

        points, face_ids = trimesh.sample.sample_surface(
            mesh_tri, count=n_samples, seed=args.seed
        )
        points = points.reshape(args.num_samples, args.num_contacts, 3)

        forces = rng.uniform(
            low=-args.force_max,
            high=args.force_max,
            size=(args.num_samples, args.num_contacts, 3),
        )

        contact_radius = get_adaptive_radius(mesh_tri, multiplier=2.0)

        graphs = make_graphs(
            mesh_mio, forces, points, msh, contact_radius, debug=args.debug
        )
        data.extend(graphs)
        meshes.extend([name] * len(graphs))

        print(f"Generated {len(graphs)} graphs for {name}.")

    out_path = (
        args.output_dir / args.filename
        if args.filename is not None
        else args.output_dir
        / f"{args.mesh_name}_{args.num_contacts}_{args.num_samples}.pt"
    )

    node_dim, edge_dim, output_dim = info(data[0], debug=False)
    torch.save(
        {
            "params": {
                "node_dim": node_dim,
                "edge_dim": edge_dim,
                "output_dim": output_dim,
            },
            "graphs": data,
            "meshes": meshes,
        },
        out_path,
    )
    print(f"Saved dataset with {len(data)} graphs to {out_path}.")

    end = time.time()
    print(f"Data generation took {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()
