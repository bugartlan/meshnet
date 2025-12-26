import argparse
import time
from pathlib import Path

import meshio
import numpy as np
import torch
import trimesh

from fem import eval, fea
from graphs import build_graph


def make_dataset(
    mesh: meshio.Mesh,
    f: np.ndarray,
    points: np.ndarray,
    stl="cantilever.stl",
    msh="cantilever.msh",
):
    """Generate n graphs from given contact points on one given mesh."""
    graphs = []
    for p, f_vec in zip(points, f):
        contacts = [(p, f_vec)]

        domain, stresses_vm = fea(
            contacts,
            contact_radius=2.0,
            debug=False,
            filename_stl=stl,
            filename_msh=msh,
        )
        y = eval(domain, stresses_vm, mesh.points)
        graphs.append(build_graph(mesh, y, contacts))

    return graphs


def parse_args():
    p = argparse.ArgumentParser(description="Generate dataset from meshes.")
    p.add_argument(
        "--name",
        type=str,
        default="cantilever",
        help="Base filename (without extension) for .stl and .msh files.",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("meshes"),
        help="Directory containing .stl and .msh.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save generated dataset.",
    )
    p.add_argument(
        "-N",
        "--num-samples",
        type=int,
        default=1,
        help="Number of surface sample points.",
    )
    p.add_argument(
        "-F",
        "--force-max",
        type=float,
        default=10.0,
        help="Uniform force range [-F, F].",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: None)."
    )
    p.add_argument(
        "--out", type=str, default=None, help="Optional explicit output .pt path."
    )
    return p.parse_args()


def main():
    args = parse_args()

    stl = args.input_dir / f"{args.name}.stl"
    msh = args.input_dir / f"{args.name}.msh"

    if not stl.exists():
        raise FileNotFoundError(f"Missing STL: {stl}")
    if not msh.exists():
        raise FileNotFoundError(f"Missing MSH: {msh}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        args.out
        if args.out is not None
        else args.output_dir / f"{args.name}_{args.num_samples}.pt"
    )

    rng = np.random.default_rng(args.seed)

    start = time.time()

    # Load mesh and generate dataset
    mesh = trimesh.load_mesh(stl)
    points, face_ids = trimesh.sample.sample_surface(mesh, count=args.num_samples)

    forces = rng.uniform(
        low=-args.force_max, high=args.force_max, size=(args.num_samples, 3)
    )
    mesh_mio = meshio.read(msh)
    graphs = make_dataset(mesh_mio, forces, points, stl=stl, msh=msh)
    torch.save(graphs, out_path)
    print(f"Saved dataset with {len(graphs)} graphs to {out_path}.")

    end = time.time()
    print(f"Data generation took {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()
