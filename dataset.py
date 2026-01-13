import argparse
import time
from pathlib import Path

import meshio
import numpy as np
import torch
import trimesh

from fem import eval, fea
from graphs import build_graph
from utils import info


def parse_args():
    p = argparse.ArgumentParser(description="Generate dataset from meshes.")
    p.add_argument(
        "--dataset",
        type=str,
        default="cantilever",
        help="Name of the dataset of meshes.",
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
        help="Number of sampled data points.",
    )
    p.add_argument(
        "--num-contacts",
        type=int,
        default=1,
        help="Number of contact points per sample.",
    )
    p.add_argument(
        "-F",
        "--force-max",
        type=float,
        default=10.0,
        help="Uniform force range [-F, F].",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--out", type=str, default=None, help="Optional explicit output .pt path."
    )
    p.add_argument("--debug", action="store_true", help="Enable debug mode.")
    return p.parse_args()


def make_graphs(
    mesh: meshio.Mesh,
    f: np.ndarray,
    points: np.ndarray,
    stl: str,
    msh: str,
    debug: bool = False,
):
    """Generate n graphs from given contact points on one given mesh."""
    graphs = []
    for p, f_vec in zip(points, f):
        contacts = list(zip(p, f_vec))

        domain, stresses_vm = fea(
            contacts,
            contact_radius=2.0,
            debug=debug,
            filename_stl=stl,
            filename_msh=msh,
        )
        y = eval(domain, stresses_vm, mesh.points)
        graphs.append(build_graph(mesh, y, contacts))

    return graphs


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stl_path = Path("meshes") / args.dataset / "stl"
    msh_path = Path("meshes") / args.dataset / "msh"

    rng = np.random.default_rng(args.seed)
    start = time.time()

    n_samples = args.num_samples * args.num_contacts
    data = []
    for f in msh_path.glob("*.msh"):
        name = f.stem
        stl = stl_path / f"{name}.stl"
        msh = msh_path / f"{name}.msh"
        if not stl.exists():
            raise FileNotFoundError(f"Missing STL: {stl}")
        if not msh.exists():
            raise FileNotFoundError(f"Missing MSH: {msh}")

        # Load mesh and generate dataset
        mesh = trimesh.load_mesh(stl)
        points, face_ids = trimesh.sample.sample_surface(
            mesh, count=n_samples, seed=args.seed
        )
        points = points.reshape(args.num_samples, args.num_contacts, 3)

        forces = rng.uniform(
            low=-args.force_max,
            high=args.force_max,
            size=(args.num_samples, args.num_contacts, 3),
        )

        mesh_mio = meshio.read(msh)
        graphs = make_graphs(mesh_mio, forces, points, stl, msh, debug=args.debug)
        data.extend(graphs)

        print(f"Generated {len(graphs)} graphs for {name}.")

    out_path = (
        args.out
        if args.out is not None
        else args.output_dir
        / f"{args.dataset}_{args.num_contacts}_{args.num_samples}.pt"
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
        },
        out_path,
    )
    print(f"Saved dataset with {len(data)} graphs to {out_path}.")

    end = time.time()
    print(f"Data generation took {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()
