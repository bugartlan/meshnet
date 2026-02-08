import argparse
from pathlib import Path
from typing import Tuple

import meshio
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from graph_builder import GraphBuilder
from simulator import Simulator
from utils import info, msh_to_trimesh


class DataGenerator:
    def __init__(
        self,
        out_dir: Path,
        num_samples: int = 1,
        num_contacts: int = 1,
        force_max: float = 1.0,
        sigma: float = 0.001,
        contact_radius: float = 0.01,
        seed: int = 42,
        debug: bool = False,
    ):
        """
        Args:
            out_dir (Path): Output directory for saving data.
            num_samples (int): Number of samples to generate per mesh.
            num_contacts (int): Number of contact points per sample.
            force_max (float): Maximum magnitude of contact forces.
            contact_radius (float): Radius around contact points for applying forces.
            seed (int): Random seed for reproducibility.
            debug (bool): If True, run in debug mode with verbose output.
        """
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.num_samples = num_samples
        self.num_contacts = num_contacts
        self.force_max = force_max
        self.contact_radius = contact_radius
        self.sigma = sigma
        self.seed = seed
        self.debug = debug

        self.rng = np.random.default_rng(seed)

        self.builder = GraphBuilder(contact_radius=contact_radius, std=sigma)

    def process(self, msh_path: Path):
        """Strategy: CG1 mesh for graph construction and CG2 mesh for simulation accuracy."""
        if not msh_path.exists():
            raise FileNotFoundError(f"Mesh file {msh_path} not found.")

        msh_path_cg1 = msh_path
        msh_path_cg2 = msh_path.with_name(
            msh_path.stem.replace("_cg1", "_cg2") + msh_path.suffix
        )
        mesh_cg1 = meshio.read(msh_path_cg1)
        mesh_cg2 = meshio.read(msh_path_cg2)

        points, forces = self._sample(mesh_cg2)
        results = self._simulate(msh_path_cg2, points, forces, mesh_cg1.points)

        graphs = []
        for y, p, f in zip(results, points, forces):
            graph = self.builder.build(
                mesh_cg1, y, radius=self.contact_radius, contacts=list(zip(p, f))
            )
            graphs.append(graph)

        return graphs

    def _sample(self, mesh: meshio.Mesh) -> Tuple[np.ndarray, np.ndarray]:
        num_total_points = self.num_samples * self.num_contacts

        # Sample 2x buffer to account for filtering
        mesh = msh_to_trimesh(mesh)
        candidates, _ = trimesh.sample.sample_surface(
            mesh, count=num_total_points * 2, seed=self.seed
        )
        # Remove point near bottom (z=0)
        candidates = candidates[candidates[:, 2] > 0.005]
        candidates = candidates[:num_total_points]

        if len(candidates) < num_total_points:
            raise ValueError(
                f"Not enough valid contact points found. Needed {num_total_points}, got {len(candidates)}."
            )

        points = candidates.reshape(self.num_samples, self.num_contacts, 3)

        # Sample random forces
        forces = self.rng.standard_normal(size=(self.num_samples, self.num_contacts, 3))
        forces = forces / np.linalg.norm(forces, axis=-1, keepdims=True)

        return points, forces

    def _simulate(
        self,
        msh_path: Path,
        points: np.ndarray,
        forces: np.ndarray,
        queries: np.ndarray,
    ):
        simulator = Simulator(
            str(msh_path), contact_radius=self.contact_radius, std=self.sigma
        )

        results = []
        for p, f in tqdm(zip(points, forces)):
            contacts = list(zip(p, f * self.force_max))
            uh = simulator.run(contacts)
            vm = simulator.compute_vm(uh)
            results.append(
                np.hstack([simulator.probe(uh, queries), simulator.probe(vm, queries)])
            )

        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate simulation data from meshes."
    )
    parser.add_argument(
        "meshes", type=Path, nargs="+", help="Paths to input mesh files or directories."
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data"),
        help="Output directory for saving data.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples per mesh."
    )
    parser.add_argument(
        "--num_contacts",
        type=int,
        default=1,
        help="Number of contact points per sample.",
    )
    parser.add_argument(
        "--force_max",
        type=float,
        default=1.0,
        help="Maximum magnitude of contact forces.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with verbose output."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    generator = DataGenerator(
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        num_contacts=args.num_contacts,
        force_max=args.force_max,
        contact_radius=0.01,
        seed=args.seed,
        debug=args.debug,
    )

    files = []
    for path in args.meshes:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(path.glob("*_cg1.msh"))
        else:
            raise RuntimeError(f"Path {path} is not a file or directory.")

    for f in files:
        graphs = generator.process(f)
        out_path = args.out_dir / (f.stem.replace("cg1", f"{args.num_samples}") + ".pt")

        node_dim, edge_dim, output_dim = info(graphs[0])
        torch.save(
            {
                "params": {
                    "node_dim": node_dim,
                    "edge_dim": edge_dim,
                    "output_dim": output_dim,
                },
                "graphs": graphs,
                "mesh": f,
            },
            out_path,
        )
        print(f"Saved {len(graphs)} samples to {out_path}")


if __name__ == "__main__":
    main()
