import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import trimesh
from tqdm import tqdm

from meshnet.mgn.graphs import GraphBuilderVirtual
from meshnet.mgn.simulator import Simulator
from meshnet.mgn.utils import info
from meshnet.utils.mesh import Mesh


class DataGenerator:
    def __init__(
        self,
        out_dir: Path,
        num_samples: int = 1,
        num_contacts: int = 1,
        sigma: float = 0.001,
        seed: int = 42,
        debug: bool = False,
    ):
        """
        Args:
            out_dir (Path): Output directory for saving data.
            num_samples (int): Number of samples to generate per mesh.
            num_contacts (int): Number of contact points per sample.
            sigma (float): Standard deviation for Gaussian kernel in contact force application.
            seed (int): Random seed for reproducibility.
            debug (bool): If True, run in debug mode with verbose output.
        """
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.num_samples = num_samples
        self.num_contacts = num_contacts
        self.sigma = sigma
        self.seed = seed
        self.debug = debug

        self.rng = np.random.default_rng(seed)

        # self.builder = GraphBuilder(std=sigma)
        self.builder = GraphBuilderVirtual(std=sigma)

    def process(self, filepath: Path) -> list[torch.Tensor]:
        """Generate graphs with ground truth labels.

        Args:
            filepath (Path): Path to the mesh file.

        Returns:
            list[torch.Tensor]: A list of graphs with ground truth labels.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Mesh file {filepath} not found.")

        mesh = Mesh.read(str(filepath))

        points, forces = self._sample(mesh)
        results = self._simulate(mesh, points, forces)

        graphs = []
        for y, p, f in zip(results, points, forces):
            graphs.append(self.builder.build(mesh.volume, y, contacts=list(zip(p, f))))

        return graphs

    def _sample(
        self, mesh: Mesh, tol=0.01
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        num_total_points = self.num_samples * self.num_contacts

        # Sample 2x buffer to account for filtering
        candidates, _ = trimesh.sample.sample_surface(
            mesh.surface, count=num_total_points * 5, seed=self.seed
        )
        # Remove point near bottom (z=0)
        candidates = candidates[candidates[:, 2] >= tol]
        candidates = candidates[:num_total_points]

        if len(candidates) < num_total_points:
            raise ValueError(
                f"Not enough valid contact points found. Needed {num_total_points}, got {len(candidates)}."
            )

        points = candidates.reshape(self.num_samples, self.num_contacts, 3)

        # Sample random forces
        directions = self.rng.standard_normal(
            size=(self.num_samples, self.num_contacts, 3)
        )
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

        magnitudes = self.rng.uniform(
            0.0, 1.0, size=(self.num_samples, self.num_contacts, 1)
        )

        forces = directions * magnitudes

        return points, forces

    def _simulate(
        self,
        mesh: Mesh,
        points: npt.ArrayLike,
        forces: npt.ArrayLike,
    ) -> list[npt.NDArray[np.float64]]:
        simulator = Simulator(mesh, std=self.sigma)

        results = []
        for p, f in tqdm(zip(points, forces)):
            contacts = list(zip(p, f))
            displacement = simulator.run(contacts)
            stress = simulator.compute_stress(displacement, degree=1)
            results.append(
                np.hstack(
                    [
                        simulator.evaluate_mesh_points(displacement),
                        simulator.evaluate_mesh_points(stress),
                    ]
                )
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
        sigma=0.01,
        seed=args.seed,
        debug=args.debug,
    )

    files = []
    for path in args.meshes:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(path.glob("*.msh"))
        else:
            raise RuntimeError(f"Path {path} is not a file or directory.")

    for f in files:
        graphs = generator.process(f)
        out_path = args.out_dir / (f.stem + f"_{len(graphs)}.pt")

        node_dim, edge_dim, output_dim = info(graphs[0])
        num_categorical = generator.builder.num_categorical
        torch.save(
            {
                "params": {
                    "node_dim": node_dim,
                    "edge_dim": edge_dim,
                    "output_dim": output_dim,
                    "num_categorical": num_categorical,
                },
                "graphs": graphs,
                "mesh": f,
            },
            out_path,
        )
        print(f"Saved {len(graphs)} samples to {out_path}")


if __name__ == "__main__":
    main()
