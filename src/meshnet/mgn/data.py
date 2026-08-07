import argparse
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

from meshnet.mgn.simulator import Simulator
from meshnet.utils.mesh import Mesh


class DataGenerator:
    """Generate self-contained vertex-aligned FEM datasets."""

    def __init__(
        self,
        out_dir: Path,
        num_samples: int = 1,
        num_contacts: int = 1,
        sigma: float = 0.001,
        seed: int = 42,
        solve_order: int = 2,
    ) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.num_contacts = num_contacts
        self.sigma = sigma
        self.seed = seed
        self.solve_order = solve_order
        self.rng = np.random.default_rng(seed)

    def process(self, filepath: Path) -> Path:
        if not filepath.exists():
            raise FileNotFoundError(f"Mesh file {filepath} not found.")

        mesh = Mesh.read(str(filepath))
        contact_points, contact_forces = self._sample(mesh)

        simulator = Simulator(
            mesh,
            order=self.solve_order,
            contact_std=self.sigma,
        )

        vertex_displacement: list[np.ndarray] = []
        vertex_stress: list[np.ndarray] = []
        vertex_forces: list[np.ndarray] = []

        for points_i, forces_i in tqdm(
            zip(contact_points, contact_forces),
            total=self.num_samples,
        ):
            result = simulator.run(zip(points_i, forces_i))
            vertex_displacement.append(result.displacement_vertices)
            vertex_stress.append(result.stress_vertices)
            vertex_forces.append(result.nodal_forces_vertices)

        displacement = np.stack(vertex_displacement)
        stress = np.stack(vertex_stress)
        nodal_forces = np.stack(vertex_forces)

        vertices = np.asarray(mesh.volume.points, dtype=np.float64)
        tetra = np.asarray(mesh.volume.cells_dict["tetra"], dtype=np.int64)
        boundary_mask = np.isclose(vertices[:, 2], 0.0, atol=1e-6, rtol=0.0)

        expected_shapes = {
            "vertex_displacement": (self.num_samples, len(vertices), 3),
            "vertex_stress": (self.num_samples, len(vertices), 6),
            "vertex_forces": (self.num_samples, len(vertices), 3),
        }
        actual = {
            "vertex_displacement": displacement.shape,
            "vertex_stress": stress.shape,
            "vertex_forces": nodal_forces.shape,
        }
        for name, expected in expected_shapes.items():
            if actual[name] != expected:
                raise RuntimeError(
                    f"{name} is not aligned with the source vertices: "
                    f"expected {expected}, got {actual[name]}."
                )

        out_path = self.out_dir / f"{filepath.stem}.npz"
        np.savez_compressed(
            out_path,
            # Static P1 graph mesh. Saving it makes the raw file self-contained.
            vertices=vertices,
            tetra=tetra,
            boundary_mask=boundary_mask,
            # Per-sample contact metadata.
            contact_points=contact_points,
            contact_forces=contact_forces,
            # Per-sample arrays, all in exactly the same vertex order.
            vertex_forces=nodal_forces,
            vertex_displacement=displacement,
            vertex_stress=stress,
            # Provenance and schema metadata.
            mesh_path=np.asarray(str(filepath)),
            contact_std=np.asarray(self.sigma),
            solve_order=np.asarray(self.solve_order),
            seed=np.asarray(self.seed),
            schema_version=np.asarray(2),
        )
        return out_path

    def _sample(self, mesh: Mesh, tol: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        total = self.num_samples * self.num_contacts

        candidates, _ = trimesh.sample.sample_surface(
            mesh.surface,
            count=total * 5,
            seed=self.seed,
        )
        candidates = candidates[candidates[:, 2] >= tol][:total]
        if len(candidates) < total:
            raise ValueError(
                f"Not enough valid contact points: needed {total}, got {len(candidates)}."
            )

        points = candidates.reshape(self.num_samples, self.num_contacts, 3)

        forces = self.rng.standard_normal(size=(self.num_samples, self.num_contacts, 3))
        forces /= np.linalg.norm(forces, axis=-1, keepdims=True)

        return points, forces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P2 FEM / P1 graph data.")
    parser.add_argument("meshes", type=Path, nargs="+")
    parser.add_argument("--out_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_contacts", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solve_order", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = DataGenerator(
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        num_contacts=args.num_contacts,
        sigma=args.sigma,
        seed=args.seed,
        solve_order=args.solve_order,
    )

    files: list[Path] = []
    for path in args.meshes:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.glob("*.msh")))
        else:
            raise RuntimeError(f"Path {path} is not a file or directory.")

    for filepath in files:
        output = generator.process(filepath)
        print(f"Saved raw FEM data to {output}")


if __name__ == "__main__":
    main()
