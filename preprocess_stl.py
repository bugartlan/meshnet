import argparse
from pathlib import Path

import trimesh


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess STL meshes.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("meshes/thingi10k/raw_stl"),
        help="Directory containing raw STL files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("meshes/thingi10k/stl"),
        help="Directory to save preprocessed STL files.",
    )
    return p.parse_args()


def preprocess(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # Center the mesh at the center of mass and translate so that the base is at z=0
    mesh.vertices -= mesh.center_mass
    mesh.vertices[:, 2] -= mesh.bounds[0][2]

    # Characteristic length of the mesh
    L = mesh.extents.max()

    # Scale the mesh to have a characteristic length of 1
    mesh.vertices /= L

    return mesh


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for stl_path in args.input_dir.glob("*.stl"):
        mesh = trimesh.load_mesh(stl_path)
        mesh = preprocess(mesh)
        out_path = args.output_dir / stl_path.name
        mesh.export(out_path)
        print(f"Processed {stl_path} -> {out_path}")


if __name__ == "__main__":
    main()
