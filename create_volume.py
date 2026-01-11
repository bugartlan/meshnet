import argparse
import math
from pathlib import Path

import gmsh

DOMAIN_TAG = 1
BOUNDARY_TAG = 2


def parse_args():
    p = argparse.ArgumentParser(description="Create volumetric meshes from STL files.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("meshes/thingi10k/stl"),
        help="Directory containing STL files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("meshes/thingi10k/msh"),
        help="Directory to save generated volumetric mesh files.",
    )
    return p.parse_args()


def create_volume(in_path: Path, out_path: Path):
    gmsh.clear()

    gmsh.model.add("mesh_volume")
    gmsh.merge(str(in_path))
    gmsh.model.geo.removeAllDuplicates()

    angle_threshold = 45  # degrees
    gmsh.model.mesh.classifySurfaces(angle_threshold * math.pi / 180.0)

    gmsh.model.mesh.createTopology()

    surface_entities = gmsh.model.getEntities(2)
    if not surface_entities:
        raise RuntimeError("No surface entities found after classification.")

    surfs = [s[1] for s in surface_entities]
    surface_loop_tag = gmsh.model.geo.addSurfaceLoop(surfs)
    vols = gmsh.model.geo.addVolume([surface_loop_tag])

    gmsh.model.addPhysicalGroup(3, [vols], DOMAIN_TAG, name="domain")

    gmsh.model.addPhysicalGroup(2, surfs, BOUNDARY_TAG, name="boundary")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    gmsh.write(str(out_path))


def main():
    args = parse_args()

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0
    for stl_path in args.input_dir.glob("*.stl"):
        out_path = args.output_dir / (stl_path.stem + ".msh")
        try:
            create_volume(stl_path, out_path)
            print(f"Processed {stl_path} -> {out_path}")
            success_count += 1
        except Exception as e:
            print(f"Failed to process {stl_path}: {e}")
            fail_count += 1
            continue
    print(f"Finished processing: {success_count} succeeded, {fail_count} failed.")
    gmsh.finalize()


if __name__ == "__main__":
    main()
