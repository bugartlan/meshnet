import argparse
import math
from pathlib import Path
from typing import Optional

import gmsh


class Mesher:
    def __init__(
        self, element_size: float = 0.001, element_order: int = 1, verbose: bool = False
    ):
        self.element_size = element_size
        self.element_order = element_order
        self.verbose = verbose

        self.TAG_DOMAIN = 1
        self.TAG_BOUNDARY = 2

    def __enter__(self):
        gmsh.initialize()
        self._configure_gmsh()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        gmsh.finalize()

    def _configure_gmsh(self):
        gmsh.option.setNumber("General.Verbosity", 2 if self.verbose else 1)
        gmsh.option.setNumber("Mesh.ElementOrder", self.element_order)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.element_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.element_size)
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")

    def process(
        self,
        in_path: Path,
        out_path: Path,
        file_format: str,
        target_size: Optional[float] = None,
    ):
        gmsh.clear()
        try:
            if file_format == "stl":
                print("Scaling is not supported for STL files.")
                self._mesh_stl(in_path, target_size)
            elif file_format == "step":
                self._mesh_step(in_path, target_size)
            else:
                raise RuntimeError(f"Unsupported file format: {file_format}")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(out_path))
            print(f"Saved mesh to {out_path}")
            return True
        except Exception as e:
            print(f"Error processing {in_path}: {e}")
            return False

    def _mesh_stl(self, path: Path, target_size: Optional[float]):
        gmsh.model.add("mesh_volume_stl")
        gmsh.merge(str(path))
        gmsh.model.geo.removeAllDuplicates()

        gmsh.model.mesh.classifySurfaces(45 * math.pi / 180.0)
        gmsh.model.mesh.createTopology()

        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            raise RuntimeError("No surfaces found after classification.")

        surf_tags = [s[1] for s in surfaces]
        surf_loop = gmsh.model.geo.addSurfaceLoop(surf_tags)

        vol_tag = gmsh.model.geo.addVolume([surf_loop])
        gmsh.model.addPhysicalGroup(3, [vol_tag], self.TAG_DOMAIN, name="domain")
        gmsh.model.addPhysicalGroup(2, surf_tags, self.TAG_BOUNDARY, name="boundary")
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)

    def _mesh_step(self, path: Path, target_size: Optional[float]):
        gmsh.model.add("mesh_volume_step")
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(3)
        if len(volumes) != 1:
            raise RuntimeError(
                f"STEP file must contain exactly 1 volume (found {len(volumes)})."
            )

        vol_tag = volumes[0][1]
        self._align_and_scale_occ(vol_tag, target_size)

        gmsh.model.addPhysicalGroup(3, [vol_tag], self.TAG_DOMAIN, name="domain")

        surfaces = gmsh.model.getEntities(2)
        surf_tags = [s[1] for s in surfaces]
        gmsh.model.addPhysicalGroup(2, surf_tags, self.TAG_BOUNDARY, name="boundary")

        gmsh.model.mesh.generate(3)

    def _align_and_scale_occ(self, vol_tag: int, target_size: Optional[float]):
        x0, y0, z0, x1, y1, z1 = gmsh.model.occ.getBoundingBox(3, vol_tag)

        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        cz = 0.5 * (z0 + z1)

        # Translate to origin
        gmsh.model.occ.translate([(3, vol_tag)], -cx, -cy, -cz)
        gmsh.model.occ.synchronize()

        # Dilate to target size
        if target_size:
            scale = target_size / max(x1 - x0, y1 - y0, z1 - z0)
            gmsh.model.occ.dilate([(3, vol_tag)], 0, 0, 0, scale, scale, scale)
            gmsh.model.occ.synchronize()

        _, _, nz0, _, _, _ = gmsh.model.occ.getBoundingBox(3, vol_tag)
        gmsh.model.occ.translate([(3, vol_tag)], 0, 0, -nz0)
        gmsh.model.occ.synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description="Volumetric Mesh Generator")
    parser.add_argument("format", choices=["stl", "step"], help="Input file format")
    parser.add_argument(
        "--input", type=Path, default=Path("meshes"), help="Input file or directory"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("meshes"), help="Output directory"
    )
    parser.add_argument("--size", type=float, default=0.005, help="Mesh element size")
    parser.add_argument(
        "--target-scale",
        type=float,
        default=None,
        help="Scale model to this max dimension (m)",
    )
    parser.add_argument(
        "--element-order",
        type=int,
        default=1,
        help="Order of mesh elements (1=linear, 2=quadratic)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.input.is_file():
        files = [args.input]
    elif args.input.is_dir():
        print("Scanning input directory for files...")
        files = list(args.input.glob(f"*.{args.format}", case_sensitive=False))
    else:
        raise RuntimeError(f"Input path {args.input} is not a file or directory.")

    with Mesher(element_size=args.size, element_order=args.element_order) as mesher:
        print(f"Start processing {len(files)} files...")
        stats = {"success": 0, "fail": 0}

        for f in files:
            suffix = "_cg1" if args.element_order == 1 else "_cg2"
            out_path = args.output / (f.stem + suffix + ".msh")
            result = mesher.process(f, out_path, args.format, args.target_scale)

            if result:
                stats["success"] += 1
            else:
                stats["fail"] += 1

        print(
            f"Processing complete. Success: {stats['success']}, Fail: {stats['fail']}"
        )


if __name__ == "__main__":
    main()
