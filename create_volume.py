import argparse
import math
from pathlib import Path

import gmsh
import meshio
import numpy as np
import trimesh

from grasp.AntipodalGrasp import AntipodalGraspSampler
from grasp.Gripper import RobotiqHandE

DOMAIN_TAG = 1
BOUNDARY_TAG = 2

TARGET_SIZE = 0.049  # (m)


def parse_args():
    p = argparse.ArgumentParser(description="Create volumetric meshes from STL files.")
    p.add_argument(
        "format",
        type=str,
        choices=["stl", "step"],
        help="Format of the input file(s).",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("meshes/thingi10k/stl"),
        help="Directory containing STL/STEP files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("meshes/thingi10k/msh"),
        help="Directory to save generated volumetric mesh files.",
    )
    p.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Optional specific file to process.",
    )
    return p.parse_args()


def scale_stl_mesh(stl_path: Path, target_size: float):
    mesh = trimesh.load(str(stl_path))
    scale = target_size / mesh.extents.max()
    mesh.apply_scale(scale)
    out_path = stl_path.parent / (stl_path.stem + "_scaled.stl")
    mesh.export(str(out_path))
    return out_path


def create_volume_stl(stl_path: Path, out_path: Path):
    scaled_stl_path = scale_stl_mesh(stl_path, TARGET_SIZE)
    gmsh.clear()

    gmsh.model.add("mesh_volume")
    gmsh.merge(str(scaled_stl_path))
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


def create_volume_step(step_path: Path, out_path: Path, debug: bool = False):
    gmsh.clear()

    gmsh.model.add("mesh_volume")
    gmsh.model.occ.importShapes(str(step_path))
    gmsh.model.occ.synchronize()

    # Check the model only consists of one volume
    volumes = gmsh.model.getEntities(3)
    vol_tag = volumes[0][1]

    if len(volumes) != 1:
        raise RuntimeError("STEP file must contain exactly one volume entity.")

    x0, y0, z0, x1, y1, z1 = gmsh.model.occ.getBoundingBox(3, vol_tag)
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    max_dim = max(dx, dy, dz)
    scale = TARGET_SIZE / max_dim

    com = gmsh.model.occ.getCenterOfMass(3, vol_tag)
    gmsh.model.occ.translate(volumes, -com[0], -com[1], -com[2])
    gmsh.model.occ.synchronize()
    gmsh.model.occ.dilate(volumes, 0.0, 0.0, 0.0, scale, scale, scale)
    gmsh.model.occ.synchronize()

    vol_tags = [v[1] for v in volumes]
    gmsh.model.addPhysicalGroup(3, vol_tags, DOMAIN_TAG, name="domain")

    surfaces = gmsh.model.getEntities(2)
    surf_tags = [s[1] for s in surfaces]
    gmsh.model.addPhysicalGroup(2, surf_tags, BOUNDARY_TAG, name="boundary")

    gmsh.model.mesh.generate(3)

    if debug:
        x0, y0, z0, x1, y1, z1 = gmsh.model.occ.getBoundingBox(3, vol_tag)
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        max_dim = max(dx, dy, dz)
        print(f"Bounding box: ({x0}, {y0}, {z0}) - ({x1}, {y1}, {z1})")
        print(f"Post-scale max dimension: {max_dim}")

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        tri_tags, _ = gmsh.model.mesh.getElementsByType(2)
        tet_tags, _ = gmsh.model.mesh.getElementsByType(4)

        print("Number of nodes:", len(node_tags))
        print("Number of triangles:", len(tri_tags))
        print("Number of elements:", len(tet_tags))

    gmsh.write(str(out_path))


def normalize_z(mesh: meshio.Mesh):
    """Translate the mesh so that the bottom of the bounding box is at z=0."""
    mesh.points[:, 2] -= mesh.points[:, 2].min()
    return mesh


# Characteristic length for mesh elements (1/20 for fine mesh, 1/10 for coarse mesh)
L = TARGET_SIZE / 10


gripper = RobotiqHandE("grasp/finger.step")


def compute_scale_factor(step_path: Path, rng=None, n=50) -> float:
    scene = trimesh.load(str(step_path))
    geometries = list(scene.geometry.values())
    mesh = trimesh.util.concatenate(geometries)
    mesh.fill_holes()
    mesh.vertices -= mesh.centroid
    mesh.vertices[:, 2] -= mesh.vertices[:, 2].min()

    if rng is None:
        rng = np.random.default_rng(42)

    scales = np.linspace(0.04, 0.08, num=5)

    for scale in scales[::-1]:
        mesh_copy = mesh.copy()
        mesh_copy.apply_scale(scale / mesh.extents.max())
        sampler = AntipodalGraspSampler(gripper, mesh_copy, {"friction_coeff": 0.1})
        grasps = sampler.sample(n)
        if len(grasps) == n:
            return scale
    return None


def process_file(in_path: Path, out_path: Path, file_format: str, debug: bool = False):
    """Process a single mesh file and convert it to volumetric .msh format."""
    if file_format == "stl":
        create_volume_stl(in_path, out_path)
    else:  # step format
        # scale_factor = compute_scale_factor(in_path)
        # if scale_factor is None:
        #     raise RuntimeError("Failed to compute scale factor.")
        # if debug:
        #     print(f"Computed scale factor: {scale_factor}")
        create_volume_step(in_path, out_path, debug=debug)
        mesh = meshio.read(out_path)
        mesh = normalize_z(mesh)
        meshio.gmsh.write(out_path, mesh, fmt_version="2.2", binary=False)


def main():
    args = parse_args()

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", L)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", L)
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")

    # Process single file or batch
    if args.file is not None:
        out_path = Path("meshes") / (args.file.stem + ".msh")
        process_file(args.file, out_path, args.format, debug=True)
        print(f"Processed {args.file} -> {out_path}")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0
        fail_count = 0

        for in_path in args.input_dir.glob(f"*.{args.format}"):
            out_path = args.output_dir / (in_path.stem + ".msh")
            try:
                process_file(in_path, out_path, args.format, debug=False)
                print(f"Processed {in_path} -> {out_path}")
                success_count += 1
            except Exception as e:
                print(f"Failed to process {in_path}: {e}")
                fail_count += 1
                # Reset gmsh state after failure
                gmsh.finalize()
                gmsh.initialize()
                gmsh.option.setNumber("General.Verbosity", 1)

        print(f"Finished: {success_count} succeeded, {fail_count} failed.")

    gmsh.finalize()


if __name__ == "__main__":
    main()
