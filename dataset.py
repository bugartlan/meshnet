"""Dataset generation module for mesh-based finite element analysis.

This module generates training datasets by:
1. Loading mesh files (.msh format)
2. Sampling contact points on the mesh surface
3. Applying forces at contact points (random or gripper-based)
4. Running FEA simulations to compute stress and displacement
5. Building graph representations for machine learning

Supports two generation modes:
- generate_random_contacts: Random contact points with random forces
- generate_antipodal_grasps: Antipodal grasp points with gripper closing forces
"""

import argparse
import time
from pathlib import Path

import meshio
import numpy as np
import torch
import trimesh

from graphs import build_graph
from grasp.AntipodalGrasp import AntipodalGraspSampler
from grasp.Gripper import RobotiqHandE
from solver import interp_pyvista, solve
from utils import info, msh_to_trimesh


def parse_args():
    """Parse command-line arguments for dataset generation.

    Returns:
        argparse.Namespace: Parsed arguments containing mesh name, sampling
            parameters, output configuration, and runtime flags.
    """
    p = argparse.ArgumentParser(
        description="Generate FEA dataset from mesh files with force/contact sampling."
    )

    # --- IO Configuration ---
    p.add_argument(
        "mesh_name",
        type=str,
        help="Name of the mesh family to sample (e.g., 'cantilever'). Matches folder in meshes/.",
    )
    p.add_argument(
        "--file", type=str, default=None, help="Optional specific mesh file to process."
    )
    p.add_argument(
        "--coarse",
        "--coarse-mesh",
        action="store_true",
        help="If set, use the coarse mesh for sampling instead of the fine mesh.",
    )

    # --- Output Configuration ---
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save the generated .pt file.",
    )
    p.add_argument(
        "--output-filename",
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
        default=15.0,
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
        domain, stresses_vm, uh = solve(
            contacts,
            contact_radius=contact_radius,
            debug=debug,
            filename_msh=str(msh_path),
        )

        # Interpolate and build graph
        y = interp_pyvista(domain, [uh, stresses_vm], mesh.points)
        graphs.append(build_graph(mesh, y, radius=contact_radius, contacts=contacts))

    return graphs


def get_adaptive_radius(mesh: trimesh.Trimesh, multiplier: float = 2.0) -> float:
    """Compute an adaptive contact radius based on mesh resolution.

    The radius scales with the average edge length to adapt to different mesh densities.

    Args:
        mesh: The trimesh object to analyze.
        multiplier: Scale factor for the average edge length (default: 2.0).

    Returns:
        Adaptive radius value for contact point neighborhoods.

    Note:
        TODO: Consider using a fixed radius for consistency across meshes.
    """
    avg_edge_length = mesh.edges_unique_length.mean()
    return multiplier * avg_edge_length


def sample(mesh: trimesh.Trimesh, num_points: int, seed: int = 42) -> np.ndarray:
    """Sample points uniformly on the mesh surface.

    Args:
        mesh: The trimesh object to sample from.
        num_points: Number of points to return.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (num_points, 3) with sampled surface points.

    Note:
        Samples 2x points initially to ensure enough remain after filtering.
    """
    points, _ = trimesh.sample.sample_surface(
        mesh, count=int(num_points * 2), seed=seed
    )
    # Filter out points near or below the bottom surface (z > 0.001)
    points = points[points[:, 2] > 0.001]
    return points[:num_points]


def generate_random_contacts(
    file: Path,
    n_samples: int,
    n_contacts: int,
    force_max: float,
    rng: np.random.Generator,
    file_coarse: Path = None,
    seed: int = 42,
    debug: bool = False,
    output_path: str = None,
):
    """Generate dataset using random contact points and forces.

    This method:
    1. Samples random points on the mesh surface
    2. Applies random force vectors at each contact point
    3. Runs FEA simulation for each sample
    4. Builds graph representations from FEA results

    Args:
        file: Path to the fine mesh .msh file.
        n_samples: Number of dataset samples to generate.
        n_contacts: Number of contact points per sample.
        force_max: Maximum magnitude for random force components.
        rng: NumPy random generator for reproducibility.
        file_coarse: Optional path to coarse mesh for faster sampling.
        seed: Random seed for point sampling.
        debug: Enable verbose logging.
        output_path: Where to save the generated .pt file.

    Raises:
        FileNotFoundError: If mesh files don't exist.
    """
    # Validate mesh files exist
    if not file.exists():
        raise FileNotFoundError(f"Missing MSH: {file}")
    if file_coarse is not None and not file_coarse.exists():
        raise FileNotFoundError(f"Missing coarse MSH: {file_coarse}")

    # Load mesh
    # Load mesh (use coarse version if specified for faster processing)
    mesh_mio = meshio.read(file_coarse if file_coarse is not None else file)
    mesh_tri = msh_to_trimesh(mesh_mio)

    # Sample contact points: n_samples × n_contacts points, then reshape
    points = sample(mesh_tri, n_samples * n_contacts, seed=seed)
    points = points.reshape(
        n_samples, n_contacts, 3
    )  # Shape: (n_samples, n_contacts, 3)

    # Generate random force vectors for each contact point
    forces = rng.uniform(
        low=-force_max,
        high=force_max,
        size=(n_samples, n_contacts, 3),  # Shape: (n_samples, n_contacts, 3)
    )

    contact_radius = get_adaptive_radius(mesh_tri, multiplier=2.0)
    graphs = make_graphs(mesh_mio, forces, points, file, contact_radius, debug=debug)

    node_dim, edge_dim, output_dim = info(graphs[0], debug=False)
    torch.save(
        {
            "params": {
                "node_dim": node_dim,
                "edge_dim": edge_dim,
                "output_dim": output_dim,
            },
            "graphs": graphs,
            "mesh": file.stem,
        },
        output_path,
    )
    print(f"Saved {len(graphs)} graphs to {output_path}.")


def generate_antipodal_grasps(
    file: Path,
    gripper,
    n_samples: int,
    force_max: float,
    rng: np.random.Generator,
    closing_force: float = 1,
    file_coarse: Path = None,
    seed: int = 42,
    debug: bool = False,
    output_path: str = None,
):
    """Generate dataset using antipodal grasp points with gripper forces.

    This method:
    1. Samples antipodal grasp points (2 contact points per sample)
    2. Computes gripper closing forces based on surface normals
    3. Adds random force perturbations
    4. Runs FEA simulation for each grasp configuration
    5. Builds graph representations from FEA results

    Args:
        file: Path to the fine mesh .msh file.
        gripper: Gripper object for antipodal grasp sampling.
        n_samples: Number of grasp samples to generate.
        force_max: Maximum magnitude for random force perturbations.
        rng: NumPy random generator for reproducibility.
        closing_force: Magnitude of gripper closing force (default: 1).
        file_coarse: Optional path to coarse mesh for faster sampling.
        seed: Random seed for grasp sampling.
        debug: Enable verbose logging.
        output_path: Where to save the generated .pt file.

    Raises:
        FileNotFoundError: If mesh files don't exist.
        RuntimeError: If insufficient valid grasps can be sampled.
    """
    # Validate mesh files exist
    if not file.exists():
        raise FileNotFoundError(f"Missing MSH: {file}")
    if file_coarse is not None and not file_coarse.exists():
        raise FileNotFoundError(f"Missing coarse MSH: {file_coarse}")

    # Load mesh
    mesh_mio = meshio.read(file_coarse if file_coarse is not None else file)
    mesh_tri = msh_to_trimesh(mesh_mio)

    sampler = AntipodalGraspSampler(gripper, mesh_tri, {"friction_coeff": 0.1})

    # Sample grasps
    if n_samples < 50:
        grasps = sampler.sample(n_samples)
    else:
        candidates = sampler.sample(50)
        grasps = rng.choice(candidates, size=n_samples, replace=True)

    if len(grasps) < n_samples:
        raise RuntimeError(f"Could only sample {len(grasps)} grasps for {file}.")

    # Extract contact point positions (2 points per grasp)
    points = np.array([[grasp.x1.position, grasp.x2.position] for grasp in grasps])

    f_random = rng.uniform(low=-force_max, high=force_max, size=(n_samples, 3))

    # Extract surface normals at contact points
    normals = np.array([[grasp.x1.normal, grasp.x2.normal] for grasp in grasps])

    # Compute gripper closing direction (normalized difference between normals)
    normal_diff = normals[:, 0, :] - normals[:, 1, :]
    normal_diff = normal_diff / np.linalg.norm(normal_diff, axis=1, keepdims=True)

    # Align normals to point toward each other (inward) for closing force
    alignment_signs = -np.sign((normal_diff[:, np.newaxis, :] * normals).sum(axis=2))
    aligned_normals = normal_diff[:, np.newaxis, :] * alignment_signs[:, :, np.newaxis]

    # Force is random force + gripper closing force
    forces = closing_force * aligned_normals + f_random[:, np.newaxis, :]

    contact_radius = get_adaptive_radius(mesh_tri, multiplier=2.0)
    graphs = make_graphs(mesh_mio, forces, points, file, contact_radius, debug=debug)

    node_dim, edge_dim, output_dim = info(graphs[0], debug=False)
    torch.save(
        {
            "params": {
                "node_dim": node_dim,
                "edge_dim": edge_dim,
                "output_dim": output_dim,
            },
            "graphs": graphs,
            "mesh": file.stem,
        },
        output_path,
    )
    print(f"Saved {len(graphs)} graphs to {output_path}.")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set up mesh paths
    msh_path = Path("meshes") / args.mesh_name / "msh"
    msh_coarse_path = Path("meshes") / args.mesh_name / "msh_coarse"

    # Initialize random number generator
    rng = np.random.default_rng(args.seed)
    start = time.time()

    # Determine which mesh files to process
    files_to_process = (
        [msh_path / args.file] if args.file is not None else msh_path.glob("*.msh")
    )

    # Load gripper model (needed for generate_antipodal_grasps)
    if args.num_contacts == 2:
        gripper = RobotiqHandE("grasp/finger.step")

    # Process each mesh file
    for f in files_to_process:
        name = f.stem
        f_coarse = msh_coarse_path / f"{name}.msh" if args.coarse else None

        # Construct output filename
        if args.output_filename is not None:
            output_path = args.output_dir / args.output_filename
        else:
            # Auto-generate filename: meshname_contacts_samples_c/f.pt
            filename = f"{name}_{args.num_contacts}_{args.num_samples}_{'c' if args.coarse else 'f'}.pt"
            output_path = args.output_dir / filename

        # Select generation method based on number of contacts
        if args.num_contacts == 2:
            # Use antipodal grasps for 2-contact scenarios
            generate_antipodal_grasps(
                f,
                gripper,
                args.num_samples,
                args.force_max,
                rng,
                closing_force=10.0,
                file_coarse=f_coarse,
                seed=args.seed,
                debug=args.debug,
                output_path=output_path,
            )
        else:
            generate_random_contacts(
                f,
                args.num_samples,
                args.num_contacts,
                args.force_max,
                rng,
                file_coarse=f_coarse,
                seed=args.seed,
                debug=args.debug,
                output_path=output_path,
            )

    end = time.time()
    print(f"Data generation took {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()
