import meshio
import numpy as np
import pyvista
import trimesh
import ufl
from dolfinx import default_scalar_type, fem, geometry, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from mpi4py import MPI

from utils import msh_to_trimesh

################################ Material Properties ###################################
E = 2.0e9  # Young's modulus
nu = 0.35  # Poisson's ratio
lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
#########################################################################################

ROOT_TOLERANCE = 1e-4


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u):
    return 2 * mu * epsilon(u) + lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(len(u))


def make_contact_patch(object_mesh, query, fdim, domain, x0, r, n, tag=1):
    """Tag boundary facets whose (sample) points lie within radius r of x0."""

    def contact(x):
        closest_points, distances, triangle_ids = query.on_surface(x.T)
        triangles = object_mesh.triangles[triangle_ids]
        vertices = object_mesh.faces[triangle_ids]
        vertex_normals = object_mesh.vertex_normals[vertices]
        bary = trimesh.triangles.points_to_barycentric(triangles, closest_points)
        n_smooth = (
            bary[:, 0:1] * vertex_normals[:, 0]
            + bary[:, 1:2] * vertex_normals[:, 1]
            + bary[:, 2:3] * vertex_normals[:, 2]
        )
        n_smooth /= np.linalg.norm(n_smooth, axis=1, keepdims=True)
        return (np.linalg.norm(x.T - x0, axis=1) < r) & (
            np.dot(n_smooth, n).reshape(-1) > 0.5
        )

    candidate_facets = mesh.locate_entities_boundary(domain, fdim, contact)
    values = np.full(candidate_facets.shape, tag, dtype=np.int32)
    return candidate_facets, values


def visualize_contact(domain, facet_tags, points=None):
    # Visualize the mesh and contact patches
    plotter = pyvista.Plotter()
    fdim = domain.topology.dim - 1
    facet_topology, facet_cell_types, facet_geometry = plot.vtk_mesh(domain, fdim)
    facet_grid = pyvista.UnstructuredGrid(
        facet_topology, facet_cell_types, facet_geometry
    )
    num_facets = (
        domain.topology.index_map(fdim).size_local
        + domain.topology.index_map(fdim).num_ghosts
    )
    facet_values = np.zeros(num_facets, dtype=np.int32)
    facet_values[facet_tags.indices] = facet_tags.values
    facet_grid.cell_data["FacetTags"] = facet_values
    facet_grid.set_active_scalars("FacetTags")
    plotter.add_mesh(facet_grid, opacity=1, show_edges=True, line_width=0.1)

    if points is not None:
        sphere = pyvista.Sphere(radius=0.001)
        for p in points:
            sph = sphere.translate(p, inplace=False)
            plotter.add_mesh(sph, color="red", opacity=1)
    plotter.show_axes()
    plotter.show()


def visualize_stress(warped_grid):
    # Visualize the Von Mises stress field
    plotter = pyvista.Plotter()
    plotter.add_mesh(warped_grid, opacity=1, scalar_bar_args={"vertical": True})
    plotter.show_axes()
    plotter.show()


def is_root(x):
    return np.isclose(x[2], 0.0, atol=1e-3)


def solve(
    loads: list[tuple],
    contact_radius: float = 2.0,
    filename_msh: str = "cantilever.msh",
    debug: bool = False,
):
    comm = MPI.COMM_WORLD
    domain, _, _ = gmshio.read_from_msh(filename_msh, comm, rank=0, gdim=3)
    x = domain.geometry.x
    xmin = x.min(axis=0)
    xmax = x.max(axis=0)

    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    fdim = domain.topology.dim - 1

    object_mesh = msh_to_trimesh(meshio.read(filename_msh))
    query = trimesh.proximity.ProximityQuery(object_mesh)

    bottom_facets = mesh.locate_entities_boundary(domain, fdim, is_root)
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    u_boundary = np.array([0, 0, 0], dtype=default_scalar_type)
    bc = fem.dirichletbc(u_boundary, bottom_dofs, V)

    f = fem.Constant(domain, default_scalar_type((0.0, 0.0, 0.0)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx

    if len(loads) > 0:
        contact_tags = list(range(2, 2 + len(loads)))
        contact_positions, contact_forces = zip(*loads)
        _, _, triangle_id = trimesh.proximity.closest_point(
            object_mesh, contact_positions
        )
        vec_n = object_mesh.face_normals[triangle_id].T
        pairs = [
            make_contact_patch(
                object_mesh, query, fdim, domain, p, contact_radius, n, tag=t
            )
            for p, t, n in zip(contact_positions, contact_tags, vec_n.T)
        ]
        facets = (
            np.concatenate([f for f, _ in pairs])
            if pairs
            else np.array([], dtype=np.int32)
        )
        values = (
            np.concatenate([v for _, v in pairs])
            if pairs
            else np.array([], dtype=np.int32)
        )

        facet_tags = mesh.meshtags(domain, fdim, facets, values)
        ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
        for f, tag in zip(contact_forces, contact_tags):
            area = fem.assemble_scalar(fem.form(1.0 * ds(tag)))
            T = fem.Constant(
                domain, f / area if area > 0 else np.array([0.0, 0.0, 0.0])
            )
            L += ufl.dot(T, v) * ds(tag)

    problem = LinearProblem(
        a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()

    if np.isnan(uh.x.array).any():
        raise RuntimeError("FEA solution contains NaN values.")

    # Stress computation
    s_von_mises = sigma(uh) - 1.0 / 3 * ufl.tr(sigma(uh)) * ufl.Identity(len(uh))
    von_mises = ufl.sqrt(3 / 2 * ufl.inner(s_von_mises, s_von_mises))
    # ("DG, 0") for cell-wise constant stress
    # ("CG, 1") for nodal stress
    V_von_mises = fem.functionspace(domain, ("CG", 1))
    stress_expr_vm = fem.Expression(
        von_mises, V_von_mises.element.interpolation_points()
    )
    stresses_vm = fem.Function(V_von_mises)
    stresses_vm.interpolate(stress_expr_vm)

    if debug:
        print(f"Domain bounds: {xmin} to {xmax} (m)")
        print("bottom_facets:", bottom_facets.size, "bottom_dofs:", bottom_dofs.size)
        visualize_contact(domain, facet_tags, points=np.array(contact_positions))
        topology, cell_types, geometry = plot.vtk_mesh(domain, domain.topology.dim)
        warped_geometry = geometry + uh.x.array.reshape(-1, 3) * 1
        warped_grid = pyvista.UnstructuredGrid(topology, cell_types, warped_geometry)
        warped_grid.cell_data["VonMises"] = stresses_vm.x.petsc_vec.array
        visualize_stress(warped_grid)

    return domain, stresses_vm, uh


def interp_pyvista(domain, funcs, points):
    topology, cell_types, geometry = plot.vtk_mesh(domain, domain.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    n = grid.n_points

    for i, func in enumerate(funcs):
        data = func.x.array.reshape(n, -1)
        grid.point_data[f"f{i}"] = data

    cloud = pyvista.PolyData(points)
    samples = cloud.sample(grid, tolerance=1e-5)

    return np.column_stack([samples[f"f{i}"] for i in range(len(funcs))])


def interp(domain, funcs, points):
    """
    Evaluate a list of functions at given points inside the domain.

    Args:
        domain: The dolfinx mesh/domain.
        funcs: A list of Functions to evaluate.
        points: (N, 3) array of points where to evaluate the functions.

    Returns:
        A (N, d) array of evaluated function values, where d is the sum of the dimensions of the function outputs.
    """
    points = np.asarray(points)
    n_points = points.shape[0]

    # Build bounding box tree and find colliding cells
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, candidates, points)

    eval_point_indices = []
    eval_cell_ids = []

    query_pts = points.copy()

    found_mask = np.zeros(n_points, dtype=bool)

    for i in range(n_points):
        cell_ids = colliding_cells.links(i)
        if len(cell_ids) > 0:
            found_mask[i] = True
            for cid in cell_ids:
                # Point is inside the domain; may belong to multiple cells (e.g. boundary)
                eval_point_indices.append(i)
                eval_cell_ids.append(cid)
        else:
            print(f"Point {points[i]} not found inside the domain.")

    # No points found inside the domain
    if len(eval_point_indices) == 0:
        return np.zeros((0,), dtype=np.float64)

    counts = np.zeros((points.shape[0], 1), dtype=np.int32)
    np.add.at(counts, eval_point_indices, 1)
    mask_found = counts.flatten() > 0

    results = []
    for func in funcs:
        raw_values = func.eval(query_pts[eval_point_indices], eval_cell_ids)
        dim = raw_values.shape[1]
        accumulated_values = np.zeros((points.shape[0], dim), dtype=raw_values.dtype)
        np.add.at(accumulated_values, eval_point_indices, raw_values)
        accumulated_values[mask_found] /= counts[mask_found]
        results.append(accumulated_values[mask_found])

    return np.hstack(results)


def eval_dg0(domain, func, points):
    points = np.asarray(points)
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, candidates, points)
    cells = np.full(points.shape[0], -1, dtype=np.int32)
    for i in range(points.shape[0]):
        cell_ids = colliding_cells.links(i)
        if len(cell_ids) > 0:
            cells[i] = cell_ids[0]

    found = cells >= 0
    if np.any(found):
        values = func.eval(points[found], cells[found])
    else:
        values = np.zeros((0,), dtype=np.float64)

    return values


if __name__ == "__main__":
    loads = [
        ((0.0, 5.0, 25.0), np.array([1.0, 0.0, 0.0])),
        ((-10.0, 5.0, 25.0), np.array([-1.0, 0.0, 0.0])),
    ]
    query_pts = np.array([[0.0, 5.0, 25.0], [-10.0, 5.0, 25.0]])
    domain, stresses_vm = solve(loads, contact_radius=2.0, debug=True)
