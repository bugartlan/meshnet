import meshio
import numpy as np
import pyvista
import trimesh
import ufl
from dolfinx import default_scalar_type, fem, geometry, mesh, plot
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from dolfinx.io import gmshio
from mpi4py import MPI
from petsc4py import PETSc

from utils import msh_to_trimesh


class Simulator:
    def __init__(
        self, filename_msh: str, contact_radius: float = 0.01, std: float = 0.001
    ):
        self.contact_radius = contact_radius
        self.std = std

        # Load mesh from .msh file
        self.comm = MPI.COMM_WORLD
        self.domain, _, _ = gmshio.read_from_msh(
            filename_msh, self.comm, rank=0, gdim=3
        )

        # Function space
        element_order = self.domain.geometry.cmap.degree
        print(f"Using Lagrange elements of order {element_order} for simulation.")
        self.V = fem.functionspace(
            self.domain, ("Lagrange", element_order, (self.domain.geometry.dim,))
        )

        # Constants
        E = 2.0e9  # Young's modulus
        nu = 0.35  # Poisson's ratio
        self.lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu_ = E / (2 * (1 + nu))

        u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        # BC
        self.fdim = self.domain.topology.dim - 1
        bottom_facets = mesh.locate_entities_boundary(
            self.domain, self.fdim, lambda x: np.isclose(x[2], 0.0, atol=1e-6)
        )
        bottom_dofs = fem.locate_dofs_topological(self.V, self.fdim, bottom_facets)
        self.bc = fem.dirichletbc(
            np.zeros((3,), dtype=default_scalar_type), bottom_dofs, self.V
        )

        # Stiffness matrix
        a = ufl.inner(self.sigma(u), self.epsilon(self.v)) * ufl.dx
        self.bilinear_form = fem.form(a)
        self.A = assemble_matrix(self.bilinear_form, bcs=[self.bc])
        self.A.assemble()

        # Linear Solver (LU Factorization)
        self.solver = PETSc.KSP().create(self.comm)
        self.solver.setOperators(self.A)
        self.solver.setType("preonly")
        self.solver.getPC().setType("lu")
        self.solver.getPC().setFactorSolverType("mumps")

        # Contact search tree
        self.object_mesh = msh_to_trimesh(meshio.read(filename_msh))
        self.query = trimesh.proximity.ProximityQuery(self.object_mesh)

        self.domain.topology.create_connectivity(self.fdim, self.domain.topology.dim)

    def epsilon(self, u):
        return ufl.sym(ufl.grad(u))

    def sigma(self, u):
        return 2 * self.mu_ * self.epsilon(u) + self.lambda_ * ufl.tr(
            self.epsilon(u)
        ) * ufl.Identity(len(u))

    def run(self, loads: list[tuple[np.ndarray, np.ndarray]]):
        x = ufl.SpatialCoordinate(self.domain)

        L_form = (
            ufl.dot(
                fem.Constant(self.domain, default_scalar_type((0.0, 0.0, 0.0))), self.v
            )
            * ufl.dx
        )

        norm_factor = 1.0 / (2 * np.pi * self.std**2)

        if len(loads) > 0:
            for point, force in loads:
                point = fem.Constant(self.domain, default_scalar_type(point))
                diff = x - point
                dist_sq = ufl.dot(diff, diff)

                weights = ufl.exp(-dist_sq / (2 * self.std**2))

                T = (
                    fem.Constant(self.domain, default_scalar_type(force * norm_factor))
                    * weights
                )
                L_form += ufl.dot(T, self.v) * ufl.ds
        L_compiled = fem.form(L_form)
        b = assemble_vector(L_compiled)

        # Apply BC to RHS
        apply_lifting(b, [self.bilinear_form], bcs=[[self.bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [self.bc])

        # Solve linear system
        uh = fem.Function(self.V)
        self.solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        return uh

    def compute_vm0(self, uh):
        V = fem.functionspace(self.domain, ("DG", 0))
        s = self.sigma(uh) - 1.0 / 3 * ufl.tr(self.sigma(uh)) * ufl.Identity(len(uh))
        vm_expr = ufl.sqrt(1.5 * ufl.inner(s, s))

        w = ufl.TestFunction(V)
        vm0 = ufl.TrialFunction(V)

        a = ufl.inner(vm0, w) * ufl.dx
        L = ufl.inner(vm_expr, w) * ufl.dx

        A = assemble_matrix(fem.form(a))
        A.assemble()
        b = assemble_vector(fem.form(L))

        solver = PETSc.KSP().create(self.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")

        vm = fem.Function(V)
        solver.solve(b, vm.x.petsc_vec)
        vm.x.scatter_forward()
        return vm

    def compute_vm1(self, uh):
        vm0 = self.compute_vm0(uh)
        V = fem.functionspace(self.domain, ("CG", 1))
        w = ufl.TestFunction(V)
        v = ufl.TrialFunction(V)

        a = ufl.inner(v, w) * ufl.dx
        L = ufl.inner(vm0, w) * ufl.dx

        A = assemble_matrix(fem.form(a))
        A.assemble()
        b = assemble_vector(fem.form(L))

        solver = PETSc.KSP().create(self.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")

        vm1 = fem.Function(V)
        solver.solve(b, vm1.x.petsc_vec)
        vm1.x.scatter_forward()
        return vm1

    # def probe(self, func: fem.Function, points: np.ndarray) -> np.ndarray:
    #     """
    #     Probe the function `func` at the given `points`.
    #     This method uses PyVista to interpolate the function values at the specified points.
    #     """
    #     topology, cell_types, geometry = plot.vtk_mesh(func.function_space)
    #     grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    #     bs = func.function_space.dofmap.index_map_bs
    #     grid.point_data["values"] = func.x.array.real.reshape(-1, bs)
    #     cloud = pyvista.PolyData(points)
    #     samples = cloud.interpolate(grid, sharpness=20)
    #     # samples = cloud.sample(grid)

    #     return samples.point_data["values"].reshape(-1, bs)

    def probe(self, func: fem.Function, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        n_points = len(points)
        bs = func.function_space.dofmap.index_map_bs

        # Always return one value per query point.
        values = np.zeros((n_points, bs), dtype=np.float64)
        found = np.zeros(n_points, dtype=bool)

        # First pass: direct FE evaluation at points with colliding cells.
        bb_tree = geometry.bb_tree(self.domain, self.domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(
            self.domain, cell_candidates, points
        )

        eval_idx = []
        eval_points = []
        eval_cells = []
        for i, point in enumerate(points):
            links = colliding_cells.links(i)
            if len(links) > 0:
                eval_idx.append(i)
                eval_points.append(point)
                eval_cells.append(links[0])

        if len(eval_points) > 0:
            eval_values = func.eval(np.asarray(eval_points), np.asarray(eval_cells))
            values[np.asarray(eval_idx)] = eval_values
            found[np.asarray(eval_idx)] = True

        # Second pass: project misses to closest surface and nudge inward.
        missing_idx = np.where(~found)[0]
        if len(missing_idx) > 0:
            missing_points = points[missing_idx]
            closest_points, _, triangle_id = self.query.on_surface(missing_points)

            normals = self.object_mesh.face_normals[np.asarray(triangle_id).ravel()]
            bbox = self.domain.geometry.x
            bbox_diag = np.linalg.norm(bbox.max(axis=0) - bbox.min(axis=0))
            eps = max(1e-12, 1e-7 * bbox_diag)
            projected_points = closest_points - eps * normals

            proj_candidates = geometry.compute_collisions_points(
                bb_tree, projected_points
            )
            proj_colliding = geometry.compute_colliding_cells(
                self.domain, proj_candidates, projected_points
            )

            proj_idx = []
            proj_points = []
            proj_cells = []
            for local_i, global_i in enumerate(missing_idx):
                links = proj_colliding.links(local_i)
                if len(links) > 0:
                    proj_idx.append(global_i)
                    proj_points.append(projected_points[local_i])
                    proj_cells.append(links[0])

            if len(proj_points) > 0:
                proj_values = func.eval(np.asarray(proj_points), np.asarray(proj_cells))
                values[np.asarray(proj_idx)] = proj_values
                found[np.asarray(proj_idx)] = True

        # Final fallback: interpolate from nodal values (guarantees no missing labels).
        missing_idx = np.where(~found)[0]
        if len(missing_idx) > 0:
            topology, cell_types, geom = plot.vtk_mesh(func.function_space)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
            grid.point_data["values"] = func.x.array.real.reshape(-1, bs)

            cloud = pyvista.PolyData(points[missing_idx])
            radius = 2.0 * np.linalg.norm(
                self.domain.geometry.x.max(axis=0) - self.domain.geometry.x.min(axis=0)
            )
            sampled = cloud.interpolate(grid, radius=radius, sharpness=1.0)
            values[missing_idx] = sampled.point_data["values"].reshape(-1, bs)

        return values.clip(min=0.0)  # Ensure non-negative values

    def plot_displacement(self, uh):
        topology, cell_types, geometry = plot.vtk_mesh(uh.function_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data["displacement"] = uh.x.array.real.reshape(-1, 3)

        plotter = pyvista.Plotter()
        plotter.add_mesh(
            grid,
            scalars="displacement",
            show_edges=True,
            scalar_bar_args={"title": "Displacement (m)"},
        )
        plotter.show_axes()
        plotter.show()

    def plot_vm(self, vm):
        topology, cell_types, geometry = plot.vtk_mesh(vm.function_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data["vm"] = vm.x.array.real

        plotter = pyvista.Plotter()
        plotter.add_mesh(
            grid,
            scalars="vm",
            show_edges=True,
            scalar_bar_args={"title": "Von Mises Stress (Pa)"},
        )
        plotter.show_axes()
        plotter.show()

    def plot_vm_bottom(self, vm):
        topology, cell_types, geometry = plot.vtk_mesh(vm.function_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data["vm"] = vm.x.array.real

        slice_z = 1e-6
        sliced = grid.slice(normal="z", origin=(0, 0, slice_z))

        plotter = pyvista.Plotter()
        plotter.add_mesh(
            sliced,
            scalars="vm",
            show_edges=True,
            scalar_bar_args={"title": "Von Mises Stress (Pa)"},
        )
        plotter.show_axes()
        plotter.show()
