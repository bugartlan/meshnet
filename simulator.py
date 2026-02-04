import meshio
import numpy as np
import pyvista
import trimesh
import ufl
from dolfinx import default_scalar_type, fem, mesh, plot
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

    def compute_facet_normals(self, facets):
        if len(facets) == 0:
            return np.empty((0, 3))

        facet_nodes = mesh.entities_to_geometry(self.domain, self.fdim, facets)
        coords = self.domain.geometry.x[facet_nodes]

        vec1 = coords[:, 1] - coords[:, 0]
        vec2 = coords[:, 2] - coords[:, 0]
        normals = np.cross(vec1, vec2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Prevent division by zero
        return normals / norms

    def make_contact_patch(self, x, normal, tag):
        x = np.asarray(x)

        candidate_facets = mesh.locate_entities_boundary(
            self.domain,
            self.fdim,
            lambda y: np.linalg.norm(y.T - x, axis=1) < self.contact_radius,
        )

        candidate_normals = self.compute_facet_normals(candidate_facets)

        mask = np.dot(candidate_normals, normal) > 0.5  # within ~60 degrees
        final_facets = candidate_facets[mask]
        values = np.full(final_facets.shape, tag, dtype=np.int32)

        return final_facets, values

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

        # cc = fem.Constant(self.domain, default_scalar_type(1.0))
        # total_force_applied = fem.assemble_scalar(fem.form(ufl.dot(T, cc) * ufl.ds))

        # print(f"Input Force: {force[0]:.4f} N")  # Assuming force is x-aligned
        # print(f"Solver Integrated: {total_force_applied:.4f} N")

        return uh

    # def run(self, loads: list[tuple[np.ndarray, np.ndarray]]):
    #     L_form = (
    #         ufl.dot(
    #             fem.Constant(self.domain, default_scalar_type((0.0, 0.0, 0.0))), self.v
    #         )
    #         * ufl.dx
    #     )

    #     all_facets = []
    #     all_values = []
    #     active_loads = []

    #     if len(loads) > 0:
    #         contact_tags = list(range(2, 2 + len(loads)))
    #         contact_positions, contact_forces = zip(*loads)

    #         for point, force, tag in zip(
    #             contact_positions, contact_forces, contact_tags
    #         ):
    #             _, _, triangle_id = trimesh.proximity.closest_point(
    #                 self.object_mesh, [point]
    #             )
    #             normal = self.object_mesh.face_normals[triangle_id].flatten()

    #             patch, tags = self.make_contact_patch(point, normal, tag)

    #             if len(patch) > 0:
    #                 all_facets.append(patch)
    #                 all_values.append(tags)
    #                 active_loads.append((force, tag))

    #         if all_facets:
    #             all_facets = np.concatenate(all_facets)
    #             all_values = np.concatenate(all_values)

    #             sorted_indices = np.argsort(all_facets)
    #             facet_tags = mesh.meshtags(
    #                 self.domain,
    #                 self.fdim,
    #                 all_facets[sorted_indices],
    #                 all_values[sorted_indices],
    #             )

    #             ds = ufl.Measure("ds", self.domain, subdomain_data=facet_tags)

    #             for force, tag in active_loads:
    #                 area = fem.assemble_scalar(fem.form(1.0 * ds(tag)))
    #                 if area > 1e-12:
    #                     T = fem.Constant(
    #                         self.domain, default_scalar_type(np.array(force) / area)
    #                     )
    #                     L_form += ufl.dot(T, self.v) * ds(tag)

    #     L_compiled = fem.form(L_form)
    #     b = assemble_vector(L_compiled)

    #     # Apply BC to RHS
    #     apply_lifting(b, [self.bilinear_form], bcs=[[self.bc]])
    #     b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    #     set_bc(b, [self.bc])

    #     # Solve linear system
    #     uh = fem.Function(self.V)
    #     self.solver.solve(b, uh.x.petsc_vec)
    #     uh.x.scatter_forward()

    #     return uh

    def compute_vm(self, uh):
        V = fem.functionspace(self.domain, ("CG", 1))
        s = self.sigma(uh) - 1.0 / 3 * ufl.tr(self.sigma(uh)) * ufl.Identity(len(uh))
        vm = ufl.sqrt(1.5 * ufl.inner(s, s))

        # Interpolate expression to function
        stress_func = fem.Function(V)
        expr = fem.Expression(vm, V.element.interpolation_points())
        stress_func.interpolate(expr)

        return stress_func

    def probe(self, vm: fem.Function, points: np.ndarray) -> np.ndarray:
        topology, cell_types, geometry = plot.vtk_mesh(vm.function_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data["vm"] = vm.x.array.real
        cloud = pyvista.PolyData(points)
        samples = cloud.sample(grid, tolerance=1e-5)

        return samples.point_data["vm"]

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
            cmap="jet",
            show_edges=True,
            scalar_bar_args={"title": "Von Mises Stress (Pa)"},
        )
        plotter.show()
