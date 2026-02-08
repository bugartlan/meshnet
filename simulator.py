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

        # Align normals
        dots = candidate_normals @ normal
        candidate_normals[dots < 0] *= -1.0
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

    def compute_vm(self, uh):
        V = fem.functionspace(self.domain, ("CG", 1))
        s = self.sigma(uh) - 1.0 / 3 * ufl.tr(self.sigma(uh)) * ufl.Identity(len(uh))
        vm_expr = ufl.sqrt(1.5 * ufl.inner(s, s))

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = ufl.inner(u, v) * ufl.dx
        L = ufl.inner(vm_expr, v) * ufl.dx

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

    def probe(self, func: fem.Function, points: np.ndarray) -> np.ndarray:
        topology, cell_types, geometry = plot.vtk_mesh(func.function_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        bs = func.function_space.dofmap.index_map_bs
        grid.point_data["values"] = func.x.array.real.reshape(-1, bs)
        cloud = pyvista.PolyData(points)
        samples = cloud.sample(grid, tolerance=1e-5)

        return samples.point_data["values"].reshape(-1, bs)

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

    def plot_patch(self, loads: list[tuple[np.ndarray, np.ndarray]]):
        all_facets = []
        all_values = []

        if len(loads) > 0:
            contact_tags = list(range(2, 2 + len(loads)))
            contact_positions, contact_forces = zip(*loads)

            for point, force, tag in zip(
                contact_positions, contact_forces, contact_tags
            ):
                _, _, triangle_id = trimesh.proximity.closest_point(
                    self.object_mesh, [point]
                )
                normal = self.object_mesh.face_normals[triangle_id].flatten()

                patch, tags = self.make_contact_patch(point, normal, tag)

                if len(patch) > 0:
                    all_facets.append(patch)
                    all_values.append(tags)

            if all_facets:
                all_facets = np.concatenate(all_facets)
                all_values = np.concatenate(all_values)

                sorted_indices = np.argsort(all_facets)
                facet_tags = mesh.meshtags(
                    self.domain,
                    self.fdim,
                    all_facets[sorted_indices],
                    all_values[sorted_indices],
                )

                topology, cell_types, geometry = plot.vtk_mesh(self.domain)
                grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
                cell_tags = np.zeros(grid.n_cells, dtype=np.int32)
                f_to_c = self.domain.topology.connectivity(
                    self.fdim, self.domain.topology.dim
                )
                for f_idx, val in zip(facet_tags.indices, facet_tags.values):
                    parent_cells = f_to_c.links(f_idx)
                    if len(parent_cells) > 0:
                        cell_tags[parent_cells[0]] = val

                grid.cell_data["facet_tags"] = cell_tags

                plotter = pyvista.Plotter()
                plotter.add_mesh(
                    grid,
                    scalars="facet_tags",
                    show_edges=True,
                    scalar_bar_args={"title": "Contact Patch Tags"},
                )
                plotter.show()
