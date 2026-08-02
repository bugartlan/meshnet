from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
from typing import Any, TypeAlias

import basix.ufl
import numpy as np
import numpy.typing as npt
import pyvista
import ufl
from dolfinx import default_scalar_type, fem, geometry, mesh, plot
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from dolfinx.mesh import create_mesh
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import KDTree

from meshnet.utils.geodesics import SurfaceGeodesics
from meshnet.utils.mesh import Mesh

YOUNG_MODULUS = 2.0e9
POISSON_RATIO = 0.35
BOUNDARY_TOLERANCE = 1e-6
PROJECTION_BISECTION_STEPS = 24

Load: TypeAlias = tuple[npt.ArrayLike, npt.ArrayLike]
MeshSource: TypeAlias = Mesh | str | PathLike[str]
ProjectionKey: TypeAlias = tuple[str, int, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class IsotropicMaterial:
    young_modulus: float = 2.0e9
    poisson_ratio: float = 0.35

    @property
    def lame_mu(self) -> float:
        """Second Lamé parameter (mu)"""
        return self.young_modulus / (2 * (1 + self.poisson_ratio))

    @property
    def lame_lambda(self) -> float:
        """First Lamé parameter (lambda)"""
        return (
            self.young_modulus
            * self.poisson_ratio
            / ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))
        )


@dataclass(frozen=True)
class _ProjectionSystem:
    function_space: Any
    test_function: Any
    mass_matrix: Any
    solver: PETSc.KSP


@dataclass
class SimulationResult:
    displacement: fem.Function
    nodal_forces: np.ndarray  # [num_nodes, 3]


class Simulator:
    """Small-strain linear-elastic FEM solver for distributed contact loads."""

    def __init__(
        self,
        obj_mesh: MeshSource,
        *,
        order: int = 2,
        contact_std: float = 0.001,
        material: IsotropicMaterial = None,
        geodesics: SurfaceGeodesics | None = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        self.comm = comm
        self.order = order
        self.contact_std = contact_std
        self.material = material or IsotropicMaterial()

        source_mesh = self._load_mesh(obj_mesh)
        self._source_points = np.asarray(source_mesh.volume.points).copy()

        self.domain = self._create_domain(source_mesh)
<<<<<<< HEAD
        self.gdim = self.domain.geometry.dim
        self.cdim = self.domain.topology.dim
        self.fdim = self.cdim - 1

        self.V = fem.FunctionSpace(self.domain, ("Lagrange", self.order, (self.gdim,)))
=======
        self.gdim = self.domain.geometry.dim  # coordinate space
        self.cdim = self.domain.topology.dim  # cell dimension (tetra=3)
        self.fdim = self.cdim - 1

        self.V = fem.functionspace(self.domain, ("Lagrange", self.order, (self.gdim,)))
>>>>>>> c2ce5de04b18df150b4e56818801e7c3418a2632

        self._surface_geodesics = geodesics or SurfaceGeodesics.from_mesh(
            source_mesh, tolerance=BOUNDARY_TOLERANCE
        )

        self.bc = self._create_fixed_bc()

<<<<<<< HEAD
=======
        self.a, self.K, self.solver = self._create_elasticity_system()

>>>>>>> c2ce5de04b18df150b4e56818801e7c3418a2632
        self._projection_systems: dict[ProjectionKey, _ProjectionSystem] = {}
        self._dof_trees: dict[int, tuple[Any, KDTree]] = {}
        self._mesh_point_dofs: dict[int, tuple[Any, np.ndarray]] = {}
        self._create_geom_search_structs()
        self._prepare_geodesic_distance_field()

        (
            self._mesh_eval_points,
            self._mesh_eval_cells,
        ) = self._prepare_mesh_point_evaluation()

    @staticmethod
    def _load_mesh(source: MeshSource) -> Mesh:
        if isinstance(source, Mesh):
            return source
        return Mesh.read(str(source))

    def _create_domain(self, source_mesh: Mesh) -> mesh.Mesh:
        points = np.asarray(source_mesh.volume.points)
        tetras = source_mesh.volume.cells_dict["tetra"]

        coordinate_element = basix.ufl.element(
            "Lagrange",
            "tetrahedron",
            1,
            shape=(points.shape[1],),
        )
        ufl_domain = ufl.Mesh(coordinate_element)

        return create_mesh(
            self.comm,
            tetras,
            points,
            ufl_domain,
        )

    def _create_fixed_bc(self) -> fem.DirichletBC:
        boundary_facets = mesh.locate_entities_boundary(
            self.domain,
            self.fdim,
            lambda x: np.isclose(
                x[2],
                0.0,
                atol=BOUNDARY_TOLERANCE,
                rtol=0.0,
            ),
        )
        boundary_dofs = fem.locate_dofs_topological(self.V, self.fdim, boundary_facets)
        zeros = np.zeros(self.gdim, dtype=default_scalar_type)
        return fem.dirichletbc(zeros, boundary_dofs, self.V)

    @staticmethod
    def epsilon(u: Any) -> Any:
        """Infinitesimal strain ε(u) = sym(grad(u))."""
        return ufl.sym(ufl.grad(u))

    def sigma(self, u: Any) -> Any:
        """Isotropic Cauchy stress σ(u)."""
        strain = self.epsilon(u)

        return (
            2.0 * self.material.lame_mu * strain
            + self.material.lame_lambda * ufl.tr(strain) * ufl.Identity(self.gdim)
        )

    def _create_elasticity_system(self) -> tuple[Any, PETSc.Mat, PETSc.KSP]:
        """Assemble the linear system for the elasticity problem."""
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # a(u, v) = ∫Ω σ(u) : ε(v) dx
        a = fem.form(ufl.inner(self.sigma(u), self.epsilon(v)) * ufl.dx)

        # stiffness matrix
        K = assemble_matrix(a, bcs=[self.bc])
        K.assemble()

        solver = self._create_direct_solver(K, factor_solver="mumps")

        return a, K, solver

    def _create_direct_solver(
        self, matrix: PETSc.Mat, *, factor_solver: str | None = None
    ) -> PETSc.KSP:
        solver = PETSc.KSP().create(self.comm)
        solver.setOperators(matrix)
        solver.setType(PETSc.KSP.Type.PREONLY)

        preconditioner = solver.getPC()
        preconditioner.setType(PETSc.PC.Type.LU)
        if factor_solver is not None:
            preconditioner.setFactorSolverType(factor_solver)

        return solver

    def _create_geom_search_structs(self) -> None:
        self.domain.topology.create_connectivity(self.fdim, self.cdim)

        cell_map = self.domain.topology.index_map(self.cdim)
        n_local_cells = cell_map.size_local + cell_map.num_ghosts

        self._cell_indices = np.arange(n_local_cells, dtype=np.int32)

        self._bounding_box_tree = geometry.bb_tree(
            self.domain, self.cdim, self._cell_indices
        )

        self._cell_midpoint_tree = geometry.create_midpoint_tree(
            self.domain, self.cdim, self._cell_indices
        )

    def _prepare_geodesic_distance_field(self) -> None:
        self._geodesic_space = fem.functionspace(self.domain, ("Lagrange", 1))

        dof_coordinates = self._geodesic_space.tabulate_dof_coordinates()
        surface_tree = KDTree(self._surface_geodesics.vertices)

        nearest_surface_vertices = surface_tree.query(dof_coordinates, k=1)[1]

        self._geodesic_dof_surface_vertices = np.asarray(
            nearest_surface_vertices, dtype=np.intp
        )

    def _geodesic_distance_field(self, point: np.ndarray) -> fem.Function:
        distances = self._surface_geodesics.distance_from(point)

        distance_field = fem.Function(self._geodesic_space)
        distance_field.x.array[:] = distances[
            self._geodesic_dof_surface_vertices
        ].astype(distance_field.x.array.dtype, copy=False)
        distance_field.x.scatter_forward()

        return distance_field

    def _traction(self, point: np.ndarray, force: np.ndarray) -> fem.Function:
        distance_field = self._geodesic_distance_field(point)
        weight = ufl.exp(-(distance_field**2) / (2 * self.contact_std**2))

        local_integral = fem.assemble_scalar(fem.form(weight * ufl.ds))
        normalization = self.comm.allreduce(local_integral, op=MPI.SUM)
        return (
            fem.Constant(
                self.domain,
                default_scalar_type(force / normalization),
            )
            * weight
        )

    @staticmethod
    def _accumulate_ghost_values(
        vector: PETSc.Vec,
        *,
        populate_ghosts: bool = False,
    ) -> None:
        vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )

        if populate_ghosts:
            vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES,
                mode=PETSc.ScatterMode.FORWARD,
            )

    def run(self, loads: Iterable[Load]) -> SimulationResult:
        """Solve for displacement under the supplied point-force pairs.

        Args:
            loads: An iterable of (point, force) pairs. Each point must be a 3D coordinate on the mesh surface, and each force must be a 3D vector.

        Returns:
            SimulationResult: Contains the displacement field and nodal forces.
        """
        zero_traction = fem.Constant(
            self.domain,
<<<<<<< HEAD
            np.zeros(
                self.geometric_dimension,
                dtype=default_scalar_type,
            ),
=======
            np.zeros(self.gdim, dtype=default_scalar_type),
>>>>>>> c2ce5de04b18df150b4e56818801e7c3418a2632
        )
        v = ufl.TestFunction(self.V)

        load_form = ufl.dot(zero_traction, v) * ufl.dx

        load_form = ufl.dot(zero_traction, self.v) * ufl.dx

        for point, force in loads:
            traction = self._traction(point, force)
<<<<<<< HEAD
            load_form += ufl.dot(traction, self.v) * ufl.ds
=======
            load_form += ufl.dot(traction, v) * ufl.ds
>>>>>>> c2ce5de04b18df150b4e56818801e7c3418a2632

        rhs = assemble_vector(fem.form(load_form))

        # Preserve a copy before lifting and Dirichlet conditions are applied
        external_load = rhs.copy()
        self._accumulate_ghost_values(
            external_load,
            populate_ghosts=True,
        )

<<<<<<< HEAD
        apply_lifting(rhs, [self.K], bcs=[[self.bc]])
=======
        apply_lifting(rhs, [self.a], bcs=[[self.bc]])
>>>>>>> c2ce5de04b18df150b4e56818801e7c3418a2632
        self._accumulate_ghost_values(rhs, populate_ghosts=True)
        set_bc(rhs, [self.bc])

        displacement = fem.Function(self.V)
        self.solver.solve(rhs, displacement.x.petsc_vec)
        displacement.x.scatter_forward()

        external_load_dofs = (
            np.asarray(external_load.array.real, dtype=np.float64)
            .reshape(-1, self.gdim)
            .copy()
        )

        return SimulationResult(
            displacement=displacement,
            nodal_forces=external_load_dofs,
        )

    def _get_projection_system(
        self,
        family: str,
        degree: int,
        shape: tuple[int, ...] = (),
    ) -> _ProjectionSystem:
        key = (family, degree, shape)

        cached = self._projection_systems.get(key)
        if cached is not None:
            return cached

        element = (family, degree) if not shape else (family, degree, shape)
        function_space = fem.functionspace(self.domain, element)
        u = ufl.TestFunction(function_space)
        v = ufl.TrialFunction(function_space)
        mass_form = fem.form(ufl.inner(u, v) * ufl.dx)
        mass_matrix = assemble_matrix(mass_form)
        mass_matrix.assemble()

        system = _ProjectionSystem(
            function_space=function_space,
            test_function=u,
            mass_matrix=mass_matrix,
            solver=self._create_direct_solver(mass_matrix, factor_solver="mumps"),
        )
        self._projection_systems[key] = system

        return system

    def _project(
        self, expression: Any, family: str, degree: int, shape: tuple[int, ...] = ()
    ) -> fem.Function:
        system = self._get_projection_system(family, degree, shape)
        rhs = assemble_vector(
            fem.form(ufl.inner(expression, system.test_function) * ufl.dx)
        )
        self._accumulate_ghost_values(rhs, populate_ghosts=True)

        result = fem.Function(system.function_space)
        system.solver.solve(rhs, result.x.petsc_vec)
        result.x.scatter_forward()

        return result

    def stress_voigt(self, displacement: fem.Function) -> fem.Function:
        """[xx, yy, zz, xy, yz, xz]"""
        stress = self.sigma(displacement)
        return ufl.as_vector(
            [
                stress[0, 0],
                stress[1, 1],
                stress[2, 2],
                stress[0, 1],
                stress[1, 2],
                stress[0, 2],
            ]
        )

    def compute_stress(self, displacement: fem.Function, degree: int) -> fem.Function:
        """Project stress onto a continuous Lagrange space.

        A continuous projection gives each mesh vertex one unambiguous stress
        value. This is required for node-based graph targets: evaluating a
        discontinuous stress field at a shared vertex otherwise selects an
        arbitrary adjacent cell's value.
        """
        sigma_voigt = self.stress_voigt(displacement)
        return self._project(
            sigma_voigt,
            family="Lagrange",
            degree=degree,
            shape=(6,),
        )

    def _colliding_cell_indices(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = geometry.compute_collisions_points(self._bounding_box_tree, points)
        collisions = geometry.compute_colliding_cells(self.domain, candidates, points)

        point_indices: list[int] = []
        cell_indices: list[int] = []
        for i in range(len(points)):
            cells = collisions.links(i)
            if len(cells):
                point_indices.append(i)
                cell_indices.append(cells[0])

        return (
            np.asarray(point_indices, dtype=np.int32),
            np.asarray(cell_indices, dtype=np.int32),
        )

    def _prepare_mesh_point_evaluation(self) -> tuple[np.ndarray, np.ndarray]:
        """Cache evaluation coordinates and cells for the volume-mesh points."""
        evaluation_points = self._source_points.copy()
        evaluation_cells = np.full(len(self._source_points), -1, dtype=np.int32)

        point_indices, cell_indices = self._colliding_cell_indices(self._source_points)
        evaluation_cells[point_indices] = cell_indices

        missing_indices = np.flatnonzero(evaluation_cells < 0)
        if missing_indices.size:
            projected_indices, projected_points, projected_cells = self._project_inside(
                self._source_points[missing_indices]
            )
            resolved_indices = missing_indices[projected_indices]
            evaluation_points[resolved_indices] = projected_points
            evaluation_cells[resolved_indices] = projected_cells

        return evaluation_points, evaluation_cells

    def _project_inside(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Move outside points to the domain boundary from an interior anchor.

        The closest volume cell supplies a mapped midpoint that is inside the
        domain. Bisection along the midpoint-to-query segment approaches the
        boundary without requiring a separate triangulated surface.

        Returns:
            Indices of successfully projected input points, their projected
            coordinates, and the cells containing those coordinates.
        """
        if not len(points):
            empty_indices = np.empty(0, dtype=np.int32)
            empty_points = np.empty((0, self.domain.geometry.dim))
            return empty_indices, empty_points, empty_indices

        closest_cells = geometry.compute_closest_entity(
            self._bounding_box_tree,
            self._cell_midpoint_tree,
            self.domain,
            points,
        )
        projectable_indices = np.flatnonzero(closest_cells >= 0).astype(np.int32)
        if not projectable_indices.size:
            empty_points = np.empty((0, self.domain.geometry.dim))
            return projectable_indices, empty_points, projectable_indices

        cell_dimension = self.domain.topology.dim
        inside_points = mesh.compute_midpoints(
            self.domain,
            cell_dimension,
            closest_cells[projectable_indices],
        )

        # A highly distorted curved element can have an unusable geometric
        # midpoint. Keep only anchors that DOLFINx confirms are inside.
        valid_anchor_indices, anchor_cells = self._colliding_cell_indices(inside_points)
        projectable_indices = projectable_indices[valid_anchor_indices]
        inside_points = inside_points[valid_anchor_indices]
        if not projectable_indices.size:
            return projectable_indices, inside_points, anchor_cells

        outside_points = points[projectable_indices].copy()
        containing_cells = anchor_cells.copy()
        for _ in range(PROJECTION_BISECTION_STEPS):
            candidates = 0.5 * (inside_points + outside_points)
            inside_candidate_indices, candidate_cells = self._colliding_cell_indices(
                candidates
            )
            candidate_is_inside = np.zeros(len(candidates), dtype=bool)
            candidate_is_inside[inside_candidate_indices] = True

            inside_points[candidate_is_inside] = candidates[candidate_is_inside]
            containing_cells[inside_candidate_indices] = candidate_cells
            outside_points[~candidate_is_inside] = candidates[~candidate_is_inside]

        return projectable_indices, inside_points, containing_cells

    def _nearest_dof_tree(self, function_space: Any) -> KDTree:
        key = id(function_space)
        cached = self._dof_trees.get(key)
        if cached is None or cached[0] is not function_space:
            cached = (
                function_space,
                KDTree(function_space.tabulate_dof_coordinates()),
            )
            self._dof_trees[key] = cached
        return cached[1]

    def evaluate_source_points(self, function: fem.Function) -> np.ndarray:
        """Evaluate a function at the volume-mesh points using cached cells.

        The geometric search is performed once when the simulator is created.
        Any mesh point that cannot be associated with a local cell uses a
        cached nearest degree of freedom, matching :meth:`probe`'s final
        fallback.
        """
        function_space = function.function_space
        block_size = function_space.dofmap.index_map_bs
        values = np.zeros((len(self._source_points), block_size), dtype=np.float64)

        resolved = self._mesh_eval_cells >= 0
        if np.any(resolved):
            values[resolved] = function.eval(
                self._mesh_eval_points[resolved],
                self._mesh_eval_cells[resolved],
            )

        missing_indices = np.flatnonzero(~resolved)
        if missing_indices.size:
            key = id(function_space)
            cached = self._mesh_point_dofs.get(key)
            if cached is None or cached[0] is not function_space:
                nearest_dofs = self._nearest_dof_tree(function_space).query(
                    self._source_points[missing_indices], k=1
                )[1]
                cached = (function_space, np.asarray(nearest_dofs, dtype=np.intp))
                self._mesh_point_dofs[key] = cached

            nodal_values = function.x.array.real.reshape(-1, block_size)
            values[missing_indices] = nodal_values[cached[1]]

        return values

    # Visualization

    @staticmethod
    def _build_grid(function_space: Any) -> pyvista.UnstructuredGrid:
        topology, cell_types, geometry_points = plot.vtk_mesh(function_space)
        return pyvista.UnstructuredGrid(topology, cell_types, geometry_points)

    def _render_grid(
        self,
        grid: pyvista.UnstructuredGrid,
        scalar_name: str,
        scalar_bar_title: str,
        slice_bottom: bool = False,
    ) -> pyvista.Plotter:
        target = (
            grid.slice(
                normal="z",
                origin=(0, 0, self.bottom_z + BOUNDARY_TOLERANCE),
            )
            if slice_bottom
            else grid
        )
        plotter = pyvista.Plotter()
        plotter.add_mesh(
            target,
            scalars=scalar_name,
            show_edges=True,
            scalar_bar_args={"title": scalar_bar_title},
        )
        plotter.show_axes()
        plotter.show()
        return plotter

    def plot_displacement(self, displacement: fem.Function) -> pyvista.Plotter:
        grid = self._build_grid(displacement.function_space)
        grid.point_data["displacement"] = displacement.x.array.real.reshape(
            -1, self.domain.geometry.dim
        )
        return self._render_grid(
            grid,
            scalar_name="displacement",
            scalar_bar_title="Displacement (m)",
        )

    def plot_stress(
        self,
        stress: fem.Function,
        component: str,
        slice_bottom: bool = False,
    ) -> pyvista.Plotter:
        """Plot a specific component of the stress tensor.

        Args:
            stress: The projected stress function in Voigt notation.
            component: The stress component to plot ('xx', 'yy', 'zz', 'xy', 'yz', 'xz').
            slice_bottom: Whether to slice the plot at the bottom.
        """
        # Map the string name to the correct Voigt array column index
        voigt_indices = {"xx": 0, "yy": 1, "zz": 2, "xy": 3, "yz": 4, "xz": 5}

        component = component.lower()
        if component not in voigt_indices:
            raise ValueError(
                f"Invalid component '{component}'. Expected one of {list(voigt_indices.keys())}"
            )

        index = voigt_indices[component]

        grid = self._build_grid(stress.function_space)
        stress_array = stress.x.array.real.reshape(-1, 6)
        grid.point_data[f"stress_{component}"] = stress_array[:, index]
        return self._render_grid(
            grid,
            scalar_name=f"stress_{component}",
            scalar_bar_title=f"Stress {component} (Pa)",
            slice_bottom=slice_bottom,
        )
