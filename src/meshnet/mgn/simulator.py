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
from dolfinx.io import gmshio
from dolfinx.mesh import create_mesh
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import KDTree

from meshnet.mgn.geodesics import SurfaceGeodesics
from meshnet.utils.mesh import Mesh

YOUNG_MODULUS = 2.0e9
POISSON_RATIO = 0.35
BOUNDARY_TOLERANCE = 1e-6
PROJECTION_BISECTION_STEPS = 24
MESHIO_TO_GMSH_TETRA10 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8], dtype=np.int32)

Load: TypeAlias = tuple[npt.ArrayLike, npt.ArrayLike]
MeshSource: TypeAlias = Mesh | str | PathLike[str]


@dataclass(frozen=True)
class _ProjectionSystem:
    function_space: Any
    test_function: Any
    mass_matrix: Any
    solver: PETSc.KSP


class Simulator:
    """Linear-elastic finite-element simulator for surface contact loads."""

    def __init__(self, obj_mesh: MeshSource, order: int = 2, std: float = 0.001):
        """Create a simulator for a tetrahedral object mesh.

        Args:
            obj_mesh: An in-memory mesh or a path readable by :class:`Mesh`.
            order: Polynomial order of the finite-element basis. It must be 1 or 2.
            std: Standard deviation of the Gaussian contact-load kernel. It
                must be positive and large enough for the surface mesh to
                resolve the applied load.
        """
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")

        self.std = float(std)
        self.comm = MPI.COMM_WORLD
        if not isinstance(obj_mesh, Mesh):
            obj_mesh = Mesh.read(str(obj_mesh))

        self._surface_geodesics = SurfaceGeodesics.from_mesh(
            obj_mesh,
            tolerance=BOUNDARY_TOLERANCE,
        )
        cells, geometry_degree = self._tetrahedral_cells(obj_mesh)

        coordinate_element = basix.ufl.element(
            "Lagrange",
            "tetrahedron",
            geometry_degree,
            shape=(obj_mesh.volume.points.shape[1],),
        )
        domain_ufl = ufl.Mesh(coordinate_element)
        self.domain = create_mesh(self.comm, cells, obj_mesh.volume.points, domain_ufl)
        self.V = fem.functionspace(
            self.domain, ("Lagrange", order, (self.domain.geometry.dim,))
        )

        self.lambda_ = (
            YOUNG_MODULUS
            * POISSON_RATIO
            / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO))
        )
        self.mu_ = YOUNG_MODULUS / (2 * (1 + POISSON_RATIO))

        displacement = ufl.TrialFunction(self.V)
        self.test_displacement = ufl.TestFunction(self.V)

        self.fdim = self.domain.topology.dim - 1
        local_z_min = self.domain.geometry.x[:, 2].min()
        self.bottom_z = self.comm.allreduce(local_z_min, op=MPI.MIN)
        bottom_facets = mesh.locate_entities_boundary(
            self.domain,
            self.fdim,
            lambda x: np.isclose(
                x[2], self.bottom_z, atol=BOUNDARY_TOLERANCE, rtol=0.0
            ),
        )
        bottom_dofs = fem.locate_dofs_topological(self.V, self.fdim, bottom_facets)
        self.bc = fem.dirichletbc(
            np.zeros(self.domain.geometry.dim, dtype=default_scalar_type),
            bottom_dofs,
            self.V,
        )

        stiffness = (
            ufl.inner(self.sigma(displacement), self.epsilon(self.test_displacement))
            * ufl.dx
        )
        self.bilinear_form = fem.form(stiffness)
        self.stiffness_matrix = assemble_matrix(self.bilinear_form, bcs=[self.bc])
        self.stiffness_matrix.assemble()
        self.solver = self._create_direct_solver(
            self.stiffness_matrix, factor_solver="mumps"
        )

        cell_dimension = self.domain.topology.dim
        self.domain.topology.create_connectivity(self.fdim, cell_dimension)
        cell_map = self.domain.topology.index_map(cell_dimension)
        self._cell_indices = np.arange(
            cell_map.size_local + cell_map.num_ghosts, dtype=np.int32
        )
        self._bb_tree = geometry.bb_tree(
            self.domain, cell_dimension, entities=self._cell_indices
        )
        self._cell_midpoint_tree = geometry.create_midpoint_tree(
            self.domain, cell_dimension, self._cell_indices
        )

        # These systems are invariant across loads, so build each one only
        # when first needed and reuse its assembled mass matrix and solver.
        self._projection_systems: dict[
            tuple[str, int, tuple[int, ...]], _ProjectionSystem
        ] = {}
        self._dof_trees: dict[int, tuple[Any, KDTree]] = {}
        self._mesh_points = np.asarray(obj_mesh.volume.points, dtype=np.float64).copy()
        (
            self._mesh_evaluation_points,
            self._mesh_evaluation_cells,
        ) = self._prepare_mesh_point_evaluation()
        self._mesh_point_dofs: dict[int, tuple[Any, np.ndarray]] = {}
        self._prepare_geodesic_distance_field()

    def _prepare_geodesic_distance_field(self) -> None:
        """Cache the map from scalar FE degrees of freedom to surface vertices."""
        self._geodesic_space = fem.functionspace(self.domain, ("Lagrange", 1))
        dof_coordinates = self._geodesic_space.tabulate_dof_coordinates()
        self._geodesic_dof_surface_vertices = np.asarray(
            KDTree(self._surface_geodesics.vertices).query(dof_coordinates, k=1)[1],
            dtype=np.intp,
        )

    def _geodesic_distance(self, point: np.ndarray) -> fem.Function:
        """Compute the surface distance from a contact and expose it as an FE field."""
        vertex_distances = self._surface_geodesics.distance_from(point)

        distance = fem.Function(self._geodesic_space)
        distance.x.array[:] = vertex_distances[
            self._geodesic_dof_surface_vertices
        ].astype(distance.x.array.dtype, copy=False)
        distance.x.scatter_forward()
        return distance

    @staticmethod
    def _tetrahedral_cells(obj_mesh: Mesh) -> tuple[np.ndarray, int]:
        """Return cells in DOLFINx ordering and their geometry degree."""
        cells_by_type = obj_mesh.volume.cells_dict
        if "tetra10" in cells_by_type:
            meshio_cells = cells_by_type["tetra10"]
            gmsh_cells = meshio_cells[:, MESHIO_TO_GMSH_TETRA10]
            permutation = gmshio.cell_perm_array(
                mesh.CellType.tetrahedron, gmsh_cells.shape[1]
            )
            return gmsh_cells[:, permutation].copy(), 2
        if "tetra" in cells_by_type:
            return cells_by_type["tetra"], 1
        raise ValueError("Mesh has no tetra or tetra10 volume cells")

    @staticmethod
    def epsilon(displacement: Any) -> Any:
        """Return the infinitesimal strain tensor."""
        return ufl.sym(ufl.grad(displacement))

    def sigma(self, displacement: Any) -> Any:
        """Return the Cauchy stress tensor."""
        strain = self.epsilon(displacement)
        return 2.0 * self.mu_ * strain + self.lambda_ * ufl.tr(strain) * ufl.Identity(
            len(displacement)
        )

    def _create_direct_solver(
        self, matrix: Any, factor_solver: str | None = None
    ) -> PETSc.KSP:
        solver = PETSc.KSP().create(self.comm)
        solver.setOperators(matrix)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        if factor_solver is not None:
            solver.getPC().setFactorSolverType(factor_solver)
        return solver

    def _as_vector(self, value: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
        vector = np.asarray(value, dtype=default_scalar_type)
        expected_shape = (self.domain.geometry.dim,)
        if vector.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        return vector

    def run(self, loads: Iterable[Load]) -> fem.Function:
        """Solve for displacement under the supplied point-force pairs.

        Args:
            loads: An iterable of (point, force) pairs. Each point must be a 3D coordinate on the mesh surface, and each force must be a 3D vector.

        Returns:
            A DOLFINx function representing the displacement field over the mesh.
        """
        load_form = (
            ufl.dot(
                fem.Constant(
                    self.domain,
                    np.zeros(self.domain.geometry.dim, dtype=default_scalar_type),
                ),
                self.test_displacement,
            )
            * ufl.dx
        )

        for point, force in loads:
            point_vector = self._as_vector(point, name="load point")
            force_vector = self._as_vector(force, name="load force")
            geodesic_distance = self._geodesic_distance(point_vector)
            weight = ufl.exp(-(geodesic_distance**2) / (2 * self.std**2))

            local_normalization = fem.assemble_scalar(fem.form(weight * ufl.ds))
            normalization = self.comm.allreduce(local_normalization, op=MPI.SUM)
            if not np.isfinite(normalization) or normalization <= np.finfo(float).tiny:
                raise ValueError(
                    "Contact load is too far from the mesh or std is too "
                    "small for the mesh to resolve it"
                )

            traction = (
                fem.Constant(
                    self.domain,
                    default_scalar_type(force_vector / normalization),
                )
                * weight
            )
            load_form += ufl.dot(traction, self.test_displacement) * ufl.ds

        rhs = assemble_vector(fem.form(load_form))
        apply_lifting(rhs, [self.bilinear_form], bcs=[[self.bc]])
        rhs.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        set_bc(rhs, [self.bc])

        displacement = fem.Function(self.V)
        self.solver.solve(rhs, displacement.x.petsc_vec)
        displacement.x.scatter_forward()
        return displacement

    def _projection_system(
        self,
        family: str,
        degree: int,
        shape: tuple[int, ...] = (),
    ) -> _ProjectionSystem:
        key = (family, degree, shape)
        if key not in self._projection_systems:
            element = (family, degree) if not shape else (family, degree, shape)

            function_space = fem.functionspace(self.domain, element)
            test_function = ufl.TestFunction(function_space)
            trial_function = ufl.TrialFunction(function_space)
            mass_form = fem.form(ufl.inner(trial_function, test_function) * ufl.dx)
            mass_matrix = assemble_matrix(mass_form)
            mass_matrix.assemble()
            solver = self._create_direct_solver(mass_matrix)
            self._projection_systems[key] = _ProjectionSystem(
                function_space=function_space,
                test_function=test_function,
                mass_matrix=mass_matrix,
                solver=solver,
            )
        return self._projection_systems[key]

    def _project(
        self, expression: Any, family: str, degree: int, shape: tuple[int, ...] = ()
    ) -> fem.Function:
        system = self._projection_system(family, degree, shape)
        rhs = assemble_vector(
            fem.form(ufl.inner(expression, system.test_function) * ufl.dx)
        )
        rhs.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )

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
        candidates = geometry.compute_collisions_points(self._bb_tree, points)
        collisions = geometry.compute_colliding_cells(self.domain, candidates, points)

        point_indices: list[int] = []
        cell_indices: list[int] = []
        for point_index in range(len(points)):
            cells = collisions.links(point_index)
            if len(cells):
                point_indices.append(point_index)
                cell_indices.append(cells[0])

        return (
            np.asarray(point_indices, dtype=np.int32),
            np.asarray(cell_indices, dtype=np.int32),
        )

    def _evaluate_found_points(
        self,
        function: fem.Function,
        points: np.ndarray,
        point_indices: np.ndarray,
        cell_indices: np.ndarray,
        values: np.ndarray,
        found: np.ndarray,
    ) -> None:
        if point_indices.size == 0:
            return
        values[point_indices] = function.eval(points[point_indices], cell_indices)
        found[point_indices] = True

    def _prepare_mesh_point_evaluation(self) -> tuple[np.ndarray, np.ndarray]:
        """Cache evaluation coordinates and cells for the volume-mesh points."""
        evaluation_points = self._mesh_points.copy()
        evaluation_cells = np.full(len(self._mesh_points), -1, dtype=np.int32)

        point_indices, cell_indices = self._colliding_cell_indices(self._mesh_points)
        evaluation_cells[point_indices] = cell_indices

        missing_indices = np.flatnonzero(evaluation_cells < 0)
        if missing_indices.size:
            projected_indices, projected_points, projected_cells = self._project_inside(
                self._mesh_points[missing_indices]
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
            self._bb_tree,
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

    def evaluate_mesh_points(self, function: fem.Function) -> np.ndarray:
        """Evaluate a function at the volume-mesh points using cached cells.

        The geometric search is performed once when the simulator is created.
        Any mesh point that cannot be associated with a local cell uses a
        cached nearest degree of freedom, matching :meth:`probe`'s final
        fallback.
        """
        function_space = function.function_space
        block_size = function_space.dofmap.index_map_bs
        values = np.zeros((len(self._mesh_points), block_size), dtype=np.float64)

        resolved = self._mesh_evaluation_cells >= 0
        if np.any(resolved):
            values[resolved] = function.eval(
                self._mesh_evaluation_points[resolved],
                self._mesh_evaluation_cells[resolved],
            )

        missing_indices = np.flatnonzero(~resolved)
        if missing_indices.size:
            key = id(function_space)
            cached = self._mesh_point_dofs.get(key)
            if cached is None or cached[0] is not function_space:
                nearest_dofs = self._nearest_dof_tree(function_space).query(
                    self._mesh_points[missing_indices], k=1
                )[1]
                cached = (function_space, np.asarray(nearest_dofs, dtype=np.intp))
                self._mesh_point_dofs[key] = cached

            nodal_values = function.x.array.real.reshape(-1, block_size)
            values[missing_indices] = nodal_values[cached[1]]

        return values

    def probe(
        self,
        function: fem.Function,
        points: npt.ArrayLike,
        clip: bool = True,
    ) -> np.ndarray:
        """Evaluate an FE function at points, including points just outside.

        Outside points are moved to the boundary using DOLFINx's volume-cell
        geometry, so probing does not depend on linear or quadratic surface
        triangle conversion. Any remaining misses use their nearest degree of
        freedom, guaranteeing one returned value per query point.
        """
        points_array = np.asarray(points, dtype=np.float64)
        if points_array.size == 0:
            points_array = np.empty((0, self.domain.geometry.dim))
        elif points_array.ndim == 1:
            points_array = points_array.reshape(1, -1)

        expected_shape = (len(points_array), self.domain.geometry.dim)
        if points_array.shape != expected_shape:
            raise ValueError(
                f"points must have shape (n, {self.domain.geometry.dim}), "
                f"got {points_array.shape}"
            )

        block_size = function.function_space.dofmap.index_map_bs
        values = np.zeros((len(points_array), block_size), dtype=np.float64)
        found = np.zeros(len(points_array), dtype=bool)
        if not len(points_array):
            return values

        point_indices, cell_indices = self._colliding_cell_indices(points_array)
        self._evaluate_found_points(
            function,
            points_array,
            point_indices,
            cell_indices,
            values,
            found,
        )

        missing_indices = np.flatnonzero(~found)
        if missing_indices.size:
            projected_local_indices, projected_points, projected_cells = (
                self._project_inside(points_array[missing_indices])
            )
            global_indices = missing_indices[projected_local_indices]
            if projected_local_indices.size:
                values[global_indices] = function.eval(
                    projected_points, projected_cells
                )
                found[global_indices] = True

        missing_indices = np.flatnonzero(~found)
        if missing_indices.size:
            nearest_dofs = self._nearest_dof_tree(function.function_space).query(
                points_array[missing_indices], k=1
            )[1]
            nodal_values = function.x.array.real.reshape(-1, block_size)
            values[missing_indices] = nodal_values[nearest_dofs]

        if clip:
            np.maximum(values, 0.0, out=values)
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
