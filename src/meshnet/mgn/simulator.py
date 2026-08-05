from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
from typing import Any, TypeAlias

import basix.ufl
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from dolfinx.mesh import create_mesh
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import KDTree

from meshnet.utils.geodesics import SurfaceGeodesics
from meshnet.utils.mesh import Mesh

BOUNDARY_TOLERANCE = 1e-6

Load: TypeAlias = tuple[npt.ArrayLike, npt.ArrayLike]
MeshSource: TypeAlias = Mesh | str | PathLike[str]


@dataclass(frozen=True, slots=True)
class IsotropicMaterial:
    young_modulus: float = 2.0e9
    poisson_ratio: float = 0.35

    @property
    def lame_mu(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def lame_lambda(self) -> float:
        nu = self.poisson_ratio
        return self.young_modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


@dataclass(slots=True)
class SimulationResult:
    """One FEM solve and its graph-aligned P1 vertex outputs.

    All three arrays use exactly the source mesh vertex order.
    """

    displacement_fem: fem.Function  # P2 solution used by FEM

    displacement_vertices: np.ndarray  # [num_vertices, 3]
    stress_vertices: np.ndarray  # [num_vertices, 6]
    nodal_forces_vertices: np.ndarray  # [num_vertices, 3]

    @property
    def targets(self) -> np.ndarray:
        """Graph target matrix: [ux, uy, uz, sxx, syy, szz, sxy, syz, sxz]."""
        return np.hstack((self.displacement_vertices, self.stress_vertices))


@dataclass(frozen=True, slots=True)
class _ProjectionSystem:
    space: Any
    test: Any
    mass_matrix: PETSc.Mat
    solver: PETSc.KSP


class Simulator:
    """P2 linear-elastic solve with P1 vertex-aligned exported fields.

    The solve space and output spaces intentionally have different roles:
      * V_solve: CG(order), normally CG2, for accurate displacement solves.
      * V_vertex: CG1 vector space for displacement export and nodal loads.
      * S_vertex: CG1 six-vector space for recovered continuous stress.

    Dataset export currently assumes a serial DOLFINx run because a graph needs
    one complete global vertex array. For MPI solves, gather owned vertex data
    to rank zero before writing the dataset.
    """

    def __init__(
        self,
        obj_mesh: MeshSource,
        *,
        order: int = 2,
        contact_std: float = 0.001,
        material: IsotropicMaterial | None = None,
        geodesics: SurfaceGeodesics | None = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        if order < 1:
            raise ValueError(f"order must be at least 1, got {order}")
        if contact_std <= 0.0:
            raise ValueError(f"contact_std must be positive, got {contact_std}")
        if comm.size != 1:
            raise NotImplementedError(
                "Graph-aligned global vertex export is implemented for serial "
                "dataset generation. Add an MPI gather for distributed runs."
            )

        self.comm = comm
        self.order = order
        self.contact_std = contact_std
        self.material = material or IsotropicMaterial()

        source_mesh = self._load_mesh(obj_mesh)
        self.source_points = np.asarray(
            source_mesh.volume.points, dtype=np.float64
        ).copy()
        self.source_tetra = np.asarray(
            source_mesh.volume.cells_dict["tetra"], dtype=np.int64
        ).copy()

        self.domain = self._create_domain(self.source_points, self.source_tetra)
        self.gdim = self.domain.geometry.dim
        self.cdim = self.domain.topology.dim
        self.fdim = self.cdim - 1

        # High-order solve space.
        self.V_solve = fem.functionspace(
            self.domain, ("Lagrange", self.order, (self.gdim,))
        )

        # P1 spaces whose block DOFs correspond to mesh vertices.
        self.V_vertex = fem.functionspace(self.domain, ("Lagrange", 1, (self.gdim,)))
        self.S_vertex = fem.functionspace(self.domain, ("Lagrange", 1, (6,)))

        self._vertex_dofs_u = self._map_source_vertices_to_block_dofs(self.V_vertex)
        self._vertex_dofs_s = self._map_source_vertices_to_block_dofs(self.S_vertex)

        self._surface_geodesics = geodesics or SurfaceGeodesics.from_mesh(
            source_mesh, tolerance=BOUNDARY_TOLERANCE
        )
        self._prepare_geodesic_distance_field()

        self.bc = self._create_fixed_bc()
        self.a, self.K, self.solver = self._create_elasticity_system()
        self._stress_projection = self._create_projection_system(self.S_vertex)

    @staticmethod
    def _load_mesh(source: MeshSource) -> Mesh:
        if isinstance(source, Mesh):
            return source
        return Mesh.read(str(source))

    def _create_domain(self, points: np.ndarray, tetra: np.ndarray) -> mesh.Mesh:
        coordinate_element = basix.ufl.element(
            "Lagrange",
            "tetrahedron",
            1,
            shape=(points.shape[1],),
        )
        ufl_domain = ufl.Mesh(coordinate_element)
        return create_mesh(self.comm, tetra, points, ufl_domain)

    def _map_source_vertices_to_block_dofs(self, space: Any) -> np.ndarray:
        """Map source mesh vertex order to a CG1 blocked-space DOF order."""
        coordinates = np.asarray(space.tabulate_dof_coordinates(), dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != self.gdim:
            raise RuntimeError(
                f"Unexpected DOF-coordinate shape for vertex space: {coordinates.shape}"
            )

        distances, indices = KDTree(coordinates).query(self.source_points, k=1)
        tolerance = max(BOUNDARY_TOLERANCE, 1e-10 * self._mesh_scale())

        if np.any(distances > tolerance):
            raise RuntimeError(
                "Could not align source vertices with CG1 DOFs. "
                f"Maximum mismatch is {float(np.max(distances)):.3e}."
            )
        if np.unique(indices).size != self.source_points.shape[0]:
            raise RuntimeError(
                "Source vertex coordinates are duplicated or multiple vertices "
                "mapped to the same CG1 DOF."
            )

        return np.asarray(indices, dtype=np.intp)

    def _mesh_scale(self) -> float:
        extent = np.ptp(self.source_points, axis=0)
        return float(max(np.max(extent), 1.0))

    def _create_fixed_bc(self) -> fem.DirichletBC:
        boundary_facets = mesh.locate_entities_boundary(
            self.domain,
            self.fdim,
            lambda x: np.isclose(x[2], 0.0, atol=BOUNDARY_TOLERANCE, rtol=0.0),
        )
        boundary_dofs = fem.locate_dofs_topological(
            self.V_solve, self.fdim, boundary_facets
        )
        value = np.zeros(self.gdim, dtype=default_scalar_type)
        return fem.dirichletbc(value, boundary_dofs, self.V_solve)

    @staticmethod
    def epsilon(u: Any) -> Any:
        return ufl.sym(ufl.grad(u))

    def sigma(self, u: Any) -> Any:
        strain = self.epsilon(u)
        return (
            2.0 * self.material.lame_mu * strain
            + self.material.lame_lambda * ufl.tr(strain) * ufl.Identity(self.gdim)
        )

    def stress_voigt(self, displacement: fem.Function) -> Any:
        stress = self.sigma(displacement)
        return ufl.as_vector(
            (
                stress[0, 0],
                stress[1, 1],
                stress[2, 2],
                stress[0, 1],
                stress[1, 2],
                stress[0, 2],
            )
        )

    def _create_elasticity_system(self) -> tuple[Any, PETSc.Mat, PETSc.KSP]:
        trial = ufl.TrialFunction(self.V_solve)
        test = ufl.TestFunction(self.V_solve)
        bilinear = fem.form(ufl.inner(self.sigma(trial), self.epsilon(test)) * ufl.dx)

        matrix = assemble_matrix(bilinear, bcs=[self.bc])
        matrix.assemble()
        return bilinear, matrix, self._create_direct_solver(matrix, "mumps")

    def _create_projection_system(self, space: Any) -> _ProjectionSystem:
        trial = ufl.TrialFunction(space)
        test = ufl.TestFunction(space)
        mass_form = fem.form(ufl.inner(trial, test) * ufl.dx)
        mass_matrix = assemble_matrix(mass_form)
        mass_matrix.assemble()
        return _ProjectionSystem(
            space=space,
            test=test,
            mass_matrix=mass_matrix,
            solver=self._create_direct_solver(mass_matrix, "mumps"),
        )

    def _create_direct_solver(
        self, matrix: PETSc.Mat, factor_solver: str | None = None
    ) -> PETSc.KSP:
        solver = PETSc.KSP().create(self.comm)
        solver.setOperators(matrix)
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        if factor_solver is not None:
            pc.setFactorSolverType(factor_solver)
        return solver

    def _prepare_geodesic_distance_field(self) -> None:
        self._geodesic_space = fem.functionspace(self.domain, ("Lagrange", 1))
        dof_coordinates = self._geodesic_space.tabulate_dof_coordinates()
        surface_tree = KDTree(self._surface_geodesics.vertices)
        nearest = surface_tree.query(dof_coordinates, k=1)[1]
        self._geodesic_dof_surface_vertices = np.asarray(nearest, dtype=np.intp)

    def _geodesic_distance_field(self, point: np.ndarray) -> fem.Function:
        distances = self._surface_geodesics.distance_from(point)
        field = fem.Function(self._geodesic_space)
        field.x.array[:] = distances[self._geodesic_dof_surface_vertices].astype(
            field.x.array.dtype, copy=False
        )
        field.x.scatter_forward()
        return field

    def _traction(self, point: np.ndarray, force: np.ndarray) -> Any:
        distance = self._geodesic_distance_field(point)
        weight = ufl.exp(-(distance**2) / (2.0 * self.contact_std**2))

        local_integral = fem.assemble_scalar(fem.form(weight * ufl.ds))
        normalization = self.comm.allreduce(local_integral, op=MPI.SUM)
        if not np.isfinite(normalization) or normalization <= 0.0:
            raise RuntimeError("Contact traction normalization failed.")

        force_constant = fem.Constant(
            self.domain,
            default_scalar_type(force / normalization),
        )
        return force_constant * weight

    def _build_load_form(self, space: Any, tractions: list[Any]) -> Any:
        test = ufl.TestFunction(space)
        zero = fem.Constant(self.domain, np.zeros(self.gdim, dtype=default_scalar_type))
        linear = ufl.dot(zero, test) * ufl.ds
        for traction in tractions:
            linear += ufl.dot(traction, test) * ufl.ds
        return fem.form(linear)

    @staticmethod
    def _accumulate_ghost_values(
        vector: PETSc.Vec, *, populate_ghosts: bool = False
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

    def _extract_vertex_values(
        self,
        function: fem.Function,
        source_to_dof: np.ndarray,
    ) -> np.ndarray:
        block_size = function.function_space.dofmap.index_map_bs
        dof_values = np.asarray(function.x.array.real).reshape(-1, block_size)
        return np.asarray(dof_values[source_to_dof], dtype=np.float64).copy()

    def _extract_vertex_vector(
        self,
        vector: PETSc.Vec,
        space: Any,
        source_to_dof: np.ndarray,
    ) -> np.ndarray:
        block_size = space.dofmap.index_map_bs
        dof_values = np.asarray(vector.array.real).reshape(-1, block_size)
        return np.asarray(dof_values[source_to_dof], dtype=np.float64).copy()

    def _interpolate_displacement_to_vertices(
        self, displacement: fem.Function
    ) -> fem.Function:
        vertex_displacement = fem.Function(self.V_vertex)
        expression = fem.Expression(
            displacement,
            self.V_vertex.element.interpolation_points(),
        )
        vertex_displacement.interpolate(expression)
        vertex_displacement.x.scatter_forward()
        return vertex_displacement

    def _recover_stress_to_vertices(self, displacement: fem.Function) -> fem.Function:
        """L2-project P2-derived stress into a continuous CG1 vertex field.

        Stress from a CG2 displacement is cellwise linear but discontinuous at
        shared vertices. The projection defines one reproducible value per
        graph vertex instead of arbitrarily selecting an adjacent cell.
        """
        system = self._stress_projection
        rhs = assemble_vector(
            fem.form(ufl.inner(self.stress_voigt(displacement), system.test) * ufl.dx)
        )
        self._accumulate_ghost_values(rhs, populate_ghosts=True)

        stress = fem.Function(system.space)
        system.solver.solve(rhs, stress.x.petsc_vec)
        stress.x.scatter_forward()
        return stress

    def run(self, loads: Iterable[Load]) -> SimulationResult:
        """Solve in CG2 and export displacement, stress, and loads at CG1 vertices."""
        normalized_loads: list[tuple[np.ndarray, np.ndarray]] = []
        for point, force in loads:
            point_array = np.asarray(point, dtype=np.float64)
            force_array = np.asarray(force, dtype=np.float64)
            if point_array.shape != (self.gdim,) or force_array.shape != (self.gdim,):
                raise ValueError("Each contact point and force must be a 3D vector.")
            if not np.all(np.isfinite(point_array)) or not np.all(
                np.isfinite(force_array)
            ):
                raise ValueError("Contact points and forces must be finite.")
            normalized_loads.append((point_array, force_array))

        # Construct each traction once and reuse it in both the P2 and P1 forms.
        tractions = [self._traction(point, force) for point, force in normalized_loads]

        # P2 RHS used by the actual elasticity solve.
        rhs = assemble_vector(self._build_load_form(self.V_solve, tractions))
        apply_lifting(rhs, [self.a], bcs=[[self.bc]])
        self._accumulate_ghost_values(rhs, populate_ghosts=True)
        set_bc(rhs, [self.bc])

        displacement_fem = fem.Function(self.V_solve)
        self.solver.solve(rhs, displacement_fem.x.petsc_vec)
        displacement_fem.x.scatter_forward()

        # Independently assemble the same continuous traction functional against
        # CG1 basis functions. This is the correct P1 consistent nodal load; it
        # is not obtained by discarding the P2 mid-edge entries.
        force_vector_p1 = assemble_vector(
            self._build_load_form(self.V_vertex, tractions)
        )
        self._accumulate_ghost_values(force_vector_p1, populate_ghosts=True)

        displacement_p1 = self._interpolate_displacement_to_vertices(displacement_fem)
        stress_p1 = self._recover_stress_to_vertices(displacement_fem)

        displacement_vertices = self._extract_vertex_values(
            displacement_p1, self._vertex_dofs_u
        )
        stress_vertices = self._extract_vertex_values(stress_p1, self._vertex_dofs_s)
        nodal_forces_vertices = self._extract_vertex_vector(
            force_vector_p1, self.V_vertex, self._vertex_dofs_u
        )

        # Partition of unity means the consistent nodal loads should preserve
        # the applied resultant, up to integration tolerance.
        expected_resultant = (
            np.sum([force for _, force in normalized_loads], axis=0)
            if normalized_loads
            else np.zeros(self.gdim)
        )
        actual_resultant = nodal_forces_vertices.sum(axis=0)
        np.testing.assert_allclose(
            actual_resultant,
            expected_resultant,
            rtol=1e-6,
            atol=1e-10,
            err_msg="P1 nodal loads do not preserve the prescribed resultant.",
        )

        return SimulationResult(
            displacement_fem=displacement_fem,
            displacement_vertices=displacement_vertices,
            stress_vertices=stress_vertices,
            nodal_forces_vertices=nodal_forces_vertices,
        )
