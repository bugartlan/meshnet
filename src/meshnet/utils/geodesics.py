from typing import Self

import meshio
import numpy as np
import numpy.typing as npt
import potpourri3d as pp3d
import pyvista

from meshnet.utils.mesh import Mesh


class SurfaceGeodesics:
    """Reusable heat-method geodesic solver for a triangular surface mesh.

    The input mesh is compacted to vertices referenced by its surface faces.
    Distances are returned in that compact ordering; ``vertex_indices`` maps
    those values back to the input vertex array.
    """

    def __init__(
        self,
        vertices: npt.ArrayLike | Mesh | meshio.Mesh,
        faces: npt.ArrayLike | None = None,
        tolerance: float = 1e-6,
    ):
        if isinstance(vertices, (Mesh, meshio.Mesh)):
            if faces is not None:
                raise ValueError("faces must be omitted when constructing from a mesh")
            vertices, faces = self._surface_arrays(vertices)
        elif faces is None:
            raise ValueError("faces are required when constructing from vertex arrays")

        vertices_array = np.asarray(vertices, dtype=np.float64)
        faces_array = np.asarray(faces, dtype=np.int64)

        if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
            raise ValueError(
                f"vertices must have shape (n, 3), got {vertices_array.shape}"
            )
        if not np.all(np.isfinite(vertices_array)):
            raise ValueError("vertices must contain only finite values")
        if faces_array.ndim != 2 or faces_array.shape[1] != 3:
            raise ValueError(f"faces must have shape (m, 3), got {faces_array.shape}")
        if not len(faces_array):
            raise ValueError("faces must contain at least one triangle")
        if np.any(faces_array < 0) or np.any(faces_array >= len(vertices_array)):
            raise ValueError("faces contain vertex indices outside the vertex array")
        if tolerance < 0 or not np.isfinite(tolerance):
            raise ValueError(
                f"tolerance must be finite and non-negative, got {tolerance}"
            )

        vertex_indices, inverse = np.unique(
            faces_array.reshape(-1), return_inverse=True
        )
        self._vertex_indices = vertex_indices.astype(np.intp, copy=False)
        self._vertices = vertices_array[self._vertex_indices].copy()
        self._faces = inverse.reshape(faces_array.shape)
        self._tolerance = float(tolerance)

        self._surface_mesh = pyvista.PolyData(
            self._vertices,
            np.column_stack(
                [
                    np.full(len(self._faces), 3, dtype=np.int64),
                    self._faces,
                ]
            ),
        )
        self._solver = pp3d.MeshHeatMethodDistanceSolver(
            self._vertices,
            self._faces,
        )

    @classmethod
    def from_mesh(
        cls,
        mesh: Mesh | meshio.Mesh,
        tolerance: float = 1e-6,
    ) -> Self:
        """Construct from the repository mesh wrapper or a meshio mesh."""
        return cls(mesh, tolerance=tolerance)

    @staticmethod
    def _surface_arrays(
        mesh: Mesh | meshio.Mesh,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.integer]]:
        if isinstance(mesh, Mesh):
            surface = mesh.surface
            return np.asarray(surface.vertices), np.asarray(surface.faces)

        faces = mesh.cells_dict.get("triangle")
        if faces is None or not len(faces):
            raise ValueError("Mesh has no linear triangular boundary cells")
        return np.asarray(mesh.points), np.asarray(faces)

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        """Compact surface vertices."""
        return self._vertices

    @property
    def faces(self) -> npt.NDArray[np.int64]:
        """Triangular faces indexing ``vertices``."""
        return self._faces

    @property
    def vertex_indices(self) -> npt.NDArray[np.intp]:
        """Indices mapping compact surface vertices to the input vertex array."""
        return self._vertex_indices

    def distance_from(self, source: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute distances from a surface point to all compact vertices.

        The source is projected onto its closest surface triangle and snapped
        to the closest vertex of that triangle. Restricting the snap candidates
        to that triangle prevents a nearby opposite surface from being chosen.

        Args:
            source: A 3D coordinate on or near the mesh surface.

        Returns:
            A 1D array of geodesic distances from the source to each compact vertex.
        """
        point = np.asarray(source, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("source must be a finite 3D coordinate")

        cell_id, closest_point = self._surface_mesh.find_closest_cell(
            point, return_closest_point=True
        )
        surface_offset = float(np.linalg.norm(point - closest_point))
        if surface_offset > self._tolerance:
            raise ValueError(
                "Source point must lie on the mesh surface; "
                f"nearest-surface distance is {surface_offset:.6g}"
            )

        triangle = self._faces[int(cell_id)]
        source_vertex = int(
            triangle[
                np.argmin(
                    np.linalg.norm(
                        self._vertices[triangle] - closest_point,
                        axis=1,
                    )
                )
            ]
        )
        distances = np.asarray(
            self._solver.compute_distance(source_vertex),
            dtype=np.float64,
        )
        if distances.shape != (len(self._vertices),) or not np.all(
            np.isfinite(distances)
        ):
            raise ValueError("Could not compute finite geodesic contact distances")
        return distances
