from dataclasses import dataclass, field

import meshio
import numpy as np
import numpy.typing as npt
import trimesh


@dataclass
class Mesh:
    volume: meshio.Mesh
    _surface: trimesh.Trimesh | None = field(default=None, init=False, repr=False)

    @classmethod
    def read(cls, path: str) -> "Mesh":
        return cls(meshio.read(path))

    @property
    def surface(self) -> trimesh.Trimesh:
        if self._surface is None:
            cells = [c.data for c in self.volume.cells if c.type == "triangle"]
            if not cells:
                raise ValueError("Mesh has no linear triangular boundary cells")

            self._surface = trimesh.Trimesh(
                vertices=self.volume.points,
                faces=np.vstack(cells),
                process=False,
            )

            if not self._surface.is_watertight:
                raise ValueError("Mesh boundary is not watertight")

        return self._surface

    @property
    def bottom_boundary_edges(self) -> npt.NDArray[np.integer]:
        vertices = self.surface.vertices
        faces = self.surface.faces

        z_min = vertices[:, 2].min()

        face_z = vertices[faces, 2]
        bottom_mask = np.isclose(face_z, z_min, atol=1e-6, rtol=0.0).all(axis=1)
        bottom_faces = faces[bottom_mask]

        if bottom_faces.size == 0:
            raise ValueError("No bottom boundary found in the mesh")

        edges = np.vstack(
            [bottom_faces[:, [0, 1]], bottom_faces[:, [1, 2]], bottom_faces[:, [2, 0]]]
        )
        # Ignore orientation of edges
        edges = np.sort(edges, axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        return unique_edges[counts == 1]
