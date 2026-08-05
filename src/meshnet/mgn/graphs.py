import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch_geometric.data import Data


@dataclass(frozen=True, slots=True)
class MeshGraphTemplate:
    """Static graph data shared by every load case on one mesh."""

    pos: torch.Tensor
    boundary: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.pos.shape[0])

    @classmethod
    def from_arrays(
        cls,
        vertices: npt.ArrayLike,
        tetra: npt.ArrayLike,
        boundary_mask: npt.ArrayLike,
    ) -> "MeshGraphTemplate":
        vertices_np = np.asarray(vertices, dtype=np.float32)
        tetra_np = np.asarray(tetra, dtype=np.int64)
        boundary_np = np.asarray(boundary_mask, dtype=np.float32).reshape(-1, 1)

        if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
            raise ValueError(
                f"vertices must have shape [N, 3], got {vertices_np.shape}"
            )
        if tetra_np.ndim != 2 or tetra_np.shape[1] != 4:
            raise ValueError(f"tetra must have shape [M, 4], got {tetra_np.shape}")
        if boundary_np.shape != (len(vertices_np), 1):
            raise ValueError(
                f"boundary_mask must have {len(vertices_np)} entries, "
                f"got {boundary_np.shape}"
            )

        edge_index, edge_attr = _tetra_edges(vertices_np, tetra_np)
        return cls(
            pos=torch.from_numpy(vertices_np),
            boundary=torch.from_numpy(boundary_np),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )


def _tetra_edges(
    vertices: np.ndarray,
    tetra: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_pairs = np.asarray(
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        dtype=np.int64,
    )
    undirected = tetra[:, local_pairs].reshape(-1, 2)
    undirected.sort(axis=1)
    undirected = np.unique(undirected, axis=0)

    src = undirected[:, 0]
    dst = undirected[:, 1]
    displacement = vertices[dst] - vertices[src]
    distance = np.linalg.norm(displacement, axis=1, keepdims=True)

    edge_index = np.hstack(
        (
            np.stack((src, dst), axis=0),
            np.stack((dst, src), axis=0),
        )
    )
    edge_attr = np.vstack(
        (
            np.hstack((displacement, distance)),
            np.hstack((-displacement, distance)),
        )
    )
    return (
        torch.from_numpy(edge_index).long(),
        torch.from_numpy(edge_attr).float(),
    )


class GraphBuilderBase:
    """Build graphs only from saved, vertex-aligned FEM arrays.

    Node features: [x, y, z, fx, fy, fz, is_boundary]
    Targets: [ux, uy, uz, sxx, syy, szz, sxy, syz, sxz]
    """

    num_categorical = 1
    edge_dim = 4

    @property
    def node_dim(self) -> int:
        return 7

    def build(
        self,
        template: MeshGraphTemplate,
        vertex_forces: npt.ArrayLike,
        vertex_displacement: npt.ArrayLike,
        vertex_stress: npt.ArrayLike,
        contacts: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> Data:
        contacts = contacts or []
        forces, displacement, stress = self._validate_fields(
            template,
            vertex_forces,
            vertex_displacement,
            vertex_stress,
        )

        x = self._physical_node_features(template, forces, contacts)
        y = torch.hstack((displacement, stress))
        target_mask = torch.ones(template.num_nodes, dtype=torch.bool)

        return Data(
            x=x,
            pos=template.pos,
            edge_index=template.edge_index,
            edge_attr=template.edge_attr,
            y=y,
            target_mask=target_mask,
            num_physical_nodes=template.num_nodes,
            contacts=contacts,
        )

    def _physical_node_features(
        self,
        template: MeshGraphTemplate,
        vertex_forces: torch.Tensor,
        contacts: list[tuple[np.ndarray, np.ndarray]],
    ) -> torch.Tensor:
        del contacts
        return torch.hstack((template.pos, vertex_forces, template.boundary))

    @staticmethod
    def _validate_fields(
        template: MeshGraphTemplate,
        vertex_forces: npt.ArrayLike,
        vertex_displacement: npt.ArrayLike,
        vertex_stress: npt.ArrayLike,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = template.num_nodes
        arrays = {
            "vertex_forces": (vertex_forces, 3),
            "vertex_displacement": (vertex_displacement, 3),
            "vertex_stress": (vertex_stress, 6),
        }
        tensors: list[torch.Tensor] = []
        for name, (values, width) in arrays.items():
            array = np.asarray(values, dtype=np.float32)
            expected = (num_nodes, width)
            if array.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {array.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")
            tensors.append(torch.from_numpy(array))
        return tensors[0], tensors[1], tensors[2]


class GraphBuilderAugment(GraphBuilderBase):
    """Add relative contact positions and contact forces to every node."""

    def __init__(self, num_contacts: int) -> None:
        self.num_contacts = num_contacts

    @property
    def node_dim(self) -> int:
        return 7 + 6 * self.num_contacts

    def _physical_node_features(
        self,
        template: MeshGraphTemplate,
        vertex_forces: torch.Tensor,
        contacts: list[tuple[np.ndarray, np.ndarray]],
    ) -> torch.Tensor:
        if len(contacts) != self.num_contacts:
            raise ValueError(
                f"Expected {self.num_contacts} contacts, got {len(contacts)}"
            )

        sorted_contacts = sorted(contacts, key=lambda item: tuple(item[0]))
        global_features: list[torch.Tensor] = []
        for point, force in sorted_contacts:
            point_tensor = torch.as_tensor(point, dtype=torch.float32)
            force_tensor = torch.as_tensor(force, dtype=torch.float32)
            relative = template.pos - point_tensor
            tiled_force = force_tensor.expand(template.num_nodes, -1)
            global_features.extend((relative, tiled_force))

        return torch.hstack(
            (
                template.pos,
                vertex_forces,
                *global_features,
                template.boundary,
            )
        )


class GraphBuilderVirtual(GraphBuilderBase):
    """Represent each contact as a virtual node connected to all mesh vertices."""

    num_categorical = 2

    @property
    def node_dim(self) -> int:
        return 8

    def build(
        self,
        template: MeshGraphTemplate,
        vertex_forces: npt.ArrayLike,
        vertex_displacement: npt.ArrayLike,
        vertex_stress: npt.ArrayLike,
        contacts: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> Data:
        contacts = contacts or []
        forces, displacement, stress = self._validate_fields(
            template,
            vertex_forces,
            vertex_displacement,
            vertex_stress,
        )

        physical_x = torch.hstack(
            (
                template.pos,
                forces,
                template.boundary,
                torch.zeros((template.num_nodes, 1), dtype=torch.float32),
            )
        )
        physical_y = torch.hstack((displacement, stress))

        if not contacts:
            return Data(
                x=physical_x,
                pos=template.pos,
                edge_index=template.edge_index,
                edge_attr=template.edge_attr,
                y=physical_y,
                target_mask=torch.ones(template.num_nodes, dtype=torch.bool),
                num_physical_nodes=template.num_nodes,
                contacts=[],
            )

        virtual_x, virtual_pos = self._virtual_nodes(contacts)
        virtual_y = torch.zeros(
            (len(contacts), physical_y.shape[1]), dtype=torch.float32
        )
        virtual_edges, virtual_edge_attr = self._virtual_edges(
            template.pos, virtual_pos
        )

        return Data(
            x=torch.vstack((physical_x, virtual_x)),
            pos=torch.vstack((template.pos, virtual_pos)),
            edge_index=torch.hstack((template.edge_index, virtual_edges)),
            edge_attr=torch.vstack((template.edge_attr, virtual_edge_attr)),
            y=torch.vstack((physical_y, virtual_y)),
            target_mask=torch.cat(
                (
                    torch.ones(template.num_nodes, dtype=torch.bool),
                    torch.zeros(len(contacts), dtype=torch.bool),
                )
            ),
            num_physical_nodes=template.num_nodes,
            contacts=contacts,
        )

    @staticmethod
    def _virtual_nodes(
        contacts: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        points = torch.as_tensor(
            np.stack([point for point, _ in contacts]), dtype=torch.float32
        )
        forces = torch.as_tensor(
            np.stack([force for _, force in contacts]), dtype=torch.float32
        )
        boundary = torch.zeros((len(contacts), 1), dtype=torch.float32)
        is_virtual = torch.ones((len(contacts), 1), dtype=torch.float32)
        return torch.hstack((points, forces, boundary, is_virtual)), points

    @staticmethod
    def _virtual_edges(
        physical_pos: torch.Tensor,
        virtual_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_physical = physical_pos.shape[0]
        num_virtual = virtual_pos.shape[0]

        physical_index = torch.arange(num_physical, dtype=torch.long)
        virtual_index = torch.arange(
            num_physical, num_physical + num_virtual, dtype=torch.long
        )

        virtual_repeated = virtual_index.repeat_interleave(num_physical)
        physical_tiled = physical_index.repeat(num_virtual)

        # Bidirectional virtual/physical edges.
        edge_index = torch.hstack(
            (
                torch.stack((virtual_repeated, physical_tiled)),
                torch.stack((physical_tiled, virtual_repeated)),
            )
        )

        all_pos = torch.vstack((physical_pos, virtual_pos))
        src, dst = edge_index
        displacement = all_pos[dst] - all_pos[src]
        distance = torch.linalg.vector_norm(displacement, dim=1, keepdim=True)
        return edge_index, torch.hstack((displacement, distance))


def generate_graph_dataset(
    raw_path: Path,
    out_path: Path,
    builder_kind: str = "base",
    num_samples: int | None = None,
) -> None:
    with np.load(raw_path, allow_pickle=False) as raw:
        vertices = raw["vertices"]
        tetra = raw["tetra"]
        boundary_mask = raw["boundary_mask"]
        contact_points = raw["contact_points"]
        contact_forces = raw["contact_forces"]
        vertex_forces = raw["vertex_forces"]
        vertex_displacement = raw["vertex_displacement"]
        vertex_stress = raw["vertex_stress"]
        mesh_path = raw["mesh_path"].item() if "mesh_path" in raw else None
        schema_version = int(raw["schema_version"]) if "schema_version" in raw else 0

    if schema_version != 2:
        raise ValueError(
            f"Expected raw schema version 2, got {schema_version}. Regenerate FEM data."
        )

    template = MeshGraphTemplate.from_arrays(vertices, tetra, boundary_mask)
    num_contacts = int(contact_points.shape[1])
    builders = {
        "base": GraphBuilderBase(),
        "augment": GraphBuilderAugment(num_contacts),
        "virtual": GraphBuilderVirtual(),
    }
    try:
        builder = builders[builder_kind]
    except KeyError as error:
        raise ValueError(f"Unknown builder kind: {builder_kind}") from error

    available = int(contact_points.shape[0])
    count = available if num_samples is None else min(num_samples, available)

    graphs: list[Data] = []
    for index in range(count):
        contacts = list(zip(contact_points[index], contact_forces[index]))
        graphs.append(
            builder.build(
                template,
                vertex_forces[index],
                vertex_displacement[index],
                vertex_stress[index],
                contacts,
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "params": {
                "node_dim": builder.node_dim,
                "edge_dim": builder.edge_dim,
                "output_dim": 9,
                "num_categorical": builder.num_categorical,
                "builder_kind": builder_kind,
            },
            "graphs": graphs,
            "mesh_path": mesh_path,
            "source_path": str(raw_path),
            "schema_version": 2,
        },
        out_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build graphs from saved vertex-aligned FEM data."
    )
    parser.add_argument("filepath", type=Path, nargs="+")
    parser.add_argument("--out_dir", type=Path, default=Path("data/graphs"))
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--builder",
        choices=("base", "augment", "virtual"),
        default="virtual",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files: list[Path] = []
    for path in args.filepath:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.glob("*.npz")))
        else:
            raise RuntimeError(f"Path {path} is not a file or directory.")

    for filepath in files:
        out_path = args.out_dir / filepath.with_suffix(".pt").name
        generate_graph_dataset(
            filepath,
            out_path,
            builder_kind=args.builder,
            num_samples=args.num_samples,
        )
        print(f"Saved graph dataset to {out_path}")


if __name__ == "__main__":
    main()
