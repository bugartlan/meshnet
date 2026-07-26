from dataclasses import dataclass, field
from importlib.resources import as_file, files
from typing import Literal

import numpy as np
import trimesh

# colors for visualization
COLOR_FINGER = [64, 255, 128, 128]
COLOR_CYLINDER = [64, 0, 128, 128]


def load_robotiq_mesh() -> trimesh.Trimesh:
    resource = files("meshnet").joinpath(
        "assets",
        "ROBOTIQ_HAND-E.step",
    )

    with as_file(resource) as path:
        return trimesh.load_mesh(path)


@dataclass(frozen=True)
class BoxFinger:
    """Primitive approximation of a Robotiq Hand-E finger as a box.

    Closing direction is along the x-axis and the approach direction is along the z-axis.
    """

    width: float = 0.021
    height: float = 0.0455
    depth: float = 0.008
    offset: float = 0.101
    mesh: trimesh.Trimesh = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        mesh = trimesh.creation.box(extents=(self.depth, self.width, self.height))
        mesh.apply_translation((0, 0, self.height / 2 + self.offset))
        mesh.visual.face_colors = COLOR_FINGER
        object.__setattr__(self, "mesh", mesh)


@dataclass(frozen=True)
class CylinderBody:
    """Primitive approximation of the Robotiq Hand-E body as a cylinder."""

    radius: float = 0.031
    height: float = 0.1005
    mesh: trimesh.Trimesh = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        mesh = trimesh.creation.cylinder(radius=self.radius, height=self.height)
        mesh.apply_translation((0, 0, self.height / 2))
        mesh.visual.face_colors = COLOR_CYLINDER
        object.__setattr__(self, "mesh", mesh)


@dataclass(frozen=True)
class RobotiqHandEGripper:
    """Configuration and lightweight geometry for the Robotiq Hand-E gripper."""

    min_width: float = 0.0
    max_width: float = 0.05
    base_to_fingertip: float = 0.146  # from base to fingertip
    palm_to_fingertip: float = 0.0455  # from palm to fingertip
    mesh: trimesh.Trimesh = field(
        default_factory=lambda: load_robotiq_mesh(),
        repr=False,
        compare=False,
    )

    box_finger_left: BoxFinger = field(default_factory=BoxFinger)
    box_finger_right: BoxFinger = field(default_factory=BoxFinger)
    cylinder_body: CylinderBody = field(default_factory=CylinderBody)

    def tf_fingertip_to_base(self) -> np.ndarray:
        tf = np.eye(4)
        tf[2, 3] = -self.base_to_fingertip
        return tf

    def tf_base_to_fingertip(self) -> np.ndarray:
        tf = np.eye(4)
        tf[2, 3] = self.base_to_fingertip
        return tf

    def tf_real_to_fake(self) -> np.ndarray:
        tf = np.array(
            [
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, -self.palm_to_fingertip + self.base_to_fingertip],
                [0, 0, 0, 1],
            ]
        )
        return tf

    def tf_fake_to_real(self) -> np.ndarray:
        tf = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, self.palm_to_fingertip - self.base_to_fingertip],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        return tf

    @staticmethod
    def _create_scene_with_axis() -> trimesh.Scene:
        scene = trimesh.Scene()
        axis = trimesh.creation.axis(
            origin_size=0.005,
            axis_radius=0.005,
            axis_length=0.1,
        )
        scene.add_geometry(axis)
        return scene

    def show(self, viewer: Literal["gl", "jupyter", "notebook"] = "gl") -> None:
        scene = self._create_scene_with_axis()
        scene.add_geometry(self.mesh)
        scene.show(viewer=viewer)

    def show_box_fingers(
        self,
        width: float,
        viewer: Literal["gl", "jupyter", "notebook"] = "gl",
    ) -> None:
        if not (self.min_width <= width <= self.max_width):
            raise ValueError(
                f"width must be in [{self.min_width}, {self.max_width}], got {width}"
            )

        offset = width / 2 + self.box_finger_left.width / 2

        left_tf = trimesh.transformations.translation_matrix([-offset, 0, 0])
        right_tf = trimesh.transformations.translation_matrix([offset, 0, 0])

        scene = self._create_scene_with_axis()
        scene.add_geometry(self.box_finger_left.mesh, transform=left_tf)
        scene.add_geometry(self.box_finger_right.mesh, transform=right_tf)
        scene.add_geometry(self.cylinder_body.mesh)
        scene.show(viewer=viewer)


# Backward-compatible alias for existing imports/usages.
ROBOTIQ_HANDE_GRIPPER = RobotiqHandEGripper
