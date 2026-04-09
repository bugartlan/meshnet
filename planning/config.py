from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass(frozen=True)
class BoxFinger:
    # Closing direction is along the x-axis and the normal is along the z-axis
    width: float = 0.021
    height: float = 0.0455
    depth: float = 0.008
    offset: float = 0.101
    mesh: trimesh.Trimesh = None

    def __post_init__(self):
        mesh = trimesh.creation.box(extents=(self.depth, self.width, self.height))
        mesh.apply_translation((0, 0, self.height / 2 + self.offset))
        mesh.visual.face_colors = [64, 255, 128, 128]
        object.__setattr__(self, "mesh", mesh)


@dataclass(frozen=True)
class CylinderBody:
    radius: float = 0.031
    height: float = 0.1005
    mesh: trimesh.Trimesh = None

    def __post_init__(self):
        mesh = trimesh.creation.cylinder(radius=0.031, height=0.1005)
        mesh.apply_translation((0, 0, self.height / 2))
        mesh.visual.face_colors = [64, 0, 128, 128]
        object.__setattr__(self, "mesh", mesh)


@dataclass(frozen=True)
class ROBOTIQ_HANDE_GRIPPER:
    min_width: float = 0.0
    max_width: float = 0.05
    base_to_fingertip: float = 0.146  # from base to fingertip
    palm_to_fingertip: float = 0.0455  # from palm to fingertip
    mesh: trimesh.Trimesh = trimesh.load_mesh("ROBOTIQ_HAND-E.step")

    box_finger_left: BoxFinger = BoxFinger()
    box_finger_right: BoxFinger = BoxFinger()
    cylinder_body: CylinderBody = CylinderBody()

    def tf_fingertip_to_base(self):
        tf = np.eye(4)
        tf[2, 3] = -self.base_to_fingertip
        return tf

    def tf_base_to_fingertip(self):
        tf = np.eye(4)
        tf[2, 3] = self.base_to_fingertip
        return tf

    def show(self, viewer="gl"):
        scene = trimesh.Scene()
        ax = trimesh.creation.axis(
            origin_size=0.005, axis_radius=0.005, axis_length=0.1
        )
        scene.add_geometry(ax)
        scene.add_geometry(self.mesh)
        scene.show(viewer=viewer)

    def show_box_fingers(self, width, viewer="gl"):
        offset = width / 2 + self.box_finger_left.width / 2

        left_tf = trimesh.transformations.translation_matrix([-offset, 0, 0])
        right_tf = trimesh.transformations.translation_matrix([offset, 0, 0])

        scene = trimesh.Scene()
        ax = trimesh.creation.axis(
            origin_size=0.005, axis_radius=0.005, axis_length=0.1
        )
        scene.add_geometry(ax)
        scene.add_geometry(self.box_finger_left.mesh, transform=left_tf)
        scene.add_geometry(self.box_finger_right.mesh, transform=right_tf)
        scene.add_geometry(self.cylinder_body.mesh)
        scene.show(viewer=viewer)
