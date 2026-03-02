from dataclasses import dataclass

import trimesh


@dataclass(frozen=True)
class BoxFinger:
    width: float = 0.021
    height: float = 0.0455
    depth: float = 0.008
    mesh: trimesh.Trimesh = None

    def __post_init__(self):
        mesh = trimesh.creation.box(extents=(self.width, self.height, self.depth))
        mesh.apply_translation((0, self.height / 2, 0))
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

        left_tf = trimesh.transformations.translation_matrix([0, 0, -offset])
        right_tf = trimesh.transformations.translation_matrix([0, 0, offset])

        scene = trimesh.Scene()
        ax = trimesh.creation.axis(
            origin_size=0.005, axis_radius=0.005, axis_length=0.1
        )
        scene.add_geometry(ax)
        scene.add_geometry(self.box_finger_left.mesh, transform=left_tf)
        scene.add_geometry(self.box_finger_right.mesh, transform=right_tf)
        scene.show(viewer=viewer)
