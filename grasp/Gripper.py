from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class Finger:
    mesh: trimesh.Trimesh
    tip: np.ndarray
    normal: np.ndarray

    def __post_init__(self):
        self.tip = np.asarray(self.tip, dtype=float)
        self.normal = np.asarray(self.normal, dtype=float)

    def transform(self, matrix: np.ndarray):
        """
        Apply a transformation matrix to the finger mesh, tip, and normal.

        Parameters
        ----------
        matrix : np.ndarray
            A 4x4 SE3 transformation matrix.
        """
        self.mesh.apply_transform(matrix)
        self.tip = trimesh.transform_points([self.tip], matrix)[0]
        self.normal = trimesh.transform_points([self.normal], matrix, translate=False)[
            0
        ]

    def move(self, distance: float):
        """
        Move the finger by translating it along its normal vector.

        Parameters
        ----------
        distance : float
            The distance to move the finger.
        """
        translation = -self.normal * distance
        self.transform(trimesh.transformations.translation_matrix(translation))


class HandEFinger(Finger):
    """Robotiq Hand-E Finger"""

    def __init__(
        self,
        mesh_path: str = "finger.step",
        tip=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    ):
        mesh = trimesh.load(mesh_path)
        mesh = trimesh.util.concatenate([g.copy() for g in mesh.geometry.values()])
        self.tip = tip
        self.normal = normal

        super().__init__(mesh=mesh, tip=tip, normal=normal)


class BoxFinger(Finger):
    """Simple Box Finger"""

    def __init__(self, extents=[0.04, 0.025, 0.012]):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation([extents[0] / 2, 0.0, extents[2] / 2 + 0.001])
        tip = np.array([0.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])

        super().__init__(mesh=mesh, tip=tip, normal=normal)


class RobotiqHandE:
    def __init__(self, mesh_path=None):
        print("-> Initializing the gripper...")
        self.name = "Robotiq Hand-E Gripper"
        self.current_width = 0.0
        self.target = np.array([0.0, 0.0, 0.0])
        self.max_opening_width = 0.05  # meters
        self.box_finger_left = BoxFinger()
        self.box_finger_left.mesh.visual.face_colors = [100, 0, 0, 255]
        self.box_finger_right = BoxFinger()
        self.box_finger_right.mesh.visual.face_colors = [0, 0, 100, 255]
        self.box_finger_right.transform(
            trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        )
        if mesh_path is None:
            self.hand_e_finger_left = HandEFinger()
            self.hand_e_finger_right = HandEFinger()
        else:
            self.hand_e_finger_left = HandEFinger(mesh_path)
            self.hand_e_finger_right = HandEFinger(mesh_path)
        self.hand_e_finger_right.transform(
            trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        )

    def open(self, width: float):
        """
        Open the gripper to a specified width.

        Parameters
        ----------
        width : float
            The desired opening width (m).
        """
        if width > self.max_opening_width:
            raise ValueError(
                f"Width {width} exceeds maximum opening width of {self.max_opening_width}"
            )
        distance = (width - self.current_width) / 2
        self.box_finger_left.move(-distance)
        self.box_finger_right.move(-distance)
        self.hand_e_finger_left.move(-distance)
        self.hand_e_finger_right.move(-distance)
        self.current_width = width

    def get_box_fingers(self) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """
        Get the meshes of the box fingers.

        Returns
        -------
        tuple[trimesh.Trimesh, trimesh.Trimesh]
            A tuple containing the left and right box finger meshes.
        """
        return self.box_finger_left.mesh, self.box_finger_right.mesh

    def get_hand_e_fingers(self) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """
        Get the meshes of the Hand-E fingers.

        Returns
        -------
        tuple[trimesh.Trimesh, trimesh.Trimesh]
            A tuple containing the left and right Hand-E finger meshes.
        """
        return self.hand_e_finger_left.mesh, self.hand_e_finger_right.mesh

    def transform(self, matrix: np.ndarray):
        """
        Apply a transformation matrix to the entire gripper.

        Parameters
        ----------
        matrix : np.ndarray
            A 4x4 SE3 transformation matrix.
        """
        self.box_finger_left.transform(matrix)
        self.box_finger_right.transform(matrix)
        self.hand_e_finger_left.transform(matrix)
        self.hand_e_finger_right.transform(matrix)
        self.target = trimesh.transform_points([self.target], matrix)[0]

    def visualize(self):
        """Visualize the gripper."""
        scale = self.max_opening_width
        axis = self.hand_e_finger_left.normal

        scene = trimesh.Scene()
        scene.add_geometry(
            [self.hand_e_finger_left.mesh, self.hand_e_finger_right.mesh]
        )

        point_visual = trimesh.creation.uv_sphere(radius=scale * 0.05)
        point_visual.vertices += self.target
        point_visual.visual.vertex_colors = [255, 255, 0, 255]
        scene.add_geometry(point_visual)

        right = trimesh.creation.uv_sphere(radius=scale * 0.05)
        right.vertices += self.target + self.current_width / 2 * axis
        right.visual.vertex_colors = [255, 255, 0, 255]
        scene.add_geometry(right)

        left = trimesh.creation.uv_sphere(radius=scale * 0.05)
        left.vertices += self.target - self.current_width / 2 * axis
        left.visual.vertex_colors = [255, 255, 0, 255]
        scene.add_geometry(left)

        ax = trimesh.creation.axis(
            origin_size=scale * 0.02, axis_radius=scale * 0.02, axis_length=scale
        )
        scene.add_geometry(ax)
        scene.show(viewer="gl")
