from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R


@dataclass
class ContactPoint:
    """Represents a contact point on an object's surface"""

    position: np.ndarray  # 3D position in world coordinates
    normal: np.ndarray  # Surface normal (unit vector pointing outward)
    friction_coeff: float  # Friction coefficient

    def __post_init__(self):
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.max_angle = np.arctan(self.friction_coeff)

    def to_dict(self):
        """Convert the contact point to a dictionary for serialization."""
        return {
            "location": {
                "x": self.position[0].item(),
                "y": self.position[1].item(),
                "z": self.position[2].item(),
            },
            "normal": {
                "x": self.normal[0].item(),
                "y": self.normal[1].item(),
                "z": self.normal[2].item(),
            },
        }

    def sample_friction_cone(self, n_samples: int = 1):
        """
        Samples unit vectors uniformly distributed within the friction cone.

        The friction cone is defined by the contact normal and the maximum friction angle.
        Vectors are sampled from the spherical sector using the following method:
        1. Uniformly sample the cosine of the angle (the height of the sector) from the normal.
        2. Uniformly sample a rotation angle around the normal.

        The surface area of spherical sector is A = 2(pi)rh, where r is the radius and h is the height,
        so for each height the area is the same, i.e. dA = 2(pi)rdh.

        Parameters
        ----------
        n_samples : int, optional
            Number of unit vectors to sample (default=1)

        Return
        ------
        * np.ndarray
            Array of sampled unit vectors with shape (3, n_samples)
        """
        cx = 1 - np.random.rand(n_samples) * (1 - np.cos(self.max_angle))  # cos(x)
        sx = np.sqrt(1 - cx**2)  # sin(x)
        u = np.random.uniform(low=0.0, high=2 * np.pi, size=(n_samples,))

        # Compute a rotation matrix
        s = np.linalg.norm(self.normal[:2])
        if s == 0 and self.normal[2] == 1:
            rot = np.eye(3)
        elif s == 0:
            rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            rot = np.array(
                [
                    [-self.normal[1] / s, self.normal[0] / s, 0],
                    [
                        -self.normal[0] * self.normal[2] / s,
                        -self.normal[1] * self.normal[2] / s,
                        s,
                    ],
                    self.normal,
                ]
            ).T

        return rot @ np.vstack([sx * np.cos(u), sx * np.sin(u), cx])

    def __str__(self):
        position_rounded = np.round(self.position, 3)
        normal_rounded = np.round(self.normal, 3)
        return f"Contact Point(location = {position_rounded}, direction = {normal_rounded})"


class AntipodalGrasp:
    def __init__(
        self, x1: ContactPoint, x2: ContactPoint, approach_direction: np.ndarray = None
    ):
        """Initialize an antipodal grasp."""
        self.x1 = x1  # Contact 1
        self.x2 = x2  # Contact 2
        self.width = np.linalg.norm(x2.position - x1.position)  # Gripper opening
        if self.width < 1e-3:
            raise ValueError("The two contact points are too close.")
        self.mid = (x1.position + x2.position) / 2  # Midpoint of the grasp
        # Closing axis; the direction is from x1 to x2
        self.axis = (x2.position - x1.position) / self.width
        if approach_direction is None:
            self.random_approach_direction()
        else:
            self.approach_direction = approach_direction / np.linalg.norm(
                approach_direction
            )

    def to_dict(self):
        """Convert the grasp to a dictionary for serialization."""
        quaternion = R.from_rotvec(self.approach_direction).as_quat()
        return {
            "contacts": [self.x1.to_dict(), self.x2.to_dict()],
            "pose": {
                "translation": {
                    "x": self.mid[0].item(),
                    "y": self.mid[1].item(),
                    "z": self.mid[2].item(),
                },
                "rotation": {
                    "x": quaternion[0].item(),
                    "y": quaternion[1].item(),
                    "z": quaternion[2].item(),
                    "w": quaternion[3].item(),
                },
            },
            # "direction": {
            #     "x": self.approach_direction[0].item(),
            #     "y": self.approach_direction[1].item(),
            #     "z": self.approach_direction[2].item(),
            # },
        }

    def random_approach_direction(self):
        """
        Sample a random approach direction orthogonal to the grasp axis.

        Return
        ------
        * np.ndarray
            A random approach direction vector
        """
        dir = np.cross(self.axis, np.random.randn(3))
        self.approach_direction = dir / np.linalg.norm(dir)

    def _validate(self):
        """Check if this is a valid antipodal grasp."""
        return np.abs(np.dot(self.x1.normal, self.axis)) >= np.cos(
            self.x1.max_angle
        ) and np.abs(np.dot(self.x2.normal, self.axis)) >= np.cos(self.x2.max_angle)

    def transform(self, mat, mesh, friction_coeff):
        """
        Applies an SE3 transformation to the grasp.

        Parameter
        ---------
        mat : np.ndarray
            An SE3 transformation matrix

        mesh : trimesh.Trimesh
            Object mesh

        friction_coeff : float
            Friction coefficient of the mesh

        Return
        ------
        * AntipodalGrasp
            The transformed grasp by the given matrix
        """
        mid = (mat @ np.concatenate([self.mid, [1]]))[:-1].reshape(1, -1)
        axis = (mat @ np.concatenate([self.axis, [0]]))[:-1].reshape(1, -1)
        approach_direction = (mat @ np.concatenate([self.approach_direction, [0]]))[:-1]
        # Emits a rays from the mid point of the grasp in the closing direction
        # to find the intersection with the object surface
        x1, _, face_index = mesh.ray.intersects_location(mid, -axis)
        if len(x1) == 0:
            return None
        idx = np.argmin(np.linalg.norm(x1 - mid, axis=1))
        contact1 = ContactPoint(
            x1[idx], mesh.face_normals[face_index][idx], friction_coeff
        )

        x2, _, face_index = mesh.ray.intersects_location(mid, axis)
        if len(x2) == 0:
            return None
        idx = np.argmin(np.linalg.norm(x2 - mid, axis=1))
        contact2 = ContactPoint(
            x2[idx], mesh.face_normals[face_index][idx], friction_coeff
        )
        return AntipodalGrasp(contact1, contact2, approach_direction)

    def get_gripper_transformation(self, gripper):
        """
        Parameter
        ---------
        gripper : Gripper
            A Gripper object for collision checks and visualization

        Return
        ------
        * np.ndarray
            An SE3 transformation matrix to position the gripper at the grasp
        """
        z_dir = self.axis  # closing direction
        y_dir = self.approach_direction
        x_dir = np.cross(y_dir, z_dir)
        x_dir = x_dir / np.linalg.norm(x_dir)

        # Translate the gripper so that the finger mid point is at the origin
        t1 = np.eye(4)
        t1[:3, 3] = -gripper.target
        # Rotate and translate the gripper to the grasp pose
        t2 = np.eye(4)
        t2[:3, :3] = np.column_stack([x_dir, y_dir, z_dir])
        t2[:3, 3] = self.mid

        return t2 @ t1

    def __str__(self):
        return f"Antipodal Grasp:\n\t{self.x1}\n\t{self.x2}\n\tApproach direction: {np.round(self.approach_direction, 3)}"


class AntipodalGraspSampler:
    """Samples antipodal pairs using rejection sampling."""

    # TODO: Check multiple ray intersections...finger width

    def __init__(self, gripper, mesh, config):
        """
        Initialize the sampler with a gripper and configuration.

        Parameters
        ----------
        gripper : Gripper
            A Gripper object for collision checks and max opening width

        mesh : Trimesh.Trimesh
            The object mesh to sample grasps from. Should be centered at the origin.

        config : dict
            Configuration
        """
        self.gripper = gripper
        self.mesh = mesh
        self.friction_coeff = config["friction_coeff"]
        self.collision_manager = trimesh.collision.CollisionManager()
        self.n_approach_direction_samples = (
            10
            if "n_approach_direction_samples" not in config
            else config["n_approach_direction_samples"]
        )

        # Add plane z = 0
        delta = 0.001  # Small offset to avoid numerical issues
        plane = trimesh.Trimesh(
            vertices=[
                [-1, -1, -delta],
                [1, -1, -delta],
                [1, 1, -delta],
                [-1, 1, -delta],
            ],
            faces=[[0, 1, 2], [0, 2, 3]],
        )
        self.collision_manager.add_object("plane", plane)
        self.collision_manager.add_object("object", self.mesh)

    def collision_check(self, grasp: AntipodalGrasp):
        """
        Checks if the given grasp collides with the object.

        Parameters
        ----------
        grasp : AntipodalGrasp
            The antipodal grasp to check for collisions

        Return
        ------
        * bool
            True if the grasp collides with the object, False otherwise
        """
        self.gripper.open(grasp.width)
        box_left_finger, box_right_finger = self.gripper.get_box_fingers()
        self.collision_manager.add_object("left", box_left_finger)
        self.collision_manager.add_object("right", box_right_finger)
        self.collision_manager.set_transform(
            "left", grasp.get_gripper_transformation(self.gripper)
        )
        self.collision_manager.set_transform(
            "right", grasp.get_gripper_transformation(self.gripper)
        )
        collision = self.collision_manager.in_collision_internal()
        self.collision_manager.remove_object("left")
        self.collision_manager.remove_object("right")

        return not collision

    def sample_approach_direction(self, grasp: AntipodalGrasp, debug=False):
        """
        Samples a random approach direction orthogonal to the grasp axis.

        Parameters
        ----------
        grasp : AntipodalGrasp
            An antipodal grasp

        Returns
        ------
        * np.ndarray
            A random approach direction vector
        """
        for _ in range(self.n_approach_direction_samples):
            grasp.random_approach_direction()
            if debug:
                print(f"-> Checking grasp: {grasp}")
                print(f"-> Collision: {grasp}")
            if self.collision_check(grasp):
                return grasp
        return None

    def sample(self, n_grasps, max_trials=10000, debug=False):
        """
        Samples a list of candidate grasps for the given object.

        Parameters
        ----------
        object : Trimesh.Trimesh
            Object mesh. Assume the adhesive layer is z = 0 and the centroid is on the z-axis.

        n_grasps : int
            The number of grasps

        Returns
        ------
        * list
            A list of candidate grasps
        """
        grasps = []
        count = 0
        while len(grasps) < n_grasps and count < max_trials:
            # Sample a random point from the surface of the object
            x1, face_index = self.mesh.sample(count=1, return_index=True)
            contact1 = ContactPoint(
                x1[0], self.mesh.face_normals[face_index][0], self.friction_coeff
            )
            # Sample grasp axes from the friction cone. Add a minus sign to flip the outward normal
            # TODO: Handling complex shapes with multiple ray intersections
            closing_axis = -contact1.sample_friction_cone().reshape(1, -1)
            x2, _, face_index = self.mesh.ray.intersects_location(
                x1 + 1e-6 * closing_axis, closing_axis
            )
            x2 = (
                x2[np.linalg.norm(x2 - x1, axis=1) < self.gripper.max_opening_width]
                if x2.size > 0
                else []
            )
            if len(x2) > 0:
                contact2 = ContactPoint(
                    x2[0], self.mesh.face_normals[face_index][0], self.friction_coeff
                )
                if (
                    np.abs(np.dot(closing_axis, contact2.normal))
                    >= np.cos(contact2.max_angle)
                    and np.linalg.norm(x2[0] - x1) >= 1e-3
                ):
                    grasp = self.sample_approach_direction(
                        AntipodalGrasp(contact1, contact2), debug=debug
                    )
                    if grasp is not None:
                        if debug:
                            print(f"-> Found valid grasp: {grasp}")
                        grasps.append(grasp)
            count += 1
        if len(grasps) < n_grasps:
            print(
                f"! Only found {len(grasps)} grasps out of {n_grasps} requested after {count} trials."
            )
            return grasps
        return grasps


# Visualize the grasp
def visualize(mesh: trimesh.Trimesh, grasp: AntipodalGrasp, gripper=None):
    """
    Visualizes the given grasp on the mesh.

    Parameters
    ----------
    mesh : Trimesh.Trimesh
        The mesh of the object

    grasp : AntipodalGrasp
        The antipodal grasp to visualize

    gripper : optional
        Default is None, which uses friction cones
    """

    SCALE = gripper.max_opening_width if gripper else np.max(mesh.extents)

    # Set the mesh to be translucent
    mesh.visual.face_colors = [100, 100, 100, 200]

    scene = trimesh.Scene()
    scene.add_geometry(mesh)

    point_radius = SCALE * 0.05
    point_visual = trimesh.creation.uv_sphere(radius=point_radius)
    point_visual.vertices += grasp.x1.position
    point_visual.visual.vertex_colors = [255, 255, 0]
    scene.add_geometry(point_visual)

    point_radius = SCALE * 0.05
    point_visual = trimesh.creation.uv_sphere(radius=point_radius)
    point_visual.vertices += grasp.x2.position
    point_visual.visual.vertex_colors = [255, 255, 0]
    scene.add_geometry(point_visual)

    # Use friction cones if gripper is not given
    # Otherwise, visualize the gripper
    if gripper is None:
        cone_height = 0.1
        rotation = trimesh.geometry.align_vectors([0, 0, 1], -grasp.x1.normal)
        cone = trimesh.creation.cone(
            radius=cone_height * np.tan(grasp.x1.max_angle),
            height=cone_height,
            sections=32,
        )
        cone.vertices -= cone.vertices[cone.vertices[:, 2].argmax()]
        cone.apply_transform(rotation)
        cone.vertices += grasp.x1.position
        cone.visual.face_colors = [255, 165, 0, 255]
        scene.add_geometry(cone)

        cone_height = 0.1
        rotation = trimesh.geometry.align_vectors([0, 0, 1], -grasp.x2.normal)
        cone = trimesh.creation.cone(
            radius=cone_height * np.tan(grasp.x2.max_angle),
            height=cone_height,
            sections=32,
        )
        cone.vertices -= cone.vertices[cone.vertices[:, 2].argmax()]
        cone.apply_transform(rotation)
        cone.vertices += grasp.x2.position
        cone.visual.face_colors = [255, 165, 0, 255]
        scene.add_geometry(cone)
    else:
        gripper.open(grasp.width)
        print("-> Visualizing the grasp with the gripper...")
        left_finger, right_finger = gripper.get_hand_e_fingers()
        # left_finger, right_finger = gripper.get_box_fingers()
        left_finger = left_finger.copy()
        right_finger = right_finger.copy()
        # Position the gripper at the midpoint of the grasp and align with the approaching direction
        left_finger.apply_transform(grasp.get_gripper_transformation(gripper))
        right_finger.apply_transform(grasp.get_gripper_transformation(gripper))
        scene.add_geometry(left_finger)
        scene.add_geometry(right_finger)

    # Add adhesive interface
    vertices = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    plane = trimesh.Trimesh(vertices, faces)
    plane.visual.face_colors = [255, 255, 255, 255]
    scene.add_geometry(plane)

    # Add coordinate axes
    axes = trimesh.creation.axis(
        origin_size=SCALE * 0.02, axis_radius=SCALE * 0.05, axis_length=SCALE
    )
    scene.add_geometry(axes)

    # TODO: Use a different viewer for macOS
    scene.show(viewer="gl")
