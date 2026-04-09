from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class Pose:
    pos: np.ndarray
    quat: np.ndarray

    def se3(self):
        """Return the SE3 transformation matrix corresponding to this pose."""
        rot = R.from_quat(self.quat).as_matrix()
        se3 = np.eye(4)
        se3[:3, :3] = rot
        se3[:3, 3] = self.pos
        return se3

    def __str__(self):
        return f"Pose(pos={self.pos}, quat={self.quat})"


@dataclass(frozen=True)
class Contact:
    pos: np.ndarray
    normal: np.ndarray  # unit vector pointing outwards from the surface
    mu: float

    def __str__(self):
        return f"Contact(pos={self.pos}, normal={self.normal}, mu={self.mu})"


@dataclass(frozen=True)
class Grasp:
    pose: Pose
    width: float
    c1: Contact
    c2: Contact
    wrench: np.ndarray = None

    def __str__(self):
        wrench_str = (
            "None"
            if self.wrench is None
            else np.array2string(
                np.asarray(self.wrench),
                precision=3,
                suppress_small=True,
                separator=", ",
            )
        )
        return (
            "Grasp(\n"
            f"  pose={self.pose},\n"
            f"  width={self.width:.3f},\n"
            f"  c1={self.c1},\n"
            f"  c2={self.c2},\n"
            f"  wrench={wrench_str}\n"
            ")"
        )
