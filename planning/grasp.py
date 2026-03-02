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


@dataclass(frozen=True)
class Contact:
    pos: np.ndarray
    normal: np.ndarray
    mu: float


@dataclass(frozen=True)
class Grasp:
    pose: Pose
    width: float
    c1: Contact
    c2: Contact
    wrench: np.ndarray = None
