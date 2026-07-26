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

    def __repr__(self) -> str:
        # Formats floats cleanly and removes unnecessary brackets/newlines
        pos_str = np.array2string(
            self.pos, precision=3, suppress_small=True, separator=", "
        )
        quat_str = np.array2string(
            self.quat, precision=3, suppress_small=True, separator=", "
        )
        return f"Pose(pos={pos_str}, quat={quat_str})"


@dataclass(frozen=True)
class Contact:
    pos: np.ndarray
    normal: np.ndarray  # unit vector pointing outwards from the surface
    mu: float
    force: np.ndarray = None

    def __repr__(self) -> str:
        # Formats floats cleanly and removes unnecessary brackets/newlines
        pos_str = np.array2string(
            self.pos, precision=3, suppress_small=True, separator=", "
        )
        normal_str = np.array2string(
            self.normal, precision=3, suppress_small=True, separator=", "
        )
        mu_str = f"{self.mu:.2f}"
        return f"Contact(pos={pos_str}, normal={normal_str}, mu={mu_str})"


@dataclass(frozen=True)
class Grasp:
    pose: Pose
    width: float
    c1: Contact
    c2: Contact
    wrench: np.ndarray = None
    score: float = None

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
        score_str = "None" if self.score is None else f"{self.score:.3f}"
        return (
            "Grasp(\n"
            f"  pose={self.pose},\n"
            f"  open width={self.width:.3f},\n"
            f"  c1={self.c1},\n"
            f"  c2={self.c2},\n"
            f"  wrench={wrench_str},\n"
            f"  score={score_str}\n"
            ")"
        )

    def _score_value(self) -> float:
        # Treat missing scores as lowest quality during comparisons.
        return float("-inf") if self.score is None else float(self.score)

    def __lt__(self, other):
        if not isinstance(other, Grasp):
            return NotImplemented
        return self._score_value() < other._score_value()

    def __le__(self, other):
        if not isinstance(other, Grasp):
            return NotImplemented
        return self._score_value() <= other._score_value()

    def __gt__(self, other):
        if not isinstance(other, Grasp):
            return NotImplemented
        return self._score_value() > other._score_value()

    def __ge__(self, other):
        if not isinstance(other, Grasp):
            return NotImplemented
        return self._score_value() >= other._score_value()
