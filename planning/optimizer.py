from abc import ABC

import numpy as np
import torch
from grasp import Contact, Grasp
from sampler import GraspSampler

from meshgraphnet.graph_builder import GraphBuilderVirtual
from meshgraphnet.utils import msh_to_trimesh


def skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric cross-product matrix for vector v."""
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def wrench_to_contact_forces(
    wrench: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a wrench into two contact forces at the given contact points.

    Solves the grasp matrix equation  w = G [f1; f2]  in the least-squares sense:

        G = [ I      I    ]   (6x6, but rank 5 for a parallel jaw gripper)
            [ p1     p2   ]

    A parallel jaw gripper cannot resist torques along the closing axis with
    contact forces alone, so G is always rank-deficient. lstsq returns the
    minimum-norm solution, projecting the wrench onto the reachable wrench space.

    Args:
        wrench: (6,) array [fx, fy, fz, tx, ty, tz] in world frame.
        p1, p2: (3,) contact points relative to the object's center of mass.

    Returns:
        f1, f2: (3,) minimum-norm force vectors at p1 and p2.
    """
    I = np.eye(3)
    G = np.block(
        [
            [I, I],
            [skew(p1), skew(p2)],
        ]
    )
    forces, _, _, _ = np.linalg.lstsq(G, wrench, rcond=None)
    return forces[:3], forces[3:]


def sample_wrenches(
    k: int, force_scale: float = 1.0, torque_scale: float = 1.0
) -> np.ndarray:
    """Sample k unit wrenches uniformly on the 6D wrench sphere.

    Forces and torques live in different units, so the two sub-vectors are
    scaled independently before normalisation so the sampling is not biased
    toward one or the other.

    Args:
        k:             Number of wrenches to sample.
        force_scale:   Characteristic force magnitude (e.g. object weight in N).
        torque_scale:  Characteristic torque magnitude (e.g. force * moment arm).

    Returns:
        (k, 6) array of wrenches [fx, fy, fz, tx, ty, tz], each with unit norm
        in the scaled sense.
    """
    w = np.random.randn(k, 6)
    w[:, :3] *= force_scale
    w[:, 3:] *= torque_scale
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    return w


def sample_contact_forces(contact: Contact, fmin: float, fmax: float) -> np.ndarray:
    """Sample a random contact force vector within the friction cone."""
    # Sample a random direction in the tangent plane
    tangent = np.random.randn(3)
    tangent -= tangent.dot(contact.normal) * contact.normal  # project to tangent plane
    tangent /= np.linalg.norm(tangent) + 1e-8  # normalize

    # Sample a random magnitude within the friction cone
    alpha = np.random.rand() * (fmax - fmin) + fmin  # scale to desired range
    return alpha * tangent + contact.normal  # total contact force vector


class GraspOptimizer(ABC):
    def __init__(self, gripper):
        self.gripper = gripper

    def optimize(self, mesh, mu):
        raise NotImplementedError


class HeuristicBasedGraspOptimizer(GraspOptimizer):
    def __init__(self, gripper):
        super().__init__(gripper)


class GNNBasedGraspOptimizer(GraspOptimizer):
    def __init__(self, gripper, model, normalizer, device="cpu"):
        super().__init__(gripper)
        self.model = model
        self.normalizer = normalizer
        self.builder = GraphBuilderVirtual()
        self.device = device
        self.epsilon = 1e-4  # small tolerance for bottom stress extraction

    def optimize(self, msh, mu, k=20):
        """Sample grasps, build graphs, and predict scores to find the best grasp.

        Args:
            msh: Trimesh mesh of the object to grasp.
            mu: Friction coefficient for grasp sampling.
            k: Number of wrenches to sample per grasp.
        Returns:
            best_grasp: Grasp with the highest predicted score, or None if no valid grasps found.
        """
        mesh = msh_to_trimesh(msh)
        pos_com = mesh.center_mass
        num_nodes = msh.points.shape[0]
        y0 = np.zeros((num_nodes, 4))  # dummy node features

        sampler = GraspSampler(mesh, self.gripper, mu)
        grasps = sampler.sample(n_samples=100)
        if not grasps:
            print("No valid grasps found!")
            return None

        # Sample wrenches
        wrenches = sample_wrenches(k, force_scale=1.0, torque_scale=0.1)

        # Build graph and predict scores for each grasp
        graphs = []
        for grasp in grasps:
            pos1 = grasp.c1.pos - pos_com
            pos2 = grasp.c2.pos - pos_com
            for wrench in wrenches:
                f1, f2 = wrench_to_contact_forces(wrench, pos1, pos2)
                grip = 0.1 * (pos1 - pos2) / grasp.width
                contacts = [(pos1, f1 - grip / 2), (pos2, f2 + grip / 2)]
                graph = self.builder.build(msh, y0, contacts)
                graphs.append(
                    (
                        graph.to(self.device),
                        Grasp(
                            pose=grasp.pose,
                            width=grasp.width,
                            c1=grasp.c1,
                            c2=grasp.c2,
                            wrench=wrench,
                        ),
                    )
                )

        best_grasp = None
        max_score = -np.inf
        for graph, grasp in graphs:
            y_pred = self.model(self.normalizer.normalize(graph))
            y_pred = self.normalizer.denormalize_y(y_pred)
            score = torch.max(y_pred[graph.x[:, 2] < self.epsilon, 3])
            if score > max_score:
                max_score = score
                best_grasp = grasp

        return best_grasp
