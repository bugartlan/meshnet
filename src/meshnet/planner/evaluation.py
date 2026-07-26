"""Evaluation of the grasp-wrench planner
Sample 100 grasps, compute their scores using the GNN, and select the top 10 and bottom 10, and 10 random from the rest of the 80.

Then evaluate using the fem solver and compare the rankings

"""

import meshio

from meshnet.utils.gripper import ROBOTIQ_HANDE_GRIPPER
from src.meshnet.mgn.utils import msh_to_trimesh
from src.meshnet.planner.sampler import GraspSampler

msh = meshio.read("../meshes/test/msh/Bushing3_cg1.msh")
mesh = msh_to_trimesh(msh)
gripper = ROBOTIQ_HANDE_GRIPPER()
sampler = GraspSampler(mesh=mesh, gripper=gripper, mu=0.01)
grasps = sampler.sample(n_samples=200, debug=False)[:100]
