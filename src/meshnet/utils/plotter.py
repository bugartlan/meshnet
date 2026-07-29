import trimesh

from meshnet.planner.grasp import Grasp
from meshnet.utils.mesh import Mesh

RADIUS = 0.01
LENGTH = 1.0

COLOR_RED = [255, 0, 0, 255]


def plot_grasp(obj: Mesh, grasp: Grasp, gripper):
    """Plot a grasp on a mesh"""
    mesh = obj.surface
    scale = mesh.scale

    origin_size = RADIUS * scale
    axis_radius = RADIUS * scale
    axis_length = LENGTH * scale

    scene = trimesh.Scene(mesh)

    ax = trimesh.creation.axis(
        origin_size=origin_size,
        axis_radius=axis_radius,
        axis_length=axis_length,
    )
    scene.add_geometry(ax)

    gripper_mesh = gripper.mesh.copy()
    tf = grasp.pose.se3() @ gripper.tf_real_to_fake()
    scene.add_geometry(gripper_mesh, transform=tf)

    # Visualize contact points
    radius = RADIUS * scale
    c1_sphere = trimesh.creation.icosphere(radius=radius)
    c1_sphere.visual.face_colors = COLOR_RED
    scene.add_geometry(
        c1_sphere,
        transform=trimesh.transformations.translation_matrix(grasp.c1.pos),
    )

    c2_sphere = trimesh.creation.icosphere(radius=radius)
    c2_sphere.visual.face_colors = COLOR_RED
    scene.add_geometry(
        c2_sphere,
        transform=trimesh.transformations.translation_matrix(grasp.c2.pos),
    )

    scene.show(viewer="gl")
