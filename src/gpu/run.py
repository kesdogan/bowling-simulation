"""This file contains the main function of the project. It sets up the scene and
starts the simulation loop."""

import igl
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch

from collision_detection import collision_detecter
from constraints import CollisionConstraint, Simplicial2DConstraint
from solver import ProjectiveDynamicsSolver
from utils import Object

# global veriables
object_list = []

vertices = (
    faces
) = (
    mesh
) = (
    initial_velocities
) = masses = external_forces = shape_constraints = solver = center_indices = None
pin_rows = 1
render = False
ball_speed = 60.0
running = False
fancy_pins = False
gpu = False

frame = 0


# set up scene for a variable number of pin rows
def set_up_scene():
    global vertices, faces, object_list, initial_velocities, masses, external_forces, simplicial_constraints, solver, ball_speed, pin_rows, fancy_pins, gpu
    device = torch.device("cuda" if gpu else "cpu")

    # set up the ball

    v, _, _, f, _, _ = igl.read_obj("./assets/simple_ball.obj")
    ball_min = v[:, 1].min()
    v = v + np.array([-6, 0, 0])
    object_list = [Object(torch.from_numpy(v), torch.from_numpy(f))]
    indices_ball = len(v)

    # create global faces and vertices np.arrays
    vertices = torch.from_numpy(v)
    faces = torch.from_numpy(f)

    # add pins to the same mesh in this array
    if fancy_pins:
        v, _, _, f, _, _ = igl.read_obj("./assets/fancy_pin.obj")
    else:
        v, _, _, f, _, _ = igl.read_obj("./assets/simple_pin.obj")
    v[:, 1] = v[:, 1] - v[:, 1].min() + ball_min
    for i in range(pin_rows):
        for j in range(-i, i + 1):
            faces = torch.concatenate((faces, torch.tensor(f + len(vertices))), axis=0)
            vertices = torch.concatenate(
                (
                    vertices,
                    (
                        torch.from_numpy(v)
                        + torch.tensor([i * 1.5, 0, 0])
                        + torch.tensor([0, 0, j * 1.5])
                    ),
                ),
                axis=0,
            )
            object_list.append(
                Object(
                    torch.from_numpy(v)
                    + torch.tensor([i * 1.5, 0, 0])
                    + torch.tensor([0, 0, j * 1.5]),
                    f,
                )
            )

    # intial all objects as static, except for ball, which is given initial velocity
    # slowly approaching pins
    initial_velocities = torch.zeros(vertices.shape).float().to(device)
    initial_velocities[:indices_ball] = torch.tensor([ball_speed, 0, 0]).float()

    vertices = vertices.float().to(device)
    faces = faces.long().to(device)

    masses = torch.ones(len(vertices)).float().to(device)
    masses *= 0.00001
    masses[:indices_ball] *= 0.1
    external_forces = torch.zeros(vertices.shape).float().to(device)

    simplicial_constraints = [
        Simplicial2DConstraint(
            triangle_indices=face, initial_positions=vertices, weight=5.0, gpu=gpu
        )
        for face in faces
    ]

    solver = ProjectiveDynamicsSolver(
        vertices,
        initial_velocities,
        masses,
        external_forces,
        simplicial_constraints,
        step_size=0.01,
        gpu=gpu,
    )


# register complete mesh w/ polyscope
set_up_scene()
ps.init()
mesh = ps.register_surface_mesh(
    "everything", vertices.cpu().numpy(), faces.cpu().numpy()
)


def callback():
    global running, pin_rows, mesh, vertices, object_list, solver, ball_speed, fancy_pins, gpu

    psim.PushItemWidth(100)
    psim.TextUnformatted("Here we do the initial set up of the scene")
    changed1, pin_rows = psim.InputInt("pin_rows", pin_rows, step=1, step_fast=1)
    changed2, ball_speed = psim.InputFloat(
        "ball_speed", ball_speed, step=0.1, step_fast=1
    )
    changed3, fancy_pins = psim.Checkbox("fancy_pins", fancy_pins)
    changed4, gpu = psim.Checkbox("gpu", gpu)

    if changed1 or changed2 or changed3 or changed4:
        set_up_scene()
        ps.remove_surface_mesh("everything", error_if_absent=False)
        mesh = ps.register_surface_mesh(
            "everything", vertices.cpu().numpy(), faces.cpu().numpy()
        )

    psim.Separator()
    psim.TextUnformatted("Here we control the simulation")
    if psim.Button("Start / Pause"):
        running = not running

    psim.PopItemWidth()

    if running:
        collisions = collision_detecter(object_list, vertices, faces, solver.q, gpu)

        collision_constraints = []

        for collision in collisions:
            for penetrating_vertex, projection_of_point_on_face in zip(
                collision.penetrating_vertex, collision.projection_of_point_on_face
            ):
                collision_constraints.append(
                    CollisionConstraint(
                        weight=1.0,
                        num_vertices=len(vertices),
                        penetrating_vertex_index=penetrating_vertex,
                        projected_vertex_positions=projection_of_point_on_face,
                        gpu=gpu,
                    )
                )

        solver.update_constraints(simplicial_constraints + collision_constraints)

        solver.perform_step(10)

        new_vertices = solver.q.cpu().numpy()
        mesh.update_vertex_positions(new_vertices)


ps.set_user_callback(callback)
ps.show()
ps.clear_user_callback()
