"""This file contains the main function of the project. It sets up the scene and
starts the simulation loop."""

import igl
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from collision_detection import collision_detecter
from solver import ProjectiveDynamicsSolver
from src.caching import Cache
from src.constraints import (CollisionConstraint, Simplicial2DConstraint,
                             VolumeConstraint)
from src.solver_fast_shape_constraints import \
    ProjectiveDynamicsSolver as FasterProjectiveDynamicsSolver
from utils import Object

SOLVER = FasterProjectiveDynamicsSolver  # or ProjectiveDynamicsSolver

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
ball_speed = 90.0
running = False
fancy_pins = True

frame = 0


def set_up_scene():
    """Sets up the scene given by the global variables."""
    global vertices, faces, object_list, initial_velocities, masses, external_forces, shape_constraints, solver, ball_speed, pin_rows, fancy_pins, center_indices

    # set up the ball
    v, _, _, f, _, _ = igl.read_obj("./assets/simple_ball.obj")
    ball_min = v[:, 1].min()
    v = v + np.array([-3, 0, 0])
    object_list = [Object(v, f)]

    vertices = np.array(v)
    faces = np.array(f)

    ball_center = np.mean(vertices, axis=0)
    vertices = np.concatenate((vertices, ball_center.reshape(1, -1)))

    indices_ball = len(vertices)
    center_indices = [indices_ball - 1]

    tetrahedrons = [np.append(face, indices_ball - 1) for face in faces]

    # set up pins

    if fancy_pins:
        v, _, _, f, _, _ = igl.read_obj("./assets/fancy_pin.obj")
    else:
        v, _, _, f, _, _ = igl.read_obj("./assets/simple_pin.obj")
    v[:, 1] = v[:, 1] - v[:, 1].min() + ball_min

    for i in range(pin_rows):
        for j in range(i + 1):
            pin_vertices = (
                v
                + np.array([i * 1.6, 0, 0])
                + np.array([0, 0, -(i / 2) * 1.6 + j * 1.6])
            )
            pin_center = np.mean(pin_vertices, axis=0)
            pin_vertices = np.concatenate((pin_vertices, pin_center.reshape(1, -1)))

            pin_faces = f + len(vertices)

            faces = np.concatenate((faces, pin_faces), axis=0)
            vertices = np.concatenate(
                (vertices, pin_vertices),
                axis=0,
            )
            object_list.append(Object(pin_vertices, f))

            tetrahedrons += [np.append(face, len(vertices) - 1) for face in pin_faces]
            center_indices.append(len(vertices) - 1)

    # set up floor

    floor_vertices = np.array(
        [
            [-10, np.min(vertices[:, 1]) - 0.001, -10],
            [-10, np.min(vertices[:, 1]) - 0.001, 10],
            [10, np.min(vertices[:, 1]) - 0.001, -10],
            [10, np.min(vertices[:, 1]) - 0.001, 10],
        ]
    )
    floor_faces = np.array([[0, 1, 2], [3, 2, 1]])
    faces = np.concatenate((faces, floor_faces + len(vertices)), axis=0)
    vertices = np.concatenate((vertices, floor_vertices), axis=0)
    object_list.append(Object(floor_vertices, floor_faces))

    # set up velocities, masses, external forces

    # intial all objects as static, except for ball, which is given initial velocity
    # slowly approaching pins
    initial_velocities = np.zeros(vertices.shape)
    initial_velocities[:indices_ball] = np.array([ball_speed, 0, 0])

    masses = np.ones(len(vertices))
    masses *= 0.000001
    masses[:indices_ball] *= 10
    masses[-len(floor_vertices) :] = 1
    external_forces = np.zeros(vertices.shape)
    external_forces[: -len(floor_vertices)] = np.array([0, -0.001, 0])

    # set up constraints and solver

    shape_constraints = [
        VolumeConstraint(
            tetrahedron_indices=tetrahedron, initial_positions=vertices, weight=0.5
        )
        for tetrahedron in tetrahedrons
    ]
    shape_constraints += [
        Simplicial2DConstraint(
            triangle_indices=face,
            initial_positions=vertices,
            weight=20,
        )
        for face in faces
    ]

    solver = SOLVER(
        vertices,
        initial_velocities,
        masses,
        external_forces,
        shape_constraints,
        step_size=0.001,
    )


set_up_scene()
cache = Cache(faces, vertices)

ps.init()
mesh = ps.register_surface_mesh("everything", vertices, faces)


def callback():
    global running, pin_rows, mesh, vertices, object_list, solver, ball_speed, fancy_pins, frame, cache, render

    # set up GUI

    psim.PushItemWidth(100)
    psim.TextUnformatted("Here we do the initial set up of the scene")
    changed1, pin_rows = psim.InputInt("pin_rows", pin_rows, step=1, step_fast=1)
    changed2, ball_speed = psim.InputFloat(
        "ball_speed", ball_speed, step=0.1, step_fast=1
    )
    changed3, fancy_pins = psim.Checkbox("fancy_pins", fancy_pins)

    psim.Separator()
    psim.TextUnformatted("Here we control the simulation")
    if psim.Button("Start / Pause"):
        running = not running
    changed4, render = psim.Checkbox("render", render)

    if changed1 or changed2 or changed3 or changed4:
        print("scene settings changed")
        print(pin_rows)
        set_up_scene()
        ps.remove_surface_mesh("everything", error_if_absent=False)
        mesh = ps.register_surface_mesh("everything", vertices, faces)
        cache = Cache(faces, vertices)

    psim.PopItemWidth()

    if running:
        # first detect collisions, then update constraints, then perform step

        collisions = collision_detecter(object_list, vertices, faces, solver.q)

        collision_constraints = []

        for collision in collisions:
            for penetrating_vertex, projection_of_point_on_face in zip(
                collision.penetrating_vertex, collision.projection_of_point_on_face
            ):
                if penetrating_vertex in center_indices:
                    continue

                collision_constraints.append(
                    CollisionConstraint(
                        weight=1,
                        num_vertices=len(vertices),
                        penetrating_vertex_index=penetrating_vertex,
                        projected_vertex_positions=projection_of_point_on_face,
                    )
                )

        solver.update_constraints(shape_constraints + collision_constraints)

        solver.perform_step(10)

        # render the new frame

        if render:
            mesh.update_vertex_positions(solver.q)

        cache.add_frame(solver.q)

        frame += 1

        if frame % 10 == 0:
            cache.store()
            mesh.update_vertex_positions(solver.q)
            print("frame", frame)


ps.set_user_callback(callback)
ps.show()
ps.clear_user_callback()
cache.store()
