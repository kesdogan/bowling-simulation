import igl
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from collision_detection import collision_detecter
from solver import (
    ProjectiveDynamicsSolver,
    CollisionConstraint,
    PDConstraint,
    Simplicial2DConstraint,
)
from utils import Object

# global veriables
object_list = []

vertices = (
    faces
) = (
    mesh
) = (
    initial_velocities
) = masses = external_forces = simplicial_constraints = solver = None
pin_rows = 1
ball_speed = 10.0
running = False
fancy_pins = False


# set up scene for a variable number of pin rows
def set_up_scene():
    global vertices, faces, object_list, initial_velocities, masses, external_forces, simplicial_constraints, solver, ball_speed, pin_rows, fancy_pins

    # load bowling ball from obj file place in field
    v, _, _, f, _, _ = igl.read_obj("./assets/simple_ball.obj")
    ball_min = v[:, 1].min()
    v = v + np.array([-3, 0, 0])
    object_list = [Object(v, f)]
    indices_ball = len(v)

    # create global faces and vertices np.arrays
    vertices = np.array(v)
    faces = np.array(f)

    # add pins to the same mesh in this array
    if fancy_pins:
        v, _, _, f, _, _ = igl.read_obj("./assets/fancy_pin.obj")
    else:
        v, _, _, f, _, _ = igl.read_obj("./assets/simple_pin.obj")
    v[:, 1] = v[:, 1] - v[:, 1].min() + ball_min
    for i in range(pin_rows):
        for j in range(-i, i + 1):
            faces = np.concatenate((faces, f + len(vertices)), axis=0)
            vertices = np.concatenate(
                (vertices, (v + np.array([i * 1.5, 0, 0]) + np.array([0, 0, j * 1.5]))),
                axis=0,
            )
            object_list.append(
                Object(v + np.array([i * 1.5, 0, 0]) + np.array([0, 0, j * 1.5]), f)
            )

    # intial all objects as static, except for ball, which is given initial velocity
    # slowly approaching pins
    initial_velocities = np.zeros(vertices.shape)
    initial_velocities[:indices_ball] = np.array([ball_speed, 0, 0])

    masses = np.ones(len(vertices))
    masses *= 0.00001
    masses[:indices_ball] *= 0.1
    external_forces = np.zeros(vertices.shape)

    simplicial_constraints = [
        Simplicial2DConstraint(
            triangle_indices=face, initial_positions=vertices, weight=5
        )
        for face in faces
    ]

    solver = ProjectiveDynamicsSolver(
        vertices,
        initial_velocities,
        masses,
        external_forces,
        simplicial_constraints,
        faces=faces,
        step_size=0.01,
    )
    solver.inverse_2d_constraints()


# register complete mesh w/ polyscope
set_up_scene()
ps.init()
mesh = ps.register_surface_mesh("everything", vertices, faces)


def callback():
    global running, pin_rows, mesh, vertices, object_list, solver, ball_speed, fancy_pins

    psim.PushItemWidth(100)
    psim.TextUnformatted("Here we do the initial set up of the scene")
    changed1, pin_rows = psim.InputInt("pin_rows", pin_rows, step=1, step_fast=1)
    changed2, ball_speed = psim.InputFloat(
        "ball_speed", ball_speed, step=0.1, step_fast=1
    )
    changed3, fancy_pins = psim.Checkbox("fancy_pins", fancy_pins)

    if changed1 or changed2 or changed3:
        print("scene settings changed")
        print(pin_rows)
        set_up_scene()
        ps.remove_surface_mesh("everything", error_if_absent=False)
        mesh = ps.register_surface_mesh("everything", vertices, faces)

    psim.Separator()
    psim.TextUnformatted("Here we control the simulation")
    if psim.Button("Start / Pause"):
        running = not running

    psim.PopItemWidth()

    if running:
        collisions = collision_detecter(object_list, vertices, faces, solver.q)

        collision_constraints = []

        for collision in collisions:
            for penetrating_vertex, projection_of_point_on_face in zip(
                collision.penetrating_vertex, collision.projection_of_point_on_face
            ):
                collision_constraints.append(
                    CollisionConstraint(
                        weight=1,
                        num_vertices=len(vertices),
                        penetrating_vertex_index=penetrating_vertex,
                        projected_vertex_positions=projection_of_point_on_face,
                    )
                )
        if collision_constraints:
            solver.update_constraints(simplicial_constraints + collision_constraints)
            solver.inverse_2d_constraints()

        solver.perform_step(10)
        mesh.update_vertex_positions(solver.q)


ps.set_user_callback(callback)
ps.show()
ps.clear_user_callback()
