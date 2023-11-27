import igl
import numpy as np
import polyscope as ps; import polyscope.imgui as psim

from collision_detection import collision_detecter
from constraints import Simplicial2DConstraint
from solver import ProjectiveDynamicsSolver
from utils import Object, Triangle

# global veriables
object_list = []
vertices = faces = mesh = initial_velocities = masses = external_forces = constraints = solver = None
pin_rows = 1; running = False


# set up scene for a variable number of pin rows
def set_up_scene():
    global vertices, faces, object_list, initial_velocities, masses, external_forces, constraints, solver

    # load bowling ball from obj file place in field
    v, _, _, f, _, _ = igl.read_obj("./assets/simple_ball.obj")
    v = v + np.array([-10, 0, 0])
    object_list.append(Object(v, f))

    # create global faces and vertices np.arrays
    vertices = np.array(v)
    faces = np.array(f)

    # add pins to the same mesh in this array
    v, _, _, f, _, _ = igl.read_obj("./assets/simple_pin.obj")
    for i in range(1):
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
    initial_velocities[:object_list[0].v.shape[0]] = np.array([1, 0, 0])

    masses = np.ones(len(vertices))
    external_forces = np.zeros(vertices.shape)

    constraints = [
        Simplicial2DConstraint(
            triangle_indices=face, initial_positions=vertices, weight=1
        )
        for face in faces
    ]

    solver = ProjectiveDynamicsSolver(
        vertices, initial_velocities, masses, external_forces, constraints
    )

# register complete mesh w/ polyscope
set_up_scene()
ps.init()
mesh = ps.register_surface_mesh("everything", vertices, faces)



def callback():
    global running, pin_rows, mesh, vertices, object_list, solver
    
    psim.PushItemWidth(100)
    psim.TextUnformatted("Here we do the initial set up of the scene")
    changed, pin_rows = psim.InputInt("pin_rows", pin_rows, step=1, step_fast=1) 

    if changed: 
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
        collision = collision_detecter(object_list, vertices, faces, solver.q)
        if collision:
            print("collision detected and passed")
            print(collision)
            pass
        else:
            solver.perform_step(5)
            mesh.update_vertex_positions(solver.q)


ps.set_user_callback(callback)
ps.show()
ps.clear_user_callback()
