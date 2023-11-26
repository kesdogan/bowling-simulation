import igl
import numpy as np
import polyscope as ps

from collision_detection import collision_detecter
from constraints import Simplicial2DConstraint
from solver import ProjectiveDynamicsSolver
from utils import Object, Triangle

object_list = []

# load bowling ball from obj file place in field
v, _, _, f, _, _ = igl.read_obj("./assets/simple_ball.obj")
v = v + np.array([-10, 0, 0])
object_list.append(Object(v, f))
indices_ball = len(v)


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

# register complete mesh w/ polyscope
ps.init()
mesh = ps.register_surface_mesh("everything", vertices, faces)

# intial all objects as static, except for ball, which is given initial velocity
# slowly approaching pins
initial_velocities = np.zeros(vertices.shape)
initial_velocities[:indices_ball] = np.array([1, 0, 0])

masses = np.ones(len(vertices))
external_forces = np.zeros(vertices.shape)

constraints = [
    Simplicial2DConstraint(
        triangle=Triangle(*face), initial_positions=vertices, weight=1
    )
    for face in faces
]

solver = ProjectiveDynamicsSolver(
    vertices, initial_velocities, masses, external_forces, constraints
)


def run_step():
    collision = collision_detecter(object_list, vertices, faces, solver.q)
    if collision:
        print("collision detected and passed")
        print(collision)
        pass
    else:
        solver.perform_step(5)
    mesh.update_vertex_positions(solver.q)


ps.set_user_callback(run_step)
ps.show()
ps.clear_user_callback()
