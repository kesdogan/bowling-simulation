import igl
import numpy as np
import polyscope as ps

from src.constraints import Simplicial2DConstraint
from src.solver import ProjectiveDynamicsSolver

# load bowling ball from obj file place in field
v, _, _, f, _, _ = igl.read_obj("../assets/simple_ball.obj")
v = v + np.array([-10, 0, 0])
indices_ball = v.shape[0]


# create global faces and vertices np.arrays
vertices = np.array(v)
faces = np.array(f)

# add pins to the same mesh in this array
v, _, _, f, _, _ = igl.read_obj("../assets/simple_pin.obj")
for i in range(1):
    for j in range(-i, i + 1):
        faces = np.concatenate((faces, f + len(vertices)), axis=0)
        vertices = np.concatenate(
            (vertices, (v + np.array([i * 1.5, 0, 0]) + np.array([0, 0, j * 1.5]))),
            axis=0,
        )


indices_ball_mask = np.zeros(len(vertices))
indices_ball_mask[:indices_ball] = 1


# register complete mesh w/ polyscope
ps.init()
ps.register_surface_mesh("everything", vertices, faces)
ps.show()


# intial all objects as static, except for ball, which is given initial velocity
# slowly approaching pins
initial_velocities = np.zeros(vertices.shape)
initial_velocities[indices_ball_mask] = np.array([1, 0, 0])

masses = np.ones(len(vertices))
external_forces = np.zeros(vertices.shape)

constraints = [
    Simplicial2DConstraint(
        triangle=face,
        initial_positions=vertices,
    )
    for face in faces
]

solver = ProjectiveDynamicsSolver(
    vertices, initial_velocities, masses, external_forces, constraints
)
