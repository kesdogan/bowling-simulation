from src.utils import Triangle, Vertex
from src.constraints import Simplicial2DConstraint, PDConstraint
from src.solver import ProjectiveDynamicsSolver

import numpy as np


def test_rotation_invariance():
    """Test if the solver is invariant w.r.t rotation of the
    mesh."""
    vertices = [
        Vertex(
            position=np.array([1.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            mass=1.0,
            external_force=np.array([0.0, 0.0, 0.0]),
        ),
        Vertex(
            position=np.array([3.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            mass=1.0,
            external_force=np.array([0.0, 0.0, 0.0]),
        ),
        Vertex(
            position=np.array([1.0, 3.0, 3.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            mass=1.0,
            external_force=np.array([0.0, 0.0, 0.0]),
        ),
    ]

    triangles = [Triangle(0, 1, 2)]

    constraints: list[PDConstraint] = [
        Simplicial2DConstraint(
            triangle=triangles[0],
            intial_positions=np.array(
                np.rot90([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 3.0, 3.0]])
            ),
            sigma_min=-1.0,
            sigma_max=1.0,
            weight=1,
        )
    ]

    solver = ProjectiveDynamicsSolver(vertices, constraints)

    for _ in range(100):
        solver.perform_step(100)

    assert np.allclose(solver.q, np.array([v.position for v in vertices]))
    assert np.allclose(solver.v, np.array([v.velocity for v in vertices]))
