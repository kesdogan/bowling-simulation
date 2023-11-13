from src.utils import Triangle, Vertex
from src.constraints import Simplicial2DConstraint
from src.solver import ProjectiveDynamicsSolver, PDConstraint

import numpy as np


def test_no_forces():
    """Test if the solver can handle no forces, leading to no
    change at all."""
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
            initial_positions=np.array(
                [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 3.0, 3.0]]
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


def test_constant_unidirectional_force():
    """Test if the solver can handle a constant unidirectional force, leading
    to a constant increase of the velocity in one direction."""
    vertices = [
        Vertex(
            position=np.array([1.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            mass=1.0,
            external_force=np.array([1.0, 0.0, 0.0]),
        ),
        Vertex(
            position=np.array([3.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            mass=1.0,
            external_force=np.array([1.0, 0.0, 0.0]),
        ),
        Vertex(
            position=np.array([1.0, 3.0, 3.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            mass=1.0,
            external_force=np.array([1.0, 0.0, 0.0]),
        ),
    ]

    triangles = [Triangle(0, 1, 2)]

    constraints: list[PDConstraint] = [
        Simplicial2DConstraint(
            triangle=triangles[0],
            initial_positions=np.array(
                [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 3.0, 3.0]]
            ),
            sigma_min=-1.0,
            sigma_max=1.0,
            weight=1,
        )
    ]

    solver = ProjectiveDynamicsSolver(vertices, constraints)

    for _ in range(100):
        solver.perform_step(100)

        for vertex, q, v in zip(vertices, solver.q, solver.v):
            vertex.velocity += vertex.external_force * solver.h
            vertex.position += vertex.velocity * solver.h

            assert np.allclose(v, vertex.velocity)
            assert np.allclose(q, vertex.position)
