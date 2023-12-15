from src.utils import Triangle, Vertex
from src.constraints import Simplicial2DConstraint
from src.solver import ProjectiveDynamicsSolver, PDConstraint

import numpy as np


def test_no_forces():
    """Test if the solver can handle no forces, leading to no
    change at all."""
    initial_positions = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 3.0, 3.0]])
    initial_velocities = np.zeros_like(initial_positions)
    masses = np.ones(initial_positions.shape[0])
    external_forces = np.zeros_like(initial_positions)

    triangles = [np.array([0, 1, 2])]

    constraints: list[PDConstraint] = [
        Simplicial2DConstraint(
            triangle_indices=triangles[0],
            initial_positions=np.array(
                [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 3.0, 3.0]]
            ),
            sigma_min=-1.0,
            sigma_max=1.0,
            weight=1,
        )
    ]

    solver = ProjectiveDynamicsSolver(
        initial_positions, initial_velocities, masses, external_forces, constraints
    )

    for _ in range(100):
        solver.perform_step(100)

    assert np.allclose(solver.q, initial_positions)
    assert np.allclose(solver.v, initial_velocities)


def test_constant_unidirectional_force():
    """Test if the solver can handle a constant unidirectional force, leading
    to a constant increase of the velocity in one direction."""
    initial_positions = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 3.0, 3.0]])
    initial_velocities = np.zeros_like(initial_positions)
    masses = np.ones(initial_positions.shape[0])
    external_forces = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0, 0]])

    triangles = [np.array([0, 1, 2])]

    constraints: list[PDConstraint] = [
        Simplicial2DConstraint(
            triangle_indices=triangles[0],
            initial_positions=np.array(
                [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 3.0, 3.0]]
            ),
            sigma_min=-1.0,
            sigma_max=1.0,
            weight=1,
        )
    ]

    solver = ProjectiveDynamicsSolver(
        initial_positions, initial_velocities, masses, external_forces, constraints
    )

    for _ in range(100):
        solver.perform_step(100)

        for v0, q0, f, q, v in zip(
            initial_velocities, initial_positions, external_forces, solver.q, solver.v
        ):
            v0 += f * solver.h
            q0 += v0 * solver.h

            assert np.allclose(v, v0)
            assert np.allclose(q, q0)
