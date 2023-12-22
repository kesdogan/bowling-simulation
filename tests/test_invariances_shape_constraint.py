from src.constraints import Simplicial2DConstraint
from src.solver_fast_shape_constraints import ProjectiveDynamicsSolver, PDConstraint
from scipy.spatial.transform import Rotation
import numpy as np


def test_rotation_invariance():
    """Test if the solver is invariant w.r.t rotation of the
    mesh."""
    initial_positions = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 3.0, 3.0]])
    initial_velocities = np.zeros_like(initial_positions)
    masses = np.ones(initial_positions.shape[0])
    external_forces = np.zeros_like(initial_positions)

    triangles = [np.array([0, 1, 2])]

    rotation = Rotation.from_euler("xy", [30, 50], degrees=True)

    constraints: list[PDConstraint] = [
        Simplicial2DConstraint(
            triangle_indices=triangles[0],
            initial_positions=rotation.apply(initial_positions),
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


def test_rotation_translation_invariance():
    """Test if the solver is invariant w.r.t rotation and translation of the
    mesh."""
    initial_positions = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 3.0, 3.0]])
    initial_velocities = np.zeros_like(initial_positions)
    masses = np.ones(initial_positions.shape[0])
    external_forces = np.zeros_like(initial_positions)

    triangles = [np.array([0, 1, 2])]

    rotation = Rotation.from_euler("xy", [30, 50], degrees=True)

    constraints: list[PDConstraint] = [
        Simplicial2DConstraint(
            triangle_indices=triangles[0],
            initial_positions=rotation.apply(
                initial_positions + np.array([[10, 20, -10]])
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
