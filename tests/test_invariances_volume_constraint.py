from src.utils import Triangle, Vertex
from src.constraints import VolumeConstraint
from src.solver import ProjectiveDynamicsSolver, PDConstraint
from scipy.spatial.transform import Rotation
import numpy as np


def test_rotation_invariance():
    """Test if the solver is invariant w.r.t rotation of the
    mesh."""
    initial_positions = np.array(
        [[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [1.0, 3.0, 3.0], [2.0, 2.0, 2.0]]
    )
    initial_velocities = np.zeros_like(initial_positions)
    masses = np.ones(initial_positions.shape[0])
    external_forces = np.zeros_like(initial_positions)

    rotation = Rotation.from_euler("xy", [30, 50], degrees=True)

    constraints: list[PDConstraint] = [
        VolumeConstraint(
            tetrahedron_indices=np.array([0, 1, 2, 3]),
            initial_positions=rotation.apply(initial_positions),
            sigma_min=0.95,
            sigma_max=1.05,
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
    initial_positions = np.array(
        [[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [1.0, 3.0, 3.0], [2.0, 2.0, 2.0]]
    )
    initial_velocities = np.zeros_like(initial_positions)
    masses = np.ones(initial_positions.shape[0])
    external_forces = np.zeros_like(initial_positions)

    rotation = Rotation.from_euler("xy", [30, 50], degrees=True)

    constraints: list[PDConstraint] = [
        VolumeConstraint(
            tetrahedron_indices=np.array([0, 1, 2, 3]),
            initial_positions=rotation.apply(initial_positions),
            sigma_min=0.95,
            sigma_max=1.05,
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
