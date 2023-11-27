from dataclasses import dataclass, field

import numpy as np

from src.solver import PDConstraint


@dataclass
class Simplicial2DConstraint(PDConstraint):
    """This class represents a 2D simplicial constraint as described in the
    paper.

    The calculation of the projection T is taken from Appendix A of the paper.
    """

    triangle_indices: np.ndarray
    initial_positions: np.ndarray

    sigma_min: float = 0.95
    sigma_max: float = 1.05

    A: np.ndarray = field(init=False)
    S: np.ndarray = field(init=False)
    X_g: np.ndarray = field(init=False)
    X_g_inv: np.ndarray = field(init=False)

    def __post_init__(self):
        n = len(self.initial_positions)

        self.A = np.array(
            [
                [1, 0, -1],
                [0, 1, -1],
            ]
        )

        self.S = np.zeros((3, n))
        self.S[0, self.triangle_indices[0]] = 1
        self.S[1, self.triangle_indices[1]] = 1
        self.S[2, self.triangle_indices[2]] = 1

        self.X_g = (self.A @ self.S @ self.initial_positions).T
        self.X_g_inv = np.linalg.pinv(self.X_g)

        self.intersting = 29 in self.triangle_indices

    def _get_auxiliary_variable(self, current_positions: np.ndarray) -> np.ndarray:
        X_f = (self.A @ self.S @ current_positions).T

        U, s, V_t = np.linalg.svd(X_f @ self.X_g_inv)

        s = np.clip(s, self.sigma_min, self.sigma_max)
        s = np.diag(s)

        T = U @ s @ V_t

        auxiliary_variable = (T @ self.X_g).T
        return auxiliary_variable


@dataclass
class CollisionConstraint(PDConstraint):
    """This class represents a constraint enforcing no collisions."""

    num_vertices: int
    penetrating_vertex_index: int
    projected_vertex_positions: np.ndarray

    A: np.ndarray = field(init=False)
    S: np.ndarray = field(init=False)

    def __post_init__(self):
        self.A = np.array([[1]])

        self.S = np.zeros((1, self.num_vertices))
        self.S[0, self.penetrating_vertex_index] = 1

    def _get_auxiliary_variable(self, current_positions: np.ndarray) -> np.ndarray:
        return np.expand_dims(self.projected_vertex_positions, axis=0)
