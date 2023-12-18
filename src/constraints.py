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

    def _get_auxiliary_variable(self, current_positions: np.ndarray) -> np.ndarray:
        X_f = (self.A @ self.S @ current_positions).T

        U, s, V_t = np.linalg.svd(X_f @ self.X_g_inv)

        s = np.clip(s, self.sigma_min, self.sigma_max)
        s = np.diag(s)

        T = U @ s @ V_t

        auxiliary_variable = (T @ self.X_g).T
        return auxiliary_variable


@dataclass
class VolumeConstraint(PDConstraint):
    """This class represents a volume constraint as described in the
    paper.

    The calculation of the projection T is taken from Appendix A of
    the paper.
    """

    tetrahedron_indices: np.ndarray
    initial_positions: np.ndarray

    sigma_min: float = 0.95
    sigma_max: float = 1.05

    A: np.ndarray = field(init=False)
    S: np.ndarray = field(init=False)
    X_g: np.ndarray = field(init=False)
    X_g_inv: np.ndarray = field(init=False)

    eps: float = 0.000001

    def __post_init__(self):
        self.A = np.array(
            [
                [1, 0, 0, -1],
                [0, 1, 0, -1],
                [0, 0, 1, -1],
            ]
        )

        n = len(self.initial_positions)
        self.S = np.zeros((4, n))
        self.S[0, self.tetrahedron_indices[0]] = 1
        self.S[1, self.tetrahedron_indices[1]] = 1
        self.S[2, self.tetrahedron_indices[2]] = 1
        self.S[3, self.tetrahedron_indices[3]] = 1

        self.X_g = (self.A @ self.S @ self.initial_positions).T
        self.X_g_inv = np.linalg.pinv(self.X_g)

    def _get_auxiliary_variable(self, current_positions: np.ndarray) -> np.ndarray:
        X_f = (self.A @ self.S @ current_positions).T

        # perform SVD

        U, sigma, V_t = np.linalg.svd(X_f @ self.X_g_inv)
        sigma = np.diag(sigma)
        sigma_p = sigma

        # check if we even have to correct

        det_sigma_p = np.linalg.det(sigma_p)

        if self.sigma_min <= det_sigma_p <= self.sigma_max:
            T = U @ sigma_p @ V_t
            auxiliary_variable = (T @ self.X_g).T
            return auxiliary_variable

        # perform iterative minimization

        D_k = np.random.random(3)

        for _ in range(10):
            sigma_p = sigma + np.diag(D_k)
            det_sigma_p = np.linalg.det(sigma_p)

            DC_D = np.repeat(det_sigma_p, 3) / np.maximum(
                np.diagonal(sigma_p), self.eps
            )
            C_D = max(det_sigma_p - self.sigma_max, det_sigma_p - self.sigma_min)

            D_k = (DC_D.T @ D_k - C_D) / max(np.dot(DC_D, DC_D), self.eps) * DC_D

        # calculate projection T

        T = U @ sigma_p @ V_t

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
