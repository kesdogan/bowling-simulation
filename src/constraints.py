from dataclasses import dataclass, field

import numpy as np

from src.solver import PDConstraint
from src.utils import Triangle


@dataclass
class Simplicial2DConstraint(PDConstraint):
    """This class represents a 2D simplicial constraint as described in the
    paper.

    The calculation of the projection T is taken from Appendix A of the paper.
    """

    triangle: Triangle
    initial_positions: np.ndarray

    sigma_min: float = 0.95
    sigma_max: float = 1.05

    A: np.ndarray = field(init=False)
    S: np.ndarray = field(init=False)

    def __post_init__(self):
        n = len(self.initial_positions)

        self.A = np.array(
            [
                [1, 0, -1],
                [0, 1, -1],
            ]
        )

        self.S = np.zeros((3, n))
        self.S[0, self.triangle.v0] = 1
        self.S[1, self.triangle.v1] = 1
        self.S[2, self.triangle.v2] = 1

    def _get_auxiliary_variable(self, current_positions: np.ndarray) -> np.ndarray:
        X_g = (self.A @ self.S @ self.initial_positions).T
        X_f = (self.A @ self.S @ current_positions).T

        U, s, V_t = np.linalg.svd(X_f @ np.linalg.pinv(X_g))

        s = np.clip(s, self.sigma_min, self.sigma_max)
        s = np.diag(s)

        T = U @ s @ V_t

        auxiliary_variable = (T @ X_g).T
        return auxiliary_variable
