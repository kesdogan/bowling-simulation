from dataclasses import dataclass

import numpy as np

from utils import PDConstraint, Triangle


@dataclass
class Simplicial2DConstraint(PDConstraint):
    """This class represents a 2D simplicial constraint as described in the
    paper.

    Note that the current implementation uses A = B = I, which means that
    the distance is measured in the Euclidean metric. This leads to slower
    convergence, but is easier to implement.

    The calculation of the projection T is taken from Appendix A of the paper.
    """

    triangle: Triangle
    intial_positions: np.ndarray
    sigma_min: float
    sigma_max: float
    weight: float

    def get_global_system_matrix_contribution(self) -> np.ndarray:
        system_matrix = np.zeros(
            (len(self.intial_positions), len(self.intial_positions))
        )

        system_matrix[self.triangle.v0, self.triangle.v0] = self.weight
        system_matrix[self.triangle.v1, self.triangle.v1] = self.weight
        system_matrix[self.triangle.v2, self.triangle.v2] = self.weight

        return system_matrix

    def get_global_system_rhs_contribution(
        self, current_positions: np.ndarray
    ) -> np.ndarray:
        X_g = np.array(
            (
                self.intial_positions[self.triangle.v0]
                - self.intial_positions[self.triangle.v1],
                self.intial_positions[self.triangle.v0]
                - self.intial_positions[self.triangle.v2],
                (0, 0, 0),
            )
        )
        X_f = np.array(
            (
                current_positions[self.triangle.v0]
                - current_positions[self.triangle.v1],
                current_positions[self.triangle.v0]
                - current_positions[self.triangle.v2],
                (0, 0, 0),
            )
        )

        U, s, V_t = np.linalg.svd(X_f @ np.linalg.inv(X_g.T))
        s = np.clip(s, self.sigma_min, self.sigma_max)
        T = U @ s @ V_t

        rhs_contribution = np.zeros((len(current_positions), 3))

        rhs_contribution[self.triangle.v0] = self.weight * T
        rhs_contribution[self.triangle.v1] = -self.weight * T
        rhs_contribution[self.triangle.v2] = -self.weight * T

        return rhs_contribution
