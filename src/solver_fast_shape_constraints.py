import numpy as np

from src.constraints import Simplicial2DConstraint
from src.solver import PDConstraint, ProjectiveDynamicsSolver


class ProjectiveDynamicsSolver(ProjectiveDynamicsSolver):
    """This class adapts the ProjectiveDynamicsSolver to process
    Simplicial2DConstraints faster by batching the caclulation of the
    auxiliary variable p (mainly the SVD decomposition).

    It can be used as a drop-in replacement for the ProjectiveDynamicsSolver."""

    def __init__(
        self,
        initial_positions: np.ndarray,
        initial_velocities: np.ndarray,
        masses: np.ndarray,
        external_forces: np.ndarray,
        constraints: list[PDConstraint],
        step_size: float = 0.1,
    ):
        super().__init__(
            initial_positions,
            initial_velocities,
            masses,
            external_forces,
            constraints,
            step_size,
        )

        shape_constraints = [
            c for c in self.constraints if isinstance(c, Simplicial2DConstraint)
        ]

        assert len(shape_constraints), "No shape constraints given."

        self.shape_A = np.array(
            [
                [1, 0, -1],
                [0, 1, -1],
            ]
        )
        self.shape_constra = np.array([c.X_g for c in shape_constraints])
        self.shape_constra_inv = np.array([c.X_g_inv for c in shape_constraints])
        self.shape_indices = np.array([c.triangle_indices for c in shape_constraints])

    def perform_step(self, num_iterations_per_step: int):
        """Performs a single step of the projective dynamics solver."""
        s = self.q + self.h * self.v + self.h**2 * self.M_inv @ self.f_ext
        q_new = self.q

        for _ in range(num_iterations_per_step):
            rhs = self.M @ s / self.h**2

            # calculate rhs contributions from shape constraints by
            # batching the calculation of the auxiliary variable p

            rhs_contributions = []
            intermediate = np.matmul(
                self.shape_A,
                self.q[self.shape_indices],
                axes=[(-2, -1), (-2, -1), (-2, -1)],
            )
            intermediate = np.transpose(intermediate, axes=[0, 2, 1])
            advanced = np.matmul(intermediate, self.shape_constra_inv)
            U, ss, V = np.linalg.svd(advanced)
            ss = np.clip(ss, 0.95, 1.05)
            sss = np.zeros((len(ss), 3, 3))
            sss[:, 0, 0] = ss[:, 0]
            sss[:, 1, 1] = ss[:, 1]
            sss[:, 2, 2] = ss[:, 2]
            T = np.matmul(np.matmul(U, sss), V)
            rr = np.matmul(T, self.shape_constra)

            rr = np.transpose(rr, axes=[0, 2, 1])

            i = 0
            for c in self.constraints:
                if isinstance(c, Simplicial2DConstraint):
                    rhs_contributions.append(c.weight * c.S.T @ c.A.T @ rr[i])
                    i += 1
                else:
                    rhs_contributions.append(
                        c.get_global_system_rhs_contribution(self.q)
                    )

            rhs += sum(rhs_contributions)

            q_new = np.linalg.solve(self.global_system_matrix, rhs)

        v_new = (q_new - self.q) / self.h

        self.q = q_new
        self.v = v_new
