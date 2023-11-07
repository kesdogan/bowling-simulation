from dataclasses import dataclass

import numpy as np

from src.utils import PDConstraint, Vertex


@dataclass
class GeneralPDConstraint(PDConstraint):
    """This class represents a general constraint for projective dynamics."""

    w: float
    A: np.ndarray
    S: np.ndarray
    B: np.ndarray

    def get_global_system_matrix_contribution(self) -> np.ndarray:
        return self.w * self.S.T @ self.A.T @ self.A @ self.S

    def get_global_system_rhs_contribution(self) -> np.ndarray:
        p = np.array([])
        # TODO
        return self.w * self.S.T @ self.A.T @ self.B @ p


class ProjectiveDynamicsSolver:
    """This class implements a projective dynamics solver as described in the
    paper "Projective Dynamics: Fusing Constraint Projections for Fast
    Simulation".

    This implementation of the solver is designed to work with triangular meshes.

    This class has the following main attributes:

    - n: The number of vertices in the mesh.

    - q: A numpy array of shape (n, 3) containing the positions of the vertices
      in the mesh.

    - v: A numpy array of shape (n, 3) containing the velocities of the vertices
      in the mesh.

    - f_ext: A numpy array of shape (n, 3) containing the external forces acting
      on the vertices in the mesh.

    - M: A diagonal numpy array of shape (n, n) containing the masses of the
      vertices in the mesh.

    - h: The step size of the solver.
    """

    def _calculate_global_system_matrix(self) -> np.ndarray:
        """Calculates the system matrix of the global optimization problem.
        This matrix is constant throughout the simulation."""
        system_matrix = self.M / self.h**2

        for c in self.constraints:
            system_matrix += c.get_global_system_matrix_contribution()

        return system_matrix

    def __init__(
        self,
        initial_vertices: list[Vertex],
        constraints: list[PDConstraint],
        step_size: float,
    ):
        """Initializes the solver."""
        self.n = len(initial_vertices)
        self.q = np.array(v.position for v in initial_vertices)
        self.v = np.array(v.velocity for v in initial_vertices)
        self.f_ext = np.array(v.external_force for v in initial_vertices)

        self.M = np.diag([v.mass for v in initial_vertices])
        self.M_inv = 1 / self.M

        self.constraints = constraints
        self.h = step_size

        self.global_system_matrix = self._calculate_global_system_matrix()

    def perform_step(self, num_iterations_per_step: int):
        """Performs a single step of the projective dynamics solver."""
        s = self.q + self.h * self.v + self.h**2 * self.M_inv @ self.f_ext
        q_new = np.copy(s)

        for _ in range(num_iterations_per_step):
            rhs = self.M @ s / self.h**2

            for c in self.constraints:
                rhs += c.get_global_system_rhs_contribution()

            q_new = np.linalg.solve(self.global_system_matrix, rhs)

        v_new = (q_new - self.q) / self.h

        self.q = q_new
        self.v = v_new
