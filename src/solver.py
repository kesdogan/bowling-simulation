from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.utils import Vertex


@dataclass
class PDConstraint(ABC):
    """This abstract class represents a general constraint for projective dynamics.
    The constraint is encoded using the mangling matrix A and the selection matrix S.

    Note that B from the paper is assumed to be the identity matrix."""

    weight: float
    A: np.ndarray
    S: np.ndarray

    @abstractmethod
    def _get_auxiliary_variable(self, current_positions: np.ndarray) -> np.ndarray:
        """Computes the auxiliary variable p for the constraint for the given
        current positions. This corresponds to the local step."""
        raise NotImplementedError

    def get_global_system_matrix_contribution(self) -> np.ndarray:
        """Returns the contribution of the constraint to the global system matrix (the
        LHS)."""
        return self.weight * self.S.T @ self.A.T @ self.A @ self.S

    def get_global_system_rhs_contribution(
        self, current_positions: np.ndarray
    ) -> np.ndarray:
        """Returns the contribution of the RHS constribution of the global system."""
        p = self._get_auxiliary_variable(current_positions)
        return self.weight * self.S.T @ self.A.T @ p


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

    def __init__(
        self,
        initial_vertices: list[Vertex],
        constraints: list[PDConstraint],
        step_size: float = 0.1,
    ):
        """Initializes the solver."""
        self.n = len(initial_vertices)
        self.q = np.array([v.position for v in initial_vertices])
        self.v = np.array([v.velocity for v in initial_vertices])
        self.f_ext = np.array([v.external_force for v in initial_vertices])

        self.M = np.diag([v.mass for v in initial_vertices])
        self.M_inv = np.linalg.inv(self.M)

        self.constraints = constraints
        self.h = step_size

        self.global_system_matrix = self.M / self.h**2 + sum(
            c.get_global_system_matrix_contribution() for c in self.constraints
        )

    def perform_step(self, num_iterations_per_step: int):
        """Performs a single step of the projective dynamics solver."""
        s = self.q + self.h * self.v + self.h**2 * self.M_inv @ self.f_ext
        q_new = self.q

        for _ in range(num_iterations_per_step):
            rhs = self.M @ s / self.h**2 + sum(
                c.get_global_system_rhs_contribution(self.q) for c in self.constraints
            )
            q_new = np.linalg.solve(self.global_system_matrix, rhs)

        v_new = (q_new - self.q) / self.h

        self.q = q_new
        self.v = v_new
