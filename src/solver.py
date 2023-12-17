from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class PDConstraint(ABC):
    """This abstract class represents a general constraint for projective dynamics.
    The constraint is encoded using the mangling matrix A and the selection matrix S.

    Note that B from the paper is assumed to be the identity matrix."""

    A: torch.Tensor
    S: torch.Tensor
    weight: float
    gpu: bool

    @abstractmethod
    def _get_auxiliary_variable(self, current_positions: torch.Tensor) -> torch.Tensor:
        """Computes the auxiliary variable p for the constraint for the given
        current positions. This corresponds to the local step."""
        raise NotImplementedError

    def get_global_system_matrix_contribution(self) -> torch.Tensor:
        """Returns the contribution of the constraint to the global system matrix (the
        LHS)."""

        return self.weight * self.S.T @ self.A.T @ self.A @ self.S

    def get_global_system_rhs_contribution(
        self, current_positions: torch.Tensor
    ) -> torch.Tensor:
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
        initial_positions: torch.Tensor,
        initial_velocities: torch.Tensor,
        masses: torch.Tensor,
        external_forces: torch.Tensor,
        constraints: list[PDConstraint],
        step_size: float = 0.1,
        gpu=False,
    ):
        """Initializes the solver."""
        self.device = torch.device("cuda" if gpu else "cpu")
        self.n = len(initial_positions)

        assert tuple(initial_positions.shape) == (self.n, 3)
        assert tuple(initial_velocities.shape) == (self.n, 3)
        assert tuple(masses.shape) == (self.n,)
        assert tuple(external_forces.shape) == (self.n, 3)

        self.q = initial_positions.float().to(self.device)
        self.v = initial_velocities.float().to(self.device)
        self.f_ext = external_forces.float().to(self.device)

        self.M = torch.diag(masses).float().to(self.device)
        self.M_inv = torch.linalg.inv(self.M).float()

        self.constraints = constraints
        self.h = step_size

        self.global_system_matrix = self.M / self.h**2 + sum(
            c.get_global_system_matrix_contribution() for c in self.constraints
        )

    def perform_step(self, num_iterations_per_step: int):
        """Performs a single step of the projective dynamics solver."""
        s = self.q + self.h * self.v + self.h**2 * self.M_inv @ self.f_ext
        s = s.float()
        q_new = self.q.float()

        for _ in range(num_iterations_per_step):
            rhs = self.M @ s / self.h**2 + torch.stack(
                [
                    c.get_global_system_rhs_contribution(self.q)
                    for c in self.constraints
                ],
                dim=0,
            ).sum(dim=0)
            q_new = torch.linalg.solve(self.global_system_matrix, rhs)

        v_new = (q_new - self.q) / self.h

        self.q = q_new.float()
        self.v = v_new.float()

    def update_constraints(self, constraints: list[PDConstraint]):
        """Updates the constraints of the solver."""
        if self.constraints is constraints:
            print(len(self.constraints))
            return

        self.constraints = constraints
        self.global_system_matrix = self.M / self.h**2 + sum(
            c.get_global_system_matrix_contribution() for c in self.constraints
        )

        print(len(self.constraints))
