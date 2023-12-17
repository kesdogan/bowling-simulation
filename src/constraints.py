from dataclasses import dataclass, field

import torch

from solver import PDConstraint


@dataclass
class Simplicial2DConstraint(PDConstraint):
    """This class represents a 2D simplicial constraint as described in the
    paper.

    The calculation of the projection T is taken from Appendix A of the paper.
    """

    triangle_indices: torch.Tensor
    initial_positions: torch.Tensor

    sigma_min: float = 0.95
    sigma_max: float = 1.05

    A: torch.Tensor = field(init=False)
    S: torch.Tensor = field(init=False)
    X_g: torch.Tensor = field(init=False)
    X_g_inv: torch.Tensor = field(init=False)

    def __post_init__(self):
        device = "cuda" if self.gpu else "cpu"
        n = len(self.initial_positions)

        self.A = (
            torch.tensor(
                [
                    [1, 0, -1],
                    [0, 1, -1],
                ]
            )
            .float()
            .to(device)
        )

        self.S = torch.zeros((3, n)).float().to(device)
        self.S[0, self.triangle_indices[0]] = 1
        self.S[1, self.triangle_indices[1]] = 1
        self.S[2, self.triangle_indices[2]] = 1

        self.X_g = (self.A @ self.S @ self.initial_positions).T
        self.X_g_inv = torch.linalg.pinv(self.X_g)

        self.intersting = 29 in self.triangle_indices

    def _get_auxiliary_variable(self, current_positions: torch.Tensor) -> torch.Tensor:
        current_positions = current_positions.float()
        X_f = (self.A @ self.S @ current_positions).T

        U, s, V_t = torch.linalg.svd(X_f @ self.X_g_inv)

        s = torch.clip(s, self.sigma_min, self.sigma_max)
        s = torch.diag(s)

        T = U @ s @ V_t

        auxiliary_variable = (T @ self.X_g).T
        return auxiliary_variable


@dataclass
class CollisionConstraint(PDConstraint):
    """This class represents a constraint enforcing no collisions."""

    num_vertices: int
    penetrating_vertex_index: int
    projected_vertex_positions: torch.Tensor

    A: torch.Tensor = field(init=False)
    S: torch.Tensor = field(init=False)

    def __post_init__(self):
        device = "cuda" if self.gpu else "cpu"
        self.A = torch.tensor([[1]]).float().to(device)

        self.S = torch.zeros((1, self.num_vertices)).float().to(device)
        self.S[0, self.penetrating_vertex_index] = 1

    def _get_auxiliary_variable(self, current_positions: torch.Tensor) -> torch.Tensor:
        return self.projected_vertex_positions[None, :]
