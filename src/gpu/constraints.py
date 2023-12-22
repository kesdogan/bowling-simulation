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
class VolumeConstraint(PDConstraint):
    """This class represents a volume constraint as described in the
    paper.

    The calculation of the projection T is taken from Appendix A of
    the paper.
    """

    tetrahedron_indices: torch.Tensor
    initial_positions: torch.Tensor

    sigma_min: float = 0.95
    sigma_max: float = 1.05

    A: torch.Tensor = field(init=False)
    S: torch.Tensor = field(init=False)
    X_g: torch.Tensor = field(init=False)
    X_g_inv: torch.Tensor = field(init=False)

    eps: float = 0.000001

    def __post_init__(self):
        device = "cuda" if self.gpu else "cpu"

        self.A = (
            torch.tensor(
                [
                    [1, 0, 0, -1],
                    [0, 1, 0, -1],
                    [0, 0, 1, -1],
                ]
            )
            .float()
            .to(device)
        )

        n = len(self.initial_positions)
        self.S = torch.zeros((4, n)).to(device)
        self.S[0, self.tetrahedron_indices[0]] = 1
        self.S[1, self.tetrahedron_indices[1]] = 1
        self.S[2, self.tetrahedron_indices[2]] = 1
        self.S[3, self.tetrahedron_indices[3]] = 1

        self.X_g = (self.A @ self.S @ self.initial_positions).T
        self.X_g_inv = torch.linalg.pinv(self.X_g)

    def _get_auxiliary_variable(self, current_positions: torch.Tensor) -> torch.Tensor:
        X_f = (self.A @ self.S @ current_positions).T

        # perform SVD

        U, sigma, V_t = torch.linalg.svd(X_f @ self.X_g_inv)
        sigma = torch.diag(sigma)
        sigma_p = sigma

        # check if we even have to correct

        det_sigma_p = torch.det(sigma_p)

        if self.sigma_min <= det_sigma_p <= self.sigma_max:
            T = U @ sigma_p @ V_t
            auxiliary_variable = (T @ self.X_g).T
            return auxiliary_variable

        # perform iterative minimization

        D_k = torch.rand(3)

        for _ in range(10):
            sigma_p = sigma + torch.diag(D_k)
            det_sigma_p = torch.det(sigma_p)

            DC_D = det_sigma_p.repeat(3) / torch.clip(
                torch.diagonal(sigma_p), min=self.eps
            )
            C_D = torch.max(
                det_sigma_p - self.sigma_max, det_sigma_p - self.sigma_min
            ).item()

            D_k = (
                (DC_D.T @ D_k - C_D)
                / torch.clip(torch.dot(DC_D, DC_D), min=self.eps)
                * DC_D
            )

        # calculate projection T

        T = U @ sigma_p @ V_t

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
