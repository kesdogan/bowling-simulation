from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

@dataclass
class PDConstraint(ABC):
    """This abstract class represents a general constraint for projective dynamics.
    The constraint is encoded using the mangling matrix A and the selection matrix S.

    Note that B from the paper is assumed to be the identity matrix."""

    A: np.ndarray
    S: np.ndarray
    weight: float

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

    def get_x_g_inv(self):
        return self.X_g_inv
    
    def _get_auxiliary_variable(self, current_positions: np.ndarray) -> np.ndarray:
        
        X_f = (self.A @ self.S @ current_positions).T

        U, s, V_t = np.linalg.svd(X_f @ self.X_g_inv)

        s = np.clip(s, self.sigma_min, self.sigma_max)
        s = np.diag(s)

        T = U @ s @ V_t

        auxiliary_variable = (T @ self.X_g).T
        return auxiliary_variable
    
    # def get_global_system_rhs_contribution(
    #     self, current_positions: np.ndarray
    # ) -> np.ndarray:
    #     """Returns the contribution of the RHS constribution of the global system."""
    #     return self.weight * self.S.T @ self.A.T @ current_positions


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
        initial_positions: np.ndarray,
        initial_velocities: np.ndarray,
        masses: np.ndarray,
        external_forces: np.ndarray,
        constraints: list[PDConstraint],
        faces: np.ndarray,
        step_size: float = 0.1,
    ):
        """Initializes the solver."""
        self.n = len(initial_positions)

        assert initial_positions.shape == (self.n, 3)
        assert initial_velocities.shape == (self.n, 3)
        assert masses.shape == (self.n,)
        assert external_forces.shape == (self.n, 3)

        self.q = initial_positions
        self.v = initial_velocities
        self.f_ext = external_forces

        self.constraints_mask = np.zeros((self.n, 3))
        self.faces = faces

        self.M = np.diag(masses)
        self.M_inv = np.linalg.inv(self.M)

        self.constraints = constraints
        self.constra_inv = None
        self.h = step_size

        self.A = np.array(
            [
                [1, 0, -1],
                [0, 1, -1],
            ]
        )

        self.global_system_matrix = self.M / self.h**2 + sum(
            c.get_global_system_matrix_contribution() for c in self.constraints
        )

    def inverse_2d_constraints(self):
        constra = np.array(self.constraints)
        self.constra_inv = np.array([c.get_x_g_inv() for c in constra[:len(self.faces)]])

    def perform_step(self, num_iterations_per_step: int):
        """Performs a single step of the projective dynamics solver."""
        s = self.q + self.h * self.v + self.h**2 * self.M_inv @ self.f_ext
        q_new = self.q

        for _ in range(num_iterations_per_step):
            rhs = self.M @ s / self.h**2

            rhs_contributions = []
            print(self.q[self.faces].shape)
            intermediate = np.matmul(self.A, self.q[self.faces], axes=[(-2, -1), (-2, -1), (-2, -1)])
            intermediate = np.transpose(intermediate, axes=[0, 2, 1])
            advanced = np.matmul(intermediate, self.constra_inv)
            U, ss, V = np.linalg.svd(advanced)
            ss = np.clip(ss, 0.95, 1.05)
            sss = np.zeros((len(ss), 3, 3))
            sss[:, 0, 0] = ss[:, 0]
            sss[:, 1, 1] = ss[:, 1]
            sss[:, 2, 2] = ss[:, 2]
            T = np.matmul(np.matmul(U, sss), V)
            rr = np.matmul(T, intermediate)
            rr = np.transpose(rr, axes=[0, 2, 1])

            for i, c in enumerate(self.constraints[:len(self.faces)]):
                rhs_contributions.append(c.get_global_system_rhs_contribution(self.q))

            for c in self.constraints[len(self.faces):]:
                tba = c.get_global_system_rhs_contribution(self.q)
                rhs_contributions.append(tba)
                
            rhs += sum(rhs_contributions)

            q_new = np.linalg.solve(self.global_system_matrix, rhs)

        v_new = (q_new - self.q) / self.h

        self.q = q_new
        self.v = v_new

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
