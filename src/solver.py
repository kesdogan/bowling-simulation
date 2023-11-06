from dataclasses import dataclass

import numpy as np


@dataclass
class Vertex:
    """This class represents a vertex in the mesh."""

    position: np.ndarray
    velocity: np.ndarray
    mass: float
    external_force: np.ndarray


@dataclass
class Triangle:
    """This class represents a triangle in the mesh."""

    v0: Vertex
    v1: Vertex
    v2: Vertex


class ProjectiveDynamicsSolver:
    """This class implements a projective dynamics solver as described in the
    paper "Projective Dynamics: Fusing Constraint Projections for Fast
    Simulation".

    This class has the following attributes:

    - q: A numpy array of shape (n, 3) containing the positions of the vertices
      in the mesh.

    - v: A numpy array of shape (n, 3) containing the velocities of the vertices
      in the mesh.

    - f_ext: A numpy array of shape (n, 3) containing the external forces acting
      on the vertices in the mesh.

    - M: A diagonal numpy array of shape (n, n) containing the masses of the
      vertices in the mesh.
    """

    def __init__(
        self, initial_triangles: list[Triangle], num_iterations: int, constraints: list
    ):
        all_vertices = [v for t in initial_triangles for v in (t.v0, t.v1, t.v2)]

        self.n = len(all_vertices)
        self.q = np.array([v.position for v in all_vertices])
        self.v = np.array([v.velocity for v in all_vertices])
        self.f_ext = np.array([v.external_force for v in all_vertices])

        self.M = np.diag([v.mass for v in all_vertices])
        self.M_inv = 1 / self.M

        self.num_iterations = num_iterations
        self.constraints = constraints

    def perform_step(self, h: float):
        """Performs a single step of the projective dynamics solver.

        Args:
            h: The time step to use.
        """
        s = self.q + h * self.v + h**2 * self.M_inv @ self.f_ext
        q_new = s

        for _ in range(self.num_iterations):
            for constraint in self.constraints:
                ...

            ...

        v_new = (q_new - self.q) / h

        self.q = q_new
        self.v = v_new
