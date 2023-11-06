from dataclasses import dataclass

import numpy as np


@dataclass
class Vertex:
    """This class represents a vertex in the mesh."""

    position: np.ndarray
    velocity: np.ndarray
    mass: float


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

    - x: A numpy array of shape (n, 3) containing the positions of the vertices
         in the mesh.

    - v: A numpy array of shape (n, 3) containing the velocities of the vertices
         in the mesh.

    - m: A numpy array of shape (n,) containing the masses of the vertices in
         the mesh.
    """

    def __init__(
        self,
        initial_triangles: list[Triangle],
    ):
        all_vertices = [v for t in initial_triangles for v in (t.v0, t.v1, t.v2)]

        self.n = len(all_vertices)
        self.x = np.array([v.position for v in all_vertices])
        self.v = np.array([v.velocity for v in all_vertices])
        self.m = np.array([v.mass for v in all_vertices])

    def perform_step(self, h: float):
        """Performs a single step of the projective dynamics solver.

        Args:
            h: The time step to use.
        """
        pass
