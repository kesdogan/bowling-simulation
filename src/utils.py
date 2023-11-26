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
    """This class represents a triangle by the indices of its vertices."""

    v0: int
    v1: int
    v2: int


@dataclass
class Object:
    v: np.ndarray
    f: np.ndarray


@dataclass
class Collision:
    """This class represents a collision between two objects."""

    # penetrating vertex, index (global)

    # pemetrated face, iondex (global)

    # projection of that point on face

    # normal
    penetrating_vertex: np.array
    penetrated_face: np.array
    projection_of_point_on_face: np.array
    normal: np.array
