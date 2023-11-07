from abc import ABC, abstractmethod
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


class PDConstraint(ABC):
    """This abstract class represents a constraint for projective dynamics."""

    @abstractmethod
    def get_global_system_matrix_contribution(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_global_system_rhs_contribution(self) -> np.ndarray:
        raise NotImplementedError
