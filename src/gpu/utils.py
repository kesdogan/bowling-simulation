from dataclasses import dataclass

import torch


@dataclass
class Vertex:
    """This class represents a vertex in the mesh."""

    position: torch.Tensor
    velocity: torch.Tensor
    mass: float
    external_force: torch.Tensor


@dataclass
class Triangle:
    """This class represents a triangle by the indices of its vertices."""

    v0: int
    v1: int
    v2: int


@dataclass
class Object:
    v: torch.Tensor
    f: torch.Tensor


@dataclass
class Collision:
    """This class represents a collision between two objects."""

    # penetrating vertex, index (global)

    # pemetrated face, iondex (global)

    # projection of that point on face

    # normal
    penetrating_vertex: torch.Tensor = torch.tensor([])
    penetrated_face: torch.Tensor = torch.tensor([])
    projection_of_point_on_face: torch.Tensor = torch.tensor([[]])
