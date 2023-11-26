import copy

import igl
import numpy as np

from utils import Collision


def collision_detecter(object_list, vertices, faces, q):
    """Detects collisions between objects in the scene."""

    # create object masks:
    object_mask = []
    vertex_counter = face_counter = 0
    object_offsets = []

    for object in object_list:
        vertex_mask = np.zeros(len(vertices), dtype=bool)
        face_mask = np.zeros(len(faces), dtype=bool)

        vertex_mask[vertex_counter : vertex_counter + len(object.v)] = True
        face_mask[face_counter : face_counter + len(object.f)] = True
        object_offsets.append((vertex_counter, face_counter))
        vertex_counter += len(object.v)
        face_counter += len(object.f)

        object_mask.append((vertex_mask, face_mask))

    # update to newest positions
    vertices = q
    for i, object in enumerate(object_list):
        object.v = vertices[object_mask[i][0]]

    for i, object in enumerate(object_list):
        vertices_i = vertices[~object_mask[i][0]]
        faces_i = copy.deepcopy(faces)
        faces_i[object_offsets[i][1] :] = faces_i[object_offsets[i][1] :] - len(
            object.v
        )
        faces_i = faces_i[~object_mask[i][1]]

        S, indices_face, C, N = igl.signed_distance(
            object.v, vertices_i, faces_i, return_normals=True
        )
        collision_mask = S < 0

        if np.any(collision_mask):
            print("COLLISION DETECTED")

            collision_indices = np.arange(len(object.v))[collision_mask]
            C = C[collision_mask]
            N = N[collision_mask]
            S = S[collision_mask]
            indices_face = indices_face[collision_mask]

            order = np.argsort(S)

            collision_indices = collision_indices[order]
            C = C[order]
            N = N[order]
            indices_face = indices_face[order]

            I_mask = indices_face >= object_offsets[i][1]
            indices_face[I_mask] = indices_face[I_mask] + object.f.shape[0]

            return Collision(
                penetrating_vertex=collision_indices + object_offsets[i][0],
                penetrated_face=indices_face,
                projection_of_point_on_face=C,
                normal=N,
            )
    return None
