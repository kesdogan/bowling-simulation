import copy

import igl
import torch

from utils import Collision

# Parameters	V #V by 3 Matrix of mesh vertex 3D positions
#               F #F by 3 Matrix of face (triangle) indices

# Returns	    N #F by 3 Matrix of mesh face (triangle) 3D normals


def per_face_normals(vertices, faces):
    face_normals = torch.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    face_normals = face_normals / torch.norm(face_normals, dim=1).unsqueeze(1)
    return face_normals


### Implemention of https://libigl.github.io/tutorial/#signed-distance
# Parameters	vertices #V by 3 tensor of mesh vertex 3D positions
#               faces #F by 3 tensor of face (triangle) indices
#               points #P by 3 tensor of query point positions
#               return_libigl: if True, return the results from libigl
#               extended_vertices: if True, add the center of each edge to the list of vertices (makes for even more accurate results)
#               winding_number: if True, use the winding number to identify the sign of the distance (slower, but most accurate)
def signed_distance(
    points, vertices, faces, return_libigl=False, extended_vertices=True
):
    # in case we want the libigl results, they will be returned here
    if return_libigl:
        true_S, true_I, true_C = igl.signed_distance(
            points.cpu().numpy(), vertices.cpu().numpy(), faces.cpu().numpy()
        )
        return true_S, true_I, true_C

    # we can expand the number of vertices by adding the center of each edge to the list of vertices
    # this leads to a better approximation of the closest point on the surface of the mesh
    if extended_vertices:
        vertices = torch.cat(
            (
                vertices,
                (vertices[faces[:, 0]] + vertices[faces[:, 2]]) / 2,
                (vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 2,
            ),
            dim=0,
        )

    # in order to identify the closest face to each point,
    # we compute the distance from each point to the center of each face
    # this works quite well in practices and returns us the face index min_fidx
    average_per_face = torch.sum(vertices[faces], dim=1) / 3
    face_dist = torch.cdist(points, average_per_face)
    _, min_fidx = torch.min(face_dist, dim=1)
    del face_dist

    # compute the sign of the distance
    normals = per_face_normals(vertices, faces)
    need_normals = normals[min_fidx]
    product = torch.sum(need_normals * (points - average_per_face[min_fidx]), dim=1)
    sign = torch.sign(product)

    # compute the distance of each point to each vertex (or extended set of vertices)
    dist = torch.cdist(points, vertices)
    min_dist, min_index = torch.min(dist, dim=1)
    del dist

    # default: assign each point to the closest vertex
    p00 = vertices[min_index]

    # implementation of these instructions:
    # https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
    # for each face, we compute whether the closest point on the mesh lies within the triangle
    # or on one of the edges of the triangle
    # we then identify the closest point there
    B = vertices[faces[min_fidx, 0]]
    E_0 = vertices[faces[min_fidx, 1]] - vertices[faces[min_fidx, 0]]
    E_1 = vertices[faces[min_fidx, 2]] - vertices[faces[min_fidx, 0]]
    a = torch.sum(E_0 * E_0, dim=1)
    b = torch.sum(E_0 * E_1, dim=1)
    c = torch.sum(E_1 * E_1, dim=1)
    d = torch.sum(E_0 * (vertices[faces[min_fidx, 0]] - points), dim=1)
    e = torch.sum(E_1 * (vertices[faces[min_fidx, 0]] - points), dim=1)
    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    reg0 = (s > 0) * (t > 0) * (s + t < det)  # inside the triangle
    reg1 = (s > 0) * (t > 0) * (s + t > det)  # outside of the top edge
    reg3 = (s < 0) * (t > 0) * (s + t < det)  # outside of the left edge
    reg5 = (s > 0) * (t < 0) * (s + t < det)  # outside of the bottom edge

    # region0
    s[reg0] = s[reg0] / det[reg0]
    t[reg0] = t[reg0] / det[reg0]

    # region1
    numer = (c + e) - (b + d)
    denom = a - 2 * b + c
    check_numer = numer <= 0
    o_check_numer = numer > 0
    check_denom_numer = numer >= denom
    o_check_denom_numer = numer < denom
    s[reg1 * check_numer] = 0
    s[reg1 * check_denom_numer * o_check_numer] = 1
    s[reg1 * o_check_denom_numer * o_check_numer] = (
        numer[reg1 * o_check_denom_numer * o_check_numer]
        / denom[reg1 * o_check_denom_numer * o_check_numer]
    )
    t[reg1] = 1 - s[reg1]

    # region3
    check_e = e >= 0
    o_check_e = e < 0
    check_e_c = -e >= c
    o_check_e_c = -e < c
    s[reg3] = 0
    t[reg3 * check_e] = 0
    t[reg3 * o_check_e * check_e_c] = 1
    t[reg3 * o_check_e * o_check_e_c] = (
        -e[reg3 * o_check_e * o_check_e_c] / c[reg3 * o_check_e * o_check_e_c]
    )

    # region5
    check_d = d >= 0
    o_check_d = d < 0
    check_d_a = -d >= a
    o_check_d_a = -d < a
    t[reg5] = 0
    s[reg5 * check_d] = 0
    s[reg5 * o_check_d * check_d_a] = 1
    s[reg5 * o_check_d * o_check_d_a] = (
        -d[reg5 * o_check_d * o_check_d_a] / a[reg5 * o_check_d * o_check_d_a]
    )

    # correct closest point for those lying on the edges and inside the triangle
    p00[reg0 + reg1 + reg3 + reg5] = (
        B[reg0 + reg1 + reg3 + reg5]
        + s[reg0 + reg1 + reg3 + reg5, None] * E_0[reg0 + reg1 + reg3 + reg5]
        + t[reg0 + reg1 + reg3 + reg5, None] * E_1[reg0 + reg1 + reg3 + reg5]
    )
    min_dist[reg0 + reg1 + reg3 + reg5] = torch.norm(
        p00[reg0 + reg1 + reg3 + reg5] - points[reg0 + reg1 + reg3 + reg5], dim=1
    )

    return sign * min_dist, min_fidx, p00


def collision_detecter(object_list, vertices, faces, q, gpu=False):
    """Detects collisions between objects in the scene."""

    device = torch.device("cuda" if gpu else "cpu")

    # create object masks:
    object_mask = []
    vertex_counter = face_counter = 0
    object_offsets = []

    for object in object_list:
        vertex_mask = torch.zeros(len(vertices), dtype=bool)
        face_mask = torch.zeros(len(faces), dtype=bool)

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

    collisions = []

    for i, object in enumerate(object_list):
        vertices_i = vertices[~object_mask[i][0]]
        faces_i = copy.deepcopy(faces)
        faces_i[object_offsets[i][1] :] = faces_i[object_offsets[i][1] :] - len(
            object.v
        )
        faces_i = faces_i[~object_mask[i][1]]

        S, indices_face, C = signed_distance(
            object.v, vertices_i, faces_i, return_libigl=not gpu
        )
        if not gpu:
            S = torch.from_numpy(S)
            indices_face = torch.from_numpy(indices_face)
            C = torch.from_numpy(C)

        collision_mask = S < 0
        collision_mask = collision_mask.to(device)

        if torch.any(collision_mask):
            print("COLLISION DETECTED")

            collision_indices = torch.arange(len(object.v)).to(device)[collision_mask]
            C = C[collision_mask]
            S = S[collision_mask]
            indices_face = indices_face[collision_mask]

            order = torch.argsort(S)

            collision_indices = collision_indices[order]
            C = C[order]
            indices_face = indices_face[order]

            I_mask = indices_face >= object_offsets[i][1]
            indices_face[I_mask] = indices_face[I_mask] + object.f.shape[0]

            collisions.append(
                Collision(
                    penetrating_vertex=collision_indices + object_offsets[i][0],
                    penetrated_face=indices_face,
                    projection_of_point_on_face=C,
                )
            )
        else:
            collisions.append(Collision())

    return collisions
