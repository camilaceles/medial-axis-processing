import igl
import numpy as np
from pygel3d import hmesh
from medial_axis_processing.medial_axis import MedialAxis


def __get_local_basis(v0, v1, v2):
    # compute local basis at v0 given 2 adjacent vertices v1 and v2
    b0 = v1 - v0
    b0 /= np.linalg.norm(b0)

    b2 = np.cross(b0, v2 - v0)
    b2 /= np.linalg.norm(b2)

    b1 = np.cross(b2, b0)
    return b0, b1, b2


def __project_point_to_basis(point, vertex, basis):
    # get point coordinates in local basis
    v_p = point - vertex
    return np.dot(v_p, basis[0]), np.dot(v_p, basis[1]), np.dot(v_p, basis[2])


def __update_point(vertex_new, local_coords, new_basis):
    # retrieve global coordinates from local basis coordinates
    return vertex_new + local_coords[0] * new_basis[0] + local_coords[1] * new_basis[1] + local_coords[2] * new_basis[2]


def __least_squares_conformal_map(m: hmesh.Manifold) -> np.ndarray:
    """Applies the igl's LSCM to the given mesh"""
    vertices = m.positions()
    faces = np.array([m.circulate_face(fid) for fid in m.faces()])

    # Fix two points on the boundary
    b = np.array([2, 1])

    bnd = igl.boundary_loop(faces)
    b[0] = bnd[0]
    b[1] = bnd[int(bnd.size / 2)]

    bc = np.array([[0.0, 0.0], [1.0, 0.0]])

    _, uv = igl.lscm(vertices, faces, b, bc)
    return uv


def __get_unfolded_positions(ma: MedialAxis) -> np.ndarray:
    """Returns the 3d coordinates for the unfolded medial axis using LSCM"""
    ma_areas = np.array([ma.mesh.area(fid) for fid in ma.mesh.faces()])
    ma_area = np.sum(ma_areas)

    uv = __least_squares_conformal_map(ma.mesh)
    uv = np.c_[uv, np.zeros(uv.shape[0])]

    uv_mesh = hmesh.Manifold(ma.mesh)
    uv_mesh.positions()[:] = uv
    uv_areas = np.array([uv_mesh.area(fid) for fid in ma.mesh.faces()])
    uv_area = np.sum(uv_areas)

    # scale uv mapping to approximate original MA area
    return uv * np.sqrt(ma_area / uv_area)


def unfold_medial_axis(ma: MedialAxis):
    uv = __get_unfolded_positions(ma)
    new_inner_points = ma.get_updated_inner_points(uv)
    face_ids = ma.inner_barycentrics[ma.sheet_indices, 0].astype(int)
    positions = ma.mesh.positions()

    new_outer_points = []

    for p_pos, q_pos, new_q, fid in zip(
            ma.outer_points.positions[ma.sheet_indices],
            ma.inner_points.positions[ma.sheet_indices],
            new_inner_points,
            face_ids
    ):
        face_vertices = ma.mesh.circulate_face(fid)

        # find two face vertices that aren't the point itself
        v1, v2 = None, None
        if np.allclose(q_pos, positions[face_vertices[0]]):
            v1, v2 = face_vertices[1], face_vertices[2]
        elif np.allclose(q_pos, positions[face_vertices[1]]):
            v1, v2 = face_vertices[0], face_vertices[2]
        else:
            v1, v2 = face_vertices[0], face_vertices[1]

        basis_old = __get_local_basis(q_pos, positions[v1], positions[v2])
        local_coords = __project_point_to_basis(p_pos, q_pos, basis_old)

        basis_new = __get_local_basis(new_q, uv[v1], uv[v2])
        new_pos = __update_point(new_q, local_coords, basis_new)
        new_outer_points.append(new_pos)

    ma.mesh.positions()[:] = uv
    ma.inner_points.positions = new_inner_points
    ma.outer_points.positions[ma.sheet_indices] = np.array(new_outer_points)
