import numpy as np
from pygel3d import hmesh
from medial_axis_processing.medial_axis import MedialAxis
from scipy.spatial.transform import Rotation as R


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


def __compute_tangents(curve):
    # Approximate tangent vectors by finite differences
    tangents = np.gradient(curve, axis=0)
    # Normalize tangent vectors
    tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]
    return tangents


def __inverse_apply_sheet(ma: MedialAxis, updated_sheet_positions: np.ndarray):
    """Applies the transformation of the medial sheet onto corresponding inner and outer points.
       Does this by projecting the points to the local basis of the medial sheet vertex, and then updating the point"""
    positions = ma.sheet.positions()

    for vid in ma.sheet.vertices():
        v1 = ma.sheet.circulate_vertex(vid)[0]
        v2 = ma.sheet.circulate_vertex(vid)[1]

        for p in ma.correspondences[vid]:
            basis_old = __get_local_basis(positions[vid], positions[v1], positions[v2])
            local_coords = __project_point_to_basis(p.pos, positions[vid], basis_old)

            basis_new = __get_local_basis(updated_sheet_positions[vid], updated_sheet_positions[v1], updated_sheet_positions[v2])
            new_pos = __update_point(updated_sheet_positions[vid], local_coords, basis_new)

            p.pos = new_pos

    ma.sheet.positions()[:] = updated_sheet_positions
    new_inner_positions = updated_sheet_positions[ma.inner_indices[ma.sheet_indices]]
    ma.inner_points.positions[ma.sheet_indices] = new_inner_positions


def parallel_transport(old_curve_pos: np.ndarray, new_curve_pos: np.ndarray, outer_points: np.ndarray):
    """Given the original curve point positions, their updated positions, and the corresponding outer points,
        returns the parallel transport of the outer points along the curve to the updated positions"""
    orig_tangents = __compute_tangents(old_curve_pos)
    new_tangents = __compute_tangents(new_curve_pos)
    new_outer_points = np.copy(outer_points)

    for i in range(len(old_curve_pos)):
        axis = np.cross(orig_tangents[i], new_tangents[i])
        axis_length = np.linalg.norm(axis)
        if axis_length > 1e-6:  # To avoid division by zero in case of parallel vectors
            axis /= axis_length
            angle = np.arccos(np.clip(np.dot(orig_tangents[i], new_tangents[i]), -1.0, 1.0))
            rotation = R.from_rotvec(axis * angle)
            new_outer_points[i] = rotation.apply(outer_points[i] - old_curve_pos[i]) + new_curve_pos[i]
        else:
            new_outer_points[i] += new_curve_pos[i] - old_curve_pos[i]

    return new_outer_points


def apply_inverse_medial_axis_transform(
        original_mesh: hmesh.Manifold,
        medial_axis: MedialAxis,
        updated_sheet_positions: np.ndarray,
        updated_curve_positions: list[np.ndarray]
):
    __inverse_apply_sheet(medial_axis, updated_sheet_positions)
    original_mesh.positions()[medial_axis.sheet_indices] = medial_axis.outer_points.positions[medial_axis.sheet_indices]

    for i, curve in enumerate(medial_axis.curves):
        curve_indices = [q.index for q in curve]
        new_outer_points = parallel_transport(
            medial_axis.inner_points.positions[curve_indices],
            updated_curve_positions[i],
            medial_axis.outer_points.positions[curve_indices]
        )
        medial_axis.inner_points.positions[curve_indices] = updated_curve_positions[i]
        original_mesh.positions()[curve_indices] = new_outer_points


