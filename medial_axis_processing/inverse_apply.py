import numpy as np
from pygel3d import hmesh, jupyter_display as jd
from trimesh.triangles import barycentric_to_points
from commons.medial_axis import MedialAxis
from scipy.spatial.transform import Rotation as R
from commons.utils import flatten, get_local_basis
import plotly.graph_objects as go


def parallel_transport_curve_framing(curve_positions, initial_normal):
    # implements parallel transport as in "Parallel Transport for Curve Framing" 1995
    n_points = len(curve_positions)

    tangents = np.zeros((n_points, 3))
    normals = np.zeros((n_points, 3))
    binormals = np.zeros((n_points, 3))

    tangents[:-1] = np.diff(curve_positions, axis=0)
    tangents[-1] = curve_positions[-1] - curve_positions[-2]
    tangents = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]

    normals[0] = initial_normal / np.linalg.norm(initial_normal)
    binormals[0] = np.cross(tangents[0], normals[0])
    binormals[0] = binormals[0] / np.linalg.norm(binormals[0])

    for i in range(1, n_points):
        u = tangents[i - 1]
        v = tangents[i]
        cos_theta = np.dot(u, v)
        sin_theta = np.linalg.norm(np.cross(u, v))

        if sin_theta > 1e-6:  # Avoid numerical instability
            axis = np.cross(u, v)
            axis = axis / np.linalg.norm(axis)
            theta = np.arctan2(sin_theta, cos_theta)

            # Compute the rotation matrix
            rotation = R.from_rotvec(axis * theta).as_matrix()

            # Rotate the normal and binormal
            normals[i] = rotation @ normals[i - 1]
            binormals[i] = np.cross(tangents[i], normals[i])
            binormals[i] = binormals[i] / np.linalg.norm(binormals[i])
        else:
            normals[i] = normals[i - 1]
            binormals[i] = binormals[i - 1]

    frames = np.zeros((n_points, 3, 3))
    frames[:, 0, :] = tangents
    frames[:, 1, :] = normals
    frames[:, 2, :] = binormals

    return frames


def __get_local_basis(v0, v1, n):
    b0 = v1 - v0
    b0 /= np.linalg.norm(b0)
    b1 = np.cross(n, b0)
    return b0, b1, n


def __project_point_to_basis(point, vertex, basis):
    # get point coordinates in local basis
    v_p = point - vertex
    return np.dot(v_p, basis[0]), np.dot(v_p, basis[1]), np.dot(v_p, basis[2])


def __update_point(vertex_new, local_coords, new_basis):
    # retrieve global coordinates from local basis coordinates
    return vertex_new + local_coords[0] * new_basis[0] + local_coords[1] * new_basis[1] + local_coords[2] * new_basis[2]


def __project_array_to_basis(v_p, basis):
    # Projects array onto the given basis vectors
    return np.dot(v_p, basis[0]), np.dot(v_p, basis[1]), np.dot(v_p, basis[2])


def __update_array(local_coords, basis):
    # Reconstructs the array from the local coordinates in the new basis
    return local_coords[0] * basis[0] + local_coords[1] * basis[1] + local_coords[2] * basis[2]


def __updated_inner_projections_sheet(medial_axis: MedialAxis, updated_sheet_positions: np.ndarray):
    new_inner_proj = np.copy(medial_axis.inner_projections)
    for outer in range(len(medial_axis.outer_points)):
        closest = medial_axis.inner_barycentrics[outer, 0].astype(int)
        closest_triangle = medial_axis.sheet.circulate_face(closest, mode='v')
        new_pos_triangle = updated_sheet_positions[closest_triangle]

        barycentrics = medial_axis.inner_barycentrics[outer, 1:]
        new_pos = barycentric_to_points(np.array([new_pos_triangle]), np.array([barycentrics]))
        new_inner_proj[outer] = new_pos
    return new_inner_proj


def inverse_apply_sheet(ma: MedialAxis, updated_sheet_positions: np.ndarray):
    """Applies the transformation of the medial sheet onto corresponding inner and outer points.
       Does this by projecting the points to the local basis of the medial sheet vertex, and then updating the point"""
    positions = ma.sheet.positions()

    updated_sheet = hmesh.Manifold(ma.sheet)
    updated_sheet.positions()[:] = updated_sheet_positions

    new_inner_pos = np.copy(ma.inner_points)
    new_outer_pos = np.copy(ma.outer_points)

    for vid in ma.sheet.vertices():
        neighbor = ma.sheet.circulate_vertex(vid)[0]

        n_old = ma.sheet.vertex_normal(vid)
        basis_old = __get_local_basis(positions[vid], positions[neighbor], n_old)

        n_new = updated_sheet.vertex_normal(vid)
        basis_new = __get_local_basis(updated_sheet_positions[vid], updated_sheet_positions[neighbor], n_new)

        corresponding_outer = ma.sheet_correspondences[vid]

        for outer_idx in corresponding_outer:
            outer_point = ma.outer_points[outer_idx]

            local_coords = __project_point_to_basis(outer_point, positions[vid], basis_old)
            new_pos = __update_point(updated_sheet_positions[vid], local_coords, basis_new)
            new_outer_pos[outer_idx] = new_pos

    # update medial axis object with new positions
    ma.sheet.positions()[:] = updated_sheet_positions
    ma.outer_points = new_outer_pos
    ma.inner_points = new_inner_pos

    ma.inner_points[~ma.curve_indices] = updated_sheet_positions[ma.inner_to_sheet_index[~ma.curve_indices]]
    ma.graph.positions()[~ma.curve_indices] = updated_sheet_positions[ma.inner_to_sheet_index[~ma.curve_indices]]

    ma.surface.positions()[:] = ma.outer_points
    ma.inner_projections[:] = __updated_inner_projections_sheet(ma, updated_sheet_positions)


def inverse_apply_sheet_v2(ma: MedialAxis, updated_sheet_positions: np.ndarray):
    """Applies the transformation of the medial sheet onto corresponding inner and outer points.
       Does this by projecting the connections to the surface to a local basis on the sheet vertex,
         and then updating the connection accordingly"""
    positions = ma.sheet.positions()

    updated_sheet = hmesh.Manifold(ma.sheet)
    updated_sheet.positions()[:] = updated_sheet_positions
    updated_inner_projs = __updated_inner_projections_sheet(ma, updated_sheet_positions)

    new_outer_pos = np.copy(ma.outer_points)

    for fid in ma.sheet.faces():
        vertices = ma.sheet.circulate_face(fid, mode='v')
        updated_sheet_vertices = updated_sheet.circulate_face(fid, mode='v')

        n_old = ma.sheet.face_normal(fid)
        basis_old = get_local_basis(positions[vertices[0]], positions[vertices[1]], n_old)

        n_new = updated_sheet.face_normal(fid)
        basis_new = get_local_basis(updated_sheet_positions[updated_sheet_vertices[0]],
                                      updated_sheet_positions[updated_sheet_vertices[1]], n_new)

        corresponding_outer = np.where(ma.inner_barycentrics[:, 0].astype(int) == fid)[0]

        for outer_idx in corresponding_outer:

            is_at_vertex = np.any(np.abs(ma.inner_barycentrics[outer_idx][1:] - 1) < 0.3)
            vid = ma.sheet.vertices()[np.argmax(ma.inner_barycentrics[outer_idx][1:])]

            # Improvement: also smooth frame for projections at edges
            if is_at_vertex and not ma.sheet.is_vertex_at_boundary(vid):
                n_old_v = ma.sheet.vertex_normal(vid)
                basis_old_v = get_local_basis(positions[vertices[0]], positions[vertices[1]], n_old_v)

                n_new_v = updated_sheet.vertex_normal(vid)
                basis_new_v = get_local_basis(updated_sheet_positions[updated_sheet_vertices[0]],
                                            updated_sheet_positions[updated_sheet_vertices[1]], n_new_v)

                diff, diff_len = ma.diffs[outer_idx], ma.diff_lens[outer_idx]

                local_coords = __project_array_to_basis(diff, basis_old_v)
                new_diff = __update_array(local_coords, basis_new_v)
                new_diff /= np.linalg.norm(new_diff)
                new_pos = updated_inner_projs[outer_idx] + new_diff * diff_len
                new_outer_pos[outer_idx] = new_pos
            else:
                diff, diff_len = ma.diffs[outer_idx], ma.diff_lens[outer_idx]

                local_coords = __project_array_to_basis(diff, basis_old)
                new_diff = __update_array(local_coords, basis_new)
                new_diff /= np.linalg.norm(new_diff)
                new_pos = updated_inner_projs[outer_idx] + new_diff * diff_len
                new_outer_pos[outer_idx] = new_pos

    # update medial axis object with new positions
    ma.sheet.positions()[:] = updated_sheet_positions
    ma.outer_points = new_outer_pos

    ma.inner_points[~ma.curve_indices] = updated_sheet_positions[ma.inner_to_sheet_index[~ma.curve_indices]]
    ma.graph.positions()[~ma.curve_indices] = updated_sheet_positions[ma.inner_to_sheet_index[~ma.curve_indices]]

    ma.surface.positions()[:] = ma.outer_points
    ma.inner_projections[:] = updated_inner_projs


def inverse_apply_curves(medial_axis: MedialAxis, updated_curve_positions: list[np.ndarray], old_sheet: hmesh.Manifold):
    for curve, new_curve_pos in zip(medial_axis.curves, updated_curve_positions):
        old_curve_pos = medial_axis.inner_points[curve]
        outer_points_indices = medial_axis.correspondences[curve]

        # if curve not updated, skip
        if np.allclose(old_curve_pos, new_curve_pos):
            continue

        new_outer_points = np.copy(medial_axis.outer_points)
        new_inner_projs = np.copy(medial_axis.inner_projections)

        # take old and new normals at sheet point connecting to start of curve
        connection = curve[0]
        vid = medial_axis.inner_to_sheet_index[connection]
        fid = medial_axis.sheet.circulate_vertex(vid, mode='f')[0]
        old_normal = old_sheet.face_normal(fid)
        new_normal = medial_axis.sheet.face_normal(fid)

        old_curve_frame = parallel_transport_curve_framing(old_curve_pos, old_normal)
        new_curve_frame = parallel_transport_curve_framing(new_curve_pos, new_normal)

        for outer in flatten(outer_points_indices):
            closest = medial_axis.inner_ts[outer, 0].astype(int)
            segment_idx = np.where(curve == closest)[0][0]
            t = medial_axis.inner_ts[outer, 2]

            diff, diff_len = medial_axis.diffs[outer], medial_axis.diff_lens[outer]
            new_inner_projs[outer] = new_curve_pos[segment_idx] * (1 - t) + new_curve_pos[segment_idx + 1] * t

            basis_old = old_curve_frame[segment_idx]
            basis_new = new_curve_frame[segment_idx]

            local_coords = __project_array_to_basis(diff, basis_old)
            new_diff = __update_array(local_coords, basis_new)
            new_diff /= np.linalg.norm(new_diff)
            new_pos = new_inner_projs[outer] + new_diff * diff_len
            new_outer_points[outer] = new_pos

        medial_axis.inner_points[curve] = new_curve_pos
        medial_axis.graph.positions()[curve] = new_curve_pos
        medial_axis.outer_points[:] = new_outer_points
        medial_axis.surface.positions()[:] = new_outer_points
        medial_axis.inner_projections[:] = new_inner_projs


def map_to_surface(medial_axis: MedialAxis, updated_positions: np.ndarray):
    # map to sheet indices
    old_sheet = hmesh.Manifold(medial_axis.sheet)

    updated_sheet_positions = np.copy(medial_axis.sheet.positions())
    updated_sheet_positions[medial_axis.inner_to_sheet_index[medial_axis.sheet_indices]] = updated_positions[medial_axis.sheet_indices]
    inverse_apply_sheet(medial_axis, updated_sheet_positions)

    # map to curve indices
    updated_curve_positions = [updated_positions[curve] for curve in medial_axis.curves]
    inverse_apply_curves(medial_axis, updated_curve_positions, old_sheet)
