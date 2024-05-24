import numpy as np
from pygel3d import hmesh
from trimesh.triangles import barycentric_to_points
from commons.medial_axis import MedialAxis
from scipy.spatial.transform import Rotation as R
from commons.utils import flatten


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


def __project_array_to_basis(point, basis):
    # get point coordinates in local basis
    return np.dot(point, basis[0]), np.dot(point, basis[1]), np.dot(point, basis[2])


def __update_array(local_coords, new_basis):
    # retrieve global coordinates from local basis coordinates
    return local_coords[0] * new_basis[0] + local_coords[1] * new_basis[1] + local_coords[2] * new_basis[2]


def __compute_tangents(curve):
    tangents = np.gradient(curve, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]
    return tangents


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


def __update_inner_projections_curve(medial_axis: MedialAxis, updated_curve_positions: list[np.ndarray]):
    for curve, new_curve_pos in zip(medial_axis.curves, updated_curve_positions):
        corr = flatten(medial_axis.correspondences[curve])

        # get updated projections of outer points with linear interpolation
        closest, next, t = medial_axis.inner_ts[corr, 0].astype(int), medial_axis.inner_ts[corr, 1].astype(int), medial_axis.inner_ts[corr, 2]

        value_to_index = {value: index for index, value in enumerate(curve)}
        closest_idx = [value_to_index[value] for value in closest]
        next_idx = [value_to_index[value] for value in next]

        pos_a = new_curve_pos[closest_idx]
        pos_b = new_curve_pos[next_idx]
        new_pos = pos_a + (pos_b - pos_a) * t.reshape(-1, 1)

        medial_axis.inner_projections[corr] = new_pos


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

        # if vertex is curve connection, also update curve inner and outer points.
        # this ensures the connecting curve points are aligned with the updated sheet
        inner = ma.sheet_to_inner_index[vid]
        if ma.curve_indices[inner]:
            # find associated curve
            curve_index = next((i for i, curve in enumerate(ma.curves) if curve and curve[0] == inner), -1)
            curve = ma.curves[curve_index]

            # map transformation to curve inner points
            for inner_idx in curve:
                inner_point = ma.inner_points[inner_idx]

                local_coords = __project_point_to_basis(inner_point, positions[vid], basis_old)
                new_pos = __update_point(updated_sheet_positions[vid], local_coords, basis_new)
                new_inner_pos[inner_idx] = new_pos

                # also add curve outer points to be updated
                corresponding_outer.extend(ma.correspondences[inner_idx])

        # map transformation to outer points
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
       Does this by projecting the points to the local basis of the medial sheet vertex, and then updating the point"""
    positions = ma.sheet.positions()

    updated_sheet = hmesh.Manifold(ma.sheet)
    updated_sheet.positions()[:] = updated_sheet_positions
    updated_inner_projs = __updated_inner_projections_sheet(ma, updated_sheet_positions)

    new_outer_pos = np.copy(ma.outer_points)

    for fid in ma.sheet.faces():
        vertices = ma.sheet.circulate_face(fid, mode='v')
        updated_sheet_vertices = updated_sheet.circulate_face(fid, mode='v')

        n_old = ma.sheet.face_normal(fid)
        basis_old = __get_local_basis(positions[vertices[0]], positions[vertices[1]], n_old)

        n_new = updated_sheet.face_normal(fid)
        basis_new = __get_local_basis(updated_sheet_positions[updated_sheet_vertices[0]],
                                      updated_sheet_positions[updated_sheet_vertices[1]], n_new)

        corresponding_outer = np.where(ma.inner_barycentrics[:, 0].astype(int) == fid)[0]

        for outer_idx in corresponding_outer:
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


def parallel_transport(old_curve_pos: np.ndarray, new_curve_pos: np.ndarray, outer_points: list[np.ndarray]):
    """Given the original curve point positions, their updated positions, and a list of corresponding outer points for
       each curve point, returns the parallel transport of the outer points along the curve to the updated positions."""
    orig_tangents = __compute_tangents(old_curve_pos)
    new_tangents = __compute_tangents(new_curve_pos)
    new_outer_points_list = []

    for i in range(len(old_curve_pos)):
        axis = np.cross(orig_tangents[i], new_tangents[i])
        axis_length = np.linalg.norm(axis)
        axis /= axis_length
        angle = np.arccos(np.clip(np.dot(orig_tangents[i], new_tangents[i]), -1.0, 1.0))
        rotation = R.from_rotvec(axis * angle)
        new_outer_points = [
            rotation.apply(point - old_curve_pos[i]) + new_curve_pos[i]
            for point in outer_points[i]
        ]
        new_outer_points_list.append(new_outer_points)

    return new_outer_points_list


def inverse_apply_curves(medial_axis: MedialAxis, updated_curve_positions: list[np.ndarray]):
    for curve, new_curve_pos in zip(medial_axis.curves, updated_curve_positions):
        inner_points = medial_axis.inner_points[curve]
        outer_points_indices = medial_axis.correspondences[curve]

        # if curve not updated, skip
        if np.allclose(inner_points, new_curve_pos):
            continue

        outer_points = []
        for indices in outer_points_indices:
            outer_points.append(medial_axis.outer_points[indices])

        new_outer_points = parallel_transport(
            inner_points,
            new_curve_pos,
            outer_points
        )
        medial_axis.inner_points[curve] = new_curve_pos
        medial_axis.graph.positions()[curve] = new_curve_pos
        for j, indices in enumerate(outer_points_indices):
            if len(indices) == 0:
                continue
            medial_axis.outer_points[indices] = new_outer_points[j]
            medial_axis.surface.positions()[indices] = new_outer_points[j]
        __update_inner_projections_curve(medial_axis, updated_curve_positions)


def inverse_apply_curves_v2(medial_axis: MedialAxis, updated_curve_positions: list[np.ndarray]):
    for curve, new_curve_pos in zip(medial_axis.curves, updated_curve_positions):
        old_curve_pos = medial_axis.inner_points[curve]
        outer_points_indices = medial_axis.correspondences[curve]

        # if curve not updated, skip
        if np.allclose(old_curve_pos, new_curve_pos):
            continue

        orig_tangents = __compute_tangents(old_curve_pos)
        new_tangents = __compute_tangents(new_curve_pos)

        new_outer_points = np.copy(medial_axis.outer_points)
        new_inner_projs = np.copy(medial_axis.inner_projections)

        for outer in flatten(outer_points_indices):
            closest = medial_axis.inner_ts[outer, 0].astype(int)
            segment_idx = np.where(curve == closest)[0][0]
            t = medial_axis.inner_ts[outer, 2]

            axis = np.cross(orig_tangents[segment_idx], new_tangents[segment_idx])
            axis_length = np.linalg.norm(axis)
            if axis_length > 1e-10:
                axis /= axis_length

            angle = np.arccos(np.clip(np.dot(orig_tangents[segment_idx], new_tangents[segment_idx]), -1.0, 1.0))
            rotation = R.from_rotvec(axis * angle)

            old_pos = old_curve_pos[segment_idx] * (1 - t) + old_curve_pos[segment_idx + 1] * t
            new_inner_projs[outer] = new_curve_pos[segment_idx] * (1 - t) + new_curve_pos[segment_idx + 1] * t

            new_point = rotation.apply(medial_axis.outer_points[outer] - old_pos) + new_inner_projs[outer]
            new_outer_points[outer] = new_point

        medial_axis.inner_points[curve] = new_curve_pos
        medial_axis.graph.positions()[curve] = new_curve_pos
        medial_axis.outer_points[:] = new_outer_points
        medial_axis.surface.positions()[:] = new_outer_points
        medial_axis.inner_projections[:] = new_inner_projs


def inverse_apply_medial_axis(
        medial_axis: MedialAxis,
        updated_sheet_positions: np.ndarray,
        updated_curve_positions: list[np.ndarray]
):
    inverse_apply_sheet(medial_axis, updated_sheet_positions)
    inverse_apply_curves(medial_axis, updated_curve_positions)
