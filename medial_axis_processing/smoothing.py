import numpy as np
import trimesh.triangles
from scipy.spatial import KDTree

from commons.medial_axis import MedialAxis
from commons.utils import flatten


def __least_squares_rbf_sheet(medial_axis: MedialAxis):
    # for each face in sheet, run lstsq to find rbf at each vertex
    outer_sheet = flatten(medial_axis.correspondences[~medial_axis.curve_indices])

    for fid in medial_axis.sheet.faces():
        vertices = medial_axis.sheet.circulate_face(fid, mode='v')
        inner_v0 = medial_axis.sheet_to_inner_index[vertices[0]]
        inner_v1 = medial_axis.sheet_to_inner_index[vertices[1]]
        inner_v2 = medial_axis.sheet_to_inner_index[vertices[2]]

        face_ids = medial_axis.inner_barycentrics[outer_sheet, 0]
        radii = medial_axis.diff_lens[outer_sheet]

        mask = face_ids == fid
        radius = radii[mask]
        if np.sum(mask) == 0:
            # skip, no outer points are corresponding to this face
            continue
        elif np.sum(mask) < 3:
            avg_radius = np.mean(radius)
            medial_axis.rbf[inner_v0] = avg_radius
            medial_axis.rbf[inner_v1] = avg_radius
            medial_axis.rbf[inner_v2] = avg_radius
        else:
            barycentrics = medial_axis.inner_barycentrics[outer_sheet, 1:][mask]

            # centralize and regularize radii to avoid numerical instability
            mean_value = np.mean(radius)
            values_centered = radius - mean_value
            B = barycentrics
            alpha = 1  # regularization parameter -- might need tuning with input
            B_reg = np.vstack([B, np.sqrt(alpha) * np.eye(3)])
            values_reg = np.concatenate([values_centered, np.zeros(3)])

            vertex_values_centered, residuals, rank, s = np.linalg.lstsq(B_reg, values_reg, rcond=None)
            vertex_values = vertex_values_centered + mean_value

            medial_axis.rbf[inner_v0] = vertex_values[0]
            medial_axis.rbf[inner_v1] = vertex_values[1]
            medial_axis.rbf[inner_v2] = vertex_values[2]

def __least_squares_rbf_curve(medial_axis: MedialAxis):
    for curve in medial_axis.curves:
        curve = np.array(curve)
        outer_curve = flatten(medial_axis.correspondences[curve])
        closest_segment = medial_axis.inner_ts[outer_curve, 0]

        if len(curve) < 2:
            medial_axis.inner_ts[outer_curve, 0] = curve[0]
            medial_axis.inner_ts[outer_curve, 1] = curve[0]
            medial_axis.inner_ts[outer_curve, 2] = 0
            medial_axis.rbf[curve[0]] = medial_axis.diff_lens[curve[0]]
            continue

        # for each segment in curve, run lstsq to find rbf at each curve point
        for i, segment in enumerate(curve[:-1]):
            mask = closest_segment == segment
            t_values = medial_axis.inner_ts[outer_curve, 2][mask]
            radius = medial_axis.diff_lens[outer_curve][mask]

            t_values = np.concatenate(([0], t_values, [1]))
            radius = np.concatenate(([np.nan], radius, [np.nan]))

            mask = ~np.isnan(radius)
            t_fit = t_values[mask]
            values_fit = radius[mask]

            A = np.vstack([t_fit, np.ones(len(t_fit))]).T
            m, c = np.linalg.lstsq(A, values_fit, rcond=None)[0]

            value_at_start = m * 0 + c
            value_at_end = m * 1 + c

            start, end = segment, curve[i + 1]
            medial_axis.rbf[start] = value_at_start
            medial_axis.rbf[end] = value_at_end


def smooth_rbf(medial_axis: MedialAxis):
    g = medial_axis.graph
    pos = g.positions()

    new_rbf = np.copy(medial_axis.rbf)
    for node in g.nodes():
        neighbors = g.neighbors(node)

        distances = np.linalg.norm(pos[node] - pos[neighbors], axis=1)
        weights = 1 / np.where(distances == 0, np.inf, distances)
        weights /= weights.sum()

        new_rbf[node] = np.dot(weights, medial_axis.rbf[neighbors])
    medial_axis.rbf[:] = new_rbf


def simple_smooth(medial_axis: MedialAxis):
    __least_squares_rbf_sheet(medial_axis)

    __least_squares_rbf_curve(medial_axis)

    __apply_rbf(medial_axis)


def __apply_rbf(medial_axis: MedialAxis):
    kd = KDTree(medial_axis.inner_points)
    _, sheet_to_inner = kd.query(medial_axis.sheet.positions())

    for i in range(len(medial_axis.inner_points)):
        corr = medial_axis.correspondences[i]
        inner_projs = medial_axis.inner_projections[corr]

        # interpolate rbf
        if medial_axis.curve_indices[i]:
            closest, next, t = medial_axis.inner_ts[corr, 0].astype(int), medial_axis.inner_ts[corr, 1].astype(int), medial_axis.inner_ts[corr, 2]
            rbf_a = medial_axis.rbf[closest]
            rbf_b = medial_axis.rbf[next]
            rbf = rbf_a + (rbf_b - rbf_a) * t
            medial_axis.outer_points[corr] = inner_projs + (medial_axis.diffs[corr] * rbf[:, np.newaxis])
        else:
            for outer in corr:
                closest = medial_axis.inner_barycentrics[outer, 0].astype(int)
                closest_triangle = medial_axis.sheet.circulate_face(closest, mode='v')
                rbf_triangle = medial_axis.rbf[sheet_to_inner[closest_triangle]]
                barycentrics = medial_axis.inner_barycentrics[outer, 1:]
                rbf = np.multiply(rbf_triangle, barycentrics).sum()
                medial_axis.outer_points[outer] = medial_axis.inner_projections[outer] + (medial_axis.diffs[outer] * rbf)

    medial_axis.surface.positions()[:] = medial_axis.outer_points
