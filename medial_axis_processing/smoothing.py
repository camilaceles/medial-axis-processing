import numpy as np
import trimesh.triangles
from scipy.spatial import KDTree

from commons.medial_axis import MedialAxis


def smooth_rbf(medial_axis: MedialAxis):
    g = medial_axis.graph

    for node in g.nodes():
        neighbors = g.neighbors(node)
        lens = medial_axis.rbf[neighbors]
        medial_axis.rbf[node] = np.mean(lens)

    __apply_rbf(medial_axis)


def simple_smooth(medial_axis: MedialAxis):
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
