import numpy as np
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
    for i in range(len(medial_axis.inner_points)):
        corr = medial_axis.correspondences[i]
        inner_pos = medial_axis.inner_points[i]

        medial_axis.outer_points[corr] = inner_pos + (medial_axis.diffs[corr] * medial_axis.rbf[i])

    medial_axis.surface.positions()[:] = medial_axis.outer_points
