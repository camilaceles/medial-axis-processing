from commons.display import *
from commons.point import PointSet
from .meso_skeleton_formation import form_meso_skeleton, regularize_curve_points
from .inner_point_sinking import sink_inner_points


def deep_points(m: hmesh.Manifold, params: dict):
    positions = m.positions()
    normals = np.array([m.vertex_normal(vid) for vid in m.vertices()])

    outer_points = PointSet(positions, normals)
    inner_points = PointSet(positions, normals)

    # step 1: sink inner points
    sink_inner_points(outer_points, inner_points, params)

    # step 2: form medial axis
    form_meso_skeleton(outer_points, inner_points, params)

    if params["run_regularization"]:
        regularize_curve_points(outer_points, inner_points, params)

    return outer_points, inner_points
