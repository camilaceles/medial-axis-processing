import numpy as np
from pygel3d import hmesh, graph
from scipy.spatial import KDTree

from commons.medial_axis import MedialAxis
from commons.utils import build_ball_correspondences


def load(
        input_mesh: hmesh.Manifold,
        medial_sheet: hmesh.Manifold,
) -> MedialAxis:
    """In case MAT is simply a sheet"""
    medial_curves = []

    vertices = medial_sheet.positions()

    correspondences = build_ball_correspondences(input_mesh, vertices, start=3.5, step=0.1)

    g = graph.from_mesh(medial_sheet)

    return MedialAxis(input_mesh, vertices, medial_sheet, medial_curves, correspondences, g)

