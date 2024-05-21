import numpy as np
from pygel3d import hmesh, graph
from scipy.spatial import KDTree

from commons.medial_axis import MedialAxis
from commons.utils import trimesh_to_manifold, build_ball_correspondences, barycentric_project


def load(
        input_mesh: hmesh.Manifold,
        filename: str,
        no_smoothing: bool = False
) -> MedialAxis:
    """In case MAT is simply a sheet"""
    outer_points = input_mesh.positions()
    medial_sheet = hmesh.load(filename)
    medial_curves = []

    vertices = medial_sheet.positions()
    face_ids, barycentrics, projected = barycentric_project(medial_sheet, outer_points)

    # snap projection to sheet points
    kd_tree = KDTree(vertices)
    _, inner_sheet_indices = kd_tree.query(projected)

    correspondences = [[] for _ in range(len(medial_sheet.vertices()))]
    for i in range(len(outer_points)):
        correspondences[inner_sheet_indices[i]].append(i)
    correspondences = np.array(correspondences, dtype=object)

    g = graph.from_mesh(medial_sheet)

    return MedialAxis(input_mesh, vertices, medial_sheet, medial_curves, correspondences, g, no_smoothing)

