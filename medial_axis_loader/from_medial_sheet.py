from pygel3d import hmesh, graph
from commons.medial_axis import MedialAxis
from commons.utils import trimesh_to_manifold, build_ball_correspondences


def load(
        input_mesh: hmesh.Manifold,
        filename: str,
        start=0.005, step=0.001
) -> MedialAxis:
    """In case MAT is simply a sheet"""
    medial_sheet = hmesh.load(filename)
    medial_curves = []

    vertices = medial_sheet.positions()
    correspondences = build_ball_correspondences(input_mesh, vertices, start=start, step=step)
    g = graph.from_mesh(medial_sheet)

    return MedialAxis(input_mesh, vertices, medial_sheet, medial_curves, correspondences, g)
