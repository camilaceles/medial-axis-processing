import numpy as np
import trimesh
from pygel3d import hmesh
from commons.medial_axis import MedialAxis
from commons.utils import trimesh_to_manifold, build_ball_correspondences
from medial_axis_loader import shared


def __read_qmat(filename: str) -> tuple[np.ndarray, list[list[int]], list[list[int]]]:
    vertices = []
    edges = []
    faces = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif line.startswith('e ') or line.startswith('l '):
                parts = line.strip().split()
                edges.append([int(parts[1]), int(parts[2])])

            elif line.startswith('f '):
                parts = line.strip().split()
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])

    return np.array(vertices), edges, faces


def load(
        input_mesh: hmesh.Manifold,
        filename: str,
        start=0.005, step=0.001
) -> MedialAxis:
    """Loads MedialAxis object from a file outputted by Q-MAT."""
    vertices, edges, faces = __read_qmat(filename)

    medial_sheet = shared.to_medial_sheet(vertices, faces)
    medial_sheet = shared.fix_normals(medial_sheet)

    medial_curves, graph = shared.to_medial_curves(vertices, edges, faces)

    correspondences = build_ball_correspondences(input_mesh, vertices, start=start, step=step)

    return MedialAxis(input_mesh, vertices, medial_sheet, medial_curves, correspondences)
