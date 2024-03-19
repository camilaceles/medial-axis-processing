import trimesh
import numpy as np
import random
from pygel3d import hmesh, graph
from commons.point import PointSet, Point
from collections import deque
from commons.utils import trimesh_to_manifold


def __precompute_face_adjacencies(mesh):
    adjacency_dict = {}
    for adj_index, faces in enumerate(mesh.face_adjacency):
        for face in faces:
            if face not in adjacency_dict:
                adjacency_dict[face] = []
            adjacency_dict[face].append((adj_index, set(faces) - {face}))
    return adjacency_dict


def __expand_from_triangle(mesh: trimesh.Trimesh, start_face: int, angle_threshold_degrees: float):
    angle_threshold_radians = np.radians(angle_threshold_degrees)

    sheet_faces = {start_face}
    to_visit = deque([start_face])

    adjacency_dict = __precompute_face_adjacencies(mesh)

    while to_visit:
        current_face = to_visit.popleft()

        if current_face in adjacency_dict:
            for adj_index, other_faces in adjacency_dict[current_face]:
                adj_face = next(iter(other_faces))

                if adj_face not in sheet_faces:
                    angle = mesh.face_adjacency_angles[adj_index]

                    if angle < angle_threshold_radians:
                        sheet_faces.add(adj_face)
                        to_visit.append(adj_face)

    return sheet_faces


def __single_sheet(m: hmesh.Manifold, dihedral_angle_threshold: float):
    """Extract a single sheet from double-sheet"""
    faces = np.array([m.circulate_face(fid) for fid in m.faces()])
    trimesh_mesh = trimesh.Trimesh(vertices=m.positions(), faces=faces)

    sheet_faces = single_sheet_faces(trimesh_mesh, dihedral_angle_threshold)
    sheet_mesh = trimesh_mesh.submesh([np.array(sheet_faces)], append=True)

    sheet = trimesh_to_manifold(sheet_mesh)
    return sheet


def single_sheet_faces(trim: trimesh.Trimesh, dihedral_angle_threshold: float):
    n_org_faces = len(trim.faces)

    sheet_faces = []
    max_faces_length = 0
    # if a bad start face is chosen, the resulting mesh is only a few triangles,
    # so we try until it results in at least 30% of the original face count
    for i in range(10):
        start_face = random.choice(range(len(trim.faces)))
        potential_sheet_faces = __expand_from_triangle(trim, start_face, dihedral_angle_threshold)

        # if good enough, return immediately
        if len(potential_sheet_faces) > 0.2 * n_org_faces:
            return list(potential_sheet_faces)

        # otherwise return the largest sheet found
        if len(potential_sheet_faces) > max_faces_length:
            sheet_faces = list(potential_sheet_faces)
            max_faces_length = len(sheet_faces)

    return sheet_faces


def to_medial_sheet(  # TODO this needs to handle multiple sheets
        input_mesh: hmesh.Manifold,
        inner_points: PointSet,
        dihedral_angle_threshold: float
) -> hmesh.Manifold:
    inner_mesh = hmesh.Manifold(input_mesh)

    sheet_indices = inner_points.is_fixed

    pos = inner_mesh.positions()
    pos[sheet_indices] = inner_points.positions[sheet_indices]

    for q in inner_points:
        if not q.is_fixed:
            inner_mesh.remove_vertex(q.index)
    inner_mesh.cleanup()
    # extract a single sheet from it and project remaining inner points to closest in the sheet
    single_sheet = __single_sheet(inner_mesh, dihedral_angle_threshold)
    return single_sheet


def to_medial_curves(inner_points: PointSet, keep_n_curves: int = None) -> list[list[int]]:
    """Find curves in the medial axis"""

    # Use graph as auxiliary data structure to find connections between curve points
    g = graph.Graph()
    for q in inner_points:
        g.add_node(q.pos)

    for q in inner_points:
        if not q.is_fixed:
            g.connect_nodes(q.index, q.front_point)
            g.connect_nodes(q.index, q.back_point)

    visited = set()

    def __expand_from_endpoint(endpoint: int) -> list[int]:
        path = [endpoint]
        while True:
            visited.add(endpoint)
            next_nodes = [n for n in g.neighbors(endpoint) if n not in visited]
            if not next_nodes:
                break
            endpoint = next_nodes[0]
            path.append(endpoint)
        return path

    curves = []
    for node in g.nodes():
        if node not in visited and len(g.neighbors(node)) == 1:
            # Start points are unvisited nodes with 1 neighbors (curve endpoint)
            curve_points = __expand_from_endpoint(node)
            curves.append(curve_points)

    curves.sort(key=lambda s: -len(s))  # Sort curves by length

    # Only keep the n longest curves
    if keep_n_curves is not None:
        for curve in curves[keep_n_curves:]:
            for point in curve:
                inner_points[point].is_fixed = True
        curves = curves[:keep_n_curves]

    return curves
