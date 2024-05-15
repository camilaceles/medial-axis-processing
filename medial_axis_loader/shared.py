from collections import deque
import random
import numpy as np
import trimesh
from pygel3d import *
from commons.utils import flatten, trimesh_to_manifold, manifold_to_trimesh


def __precompute_face_adjacencies(mesh):
    adjacency_dict = {}
    for adj_index, faces in enumerate(mesh.face_adjacency):
        for face in faces:
            if face not in adjacency_dict:
                adjacency_dict[face] = []
            adjacency_dict[face].append((adj_index, set(faces) - {face}))
    return adjacency_dict


def __remove_edges_in_faces(edges, faces):
    edges_from_faces = set()
    for face in faces:
        num_vertices = len(face)
        for i in range(num_vertices):
            edge = (face[i], face[(i + 1) % num_vertices])
            edges_from_faces.add(frozenset(edge))

    filtered_edges = [edge for edge in edges if frozenset(edge) not in edges_from_faces]
    return filtered_edges


def fix_normals(m: hmesh.Manifold) -> hmesh.Manifold:
    mesh = manifold_to_trimesh(m, process=True)

    start_face = random.choice(range(len(mesh.faces)))
    visited = {start_face}
    to_visit = deque([start_face])

    adjacency_dict = __precompute_face_adjacencies(mesh)

    while to_visit:
        current_face = to_visit.popleft()

        if current_face in adjacency_dict:
            for adj_index, other_faces in adjacency_dict[current_face]:
                adj_face = next(iter(other_faces))

                if adj_face not in visited:
                    angle = mesh.face_adjacency_angles[adj_index]
                    if angle > np.pi / 2:
                        mesh.faces[adj_face] = mesh.faces[adj_face][[1, 0, 2]]

                    visited.add(adj_face)
                    to_visit.append(adj_face)
    mesh.fix_normals()

    return trimesh_to_manifold(mesh)


def to_graph(vertices, edges):
    g = graph.Graph()
    for v in vertices:
        g.add_node(v)
    for (v1, v2) in edges:
        g.connect_nodes(v1, v2)
    return g


def to_medial_curves(vertices, edges, faces):
    sheet_vertices = set(flatten(faces))
    curve_edges = __remove_edges_in_faces(edges, faces)

    g = graph.Graph()
    for v in vertices:
        g.add_node(v)
    for (v1, v2) in curve_edges:
        g.connect_nodes(v1, v2)

    visited = set()

    def __expand_from_endpoint(endpoint: int) -> list[int]:
        path = [endpoint]
        while True:
            visited.add(endpoint)
            next_nodes = [n for n in g.neighbors(endpoint) if n not in visited]
            if not next_nodes or len(next_nodes) > 1:
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

    # ensure curve[0] is connection to medial sheet
    for i, curve in enumerate(curves):
        if curve[-1] in sheet_vertices:
            curves[i] = curve[::-1]

    curves.sort(key=lambda s: -len(s))  # Sort curves by length
    return curves


def to_medial_sheet(vertices, faces):
    return hmesh.Manifold.from_triangles(vertices, faces)

    # connected_components = list(trim.split(only_watertight=False))
    # connected_components.sort(key=lambda x: -len(x.faces))
    # to_keep = connected_components[0]
    #
    # face_areas = to_keep.area_faces
    # faces_to_keep = face_areas > (0.1 * np.mean(face_areas))
    # to_keep.update_faces(faces_to_keep)
    # to_keep.remove_unreferenced_vertices()
    #
    # return trimesh_to_manifold(to_keep)

