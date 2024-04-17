import numpy as np
import trimesh
from scipy.spatial import KDTree
from pygel3d import *

from commons.utils import flatten, trimesh_to_manifold


class MedialAxis:
    def __init__(
            self,
            surface: hmesh.Manifold,
            inner_points: np.ndarray,
            medial_sheet: hmesh.Manifold,
            medial_curves: list[list[int]],
            correspondences: list[list[int]]
    ):
        self.surface: hmesh.Manifold = hmesh.Manifold(surface)
        self.inner_points: np.ndarray = np.copy(inner_points)
        self.outer_points: np.ndarray = np.copy(surface.positions())

        self.sheet: hmesh.Manifold = hmesh.Manifold(medial_sheet)
        self.curves = medial_curves

        self.curve_indices = np.zeros(inner_points.shape[0], dtype=bool)
        self.curve_indices[list(set(flatten(self.curves)))] = True

        # Store the indices of corresponding inner points in the medial sheet
        self.inner_indices: np.ndarray = np.zeros(self.inner_points.shape[0], dtype=int)

        # Store list of surface points corresponding to each sheet point
        self.correspondences = correspondences
        self.sheet_correspondences: list[list[int]] = [[] for _ in range(len(self.sheet.vertices()))]
        self.__map_sheet_correspondences()

    def __map_sheet_correspondences(self):
        # for each point in medial sheet, store corresponding outer points
        sheet_pos = self.sheet.positions()

        kd_tree = KDTree(sheet_pos)
        _, sheet_inner_indices = kd_tree.query(self.inner_points)
        self.inner_indices = sheet_inner_indices

        for inner_idx, outer_points in enumerate(self.correspondences):
            if not self.curve_indices[inner_idx]:
                for p in outer_points:
                    self.sheet_correspondences[sheet_inner_indices[inner_idx]].append(p)


def to_medial_curves(vertices, edges, faces):
    sheet_vertices = set(flatten(faces))

    g = graph.Graph()
    for v in vertices:
        g.add_node(v)
    for (v1, v2) in edges:
        g.connect_nodes(v1, v2)
    g2 = graph.Graph(g)

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
    return curves, g2


def to_medial_sheet(vertices, faces):
    trim = trimesh.Trimesh(vertices, faces)
    return trimesh_to_manifold(trim)

    connected_components = list(trim.split(only_watertight=False))
    connected_components.sort(key=lambda x: -len(x.faces))
    to_keep = connected_components[0]

    # face_areas = to_keep.area_faces
    # faces_to_keep = face_areas > (0.1 * np.mean(face_areas))
    # to_keep.update_faces(faces_to_keep)
    # to_keep.remove_unreferenced_vertices()

    return trimesh_to_manifold(to_keep)
    # return to_keep.vertices, to_keep.faces
