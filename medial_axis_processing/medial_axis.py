import numpy as np
from scipy.spatial import KDTree
from pygel3d import *
from commons.point import Point, PointSet
import copy


class MedialAxis:
    def __init__(
            self,
            medial_sheet: hmesh.Manifold,
            medial_curves: list[list[int]],
            inner_points: PointSet,
            outer_points: PointSet
    ):
        self.inner_points: PointSet = copy.deepcopy(inner_points)
        self.outer_points: PointSet = copy.deepcopy(outer_points)

        self.sheet: hmesh.Manifold = hmesh.Manifold(medial_sheet)
        self.curves: list[list[Point]] = [[] for _ in range(len(medial_curves))]
        for i, curve in enumerate(medial_curves):
            for q_idx in curve:
                self.curves[i].append(self.inner_points[q_idx])

        # find sheet to curve connection points
        self.__find_connection_points()

        # build indices
        self.sheet_indices: np.ndarray = self.inner_points.is_fixed
        self.curve_indices: np.ndarray = ~self.inner_points.is_fixed

        # build correspondences maps
        self.inner_indices: np.ndarray = np.arange(self.inner_points.N)  # stores the indices of corresponding inner points in the medial structure
        self.correspondences: list[list[Point]] = [[] for _ in range(self.inner_points.N)]  # stores the outer points corresponding to each inner point
        self.__map_correspondences()

    def __map_correspondences(self):
        # Project sheet inner points to medial sheet.
        # Curve inner points are not projected, and instead their original position and correspondence.
        single_sheet_pos = self.sheet.positions()

        kd_tree = KDTree(single_sheet_pos)
        _, sheet_inner_indices = kd_tree.query(self.inner_points.positions[self.sheet_indices])
        projected = single_sheet_pos[sheet_inner_indices]
        self.inner_points.positions[self.sheet_indices] = projected
        self.inner_indices[self.sheet_indices] = sheet_inner_indices

        for i, outer_point in enumerate(self.outer_points):
            self.correspondences[self.inner_indices[i]].append(outer_point)

    def __find_connection_points(self):
        # TODO not sure what to store as connection, for now sheet index, figure out in pipeline
        kd = KDTree(self.sheet.positions())

        # for each curve, find point in medial sheet to which is connected (closest)
        for i, curve in enumerate(self.curves):
            start, end = curve[0], curve[-1]
            dist_start, closest_start = kd.query(start.pos, k=1)
            dist_end, closest_end = kd.query(end.pos, k=1)
            if dist_start < dist_end:  # take endpoint closest to sheet as connection
                self.inner_points.is_connection[start.index] = closest_start
            else:
                self.curves[i] = curve[::-1]  # reverse curve so first index is where it connects to sheet
                self.inner_points.is_connection[end.index] = closest_end

    def to_graph(self):
        g = graph.Graph()

        # add curve connections
        for q in self.inner_points:
            g.add_node(q.pos)

        for curve in self.curves:
            for i in range(len(curve) - 1):
                g.connect_nodes(curve[i].index, curve[i+1].index)

        # add sheet connections
        vid_offset = len(g.nodes())
        for vid in self.sheet.vertices():
            g.add_node(self.sheet.positions()[vid])

        for vid in self.sheet.vertices():
            neighbors = self.sheet.circulate_vertex(vid)
            for n in neighbors:
                g.connect_nodes(vid + vid_offset, n + vid_offset)

        # connect curves to sheet
        for curve in self.curves:
            start = curve[0]
            connection = start.is_connection
            g.connect_nodes(start.index, connection + vid_offset)

        return g


########################################################################################################################
# Graph with multiple sheets
########################################################################################################################

# g = graph.Graph()
# for q in inner_points:
#     g.add_node(q.pos)
#
# dihedral_angle_threshold = 90
# r = outer_points.get_average_sparsity()
#
# # build manifold from fixed inner points (medial sheet points)
#
# inner_mesh = hmesh.Manifold(input_mesh)
# inner_mesh.positions()[:] = inner_points.positions
# for p in inner_points:
#     if not p.is_fixed:
#         inner_mesh.remove_vertex(p.index)
#
# faces = np.array([inner_mesh.circulate_face(fid) for fid in inner_mesh.faces()])
# trim = trimesh.Trimesh(vertices=inner_mesh.positions(), faces=faces, process=False)
#
# # trim = manifold_to_trimesh(inner_mesh)
#
# connected_components = trimesh.graph.connected_components(
#     edges=trim.face_adjacency, nodes=np.arange(len(trim.faces))
# )
#
# edge_lengths = []
# medial_sheet_faces = []
# # get submeshes keeping vertex index correspondaces
# for component in connected_components:
#     sub_mesh = trimesh.Trimesh(vertices=trim.vertices, faces=trim.faces[component], process=False)
#     sheet_faces_indexes = single_sheet_faces(sub_mesh, dihedral_angle_threshold)
#     sheet_faces = sub_mesh.faces[sheet_faces_indexes]
#     medial_sheet_faces.append(sheet_faces)
#
#     edges = np.vstack([sheet_faces[:, [0, 1]], sheet_faces[:, [1, 2]], sheet_faces[:, [0, 2]]])
#
#     for edge in edges:
#         g.connect_nodes(edge[0], edge[1])
#         edge_lengths.append(np.linalg.norm(g.positions()[edge[0]] - g.positions()[edge[1]]))
#
# sheets_pos = np.array([g.positions()[node] for node in g.nodes() if len(g.neighbors(node)) > 0])
# sheets_idx = [node for node in g.nodes() if len(g.neighbors(node)) > 0]
#
# for q in inner_points:
#     if not q.is_fixed:
#         g.connect_nodes(q.index, q.front_point)
#         g.connect_nodes(q.index, q.back_point)
#
# kd = KDTree(sheets_pos)
#
# # for each curve, connection is the one closest to medial sheet
# for end1, end2 in endpoints:
#     dist1, closest1 = kd.query(g.positions()[end1], k=1)
#     dist2, closest2 = kd.query(g.positions()[end2], k=1)
#     if dist1 < dist2:
#         g.connect_nodes(end1, sheets_idx[closest1])
#         inner_points.is_connection[end1] = sheets_idx[closest1]
#     else:
#         g.connect_nodes(end2, sheets_idx[closest2])
#         inner_points.is_connection[end2] = sheets_idx[closest2]