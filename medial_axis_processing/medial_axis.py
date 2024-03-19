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

