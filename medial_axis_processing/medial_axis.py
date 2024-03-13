import numpy as np
from scipy.spatial import KDTree
from pygel3d import *
from commons.point import Point, PointSet
import copy


class MedialAxis:
    def __init__(self, medial_sheet: hmesh.Manifold, inner_points: PointSet, outer_points: PointSet):
        self.mesh = hmesh.Manifold(medial_sheet)
        self.inner_points = copy.deepcopy(inner_points)
        self.outer_points = copy.deepcopy(outer_points)

        self.sheet_indices = self.inner_points.is_fixed

        # build correspondences maps
        self.inner_indices: np.ndarray = np.arange(self.inner_points.N)  # stores the indices of corresponding inner points in the medial structure
        self.correspondences: list[list[Point]] = [[] for _ in range(self.inner_points.N)]  # stores the outer points corresponding to each inner point
        self.__map_correspondences()

    def __map_correspondences(self):
        # Project sheet inner points to medial sheet.
        # Curve inner points are not projected, and instead their original position and correspondence.
        single_sheet_pos = self.mesh.positions()

        kd_tree = KDTree(single_sheet_pos)
        _, sheet_inner_indices = kd_tree.query(self.inner_points.positions[self.sheet_indices])
        projected = single_sheet_pos[sheet_inner_indices]
        self.inner_points.positions[self.sheet_indices] = projected
        self.inner_indices[self.sheet_indices] = sheet_inner_indices

        for i, outer_point in enumerate(self.outer_points):
            self.correspondences[self.inner_indices[i]].append(outer_point)
