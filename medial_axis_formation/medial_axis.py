from typing import Optional
import numpy as np
from numpy import ndarray as array
from scipy.spatial import KDTree
from pygel3d import *
from medial_axis_formation.point import Point, PointSet
import copy


class MedialAxis:
    def __init__(self, medial_sheet: hmesh.Manifold, inner_points: PointSet, outer_points: PointSet):
        self.mesh = hmesh.Manifold(medial_sheet)
        self.indices: np.ndarray = np.zeros(inner_points.N)
        self.correspondences: list[list[Point]] = [[] for _ in range(len(self.mesh.vertices()))]

        # build correspondences map
        self.__map_correspondences(inner_points, outer_points)

    def __map_correspondences(self, inner_points: PointSet, outer_points: PointSet):
        # project inner points to medial sheet
        single_sheet_pos = self.mesh.positions()

        kd_tree = KDTree(single_sheet_pos)
        _, self.indices = kd_tree.query(inner_points.positions)

        for i, outer_point in enumerate(outer_points):
            self.correspondences[self.indices[i]].append(outer_point)

