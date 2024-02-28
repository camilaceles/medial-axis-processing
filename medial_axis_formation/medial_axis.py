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
        self.inner_points = copy.deepcopy(inner_points)
        self.outer_points = copy.deepcopy(outer_points)

        # build correspondences maps
        self.inner_indices: np.ndarray = np.zeros(inner_points.N)
        self.correspondences: list[list[Point]] = [[] for _ in range(len(self.mesh.vertices()))]
        self.__map_correspondences()

    def __map_correspondences(self):
        # project inner points to medial sheet
        single_sheet_pos = self.mesh.positions()

        kd_tree = KDTree(single_sheet_pos)
        _, self.inner_indices = kd_tree.query(self.inner_points.positions)
        projected = single_sheet_pos[self.inner_indices]
        self.inner_points.positions = projected

        for i, outer_point in enumerate(self.outer_points):
            self.correspondences[self.inner_indices[i]].append(outer_point)

    def update_medial_sheet_positions(self, new_positions: np.ndarray):
        self.mesh.positions()[:] = new_positions
        new_inner_positions = new_positions[self.inner_indices]
        self.inner_points.positions = new_inner_positions

