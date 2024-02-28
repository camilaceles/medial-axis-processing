from typing import Optional
import numpy as np
from numpy import ndarray as array
from scipy.spatial import KDTree
from pygel3d import *
from medial_axis_formation.point import Point, PointSet


class MedialAxis:
    def __init__(self, medial_sheet: hmesh.Manifold, inner_points: PointSet, outer_points: PointSet):
        self.mesh = medial_sheet
        self.map: dict[tuple, dict] = {}

        # project remaining inner points to medial sheet
        self.__project_to_medial_sheet(medial_sheet, inner_points)
        # build correspondences map
        self.__map_points(inner_points, outer_points)

    @staticmethod
    def __project_to_medial_sheet(medial_sheet: hmesh.Manifold, inner_points: PointSet):
        single_sheet_pos = medial_sheet.positions()

        kd_tree = KDTree(single_sheet_pos)
        _, indices = kd_tree.query(inner_points.positions)
        projected = single_sheet_pos[indices]
        inner_points.positions = projected

    def __map_points(self, inner_points: PointSet, outer_points: PointSet):
        for q, p in zip(inner_points, outer_points):
            key = tuple(q.pos)
            if key not in self.map:
                self.map[key] = {
                    'inner_point': q,
                    'outer_points': [p]
                }
            else:
                self.map[key]['outer_points'].append(p)

    def medial_sheet_idx_to_outer_points(self, idx: int) -> list[Point]:
        key = tuple(self.mesh.positions()[idx])
        return self.map[key]['outer_points']
