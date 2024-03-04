import numpy as np
from numpy import ndarray as array
from pygel3d import *
from commons.point import Point, PointSet
import copy
import trimesh
from commons.utils import manifold_to_trimesh, barycentric_project


class MedialAxis:
    def __init__(self, medial_sheet: hmesh.Manifold, inner_points: PointSet, outer_points: PointSet):
        self.mesh = hmesh.Manifold(medial_sheet)
        self.inner_points = copy.deepcopy(inner_points)
        self.outer_points = copy.deepcopy(outer_points)

        # build correspondences maps
        self.inner_barycentrics: array = np.zeros((inner_points.N,  4))
        self.correspondences: list[list[Point]] = [[] for _ in range(len(self.mesh.vertices()))]
        self.__map_correspondences()

    def __map_correspondences(self):
        # project inner points to medial sheet using barycentric coordinates
        face_ids, barycentrics = barycentric_project(self.mesh, self.inner_points.positions)
        self.inner_barycentrics[:, 0] = face_ids
        self.inner_barycentrics[:, 1:] = barycentrics

    def get_updates_inner_points(self, new_positions: array) -> array:
        updated_m = hmesh.Manifold(self.mesh)
        updated_m.positions()[:] = new_positions
        trim = manifold_to_trimesh(updated_m)
        triangles = trim.triangles[self.inner_barycentrics[:, 0].astype(int)]

        new_inner_positions = trimesh.triangles.barycentric_to_points(
            triangles,
            self.inner_barycentrics[:, 1:]
        )

        return new_inner_positions
