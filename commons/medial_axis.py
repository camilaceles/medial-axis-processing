import numpy as np
import trimesh
from scipy.spatial import KDTree
from pygel3d import *
from commons.utils import flatten, trimesh_to_manifold, project_points_to_curve, barycentric_project


class MedialAxis:
    def __init__(
            self,
            surface: hmesh.Manifold,
            inner_points: np.ndarray,
            medial_sheet: hmesh.Manifold,
            medial_curves: list[list[int]],
            correspondences: list[list[int]],
            medial_graph: graph.Graph = None
    ):
        self.surface: hmesh.Manifold = hmesh.Manifold(surface)
        self.inner_points: np.ndarray = np.copy(inner_points)
        self.outer_points: np.ndarray = np.copy(surface.positions())
        self.graph = medial_graph

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

        # Compute projections
        self.inner_projections = np.zeros(self.outer_points.shape)
        self.__compute_projections()

        # Compute average radial basis function
        self.rbf = np.zeros(self.inner_points.shape[0])
        self.diffs = np.zeros(self.outer_points.shape)
        self.__radial_basis_function()

    def update_correspondences(self, correspondences: list[list[int]]):
        self.correspondences = correspondences
        self.sheet_correspondences = [[] for _ in range(len(self.sheet.vertices()))]
        self.__map_sheet_correspondences()
        self.__radial_basis_function()

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

    def __radial_basis_function(self):
        for i in range(len(self.inner_points)):
            corr = self.correspondences[i]
            inner_projections = self.inner_projections[corr]
            outer_pos = self.outer_points[corr]

            diffs = outer_pos - inner_projections
            lens = np.linalg.norm(diffs, axis=1)

            # weights = 1 / (lens + 1e-8)
            # weighted_avg_len = np.average(lens, weights=weights)
            if len(lens) == 0:
                continue

            avg_len = np.mean(lens)
            norm_diffs = diffs / lens[:, np.newaxis]

            self.rbf[i] = avg_len
            self.diffs[corr] = norm_diffs

    def __compute_projections(self):
        outer_sheet = flatten(self.correspondences[~self.curve_indices])
        outer_sheet_pos = self.outer_points[outer_sheet]
        face_ids, barycentrics, projected = barycentric_project(self.sheet, outer_sheet_pos)
        self.inner_projections[outer_sheet] = projected

        for curve in self.curves:
            curve_pos = self.inner_points[curve]

            outer_curve = flatten(self.correspondences[curve])
            outer_curve_pos = self.outer_points[outer_curve]

            closest_segment, t_values, projected = project_points_to_curve(outer_curve_pos, curve_pos)
            self.inner_projections[outer_curve] = projected
