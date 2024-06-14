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
            medial_graph: graph.Graph = None,
    ):
        self.surface: hmesh.Manifold = hmesh.Manifold(surface)
        self.inner_points: np.ndarray = np.copy(inner_points)
        self.outer_points: np.ndarray = np.copy(surface.positions())
        self.graph = medial_graph

        self.sheet: hmesh.Manifold = hmesh.Manifold(medial_sheet)
        self.curves = medial_curves

        self.curve_indices = np.zeros(inner_points.shape[0], dtype=bool)
        self.curve_indices[list(set(flatten(self.curves)))] = True

        # connection is also in sheet (first pos of curve)
        self.sheet_indices = ~self.curve_indices
        for curve in self.curves:
            self.sheet_indices[curve[0]] = True

        # Store the indices of corresponding inner points in the medial sheet
        self.inner_to_sheet_index: np.ndarray = np.zeros(self.inner_points.shape[0], dtype=int)
        self.sheet_to_inner_index: np.ndarray = np.zeros(len(self.sheet.vertices()), dtype=int)

        # Store list of surface points corresponding to each point in MA
        self.correspondences = correspondences
        self.sheet_correspondences: list[list[int]] = [[] for _ in range(len(self.sheet.vertices()))]
        self.__map_sheet_correspondences()

        # Compute average radial basis function
        self.rf = np.zeros(self.inner_points.shape[0])
        self.diffs = np.zeros(self.outer_points.shape)
        self.diff_lens = np.zeros(self.outer_points.shape[0])

        # Compute projections
        self.inner_projections = np.zeros(self.outer_points.shape)
        self.inner_barycentrics = -1 * np.ones((self.outer_points.shape[0], 4))
        self.inner_ts = np.zeros((self.outer_points.shape[0], 3))
        self.__compute_projections()
        # self.update_radial_basis_function()

    def update_correspondences(self, correspondences: list[list[int]]):
        self.correspondences = correspondences
        self.sheet_correspondences = [[] for _ in range(len(self.sheet.vertices()))]
        self.__map_sheet_correspondences()
        self.__compute_projections()

    def __map_sheet_correspondences(self):
        sheet_pos = self.sheet.positions()

        # map each sheet point to corresponding inner points
        kd_tree = KDTree(self.inner_points)
        _, inner_sheet_indices = kd_tree.query(sheet_pos)
        self.sheet_to_inner_index = inner_sheet_indices

        # for each point in medial sheet, store corresponding outer points
        kd_tree = KDTree(sheet_pos)
        _, sheet_inner_indices = kd_tree.query(self.inner_points)
        self.inner_to_sheet_index = sheet_inner_indices

        for inner_idx, outer_points in enumerate(self.correspondences):
            if not self.curve_indices[inner_idx]:
                for p in outer_points:
                    self.sheet_correspondences[sheet_inner_indices[inner_idx]].append(p)

    def update_radial_basis_function(self):
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

            self.rf[i] = avg_len
            self.diffs[corr] = norm_diffs

    def __compute_projections(self):
        # Project relevant outer points to medial sheet
        outer_sheet = flatten(self.correspondences[~self.curve_indices])
        outer_sheet_pos = self.outer_points[outer_sheet]
        face_ids, barycentrics, projected = barycentric_project(self.sheet, outer_sheet_pos)
        # face_ids, barycentrics, projected = barycentric_project_v2(self.sheet, self.sheet_correspondences, self.outer_points)
        # face_ids, barycentrics, projected = face_ids[outer_sheet], barycentrics[outer_sheet], projected[outer_sheet]
        self.inner_projections[outer_sheet] = projected
        self.inner_barycentrics[outer_sheet, 0] = face_ids
        self.inner_barycentrics[outer_sheet, 1:] = barycentrics

        diffs = outer_sheet_pos - projected
        radii = np.linalg.norm(diffs, axis=1)
        norm_diffs = diffs / radii[:, np.newaxis]
        self.diffs[outer_sheet] = norm_diffs
        self.diff_lens[outer_sheet] = radii

        # Project relevant outer points to medial curves
        for curve in self.curves:
            curve = np.array(curve)
            curve_pos = self.inner_points[curve]

            outer_curve = flatten(self.correspondences[curve])
            outer_curve_pos = self.outer_points[outer_curve]

            # project to curve and store corresponding segment and t values
            closest_segment, t_values, projected = project_points_to_curve(outer_curve_pos, curve_pos)
            if len(curve_pos) < 2:
                self.inner_projections[outer_curve] = projected
                self.inner_ts[outer_curve, 0] = curve[closest_segment]
                self.inner_ts[outer_curve, 1] = curve[closest_segment]
                self.inner_ts[outer_curve, 2] = t_values
            else:
                self.inner_projections[outer_curve] = projected
                self.inner_ts[outer_curve, 0] = curve[closest_segment]
                self.inner_ts[outer_curve, 1] = curve[closest_segment+1]
                self.inner_ts[outer_curve, 2] = t_values

            diffs = outer_curve_pos - projected
            radii = np.linalg.norm(diffs, axis=1)
            norm_diffs = diffs / radii[:, np.newaxis]
            self.diffs[outer_curve] = norm_diffs
            self.diff_lens[outer_curve] = radii
