import numpy as np
from numpy import ndarray as array
from scipy.spatial import KDTree


class Point:
    def __init__(self, point_set: 'PointSet', index: int):
        self.point_set: 'PointSet' = point_set
        self.index = index

    @property
    def pos(self) -> array:
        return self.point_set.positions[self.index]

    @pos.setter
    def pos(self, value):
        self.point_set.positions[self.index] = value

    @property
    def normal(self) -> array:
        return self.point_set.normals[self.index]

    @normal.setter
    def normal(self, value):
        self.point_set.normals[self.index] = value

    @property
    def neighborhood(self) -> array:
        return self.point_set.neighborhoods[self.index]

    @neighborhood.setter
    def neighborhood(self, value):
        self.point_set.neighborhoods[self.index] = value

    @property
    def is_fixed(self) -> bool:
        return self.point_set.is_fixed[self.index]

    @is_fixed.setter
    def is_fixed(self, value):
        self.point_set.is_fixed[self.index] = value

    @property
    def skel_radius(self) -> float:
        return self.point_set.skel_radius[self.index]

    @skel_radius.setter
    def skel_radius(self, value):
        self.point_set.skel_radius[self.index] = value

    @property
    def principal_axis(self) -> array:
        return self.point_set.principal_axis[self.index]

    @principal_axis.setter
    def principal_axis(self, value):
        self.point_set.principal_axis[self.index] = value

    @property
    def semi_axis_lengths(self) -> array:
        return self.point_set.semi_axis_lengths[self.index]

    @semi_axis_lengths.setter
    def semi_axis_lengths(self, value):
        self.point_set.semi_axis_lengths[self.index] = value

    @property
    def eigen_confidence(self) -> array:
        return self.point_set.eigen_confidence[self.index]

    @eigen_confidence.setter
    def eigen_confidence(self, value):
        self.point_set.eigen_confidence[self.index] = value

    @property
    def average(self) -> array:
        return self.point_set.average[self.index]

    @average.setter
    def average(self, value):
        self.point_set.average[self.index] = value

    @property
    def average_weight_sum(self) -> float:
        return self.point_set.average_weight_sum[self.index]

    @average_weight_sum.setter
    def average_weight_sum(self, value):
        self.point_set.average_weight_sum[self.index] = value

    @property
    def repulsion(self) -> array:
        return self.point_set.repulsion[self.index]

    @repulsion.setter
    def repulsion(self, value):
        self.point_set.repulsion[self.index] = value

    @property
    def repulsion_weight_sum(self) -> float:
        return self.point_set.repulsion_weight_sum[self.index]

    @repulsion_weight_sum.setter
    def repulsion_weight_sum(self, value):
        self.point_set.repulsion_weight_sum[self.index] = value

    @property
    def front_point(self) -> int:
        return self.point_set.front_point[self.index]

    @front_point.setter
    def front_point(self, value):
        self.point_set.front_point[self.index] = value

    @property
    def back_point(self) -> int:
        return self.point_set.back_point[self.index]

    @back_point.setter
    def back_point(self, value):
        self.point_set.back_point[self.index] = value

    @property
    def graph_index(self) -> int:
        return self.point_set.graph_index[self.index]

    @graph_index.setter
    def graph_index(self, value):
        self.point_set.graph_index[self.index] = value

    @property
    def is_connection(self) -> bool:
        return self.point_set.is_connection[self.index]

    @is_connection.setter
    def is_connection(self, value):
        self.point_set.is_connection[self.index] = value


class PointSet:
    class _PointSetIterator:
        def __init__(self, point_set):
            self._point_set = point_set
            self._index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._index < self._point_set.N:
                result = self._point_set[self._index]
                self._index += 1
                return result
            else:
                raise StopIteration

    def __init__(self, positions: array, normals: array):
        self.N: int = positions.shape[0]
        self.positions: array = np.copy(positions)
        self.normals: array = np.copy(normals) / np.linalg.norm(normals, axis=1, keepdims=True)

        self.neighborhoods: list[array] = [np.empty(0)] * self.N
        self.is_fixed: array = np.zeros(self.N, dtype=bool)
        self.skel_radius: array = np.zeros(self.N)

        self.principal_axis: array = np.zeros((self.N, 3, 3))
        self.semi_axis_lengths: array = np.zeros((self.N, 3))
        self.eigen_confidence: array = np.zeros(self.N)

        self.average: array = np.zeros((self.N, 3))
        self.average_weight_sum: array = np.zeros(self.N)
        self.repulsion: array = np.zeros((self.N, 3))
        self.repulsion_weight_sum: array = np.zeros(self.N)

        self.graph_index: array = np.ones(self.N, dtype=int) * -1
        self.is_connection: array = np.ones(self.N, dtype=int) * -1
        self.front_point: array = np.ones(self.N, dtype=int) * -1
        self.back_point: array = np.ones(self.N, dtype=int) * -1

    def __getitem__(self, index):
        return Point(self, index)

    def __iter__(self):
        return PointSet._PointSetIterator(self)

    def update_ball_neighborhood(self, neighbourhood_scale: float) -> None:
        kd = KDTree(self.positions)
        kd2 = KDTree(self.positions)

        # for each point, find points within `neighbourhood_scale` distance of it
        neighborhoods = kd.query_ball_tree(kd2, neighbourhood_scale)

        # remove point itself from neighbourhood and update it
        for i, neighborhood in enumerate(neighborhoods):
            neighborhood.remove(i)
            self.neighborhoods[i] = np.array(neighborhood, dtype=int)

    def update_knn_neighborhood(self, n: int) -> None:
        kd = KDTree(self.positions)

        for i, point in enumerate(self.positions):
            _, neighbors = kd.query(point, k=list(range(2, n+3)))
            self.neighborhoods[i] = np.array(neighbors, dtype=int)

    def angles_to_neighbors(self):
        angles = []
        for i, point in enumerate(self):
            neighborhood = self.neighborhoods[i]
            neighbor_normals = self.normals[neighborhood]
            dot_product = np.dot(neighbor_normals, point.normal)
            angles.append(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        return angles

    def get_average_sparsity(self) -> float:
        # Computes the average distance to the closest neighbor
        kd = KDTree(self.positions)
        acc = 0
        for point in self.positions:
            ds, _ = kd.query(point, k=[2])
            acc += ds[0]
        return acc / self.N
