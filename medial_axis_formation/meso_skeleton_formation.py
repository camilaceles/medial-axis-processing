from commons.point import *


def __run_PCA(inner_points: PointSet, radius: float) -> None:
    radius_squared = radius * radius

    for q in inner_points:
        neighborhood_indices = q.neighborhood
        if len(neighborhood_indices) < 4:
            q.eigen_confidence = 0.5
            continue

        neighbors = inner_points.positions[neighborhood_indices]
        diffs = neighbors - q.pos
        dists_squared = np.sum(diffs ** 2, axis=1)
        thetas = np.exp(-dists_squared / radius_squared)[:, np.newaxis]

        # Vectorized computation of covariance matrix
        cov_matrix = diffs.T @ (diffs * thetas)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(-eigenvalues)

        # Normalize eigenvalues and store results
        normalized_eigenvalues = eigenvalues[sorted_indices] / np.sum(eigenvalues)
        q.principal_axis = eigenvectors[:, sorted_indices].T
        q.semi_axis_lengths = normalized_eigenvalues
        q.eigen_confidence = normalized_eigenvalues[0]


def __compute_eigen_directions(outer_points: PointSet, inner_points: PointSet, radius: float):
    inner_points.update_ball_neighborhood(radius)
    __run_PCA(inner_points, radius)

    for q in inner_points:
        dirs = [outer_points.positions[q.index] - q.pos]
        neighbor_positions = outer_points.positions[q.neighborhood]
        neighbor_dirs = neighbor_positions - q.pos
        dirs.extend(neighbor_dirs)

        dirs = np.array(dirs)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        abs_projections = np.abs(dirs @ q.principal_axis.T)
        mean_projections = np.mean(abs_projections, axis=0)

        sum_projs = np.sum(mean_projections)
        eigenvalues = (sum_projs - mean_projections) / sum_projs
        eigenvalues /= np.sum(eigenvalues)

        eigenvalues = np.power(eigenvalues, 8)
        eigenvalues /= np.sum(eigenvalues)

        q.semi_axis_lengths = eigenvalues


def __compute_eigen_neighborhood(outer_points: PointSet, inner_points: PointSet, radius: float):
    __compute_eigen_directions(outer_points, inner_points, radius)

    eigen_para_1 = 0.5
    eigen_para_2 = 3.0

    principal_axes_scaled = inner_points.principal_axis * (radius * eigen_para_1)
    semi_axes_scaled = inner_points.principal_axis * (inner_points.semi_axis_lengths[:, np.newaxis, :] * radius * eigen_para_2)
    total_diff = principal_axes_scaled + semi_axes_scaled
    ranges = np.linalg.norm(total_diff, axis=2)

    for q in inner_points:
        q_range = ranges[q.index]

        diffs = inner_points.positions[q.neighborhood] - q.pos
        proj = np.abs(diffs @ q.principal_axis.T)

        valid_neighbors = (proj[:, 0] < q_range[0]) & (proj[:, 1] < q_range[1]) & (proj[:, 2] < q_range[2])
        filtered_neighborhood = q.neighborhood[valid_neighbors]
        q.neighborhood = filtered_neighborhood


def __average_term(outer_points: PointSet, inner_points: PointSet, radius: float):
    inner_points.average[:, :] = 0
    inner_points.average_weight_sum[:] = 0

    for q1 in inner_points:
        # neighbors = [inner_points[q2] for q2 in outer_points[q1.index].neighborhood]
        neighbors = [inner_points[q2] for q2 in q1.neighborhood]
        for neighbor in neighbors:
            diff = neighbor.pos - q1.pos
            dist_squared = np.dot(diff, diff)
            w = np.exp(-dist_squared * 4 / (radius * radius))

            q1.average += neighbor.pos * w
            q1.average_weight_sum += w


def __repulsion_term(inner_points: PointSet, radius: float):
    inner_points.repulsion[:, :] = 0
    inner_points.repulsion_weight_sum[:] = 0

    for q1 in inner_points:
        neighbors = [inner_points[q2] for q2 in q1.neighborhood]
        for neighbor in neighbors:
            diff = q1.pos - neighbor.pos
            dist = max(np.linalg.norm(diff), 0.001 * radius)

            ellipsoid_dists = np.dot(q1.principal_axis, diff) / q1.semi_axis_lengths
            ellipsoid_dist_square = np.sum(ellipsoid_dists ** 2)

            w = np.exp(-ellipsoid_dist_square * 4 / (radius * radius))
            repulsion_weight = w * ((1 / dist) ** 5)

            q1.repulsion += np.dot(q1.principal_axis, diff) * repulsion_weight
            q1.repulsion_weight_sum += repulsion_weight


def __fixed_point_update(outer_points: PointSet, inner_points: PointSet, radius: float) -> None:
    mu = 0.8
    threshold = 0.35

    __compute_eigen_neighborhood(outer_points, inner_points, radius)
    __average_term(outer_points, inner_points, radius)

    __run_PCA(inner_points, radius)
    __repulsion_term(inner_points, radius)

    for q1 in inner_points:
        if q1.average_weight_sum > 1e-20:
            q1.pos = q1.average / q1.average_weight_sum

        if q1.repulsion_weight_sum > 1e-20:
            repulsion_dir = np.zeros(3)
            for i in range(3):
                if q1.semi_axis_lengths[i] > threshold:
                    repulsion_dir += q1.semi_axis_lengths[i] * np.dot(q1.principal_axis[i], q1.repulsion[i])
            q1.pos += (mu / q1.repulsion_weight_sum) * repulsion_dir


def __regularize_curve_points(q1: Point, inner_points: PointSet, curve_regularization_threshold: float) -> None:
    if q1.eigen_confidence < curve_regularization_threshold:
        q1.is_fixed = True
        return

    neighbors = [inner_points[q2] for q2 in q1.neighborhood]

    proj_lengths = np.array([np.dot(q1.principal_axis[0], q2.pos - q1.pos) for q2 in neighbors])
    pos_lengths_idx = np.where(proj_lengths > 0)[0]
    neg_lengths_idx = np.where(proj_lengths <= 0)[0]

    if len(pos_lengths_idx) == 0 and len(neg_lengths_idx) == 0:
        q1.is_fixed = True
        return
    if len(pos_lengths_idx) == 0 or len(neg_lengths_idx) == 0:
        return

    front_nearest_idx = pos_lengths_idx[proj_lengths[pos_lengths_idx].argmin()]
    front_nearest_pos = neighbors[front_nearest_idx].pos

    back_nearest_idx = neg_lengths_idx[proj_lengths[neg_lengths_idx].argmax()]
    back_nearest_pos = neighbors[back_nearest_idx].pos

    q1.front_point = neighbors[front_nearest_idx].index
    q1.back_point = neighbors[back_nearest_idx].index

    q1.pos = (front_nearest_pos + back_nearest_pos) / 2.0


def regularize_curve_points(outer_points: PointSet, inner_points: PointSet, params: dict) -> None:
    r = outer_points.get_average_sparsity()
    radius = r * params["sigma_s"]
    curve_regularization_threshold = params["curve_regularization_threshold"]

    print("Running curve point regularization...")
    for i in range(2):
        inner_points.update_ball_neighborhood(radius)
        __run_PCA(inner_points, radius)
        inner_points.update_knn_neighborhood(16)

        for q in inner_points:
            __regularize_curve_points(q, inner_points, curve_regularization_threshold)


def form_meso_skeleton(outer_points: PointSet, inner_points: PointSet, params: dict) -> None:
    r = outer_points.get_average_sparsity()

    radius = r * params["sigma_s"]

    for q in inner_points:
        q.is_fixed = False

    print("Running skeleton formation...")
    for i in range(3):
        # old_pos = np.copy(inner_points.positions)
        __fixed_point_update(outer_points, inner_points, radius)
        # res = np.linalg.norm(old_pos - inner_points.positions, axis=1)
        # print(f"iteration {i}, avg residual {np.sum(res) / inner_points.N:.03g}")
