import numpy as np
from commons.point import PointSet

N_ITERS_SINKING = 300
params: dict = {}


def __sink_one_step(inner_points: PointSet, r: float, check_collision: bool = True) -> None:
    step_size_scale = 0.1
    step_size = r * step_size_scale

    if not check_collision:
        inner_points.positions -= inner_points.normals * step_size
        return

    inner_points.update_ball_neighborhood(r * params["sigma_q"])

    # Check for collisions (angle is bigger than omega) - reached the medial axis
    angles = inner_points.angles_to_neighbors()
    collision_indices = np.array([
        point.index for point, angle in zip(inner_points, angles)
        if np.any(angle > params["omega"])
    ])
    if np.any(collision_indices):
        inner_points.is_fixed[collision_indices] = True

    # Update point positions
    non_fixed_indices = ~inner_points.is_fixed
    inner_points.positions[non_fixed_indices] -= inner_points.normals[non_fixed_indices] * step_size


def __normal_smooth(outer_points: PointSet, inner_points: PointSet, radius: float) -> None:
    omega = params["omega"]
    for i, p in enumerate(outer_points):
        inner_point = inner_points[i]
        normal_weight_sum = 0
        normal_sum = np.zeros(3)
        neighbors = [outer_points[p2] for p2 in p.neighborhood]
        for neighbor in neighbors:
            diff = p.pos - neighbor.pos
            dist_squared = np.dot(diff, diff)

            n_dot = np.dot(p.normal, neighbor.normal)
            cos = max(1e-8, 1 - np.cos(omega))
            psi = np.exp(-((1-n_dot) / cos)*((1-n_dot) / cos))
            theta = np.exp(-dist_squared * 4 / (radius*radius))
            rep = max(psi*theta, 1e-10)

            normal_weight_sum += rep
            normal_sum += neighbor.normal * rep

        if normal_weight_sum > 1e-6 and len(p.neighborhood) > 0:
            new_normal = normal_sum / normal_weight_sum
            p.normal = new_normal / np.linalg.norm(new_normal)
            inner_point.normal = p.normal


def __back_project_non_fixed(outer_points: PointSet, inner_points: PointSet) -> None:
    for it in range(10):
        not_fixed = np.invert(np.array([q.is_fixed for q in inner_points]))
        not_fixed_count = np.count_nonzero(not_fixed)
        if not_fixed_count < 1:
            break

        need_change_index = []
        for i in range(outer_points.N):
            p = outer_points[i]
            q = inner_points[i]

            if q.is_fixed:
                continue

            min_length = -1
            min_dist = 1000000
            for neighbor in p.neighborhood:
                inner_neighbor = inner_points[neighbor]
                outer_neighbor = outer_points[neighbor]

                if not inner_neighbor.is_fixed:
                    continue

                dist = np.linalg.norm(inner_neighbor.pos - outer_neighbor.pos)
                if dist < min_dist:
                    min_dist = dist
                    min_length = outer_neighbor.skel_radius

            q.pos = p.pos - p.normal * min_length
            p.skel_radius = min_length
            need_change_index.append(i)

        for i in need_change_index:
            q = inner_points[i]
            q.is_fixed = True


def __smooth(outer_points: PointSet, inner_points: PointSet):
    # Update skel radius
    diff = outer_points.positions - inner_points.positions
    outer_points.skel_radius = np.linalg.norm(diff, axis=1)

    new_radius = np.zeros(outer_points.N)
    for i, neighbors in enumerate(outer_points.neighborhoods):
        fixed_neighbors = inner_points.is_fixed[neighbors]
        relevant_skel_radius = outer_points.skel_radius[neighbors][fixed_neighbors]
        if len(relevant_skel_radius) > 0:
            new_radius[i] = np.mean(relevant_skel_radius)
        else:
            new_radius[i] = outer_points.skel_radius[i]

    # Update the positions of inner_points based on new_radius
    not_fixed = ~inner_points.is_fixed
    v = (inner_points.positions - outer_points.positions)[not_fixed]
    v_unit = v / np.linalg.norm(v, axis=1, keepdims=True)
    inner_points.positions[not_fixed] = outer_points.positions[not_fixed] + v_unit * new_radius[not_fixed][:, np.newaxis]
    outer_points.skel_radius[not_fixed] = new_radius[not_fixed]


def sink_inner_points(outer_points: PointSet, inner_points: PointSet, parameters: dict) -> None:
    global params
    params = parameters

    r = outer_points.get_average_sparsity()
    radius = r * parameters["sigma_p"]
    outer_points.update_ball_neighborhood(radius)

    print("Running normal smoothing...")
    __normal_smooth(outer_points, inner_points, radius)

    print("Running sinking of inner points...")
    for i in range(10):  # perform 10 sinking steps without collision check
        __sink_one_step(inner_points, r, check_collision=False)

    for i in range(N_ITERS_SINKING):
        __sink_one_step(inner_points, r, check_collision=True)

    if params["run_sinking_smoothing"]:
        __smooth(outer_points, inner_points)
        __smooth(outer_points, inner_points)

        # __back_project_non_fixed(outer_points, inner_points)
        #
        # __smooth(outer_points, inner_points)
        # __smooth(outer_points, inner_points)

