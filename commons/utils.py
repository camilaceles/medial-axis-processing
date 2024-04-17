import trimesh
from pygel3d import hmesh
import numpy as np
from scipy.spatial import KDTree


def smooth(m, max_iter=1):
    pos = m.positions()
    for i in range(0, max_iter):
        new_pos = np.zeros(pos.shape)
        for vertex in m.vertices():
            adjacent_vertices = m.circulate_vertex(vertex, 'v')
            for adj in adjacent_vertices:
                new_pos[vertex] += pos[adj]
            new_pos[vertex] /= len(adjacent_vertices)
        pos[:] = new_pos[:]

    # remove zero area faces after smooth
    for fid in m.faces():
        if m.area(fid) < 1e-6:
            m.remove_face(fid)
    m.cleanup()


def __sample_point_in_face(m: hmesh.Manifold, fid: int):
    vertices = m.circulate_face(fid, mode='v')
    vertices_pos = m.positions()[vertices]
    s, t = sorted([np.random.random(), np.random.random()])
    return s * vertices_pos[0] + (t - s) * vertices_pos[1] + (1 - t) * vertices_pos[2]


def poisson_disk_sampling_on_mesh(m, n):
    face_areas = np.array([m.area(fid) for fid in m.faces()])
    face_probs = face_areas / face_areas.sum()
    total_area = face_areas.sum()
    threshold = np.sqrt(total_area / (np.pi * n))

    face_normals = np.array([m.face_normal(fid) for fid in m.faces()])

    sampled_points = np.zeros((0, 3))  # Initialize an empty array for points
    normals = np.zeros((0, 3))  # Initialize an empty array for normals

    # Optimization - build an array of random indices according to the face probabilities
    random_indices = np.random.choice(len(face_probs), size=10*n, p=face_probs)
    for drawn_fid in random_indices:
        new_point = __sample_point_in_face(m, drawn_fid)
        new_normal = face_normals[drawn_fid]

        # Check if this new point is far enough from all other points using broadcasting
        if sampled_points.size == 0 or np.all(np.linalg.norm(sampled_points - new_point, axis=1) >= threshold):
            sampled_points = np.vstack([sampled_points, new_point])
            normals = np.vstack([normals, new_normal])
            if len(sampled_points) == n:
                break

    return sampled_points, normals


def manifold_to_trimesh(m: hmesh.Manifold, process=False) -> trimesh.Trimesh:
    faces = np.array([m.circulate_face(fid) for fid in m.faces()])
    trim = trimesh.Trimesh(vertices=m.positions(), faces=faces, process=process)
    return trim


def trimesh_to_manifold(trim: trimesh.Trimesh) -> hmesh.Manifold:
    return hmesh.Manifold.from_triangles(trim.vertices, trim.faces)


def barycentric_project(m: hmesh.Manifold, points: np.ndarray):
    trim = manifold_to_trimesh(m)
    prox_query = trimesh.proximity.ProximityQuery(trim)
    _, _, face_ids = prox_query.on_surface(points)

    triangles = trim.triangles[face_ids]
    barycentrics = trimesh.triangles.points_to_barycentric(triangles, points)

    return face_ids, barycentrics


def flatten(xss):
    return [x for xs in xss for x in xs]


def read_obj(filename: str) -> np.ndarray:
    vertices = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
    return np.array(vertices)


def read_correspondences(filename: str) -> list[list[int]]:
    with open(filename) as f:
        correspondences = [[int(x) for x in line.strip().split(',')] for line in f]
    return correspondences


def read_CA_MA(filename: str) -> hmesh.Manifold:
    mesh = hmesh.load(filename)
    trim = manifold_to_trimesh(mesh, process=True)
    mesh = trimesh_to_manifold(trim)
    mesh.cleanup()
    return trimesh_to_manifold(trim)


def read_ma(file_path):
    vertices = []
    edges = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif line.startswith('e ') or line.startswith('l '):
                parts = line.strip().split()
                edges.append([int(parts[1]), int(parts[2])])

            elif line.startswith('f '):
                parts = line.strip().split()
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])

    return np.array(vertices), edges, faces


def read_ma_ca(file_path):
    vertices = []
    edges = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif line.startswith('e ') or line.startswith('l '):
                parts = line.strip().split()
                edges.append([int(parts[1])-1, int(parts[2])-1])

            elif line.startswith('f '):
                parts = line.strip().split()
                faces.append([int(parts[1])-1, int(parts[2])-1, int(parts[3])-1])

    return np.array(vertices), edges, faces


def find_minimum_gamma(mesh, inner_points, start=0.01, step=0.01):
    """Finds minimum addition to radii to cover all points in the mesh with given inner points"""
    tree = KDTree(mesh.positions())
    total_points = len(mesh.positions())
    addition = start
    covered_points = set()

    while len(covered_points) < total_points:
        R = tree.query(inner_points, k=1)[0] + addition
        for i, radius in enumerate(R):
            indices = tree.query_ball_point(inner_points[i], radius)
            covered_points.update(indices)

        if len(covered_points) >= total_points:
            break
        addition += step

    return addition


def __build_opposite_dict(nested_lists):
    opposite_dict = {}
    for outer_index, inner_list in enumerate(nested_lists):
        for element in inner_list:
            if element in opposite_dict:
                opposite_dict[element].append(outer_index)
            else:
                opposite_dict[element] = [outer_index]
    return opposite_dict


def build_ball_correspondences(mesh: hmesh.Manifold, inner_points: np.ndarray, gamma: float = None):
    # Find minimum gamma to cover all surface points and map correspondences from it
    if gamma is None:
        gamma = find_minimum_gamma(mesh, inner_points)
        print("Chosen minimum gamma:", gamma)

    tree = KDTree(mesh.positions())
    dist, _ = tree.query(inner_points, k=1)
    R = dist + gamma
    correspondences = tree.query_ball_point(inner_points, R)

    # Ensure each outer point is only associated to one inner point
    # Choose inner point where inner-outer connection is best aligned with surface normal
    pos = mesh.positions()

    for (outer, inners) in __build_opposite_dict(correspondences).items():
        if len(inners) == 1:
            continue
        best_inner = None
        best_angle = np.inf

        # find the inner point with the best normal alignment
        for inner in inners:
            # calculate normal alignment
            outer_normal = mesh.vertex_normal(outer)
            outer_normal /= np.linalg.norm(outer_normal)
            corr_normal = pos[outer] - inner_points[inner]
            corr_normal /= np.linalg.norm(corr_normal)
            angle = np.arccos(np.clip(np.dot(outer_normal, corr_normal), -1.0, 1.0))

            if angle < best_angle:
                best_angle = angle
                best_inner = inner

        # remove non optimal correspondences
        for inner, corr in enumerate(correspondences):
            if outer in corr and inner != best_inner:
                corr.remove(outer)

    return correspondences
