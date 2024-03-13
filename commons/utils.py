import trimesh
from pygel3d import hmesh
import numpy as np
import pickle
from commons.point import PointSet


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


def manifold_to_trimesh(m: hmesh.Manifold)-> trimesh.Trimesh:
    faces = np.array([m.circulate_face(fid) for fid in m.faces()])
    trim = trimesh.Trimesh(vertices=m.positions(), faces=faces, process=False)
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


def save_pointset_to_file(point_set: PointSet, file_path: str) -> None:
    with open(file_path, 'wb') as file:
        pickle.dump(point_set, file)


def load_pointset_from_file(file_path: str) -> PointSet:
    with open(file_path, 'rb') as file:
        point_set = pickle.load(file)
    return point_set

