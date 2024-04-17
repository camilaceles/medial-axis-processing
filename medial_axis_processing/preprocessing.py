import trimesh
import numpy as np
import random
from pygel3d import hmesh
from collections import deque
from commons.utils import trimesh_to_manifold, manifold_to_trimesh


def __precompute_face_adjacencies(mesh):
    adjacency_dict = {}
    for adj_index, faces in enumerate(mesh.face_adjacency):
        for face in faces:
            if face not in adjacency_dict:
                adjacency_dict[face] = []
            adjacency_dict[face].append((adj_index, set(faces) - {face}))
    return adjacency_dict


def fix_normals(m: hmesh.Manifold) -> hmesh.Manifold:
    mesh = manifold_to_trimesh(m, process=True)

    start_face = random.choice(range(len(mesh.faces)))
    visited = {start_face}
    to_visit = deque([start_face])

    adjacency_dict = __precompute_face_adjacencies(mesh)

    while to_visit:
        current_face = to_visit.popleft()

        if current_face in adjacency_dict:
            for adj_index, other_faces in adjacency_dict[current_face]:
                adj_face = next(iter(other_faces))

                if adj_face not in visited:
                    angle = mesh.face_adjacency_angles[adj_index]
                    if angle > np.pi / 2:
                        mesh.faces[adj_face] = mesh.faces[adj_face][[1, 0, 2]]

                    visited.add(adj_face)
                    to_visit.append(adj_face)
    mesh.fix_normals()

    return trimesh_to_manifold(mesh)



