import trimesh
import numpy as np
import random
from pygel3d import hmesh
from commons.point import PointSet
from collections import deque
from commons.utils import trimesh_to_manifold


def __precompute_face_adjacencies(mesh):
    adjacency_dict = {}
    for adj_index, faces in enumerate(mesh.face_adjacency):
        for face in faces:
            if face not in adjacency_dict:
                adjacency_dict[face] = []
            adjacency_dict[face].append((adj_index, set(faces) - {face}))
    return adjacency_dict


def __expand_from_triangle(mesh: trimesh.Trimesh, start_face: int, angle_threshold_degrees: float):
    angle_threshold_radians = np.radians(angle_threshold_degrees)

    sheet_faces = {start_face}
    to_visit = deque([start_face])

    adjacency_dict = __precompute_face_adjacencies(mesh)

    while to_visit:
        current_face = to_visit.popleft()

        if current_face in adjacency_dict:
            for adj_index, other_faces in adjacency_dict[current_face]:
                adj_face = next(iter(other_faces))

                if adj_face not in sheet_faces:
                    angle = mesh.face_adjacency_angles[adj_index]

                    if angle < angle_threshold_radians:
                        sheet_faces.add(adj_face)
                        to_visit.append(adj_face)

    return sheet_faces


def __single_sheet(m: hmesh.Manifold, dihedral_angle_threshold: float):
    """Extract a single sheet from double-sheet"""
    faces = np.array([m.circulate_face(fid) for fid in m.faces()])
    trimesh_mesh = trimesh.Trimesh(vertices=m.positions(), faces=faces)

    sheet_faces = single_sheet_faces(trimesh_mesh, dihedral_angle_threshold)
    sheet_mesh = trimesh_mesh.submesh([np.array(sheet_faces)], append=True)

    sheet = trimesh_to_manifold(sheet_mesh)
    return sheet


def single_sheet_faces(trim: trimesh.Trimesh, dihedral_angle_threshold: float):
    n_org_faces = len(trim.faces)

    sheet_faces = []
    # if a bad start face is chosen, the resulting mesh is only a few triangles,
    # so we try until it results in at least 30% of the original face count
    for i in range(20):
        start_face = random.choice(range(len(trim.faces)))
        sheet_faces = __expand_from_triangle(trim, start_face, dihedral_angle_threshold)

        if len(sheet_faces) > 0.3 * n_org_faces:
            break
        if i == 19:
            print("Couldn't extract a single sheet. Try a bigger `dihedral_angle_threshold`")

    return list(sheet_faces)


def to_medial_sheet(
        m: hmesh.Manifold,
        inner_points: PointSet,
        dihedral_angle_threshold: float
) -> hmesh.Manifold:
    inner_mesh = hmesh.Manifold(m)
    pos = inner_mesh.positions()
    pos[:] = inner_points.positions

    # extract a single sheet from it and project remaining inner points to closest in the sheet
    single_sheet = __single_sheet(inner_mesh, dihedral_angle_threshold)
    return single_sheet
