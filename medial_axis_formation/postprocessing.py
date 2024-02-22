import trimesh
import numpy as np
import random
from pygel3d import hmesh
from medial_axis_formation.point import PointSet


def __expand_from_triangle(mesh: trimesh.Trimesh, start_face: int, angle_threshold_degrees: float):
    angle_threshold_radians = np.radians(angle_threshold_degrees)

    sheet_faces = {start_face}  # set of faces in same sheet as start_face
    to_visit = [start_face]  # queue of faces to visit

    while to_visit:
        # visit face in front of queue
        current_face = to_visit.pop(0)

        # find indices of adjacent faces
        adjacent_indices = np.where(mesh.face_adjacency[:, 0] == current_face)[0]
        adjacent_indices = np.append(adjacent_indices, np.where(mesh.face_adjacency[:, 1] == current_face)[0])

        for adj_index in adjacent_indices:
            # find adjacent face
            adj_face = mesh.face_adjacency[adj_index][1] \
                if mesh.face_adjacency[adj_index][0] == current_face \
                else mesh.face_adjacency[adj_index][0]

            if adj_face not in sheet_faces:
                # check if adjacent face is in same sheet as currently visited face
                angle = mesh.face_adjacency_angles[adj_index]

                if angle < angle_threshold_radians:
                    # if yes, add it to the sheet set, and to the queue to be visited
                    sheet_faces.add(adj_face)
                    to_visit.append(adj_face)

    return sheet_faces


def __single_sheet(m: hmesh.Manifold, dihedral_angle_threshold: float):
    """Extract a single sheet from double-sheet"""
    faces = np.array([m.circulate_face(fid) for fid in m.faces()])
    trimesh_mesh = trimesh.Trimesh(vertices=m.positions(), faces=faces)

    n_org_faces = len(m.faces())

    sheet_faces = []
    # if a bad start face is chosen, the resulting mesh is only a few triangles,
    # so we try until it results in at least 40% of the original face count
    for i in range(10):
        start_face = random.choice(range(len(trimesh_mesh.faces)))
        sheet_faces = __expand_from_triangle(trimesh_mesh, start_face, dihedral_angle_threshold)

        if len(sheet_faces) > 0.4 * n_org_faces:
            break
        if i == 9:
            print("Couldn't extract a single sheet. Try a bigger `dihedral_angle_threshold`")

    sheet_mesh = trimesh_mesh.submesh([np.array(list(sheet_faces))], append=True)

    sheet = hmesh.Manifold.from_triangles(sheet_mesh.vertices, sheet_mesh.faces)
    return sheet


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
