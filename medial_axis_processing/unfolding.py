import igl
import numpy as np
from pygel3d import hmesh
from commons.medial_axis import MedialAxis
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from commons.utils import get_local_basis


def __project_point_to_basis(point, vertex, basis):
    # get point coordinates in local basis
    v_p = point - vertex
    return np.dot(v_p, basis[0]), np.dot(v_p, basis[1]), np.dot(v_p, basis[2])


def __update_point(vertex_new, local_coords, new_basis):
    # retrieve global coordinates from local basis coordinates
    return vertex_new + local_coords[0] * new_basis[0] + local_coords[1] * new_basis[1] + local_coords[2] * new_basis[2]


def __compute_principal_axes(vertices):
    pca = PCA(n_components=3)
    pca.fit(vertices)
    return pca.components_


def __compute_rotation_matrix(src_axes, dst_axes):
    return -1 * R.align_vectors(src_axes, dst_axes)[0].as_matrix()


def least_squares_conformal_map(m: hmesh.Manifold) -> np.ndarray:
    """Applies the igl's LSCM to the given mesh"""
    vertices = m.positions()
    faces = np.array([m.circulate_face(fid) for fid in m.faces()])

    # Fix two points on the boundary
    b = np.array([2, 1])

    bnd = igl.boundary_loop(faces)
    b[0] = bnd[0]
    b[1] = bnd[int(bnd.size / 2)]

    bc = np.array([[0.0, 0.0], [1.0, 0.0]])

    _, uv = igl.lscm(vertices, faces, b, bc)
    return uv


def get_unfolded_sheet_positions(ma: MedialAxis, sheet: hmesh.Manifold = None) -> np.ndarray:
    if sheet is None:
        sheet = ma.sheet
        
    """Returns the 3d coordinates for the unfolded medial axis using LSCM"""
    original_axes = __compute_principal_axes(np.c_[sheet.positions()[:,:2], np.zeros(sheet.positions().shape[0])])
    original_centroid = np.mean(sheet.positions(), axis=0)
    original_centroid[2] = 0
    # original_axes = sheet.positions()[:3, :2] - sheet.positions()[3:6, :2]
    # original_axes = np.c_[original_axes, np.zeros(original_axes.shape[0])]

    ma_areas = np.array([sheet.area(fid) for fid in sheet.faces()])
    ma_area = np.sum(ma_areas)

    uv = least_squares_conformal_map(sheet)
    uv = np.c_[uv, np.zeros(uv.shape[0])]

    # scale uv mapping to approximate original MA area
    uv_mesh = hmesh.Manifold(sheet)
    uv_mesh.positions()[:] = uv
    uv_areas = np.array([uv_mesh.area(fid) for fid in sheet.faces()])
    uv_area = np.sum(uv_areas)

    uv = uv * np.sqrt(ma_area / uv_area)

    # rotate uv mapping to align with original axes
    unfolded_axes = __compute_principal_axes(uv)
    # unfolded_axes = uv[:3] - uv[3:6]
    rotation_matrix = __compute_rotation_matrix(unfolded_axes, original_axes)
    uv[:] = np.dot(uv, rotation_matrix.T)

    # unfolded_centroid = np.mean(uv, axis=0)
    # uv = uv - unfolded_centroid + original_centroid

    return uv


def get_unfolded_medial_axis_positions(ma: MedialAxis) -> np.ndarray:
    unfolded_inner_pos = np.copy(ma.inner_points)

    unfolded_sheet_positions = get_unfolded_sheet_positions(ma)
    unfolded_inner_pos[~ma.curve_indices] = unfolded_sheet_positions[ma.inner_to_sheet_index[~ma.curve_indices]]

    positions = ma.sheet.positions()
    updated_sheet = hmesh.Manifold(ma.sheet)
    updated_sheet.positions()[:] = unfolded_sheet_positions

    # for sheet-curve connection points, carry the sheet update to the curve points.
    # this ensures the connecting they are aligned with the updated sheet positions
    for vid in ma.sheet.vertices():
        neighbor = ma.sheet.circulate_vertex(vid)[0]

        n_old = ma.sheet.vertex_normal(vid)
        basis_old = get_local_basis(positions[vid], positions[neighbor], n_old)

        n_new = updated_sheet.vertex_normal(vid)
        basis_new = get_local_basis(unfolded_sheet_positions[vid], unfolded_sheet_positions[neighbor], n_new)

        inner = ma.sheet_to_inner_index[vid]
        if ma.curve_indices[inner]:
            # find associated curve
            curve_index = next((i for i, curve in enumerate(ma.curves) if curve and curve[0] == inner), -1)
            curve = ma.curves[curve_index]

            # map transformation to curve inner points
            for inner_idx in curve:
                inner_point = ma.inner_points[inner_idx]

                local_coords = __project_point_to_basis(inner_point, positions[vid], basis_old)
                new_pos = __update_point(unfolded_sheet_positions[vid], local_coords, basis_new)
                unfolded_inner_pos[inner_idx] = new_pos

    # project curve to plane z=0
    unfolded_inner_pos[ma.curve_indices, 2] = 0

    return unfolded_inner_pos
