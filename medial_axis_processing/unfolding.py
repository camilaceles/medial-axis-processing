import igl
import numpy as np
from pygel3d import hmesh
from medial_axis_processing.medial_axis import MedialAxis
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA


def __compute_principal_axes(vertices):
    pca = PCA(n_components=3)
    pca.fit(vertices)
    return pca.components_


def __compute_rotation_matrix(src_axes, dst_axes):
    return -1 * R.align_vectors(src_axes, dst_axes)[0].as_matrix()


def __least_squares_conformal_map(m: hmesh.Manifold) -> np.ndarray:
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


def get_unfolded_sheet_positions(ma: MedialAxis) -> np.ndarray:
    """Returns the 3d coordinates for the unfolded medial axis using LSCM"""
    original_axes = __compute_principal_axes(np.c_[ma.sheet.positions()[:,:2], np.zeros(ma.sheet.positions().shape[0])])
    ma_areas = np.array([ma.sheet.area(fid) for fid in ma.sheet.faces()])
    ma_area = np.sum(ma_areas)

    uv = __least_squares_conformal_map(ma.sheet)
    uv = np.c_[uv, np.zeros(uv.shape[0])]

    # rotate uv mapping to align with original axes
    unfolded_axes = __compute_principal_axes(uv)
    rotation_matrix = __compute_rotation_matrix(unfolded_axes, original_axes)
    uv[:] = np.dot(uv, rotation_matrix.T)

    # compute area of uv mapped mesh
    uv_mesh = hmesh.Manifold(ma.sheet)
    uv_mesh.positions()[:] = uv
    uv_areas = np.array([uv_mesh.area(fid) for fid in ma.sheet.faces()])
    uv_area = np.sum(uv_areas)

    # scale uv mapping to approximate original MA area
    return uv * np.sqrt(ma_area / uv_area)


def get_unfolded_curve_positions(
        ma: MedialAxis,
        original_medial_sheet: np.ndarray,
        unfolded_medial_sheet: np.ndarray
) -> list[np.ndarray]:
    unfolded_curve_positions = []
    for curve in ma.curves:
        medial_connection = ma.inner_indices[curve[0]]
        old_connection_pos = original_medial_sheet[medial_connection]
        new_connection_pos = unfolded_medial_sheet[medial_connection]

        new_curve_pos = np.copy(ma.inner_points[curve])
        new_curve_pos += new_connection_pos - old_connection_pos
        new_curve_pos[:, 2] = 0  # project to z=0 plane
        unfolded_curve_positions.append(new_curve_pos)

    return unfolded_curve_positions
