import pickle
from pygel3d import hmesh
from commons.medial_axis import MedialAxis


def load(input_mesh: hmesh.Manifold, filename: str) -> MedialAxis:
    """Loads MedialAxis object from a file outputted by DPC.
        File is simply an object with the required attributes.
        See `dpc-medial-axis` repo for how this is obtained.
    """
    with open(filename, 'rb') as file:
        ma = pickle.load(file)

    medial_sheet = hmesh.Manifold.from_triangles(ma["medial_sheet_vertices"], ma["medial_sheet_faces"])

    return MedialAxis(input_mesh, ma["vertices"], medial_sheet, ma["medial_curves"], ma["correspondences"])
