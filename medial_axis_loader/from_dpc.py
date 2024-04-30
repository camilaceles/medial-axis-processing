import pickle
from pygel3d import hmesh, graph
from commons.medial_axis import MedialAxis


def load(input_mesh: hmesh.Manifold, filename: str) -> MedialAxis:
    """Loads MedialAxis object from a file outputted by DPC.
        File is simply an object with the required attributes.
        See `dpc-medial-axis` repo for how this is obtained.
    """
    with open(filename, 'rb') as file:
        ma = pickle.load(file)

    medial_sheet = hmesh.Manifold.from_triangles(ma["medial_sheet_vertices"], ma["medial_sheet_faces"])
    vertices = ma["vertices"]
    curves = ma["medial_curves"]
    sheet_faces = ma["medial_sheet_faces"]

    g = graph.Graph()
    for v in vertices:
        g.add_node(v)
    for c in curves:
        for i in range(len(c) - 1):
            g.connect_nodes(c[i], c[i+1])
    for f in sheet_faces:
        for i in range(3):
            g.connect_nodes(f[i], f[(i+1) % 3])

    return MedialAxis(input_mesh, vertices, medial_sheet, curves, ma["correspondences"], g)
