from pygel3d import hmesh
from medial_axis_processing.medial_axis import MedialAxis


def apply_inverse_medial_axis_transform(original_mesh: hmesh.Manifold, medial_axis: MedialAxis):
    # TODO this goes back to before, input positions just need to be updated
    original_mesh.positions()[medial_axis.sheet_indices] = medial_axis.outer_points.positions[medial_axis.sheet_indices]
