import numpy as np
from pygel3d import hmesh
import trimesh
from trimesh.voxel.creation import voxelize
from commons.utils import trimesh_to_manifold


def voxel_remesh(m: hmesh.Manifold, voxel_size: float):
    # Convert hmesh to trimesh
    faces = np.array([m.circulate_face(fid) for fid in m.faces()])
    trimesh_mesh = trimesh.Trimesh(vertices=m.positions(), faces=faces)

    # Voxelize the mesh
    voxel_grid = voxelize(trimesh_mesh, pitch=voxel_size)
    remeshed = voxel_grid.marching_cubes

    # Scale and translate the remeshed mesh to match the original mesh
    original_size = trimesh_mesh.bounds[1] - trimesh_mesh.bounds[0]
    remeshed_size = remeshed.bounds[1] - remeshed.bounds[0]
    scale_factors = original_size / remeshed_size
    remeshed.apply_scale(np.min(scale_factors))

    original_min = trimesh_mesh.bounds[0]
    remeshed_min = remeshed.bounds[0]
    translation_vector = original_min - remeshed_min
    remeshed.apply_translation(translation_vector)

    # Take only the largest connected component
    connected_components = list(remeshed.split(only_watertight=False))
    connected_components.sort(key=lambda x: len(x.faces), reverse=True)
    largest_component = connected_components[0]

    # Convert back to original mesh format if needed
    voxelized = trimesh_to_manifold(largest_component)
    return voxelized

