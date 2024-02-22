import numpy as np
from pygel3d import hmesh
import trimesh
from trimesh.voxel.creation import voxelize


def voxel_remesh(m: hmesh.Manifold, voxel_size: float):
    faces = np.array([m.circulate_face(fid) for fid in m.faces()])
    trimesh_mesh = trimesh.Trimesh(vertices=m.positions(), faces=faces)

    # voxelize mesh
    voxel_grid = voxelize(trimesh_mesh, pitch=voxel_size)
    remeshed = voxel_grid.marching_cubes

    # translate back to original position
    original_center = trimesh_mesh.bounding_box.centroid
    remeshed_center = remeshed.bounding_box.centroid

    translation_vector = original_center - remeshed_center
    remeshed.apply_translation(translation_vector)

    # take only largest connected component
    connected_components = remeshed.split(only_watertight=False)
    largest_component = connected_components[0]

    voxelized = hmesh.Manifold.from_triangles(largest_component.vertices, largest_component.faces)
    return voxelized
