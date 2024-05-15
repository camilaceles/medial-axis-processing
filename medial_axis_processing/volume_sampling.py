import numpy as np
import nibabel as nib
from time import perf_counter as time
from scipy.interpolate import RegularGridInterpolator
from skimage import measure
import scipy.ndimage


def get_volume_interpolator(file_path: str, downsample_factor: int = 4):
    niiVol = nib.load(file_path)

    affine = niiVol.affine
    imgSpacing = niiVol.header['pixdim'][1:4]

    # Downsample
    affine[0, 0] = affine[0, 0] * downsample_factor
    affine[1, 1] = affine[1, 1] * downsample_factor
    affine[2, 2] = affine[2, 2] * downsample_factor
    imgSpacing = imgSpacing * downsample_factor

    vol = scipy.ndimage.zoom(niiVol.get_fdata().astype('float32'), 1 / downsample_factor, order=2)
    imgDim = vol.shape

    intMethod = 'linear'  # Options: "linear", "nearest", "slinear", "cubic", "quintic" and "pchip"
    expVal = 0.0  # Value for extrapolation (i.e. values outside volume domain)
    x = np.arange(start=0, stop=imgDim[0], step=1) * imgSpacing[0] + affine[0, 3]
    y = np.arange(start=0, stop=imgDim[1], step=1) * imgSpacing[1] + affine[1, 3]
    z = np.arange(start=0, stop=imgDim[2], step=1) * imgSpacing[2] + affine[2, 3]

    F_vol = RegularGridInterpolator((x, y, z), vol, method=intMethod, bounds_error=False, fill_value=expVal)
    return F_vol, affine


def sample_points_intensity(F_vol, vertices, affine):
    # Convert vertices to physical coordinates
    vertices_ft = vertices @ affine[0:3, 0:3] + np.transpose(affine[0:3, 3])
    return F_vol(vertices_ft)
