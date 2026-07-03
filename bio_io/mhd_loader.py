import pathlib

import numpy as np
import SimpleITK as sitk
from PIL import Image


AXIS_LABELS = {
    0: "0: Z (axial)",
    1: "1: Y (coronal)",
    2: "2: X (sagittal)",
}


def _resolve_mhd_path(path):
    path = pathlib.Path(path)
    if path.suffix.lower() == ".raw":
        mhd_files = sorted(path.parent.glob("*.mhd"))
        if not mhd_files:
            raise ValueError(f"No .mhd header found for raw file: {path}")
        path = mhd_files[0]
    return path


def load_mhd_volume(path):
    path = _resolve_mhd_path(path)
    image = sitk.ReadImage(str(path))
    volume = sitk.GetArrayFromImage(image).astype(np.float64)
    return volume, image


def save_mhd_volume(volume, reference_image, path):
    output = sitk.GetImageFromArray(np.asarray(volume, dtype=np.float32))
    output.CopyInformation(reference_image)
    sitk.WriteImage(output, str(path))


def load_mhd_slice(path, slice_index=None, axis=0):
    volume, image = load_mhd_volume(path)

    if volume.ndim == 2:
        return volume, image, {
            "is_volume": False,
            "slice_index": 0,
            "shape": volume.shape,
            "slice_shape": volume.shape,
        }

    if slice_index is None:
        slice_index = volume.shape[axis] // 2

    max_index = volume.shape[axis] - 1
    if slice_index < 0 or slice_index > max_index:
        raise ValueError(f"Slice index {slice_index} out of range [0, {max_index}]")

    if axis == 0:
        slice_data = volume[slice_index]
    elif axis == 1:
        slice_data = volume[:, slice_index, :]
    else:
        slice_data = volume[:, :, slice_index]

    return slice_data, image, {
        "is_volume": True,
        "slice_index": slice_index,
        "shape": volume.shape,
        "slice_shape": slice_data.shape,
        "axis": axis,
    }


def normalize_to_uint8(array):
    array = array.astype(np.float64)
    minimum = array.min()
    maximum = array.max()
    if maximum - minimum < 1e-8:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array - minimum) / (maximum - minimum) * 255.0
    return scaled.astype(np.uint8)


def window_to_uint8(array, window_min=None, window_max=None):
    array = array.astype(np.float64)
    if window_min is None or window_max is None:
        return normalize_to_uint8(array)
    if window_max <= window_min:
        window_max = window_min + 1.0
    clipped = np.clip(array, window_min, window_max)
    scaled = (clipped - window_min) / (window_max - window_min) * 255.0
    return scaled.astype(np.uint8)


def paired_window_to_uint8(original, filtered, window_min=None, window_max=None):
    if window_min is None or window_max is None:
        combined = np.concatenate([original.ravel(), filtered.ravel()])
        window_min = float(combined.min())
        window_max = float(combined.max())
        if window_max <= window_min:
            window_max = window_min + 1.0
    return (
        window_to_uint8(original, window_min, window_max),
        window_to_uint8(filtered, window_min, window_max),
        window_min,
        window_max,
    )


def save_slice_png(array, path, window_min=None, window_max=None):
    Image.fromarray(window_to_uint8(array, window_min, window_max)).save(path)
