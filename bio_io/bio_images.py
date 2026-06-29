import pathlib

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_BIO_EXTENSIONS = {".nii", ".nii.gz", ".mhd", ".mha", ".raw"}
STANDARD_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _path_suffixes(path):
    path = pathlib.Path(path)
    joined = "".join(suffix.lower() for suffix in path.suffixes)
    if joined.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def is_bio_format(path):
    return _path_suffixes(path) in SUPPORTED_BIO_EXTENSIONS


def is_standard_image(path):
    return _path_suffixes(path) in STANDARD_IMAGE_EXTENSIONS


def load_volume(path):
    import SimpleITK as sitk

    path = pathlib.Path(path)
    if path.suffix.lower() == ".raw":
        mhd_candidates = sorted(path.parent.glob("*.mhd"))
        if not mhd_candidates:
            raise ValueError(
                f".raw 文件需要同目录下的 .mhd 头文件；未在 {path.parent} 中找到。"
            )
        path = mhd_candidates[0]

    image = sitk.ReadImage(str(path))
    volume = sitk.GetArrayFromImage(image).astype(np.float64)
    metadata = {
        "path": str(path),
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
        "size": image.GetSize(),
        "dimension": image.GetDimension(),
    }
    return volume, image, metadata


def get_slice(volume, slice_index=None, axis=0):
    if volume.ndim == 2:
        return volume.astype(np.float64)

    if slice_index is None:
        slice_index = volume.shape[axis] // 2

    max_index = volume.shape[axis] - 1
    if slice_index < 0 or slice_index > max_index:
        raise ValueError(f"切片索引 {slice_index} 超出范围 [0, {max_index}]")

    if axis == 0:
        return volume[slice_index].astype(np.float64)
    if axis == 1:
        return volume[:, slice_index, :].astype(np.float64)
    return volume[:, :, slice_index].astype(np.float64)


def normalize_to_uint8(array):
    array = array.astype(np.float64)
    minimum = array.min()
    maximum = array.max()
    if maximum - minimum < 1e-8:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array - minimum) / (maximum - minimum) * 255.0
    return scaled.astype(np.uint8)


def save_slice_png(array, path):
    Image.fromarray(normalize_to_uint8(array)).save(path)


def load_standard_image(path):
    return np.asarray(ImageOps.grayscale(Image.open(path)), dtype=np.float64)


def load_image(path, slice_index=None, axis=0):
    path = pathlib.Path(path)
    if is_bio_format(path):
        volume, _, metadata = load_volume(path)
        slice_data = get_slice(volume, slice_index=slice_index, axis=axis)
        return slice_data, {
            "is_volume": volume.ndim == 3,
            "volume_shape": volume.shape,
            "slice_index": slice_index if slice_index is not None else volume.shape[axis] // 2,
            "axis": axis,
            "metadata": metadata,
            "source_path": str(path),
        }
    if is_standard_image(path):
        return load_standard_image(path), {
            "is_volume": False,
            "volume_shape": None,
            "slice_index": None,
            "axis": axis,
            "metadata": None,
            "source_path": str(path),
        }
    raise ValueError(f"不支持的图像格式: {path}")


def save_volume(array, reference_image, path):
    import SimpleITK as sitk

    output = sitk.GetImageFromArray(array.astype(np.float32))
    output.CopyInformation(reference_image)
    sitk.WriteImage(output, str(path))
