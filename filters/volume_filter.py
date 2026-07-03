import numpy as np

from filters.notch_core import apply_filter_specs, compute_fshift


def filter_volume_slice(slice_2d, filter_specs):
    fshift = compute_fshift(slice_2d)
    filtered, _ = apply_filter_specs(fshift, filter_specs)
    return filtered


def apply_filters_to_volume(volume, filter_specs, axis=0, progress_callback=None):
    volume = np.asarray(volume, dtype=np.float64)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    filtered = np.empty_like(volume)
    if axis == 0:
        total = volume.shape[0]
        for index in range(total):
            filtered[index] = filter_volume_slice(volume[index], filter_specs)
            if progress_callback is not None:
                progress_callback(index + 1, total)
    elif axis == 1:
        total = volume.shape[1]
        for index in range(total):
            filtered[:, index, :] = filter_volume_slice(volume[:, index, :], filter_specs)
            if progress_callback is not None:
                progress_callback(index + 1, total)
    else:
        total = volume.shape[2]
        for index in range(total):
            filtered[:, :, index] = filter_volume_slice(volume[:, :, index], filter_specs)
            if progress_callback is not None:
                progress_callback(index + 1, total)
    return filtered
