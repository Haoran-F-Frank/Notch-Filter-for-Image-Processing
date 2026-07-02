from dataclasses import dataclass

import numpy as np


FILTER_TYPE_BUTTERWORTH = 0
FILTER_TYPE_GAUSSIAN = 1
FILTER_TYPE_LABELS = {
    FILTER_TYPE_BUTTERWORTH: "Butterworth (0)",
    FILTER_TYPE_GAUSSIAN: "Gaussian (1)",
}


@dataclass
class NotchFilterSpec:
    filter_type: int = FILTER_TYPE_BUTTERWORTH
    x: float = 64.0
    y: float = 64.0
    radius: float = 30.0
    order: int = 2
    enabled: bool = True
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = "Filter"

    @property
    def type_label(self):
        return FILTER_TYPE_LABELS.get(self.filter_type, str(self.filter_type))

    def summary(self):
        return (
            f"{self.name}: {self.type_label}, "
            f"({self.x:.1f},{self.y:.1f}), r={self.radius:.1f}, order={self.order}"
        )


def compute_fshift(image_2d):
    image = np.asarray(image_2d, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D slice, got shape {image.shape}")
    fshift = np.fft.fftshift(np.fft.fft2(image))
    if fshift.shape != image.shape:
        raise ValueError(
            f"Frequency domain shape {fshift.shape} does not match image shape {image.shape}"
        )
    return fshift


def compute_log_magnitude(fshift):
    spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    if spectrum.shape != fshift.shape:
        raise ValueError("Spectrum shape must match frequency-domain input shape")
    return spectrum


def build_notch_mask(height, width, point_x, point_y, radius, filter_type, order=1):
    u0 = point_y
    v0 = point_x
    rows = np.arange(height, dtype=np.float64)[:, None]
    cols = np.arange(width, dtype=np.float64)[None, :]

    d1 = np.maximum(np.sqrt((rows - u0) ** 2 + (cols - v0) ** 2), 1e-8)
    d2 = np.maximum(np.sqrt((rows + u0) ** 2 + (cols + v0) ** 2), 1e-8)

    if filter_type == FILTER_TYPE_BUTTERWORTH:
        return 1.0 / (1.0 + (radius * radius / (d1 * d2)) ** order)
    if filter_type == FILTER_TYPE_GAUSSIAN:
        return 1.0 - np.exp(-0.5 * (d1 * d2 / (radius * radius)))
    raise ValueError(f"Unknown filter type: {filter_type}")


def apply_filter_specs(fshift_base, filter_specs):
    fshift_base = np.asarray(fshift_base)
    if fshift_base.ndim != 2:
        raise ValueError(f"Expected 2D frequency data, got shape {fshift_base.shape}")

    fshift = np.array(fshift_base, copy=True)
    height, width = fshift.shape

    for spec in filter_specs:
        if not spec.enabled:
            continue
        mask = build_notch_mask(
            height,
            width,
            spec.x,
            spec.y,
            spec.radius,
            spec.filter_type,
            order=spec.order,
        )
        if mask.shape != (height, width):
            raise ValueError("Notch mask shape must match the input slice shape")
        fshift *= mask

    reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
    if reconstructed.shape != fshift_base.shape:
        raise ValueError(
            f"Filtered image shape {reconstructed.shape} does not match input shape {fshift_base.shape}"
        )
    return reconstructed, fshift
