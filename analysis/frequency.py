import numpy as np


def compute_fshift(image_2d):
    image = np.asarray(image_2d, dtype=np.float64)
    return np.fft.fftshift(np.fft.fft2(image))


def compute_log_magnitude(fshift):
    return 20 * np.log(np.abs(fshift) + 1e-8)


def save_dft_image(log_magnitude, path):
    import matplotlib.image as mpimg

    mpimg.imsave(path, log_magnitude, cmap="gray")


def analyze_noise(fshift, top_n=6, dc_radius_ratio=0.02):
    magnitude = np.abs(fshift)
    height, width = magnitude.shape
    center_y, center_x = height // 2, width // 2
    radius = int(min(height, width) * dc_radius_ratio)

    y_grid, x_grid = np.ogrid[:height, :width]
    distance_from_dc = np.sqrt((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2)
    off_dc_mask = distance_from_dc > radius

    masked_magnitude = magnitude.copy()
    masked_magnitude[~off_dc_mask] = 0

    peak_indices = np.argsort(masked_magnitude.ravel())[::-1][:top_n]
    peaks = []
    for index in peak_indices:
        peak_y, peak_x = np.unravel_index(index, magnitude.shape)
        peak_value = float(masked_magnitude[peak_y, peak_x])
        if peak_value <= 0:
            break
        peaks.append(
            {
                "y": int(peak_y),
                "x": int(peak_x),
                "magnitude": float(magnitude[peak_y, peak_x]),
                "distance_from_dc": float(distance_from_dc[peak_y, peak_x]),
                "symmetric_y": int(2 * center_y - peak_y),
                "symmetric_x": int(2 * center_x - peak_x),
            }
        )

    radial_bins = np.linspace(radius, min(height, width) / 2, num=32)
    radial_profile = []
    for index in range(len(radial_bins) - 1):
        ring_mask = (distance_from_dc >= radial_bins[index]) & (
            distance_from_dc < radial_bins[index + 1]
        )
        if np.any(ring_mask):
            radial_profile.append(
                {
                    "radius": float((radial_bins[index] + radial_bins[index + 1]) / 2),
                    "mean_magnitude": float(magnitude[ring_mask].mean()),
                }
            )

    return {
        "shape": (height, width),
        "dc_magnitude": float(magnitude[center_y, center_x]),
        "mean_off_dc_magnitude": float(magnitude[off_dc_mask].mean()) if np.any(off_dc_mask) else 0.0,
        "max_off_dc_magnitude": float(masked_magnitude.max()),
        "peaks": peaks,
        "radial_profile": radial_profile,
    }


def peaks_to_points(peaks):
    return np.asarray([[peak["x"], peak["y"]] for peak in peaks], dtype=np.float64)


def parse_points(points_text):
    if not points_text:
        raise ValueError("请使用 --points 提供坐标，格式如: 120,80;300,200")

    points = []
    for item in points_text.split(";"):
        item = item.strip()
        if not item:
            continue
        x_text, y_text = item.split(",")
        points.append([float(x_text), float(y_text)])
    if not points:
        raise ValueError("未解析到有效坐标点")
    return np.asarray(points, dtype=np.float64)


def print_noise_report(stats, output_path=None):
    print("=== 频域噪声分析报告 ===")
    print(f"图像尺寸: {stats['shape'][1]} x {stats['shape'][0]}")
    print(f"直流分量幅度: {stats['dc_magnitude']:.4f}")
    print(f"非直流区域平均幅度: {stats['mean_off_dc_magnitude']:.4f}")
    print(f"非直流区域最大幅度: {stats['max_off_dc_magnitude']:.4f}")
    if output_path:
        print(f"频谱图已保存: {output_path}")

    if stats["peaks"]:
        print("\n疑似周期性噪声峰值 (可用于陷波滤波):")
        for index, peak in enumerate(stats["peaks"], start=1):
            print(
                f"  {index}. 坐标(x,y)=({peak['x']}, {peak['y']}), "
                f"幅度={peak['magnitude']:.4f}, "
                f"距中心={peak['distance_from_dc']:.2f}, "
                f"对称点=({peak['symmetric_x']}, {peak['symmetric_y']})"
            )
    else:
        print("\n未检测到明显的离轴噪声峰值。")

    if stats["radial_profile"]:
        print("\n径向平均幅度 (前 5 个频带):")
        for band in stats["radial_profile"][:5]:
            print(
                f"  半径 {band['radius']:.1f}: 平均幅度 {band['mean_magnitude']:.4f}"
            )
