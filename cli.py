#!/usr/bin/env python3
import argparse
import pathlib
import sys

import numpy as np

from analysis.frequency import (
    analyze_noise,
    compute_fshift,
    compute_log_magnitude,
    parse_points,
    peaks_to_points,
    print_noise_report,
    save_dft_image,
)
from filters.notch_filters import ButterworthNotchFilter, GaussianNotchFilter, IdealNotchFilter
from bio_io.bio_images import load_image, save_slice_png


def _ensure_tmp_dir():
    pathlib.Path("tmp").mkdir(exist_ok=True)


def _load_slice(args):
    slice_index = args.slice
    image, info = load_image(args.input, slice_index=slice_index, axis=args.axis)
    if info["is_volume"]:
        print(
            f"已加载体数据 {info['volume_shape']}，使用切片 {info['slice_index']} (axis={args.axis})"
        )
    else:
        print(f"已加载 2D 图像: {args.input}")
    return image, info


def command_freq(args):
    _ensure_tmp_dir()
    image, _ = _load_slice(args)
    fshift = compute_fshift(image)
    log_magnitude = compute_log_magnitude(fshift)
    output_path = pathlib.Path(args.output)
    save_dft_image(log_magnitude, output_path)

    stats = analyze_noise(fshift, top_n=args.top_peaks)
    print_noise_report(stats, output_path=output_path)

    if args.show:
        import matplotlib.pyplot as plt

        plt.imshow(log_magnitude, cmap="gray")
        plt.title("Frequency Spectrum")
        plt.show()


def _apply_filter(filter_name, fshift, points, radius, order):
    fshift = fshift.copy()
    if filter_name == "ideal":
        return IdealNotchFilter().apply_filter(fshift, points, radius)
    if filter_name == "butterworth":
        return ButterworthNotchFilter().apply_filter(fshift, points, radius, order=order)
    if filter_name == "gaussian":
        return GaussianNotchFilter().apply_filter(fshift, points, radius)
    raise ValueError(f"未知滤波器类型: {filter_name}")


def command_denoise(args):
    _ensure_tmp_dir()
    image, info = _load_slice(args)
    fshift = compute_fshift(image)

    if args.auto_peaks:
        stats = analyze_noise(fshift, top_n=args.auto_peaks)
        if not stats["peaks"]:
            raise SystemExit("未检测到可用于自动陷波的噪声峰值。")
        points = peaks_to_points(stats["peaks"])
        print(f"自动选取 {len(points)} 个陷波点:")
        for index, point in enumerate(points, start=1):
            print(f"  {index}. x={point[0]:.1f}, y={point[1]:.1f}")
    else:
        points = parse_points(args.points)

    filtered = _apply_filter(args.filter, fshift, points, args.radius, args.order)
    output_path = pathlib.Path(args.output)
    save_slice_png(filtered, output_path)
    print(f"去噪结果已保存: {output_path}")
    print(f"源文件: {info['source_path']}")

    if args.save_dft:
        filtered_fshift = compute_fshift(filtered)
        dft_path = pathlib.Path(args.save_dft)
        save_dft_image(compute_log_magnitude(filtered_fshift), dft_path)
        print(f"滤波后频谱图已保存: {dft_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="生物医学图像陷波滤波与频域噪声分析工具"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freq_parser = subparsers.add_parser("freq", help="在终端分析频域噪声")
    freq_parser.add_argument("input", help="输入图像路径 (.nii.gz / .mhd / .raw / png 等)")
    freq_parser.add_argument("--slice", type=int, default=None, help="3D 体数据切片索引")
    freq_parser.add_argument("--axis", type=int, default=0, choices=[0, 1, 2], help="切片轴")
    freq_parser.add_argument("--output", default="tmp/dft.png", help="频谱图输出路径")
    freq_parser.add_argument("--top-peaks", type=int, default=6, help="输出的疑似噪声峰值数量")
    freq_parser.add_argument("--show", action="store_true", help="显示频谱图窗口")
    freq_parser.set_defaults(func=command_freq)

    denoise_parser = subparsers.add_parser("denoise", help="对图像应用陷波滤波去噪")
    denoise_parser.add_argument("input", help="输入图像路径")
    denoise_parser.add_argument("--slice", type=int, default=None, help="3D 体数据切片索引")
    denoise_parser.add_argument("--axis", type=int, default=0, choices=[0, 1, 2], help="切片轴")
    denoise_parser.add_argument(
        "--filter",
        choices=["ideal", "butterworth", "gaussian"],
        default="butterworth",
        help="陷波滤波器类型",
    )
    denoise_parser.add_argument(
        "--points",
        help='陷波中心坐标，格式: "x1,y1;x2,y2"（与 GUI 中点击坐标一致）',
    )
    denoise_parser.add_argument(
        "--auto-peaks",
        type=int,
        help="自动从频域峰值选取陷波点数量（与 --points 二选一）",
    )
    denoise_parser.add_argument("--radius", type=float, default=121.0, help="陷波半径")
    denoise_parser.add_argument("--order", type=int, default=1, help="Butterworth 滤波器阶数")
    denoise_parser.add_argument("--output", default="tmp/filtered_img.png", help="输出图像路径")
    denoise_parser.add_argument("--save-dft", help="可选，保存滤波后频谱图路径")
    denoise_parser.set_defaults(func=command_denoise)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "denoise" and not args.points and not args.auto_peaks:
        parser.error("denoise 命令需要 --points 或 --auto-peaks 之一")

    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
