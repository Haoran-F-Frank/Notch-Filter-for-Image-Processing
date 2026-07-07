#!/usr/bin/env python3
"""
End-to-end FoundDiff denoising: NIfTI -> process in memory -> NIfTI.

Intermediate .npy files are NOT required. This script:
  1. reads a .nii/.nii.gz volume (HU values),
  2. resizes each slice to 512x512, applies the official linear HU transform,
  3. runs FoundDiff denoising,
  4. converts back to HU with the inverse linear transform,
  5. writes a new NIfTI and a per-slice stats CSV (min/max/mean).
"""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from ema_pytorch import EMA
from skimage.transform import resize
from tqdm import tqdm

from data import transforms
from src.DADiff import ResidualDiffusion, UnetRes, set_seed

# Official FoundDiff linear transforms (fixed HU window, NOT per-slice min-max).
HU_OFFSET = 1024
HU_MIN = -1000
HU_MAX = 2000
HU_RANGE = HU_MAX - HU_MIN  # 3000


def hu_to_model_norm(hu_slice):
    """HU -> [0, 1] using the same formula as data/transforms.Normalize."""
    m = hu_slice.astype(np.float32) - HU_OFFSET
    return np.clip((m - HU_MIN) / HU_RANGE, 0.0, 1.0)


def model_norm_to_hu(norm_slice):
    """[0, 1] -> HU (inverse of official test post-processing)."""
    return norm_slice.astype(np.float32) * HU_RANGE - 1000.0


def slice_stats(arr, name):
    return {
        'name': name,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
    }


def build_model(checkpoint_path, device):
    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=1,
        condition=True,
        input_condition=False,
        objective='pred_res',
        test_res_or_noise='res',
    )
    diffusion = ResidualDiffusion(
        model,
        image_size=512,
        timesteps=1000,
        sampling_timesteps=2,
        objective='pred_res',
        loss_type='l2',
        condition=True,
        sum_scale=0.01,
        input_condition=False,
        input_condition_mask=False,
        test_res_or_noise='res',
    )

    print(f'Loading FoundDiff checkpoint: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    ema = EMA(diffusion, beta=0.995, update_every=10)
    ema.load_state_dict(ckpt['ema'])
    ema.ema_model.init()
    ema.to(device)
    ema.ema_model.eval()
    print('FoundDiff EMA model loaded.')
    return ema.ema_model


def preprocess_slice(hu_slice, size=512):
    """Resize to model size and build a (1, 1, H, W) tensor."""
    resized = resize(
        hu_slice,
        (size, size),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    arr = np.expand_dims(resized, axis=0)  # (1, H, W) for transforms
    val_transform = transforms.Compose([
        transforms.Normalize(min_value=HU_MIN, max_value=HU_MAX),
        transforms.ToTensor(expand_dims=False),
    ])
    tensor = val_transform(arr)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor.unsqueeze(0), resized.shape


@torch.no_grad()
def denoise_tensor(diffusion, tensor, device):
    x_in = tensor.to(device)
    out = diffusion.sample([x_in], batch_size=1, last=True)
    return out[-1].squeeze(0).squeeze(0).cpu().numpy()


def resize_back(slice_2d, out_shape):
    return resize(
        slice_2d,
        out_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def iter_slices(volume, axis):
    """Yield (index, 2d_slice, insert_axis) for each slice along `axis`."""
    for i in range(volume.shape[axis]):
        sl = np.take(volume, i, axis=axis)
        yield i, sl


def insert_slice(volume, axis, index, slice_2d):
    idx = [slice(None)] * volume.ndim
    idx[axis] = index
    volume[tuple(idx)] = slice_2d


def main():
    parser = argparse.ArgumentParser(
        description='FoundDiff end-to-end NIfTI denoising (no intermediate .npy needed)'
    )
    parser.add_argument('--in_nii', required=True, help='Input .nii or .nii.gz (HU values)')
    parser.add_argument('--out_nii', required=True, help='Output denoised .nii.gz')
    parser.add_argument('--stats_csv', default=None, help='Per-slice stats CSV (default: <out_nii>.stats.csv)')
    parser.add_argument('--checkpoint', default='checkpoints/FoundDiff/sample/model-400.pt')
    parser.add_argument('--axis', type=int, default=2, help='Slice axis in the volume (default: 2)')
    parser.add_argument('--z_start', type=int, default=None, help='First slice index (inclusive)')
    parser.add_argument('--z_end', type=int, default=None, help='Last slice index (exclusive)')
    parser.add_argument('--skip_air', action='store_true', help='Skip slices with mean HU < -900')
    parser.add_argument('--size', type=int, default=512, help='Model input size')
    args = parser.parse_args()

    set_seed(10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    nii_in = nib.load(args.in_nii)
    volume_hu = nii_in.get_fdata().astype(np.float32)
    affine = nii_in.affine.copy()
    print(f'Input volume shape: {volume_hu.shape}, dtype float32 HU')

    z_start = args.z_start if args.z_start is not None else 0
    z_end = args.z_end if args.z_end is not None else volume_hu.shape[args.axis]
    z_end = min(z_end, volume_hu.shape[args.axis])

    out_volume = np.zeros_like(volume_hu, dtype=np.float32)
    diffusion = build_model(args.checkpoint, device)

    stats_path = args.stats_csv or (str(args.out_nii) + '.stats.csv')
    rows = []

    for z in tqdm(range(z_start, z_end), desc='denoising slices'):
        sl_hu = np.take(volume_hu, z, axis=args.axis)
        in_stat = slice_stats(sl_hu, 'input_hu')

        if args.skip_air and in_stat['mean'] < -900:
            insert_slice(out_volume, args.axis, z, sl_hu)
            rows.append({'slice': z, 'skipped': True, **{f'in_{k}': v for k, v in in_stat.items() if k != 'name'}})
            continue

        tensor, _ = preprocess_slice(sl_hu, size=args.size)
        denoised_norm = denoise_tensor(diffusion, tensor, device)
        denoised_hu_512 = model_norm_to_hu(denoised_norm)
        denoised_hu = resize_back(denoised_hu_512, sl_hu.shape)
        insert_slice(out_volume, args.axis, z, denoised_hu)

        out_stat = slice_stats(denoised_hu, 'output_hu')
        rows.append({
            'slice': z,
            'skipped': False,
            'in_min': in_stat['min'],
            'in_max': in_stat['max'],
            'in_mean': in_stat['mean'],
            'out_min': out_stat['min'],
            'out_max': out_stat['max'],
            'out_mean': out_stat['mean'],
            'norm_min': float(denoised_norm.min()),
            'norm_max': float(denoised_norm.max()),
            'norm_mean': float(denoised_norm.mean()),
        })

    # Copy slices outside [z_start, z_end) unchanged
    for z in list(range(0, z_start)) + list(range(z_end, volume_hu.shape[args.axis])):
        insert_slice(out_volume, args.axis, z, np.take(volume_hu, z, axis=args.axis))

    out_path = Path(args.out_nii)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out_volume, affine), str(out_path))

    with open(stats_path, 'w', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f'Saved denoised NIfTI: {out_path}')
    print(f'Saved per-slice stats: {stats_path}')
    print(
        'Linear transforms used:\n'
        f'  HU -> model: norm = clip((hu - {HU_OFFSET} - ({HU_MIN})) / {HU_RANGE}, 0, 1)\n'
        f'  model -> HU: hu = norm * {HU_RANGE} - 1000'
    )


if __name__ == '__main__':
    main()
