#!/usr/bin/env python3
"""
End-to-end FoundDiff denoising: NIfTI -> process in memory -> NIfTI.

Intermediate .npy files are NOT required.
Typical speed on GPU: ~20-30 sec/slice (batch_size=1). Use --batch_size 2 or 4 if VRAM allows.
"""

import argparse
import csv
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from ema_pytorch import EMA
from skimage.transform import resize
from tqdm import tqdm

from data import transforms
from src.DADiff import ResidualDiffusion, UnetRes, set_seed

HU_OFFSET = 1024
HU_MIN = -1000
HU_MAX = 2000
HU_RANGE = HU_MAX - HU_MIN  # 3000

VAL_TRANSFORM = transforms.Compose([
    transforms.Normalize(min_value=HU_MIN, max_value=HU_MAX),
    transforms.ToTensor(expand_dims=False),
])


def model_norm_to_hu(norm_slice):
    return norm_slice.astype(np.float32) * HU_RANGE - 1000.0


def slice_stats(arr):
    return {
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


def maybe_resize(hu_slice, size=512):
    if hu_slice.shape == (size, size):
        return hu_slice.astype(np.float32)
    return resize(
        hu_slice,
        (size, size),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def preprocess_slice(hu_slice, size=512):
    resized = maybe_resize(hu_slice, size=size)
    arr = np.expand_dims(resized, axis=0)
    tensor = VAL_TRANSFORM(arr)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def preprocess_batch(slices, size=512):
    tensors = [preprocess_slice(sl, size=size) for sl in slices]
    return torch.stack(tensors, dim=0)


@torch.no_grad()
def denoise_batch(diffusion, tensor, device):
    x_in = tensor.to(device, non_blocking=True)
    out = diffusion.sample([x_in], batch_size=x_in.shape[0], last=True)
    result = out[-1].cpu().numpy()
    if result.ndim == 4:
        return result[:, 0]
    return result


def resize_back(slice_2d, out_shape):
    if slice_2d.shape == out_shape:
        return slice_2d.astype(np.float32)
    return resize(
        slice_2d,
        out_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def insert_slice(volume, axis, index, slice_2d):
    idx = [slice(None)] * volume.ndim
    idx[axis] = index
    volume[tuple(idx)] = slice_2d


def warmup_gpu(diffusion, device, size=512):
    dummy = torch.zeros(1, 1, size, size)
    denoise_batch(diffusion, dummy, device)
    if device.type == 'cuda':
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description='FoundDiff end-to-end NIfTI denoising (no intermediate .npy needed)'
    )
    parser.add_argument('--in_nii', required=True, help='Input .nii or .nii.gz (HU values)')
    parser.add_argument('--out_nii', required=True, help='Output denoised .nii.gz')
    parser.add_argument('--stats_csv', default=None, help='Per-slice stats CSV')
    parser.add_argument('--checkpoint', default='checkpoints/FoundDiff/sample/model-400.pt')
    parser.add_argument('--axis', type=int, default=2, help='Slice axis (default: 2)')
    parser.add_argument('--z_start', type=int, default=None, help='First slice index (inclusive)')
    parser.add_argument('--z_end', type=int, default=None, help='Last slice index (exclusive)')
    parser.add_argument('--skip_air', action='store_true', help='Skip slices with mean HU < -900')
    parser.add_argument('--size', type=int, default=512, help='Model input size')
    parser.add_argument('--batch_size', type=int, default=1, help='Slices per GPU forward pass (try 2 or 4)')
    args = parser.parse_args()

    if not Path(args.in_nii).exists():
        raise FileNotFoundError(
            f'Input not found: {args.in_nii}\n'
            'Use your real path, e.g. data/mydata/APNHC00002_CT.nii.gz'
        )

    set_seed(10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f'Using device: {device}, batch_size={args.batch_size}')

    nii_in = nib.load(args.in_nii)
    volume_hu = nii_in.get_fdata().astype(np.float32)
    affine = nii_in.affine.copy()
    print(f'Input volume shape: {volume_hu.shape}, dtype float32 HU')

    z_start = args.z_start if args.z_start is not None else 0
    z_end = args.z_end if args.z_end is not None else volume_hu.shape[args.axis]
    z_end = min(z_end, volume_hu.shape[args.axis])
    num_slices = z_end - z_start
    print(f'Processing slices [{z_start}, {z_end}) = {num_slices} slices')

    out_volume = np.zeros_like(volume_hu, dtype=np.float32)
    diffusion = build_model(args.checkpoint, device)
    if device.type == 'cuda':
        print('GPU warmup...')
        warmup_gpu(diffusion, device, size=args.size)

    stats_path = args.stats_csv or (str(args.out_nii) + '.stats.csv')
    rows = []
    t0 = time.time()
    processed = 0

    z_indices = list(range(z_start, z_end))
    pbar = tqdm(total=num_slices, desc='denoising slices', unit='slice')

    batch_z = []
    batch_sl = []

    def flush_batch():
        nonlocal processed
        if not batch_z:
            return
        tensor = preprocess_batch(batch_sl, size=args.size)
        denoised_norm = denoise_batch(diffusion, tensor, device)
        for i, z in enumerate(batch_z):
            sl_hu = batch_sl[i]
            in_stat = slice_stats(sl_hu)
            denoised_hu = resize_back(model_norm_to_hu(denoised_norm[i]), sl_hu.shape)
            insert_slice(out_volume, args.axis, z, denoised_hu)
            out_stat = slice_stats(denoised_hu)
            rows.append({
                'slice': z,
                'skipped': False,
                'in_min': in_stat['min'],
                'in_max': in_stat['max'],
                'in_mean': in_stat['mean'],
                'out_min': out_stat['min'],
                'out_max': out_stat['max'],
                'out_mean': out_stat['mean'],
                'norm_min': float(denoised_norm[i].min()),
                'norm_max': float(denoised_norm[i].max()),
                'norm_mean': float(denoised_norm[i].mean()),
            })
            processed += 1
            pbar.update(1)
        batch_z.clear()
        batch_sl.clear()

    for z in z_indices:
        sl_hu = np.take(volume_hu, z, axis=args.axis)
        in_stat = slice_stats(sl_hu)

        if args.skip_air and in_stat['mean'] < -900:
            insert_slice(out_volume, args.axis, z, sl_hu)
            rows.append({
                'slice': z, 'skipped': True,
                'in_min': in_stat['min'], 'in_max': in_stat['max'], 'in_mean': in_stat['mean'],
            })
            pbar.update(1)
            continue

        batch_z.append(z)
        batch_sl.append(sl_hu)
        if len(batch_z) >= args.batch_size:
            flush_batch()

    flush_batch()
    pbar.close()

    for z in list(range(0, z_start)) + list(range(z_end, volume_hu.shape[args.axis])):
        insert_slice(out_volume, args.axis, z, np.take(volume_hu, z, axis=args.axis))

    elapsed = time.time() - t0
    if processed > 0:
        sec_per = elapsed / processed
        print(
            f'Denoised {processed} slices in {elapsed/60:.1f} min '
            f'({sec_per:.1f} s/slice). '
            f'Full volume estimate ({volume_hu.shape[args.axis]} slices): '
            f'{sec_per * volume_hu.shape[args.axis] / 3600:.1f} hours'
        )

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


if __name__ == '__main__':
    main()
