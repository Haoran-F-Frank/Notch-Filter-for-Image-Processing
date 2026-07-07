#!/usr/bin/env python3
"""Run FoundDiff denoising on a single .npy CT slice."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ema_pytorch import EMA

from data import transforms
from src.DADiff import ResidualDiffusion, UnetRes, set_seed


def build_model(checkpoint_path, device):
    """Load EMA weights and prepare the model for inference."""
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

    # init() rebuilds diffusion schedule buffers on CPU; move to GPU after init.
    ema.ema_model.init()
    ema.to(device)
    ema.ema_model.eval()
    print('FoundDiff EMA model loaded.')
    return ema.ema_model


def load_npy_slice(npy_path):
    """Load and preprocess a CT slice using the same transforms as validation."""
    arr = np.load(npy_path).astype(np.float32)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)

    val_transform = transforms.Compose([
        transforms.Normalize(min_value=-1000, max_value=2000),
        transforms.ToTensor(expand_dims=False),
    ])
    tensor = val_transform(arr)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor.unsqueeze(0)


@torch.no_grad()
def denoise_one(diffusion, x, device):
    x_in = x.to(device)
    out = diffusion.sample([x_in], batch_size=1, last=True)
    return out[-1].squeeze(0).squeeze(0).cpu().numpy()


def save_comparison(input_hu, denoised_norm, out_png):
  # Exact inverse of transforms.Normalize (NOT norm*3000-1000 which is off by 1024 HU)
    denoised_hu = denoised_norm * 3000.0 + 24.0
    vmin, vmax = -160, 240

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(input_hu, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('Input (HU)')
    axes[0].axis('off')

    axes[1].imshow(denoised_hu, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title('Denoised (HU)')
    axes[1].axis('off')

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='FoundDiff single-slice denoising demo')
    parser.add_argument('--npy', required=True, help='Input .npy slice (HU values, 512x512)')
    parser.add_argument('--out_npy', required=True, help='Output denoised .npy path')
    parser.add_argument('--out_png', required=True, help='Output side-by-side comparison PNG')
    parser.add_argument(
        '--checkpoint',
        default='checkpoints/FoundDiff/sample/model-400.pt',
        help='FoundDiff checkpoint path',
    )
    args = parser.parse_args()

    set_seed(10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    input_hu = np.load(args.npy).astype(np.float32)
    print(
        f'Input stats: shape={input_hu.shape}, '
        f'min={input_hu.min():.4f}, max={input_hu.max():.4f}, mean={input_hu.mean():.4f}'
    )
    if input_hu.mean() < -900:
        print(
            'WARNING: input mean is near -1000 HU (mostly air). '
            'Pick a slice with visible anatomy (mean roughly -800 to -200).'
        )

    diffusion = build_model(args.checkpoint, device)
    x = load_npy_slice(args.npy)
    denoised_norm = denoise_one(diffusion, x, device)

    out_npy = Path(args.out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, denoised_norm.astype(np.float32))

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    save_comparison(input_hu, denoised_norm, out_png)

    print(
        f'Output stats: min={denoised_norm.min():.6f}, '
        f'max={denoised_norm.max():.6f}, mean={denoised_norm.mean():.6f}'
    )
    print(f'Saved denoised npy: {out_npy}')
    print(f'Saved comparison png: {out_png}')


if __name__ == '__main__':
    main()
