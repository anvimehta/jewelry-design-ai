"""Generate a sample grid from a saved DDPM-64 checkpoint.

Run:
    python sample.py --ckpt checkpoints/ddpm_unet_b64_ep3000.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffusion import GaussianDiffusion
from models import UNet
from utils import pick_device, save_grid


def sample(args: argparse.Namespace) -> None:
    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    # Notebook-era checkpoints store config under 'cfg'; train.py stores it under 'args'.
    cfg = ckpt.get('args') or ckpt['cfg']

    diffusion = GaussianDiffusion(T=cfg['timesteps'], device=device)
    ema_model = UNet(base=cfg['base_channels']).to(device)
    ema_model.load_state_dict(ckpt['ema_state_dict'])
    ema_model.eval()

    samples = diffusion.ddim_sample(ema_model, n=args.n_samples,
                                    image_size=cfg['image_size'],
                                    steps=args.sample_steps, device=device)
    out_path = Path(args.out_dir) / f'samples_{Path(args.ckpt).stem}.png'
    save_grid((samples + 1) / 2, out_path, nrow=8)
    print(f'saved {args.n_samples} samples: {out_path}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out-dir', default='outputs')
    p.add_argument('--n-samples', type=int, default=32)
    p.add_argument('--sample-steps', type=int, default=250)
    return p.parse_args()


if __name__ == '__main__':
    sample(parse_args())
