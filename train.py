"""Train the DDPM-64 necklace generator and emit a sample grid at the end.

Training objective: predict the Gaussian noise added at a random timestep,
weighted by min-SNR-gamma (Hang et al. 2023) so low-noise timesteps don't dominate
gradients. EMA-averaged weights are saved alongside the live weights.

Run:
    python train.py --epochs 3000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import NecklaceDataset
from diffusion import EMA, GaussianDiffusion
from models import UNet
from utils import pick_device, save_grid, set_seed


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = pick_device()
    print(f'device: {device}')

    ds = NecklaceDataset(args.data_dir, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)
    print(f'training images: {len(ds)}  |  batches/epoch: {len(loader)}')

    model = UNet(base=args.base_channels).to(device)
    diffusion = GaussianDiffusion(T=args.timesteps, device=device)
    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'UNet params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, n = 0.0, 0
        for x in loader:
            x = x.to(device) * 2 - 1                                 # [0,1] -> [-1,1]
            t = torch.randint(0, diffusion.T, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)
            eps_pred = model(x_t, t)
            # Min-SNR-gamma loss weighting for epsilon-prediction.
            snr = diffusion.alphas_cumprod[t] / (1 - diffusion.alphas_cumprod[t])
            weight = torch.clamp(snr, max=args.min_snr_gamma) / snr
            per_sample = F.mse_loss(eps_pred, noise, reduction='none').mean(dim=(1, 2, 3))
            loss = (weight * per_sample).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            ep_loss += loss.item() * x.size(0); n += x.size(0)
        losses.append(ep_loss / max(n, 1))
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f'epoch {epoch:5d}  loss {losses[-1]:.4f}')

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'ddpm_unet_b{args.base_channels}_ep{args.epochs}.pt'
    torch.save({
        'state_dict': model.state_dict(),
        'ema_state_dict': ema.shadow,
        'args': vars(args),
        'losses': losses,
    }, ckpt_path)
    print(f'saved checkpoint: {ckpt_path}')

    # Final sample grid using EMA weights.
    ema_model = UNet(base=args.base_channels).to(device)
    ema.copy_to(ema_model)
    samples = diffusion.ddim_sample(ema_model, n=args.n_samples,
                                    image_size=args.image_size,
                                    steps=args.sample_steps, device=device)
    out_path = Path(args.out_dir) / f'samples_ep{args.epochs}.png'
    save_grid((samples + 1) / 2, out_path, nrow=8)
    print(f'saved {args.n_samples} samples: {out_path}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='data/necklaces')
    p.add_argument('--ckpt-dir', default='checkpoints')
    p.add_argument('--out-dir', default='outputs')
    p.add_argument('--image-size', type=int, default=64)
    p.add_argument('--base-channels', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--epochs', type=int, default=3000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--timesteps', type=int, default=1000)
    p.add_argument('--ema-decay', type=float, default=0.999)
    p.add_argument('--min-snr-gamma', type=float, default=5.0)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--n-samples', type=int, default=32)
    p.add_argument('--sample-steps', type=int, default=250)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
