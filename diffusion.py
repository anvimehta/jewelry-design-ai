"""DDPM math: cosine noise schedule, forward diffusion, DDIM sampler, and EMA.

Forward diffusion adds Gaussian noise according to a *cosine schedule* — a smoother
ramp than the original linear schedule, important for small datasets because it
keeps useful signal at the highest timesteps instead of obliterating it.

Reverse sampling uses DDIM (Song et al. 2021), which is a deterministic non-Markovian
sampler that lets us trade fewer denoising steps (~50-250) for marginal quality loss
relative to the full T-step DDPM ancestral sampler.

Sample quality at inference uses an exponential moving average of the model's weights;
this is standard practice for diffusion and important for sharpness with our limited
training set.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class GaussianDiffusion:
    """Cosine-schedule DDPM with a DDIM sampler. Operates on data in [-1, 1]."""

    def __init__(self, T: int = 1000, device: torch.device = torch.device('cpu')):
        self.T = T
        # Nichol & Dhariwal cosine schedule: alphas_cumprod = cos^2((t/T + s)/(1+s) * pi/2).
        s = 0.008
        x = torch.linspace(0, T, T + 1)
        acp = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        acp = acp / acp[0]
        betas = (1 - acp[1:] / acp[:-1]).clamp(1e-4, 0.999)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_acp = alphas_cumprod.sqrt()
        self.sqrt_one_minus_acp = (1 - alphas_cumprod).sqrt()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(acp_t) * x0 + sqrt(1 - acp_t) * noise."""
        return (self.sqrt_acp[t][:, None, None, None] * x0
                + self.sqrt_one_minus_acp[t][:, None, None, None] * noise)

    @torch.no_grad()
    def ddim_sample(self, model: nn.Module, n: int, image_size: int,
                    steps: int = 250, eta: float = 0.0,
                    device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Deterministic DDIM sampler (eta=0). Returns images in [-1, 1]."""
        model.eval()
        ts = torch.linspace(0, self.T - 1, steps, dtype=torch.long, device=device).flip(0)
        x = torch.randn(n, 3, image_size, image_size, device=device)
        for i, t in enumerate(ts):
            t_batch = torch.full((n,), int(t), dtype=torch.long, device=device)
            eps = model(x, t_batch)
            acp_t = self.alphas_cumprod[t]
            x0_pred = ((x - (1 - acp_t).sqrt() * eps) / acp_t.sqrt()).clamp(-1.0, 1.0)
            if i < len(ts) - 1:
                t_prev = ts[i + 1]
                acp_prev = self.alphas_cumprod[t_prev]
                sigma = eta * ((1 - acp_prev) / (1 - acp_t)).sqrt() * (1 - acp_t / acp_prev).sqrt()
                dir_xt = (1 - acp_prev - sigma ** 2).clamp(min=0).sqrt() * eps
                x = acp_prev.sqrt() * x0_pred + dir_xt + sigma * torch.randn_like(x)
            else:
                x = x0_pred
        return x


class EMA:
    """Exponential moving average of model parameters. Sampling uses EMA weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)
