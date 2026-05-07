"""UNet architecture for the 64x64 DDPM.

Standard small UNet recipe: sinusoidal timestep embedding -> MLP, three downsampling
stages with GroupNorm + SiLU residual blocks, a self-attention layer at the 8x8
bottleneck, and a mirrored decoder with skip connections.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding of integer timesteps -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv, with timestep-conditioning added in the middle."""

    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over spatial positions of a feature map."""

    def __init__(self, ch: int, heads: int = 4, groups: int = 8):
        super().__init__()
        assert ch % heads == 0
        self.heads = heads
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv.unbind(dim=1)
        scale = (c // self.heads) ** -0.5
        attn = torch.softmax((q.transpose(-2, -1) @ k) * scale, dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    """64x64 RGB UNet. ~11M params at base=64.

    Layout (input is the noised image x_t and the timestep t):
        in_conv:  3      ->   base    @ 64x64
        d1:       base   ->   base    @ 64x64
        down1:    base   ->   base*2  @ 32x32
        d2:       base*2 ->   base*2  @ 32x32
        down2:    base*2 ->   base*4  @ 16x16
        d3:       base*4 ->   base*4  @ 16x16
        down3:    base*4 ->   base*4  @ 8x8
        m1:       base*4 ->   base*4  @ 8x8
        attn:     self-attention      @ 8x8
        m2:       base*4 ->   base*4  @ 8x8
        up3 + u3: base*4 ->   base*4  @ 16x16  (skip from d3)
        up2 + u2: base*4 ->   base*2  @ 32x32  (skip from d2)
        up1 + u1: base*2 ->   base    @ 64x64  (skip from d1)
        out_conv: base   ->   3       @ 64x64

    Output is the predicted noise (epsilon) of the same shape as x_t.
    """

    def __init__(self, in_ch: int = 3, base: int = 64, t_dim: int = 128):
        super().__init__()
        self.t_dim = t_dim
        emb = t_dim * 4
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, emb), nn.SiLU(), nn.Linear(emb, emb))

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.d1 = ResBlock(base, base, emb)
        self.down1 = nn.Conv2d(base, base * 2, 4, 2, 1)
        self.d2 = ResBlock(base * 2, base * 2, emb)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, 2, 1)
        self.d3 = ResBlock(base * 4, base * 4, emb)
        self.down3 = nn.Conv2d(base * 4, base * 4, 4, 2, 1)

        self.m1 = ResBlock(base * 4, base * 4, emb)
        self.attn = SelfAttention(base * 4)
        self.m2 = ResBlock(base * 4, base * 4, emb)

        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 4, 2, 1)
        self.u3 = ResBlock(base * 4 + base * 4, base * 4, emb)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1)
        self.u2 = ResBlock(base * 2 + base * 2, base * 2, emb)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.u1 = ResBlock(base + base, base, emb)

        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_mlp(timestep_embedding(t, self.t_dim))
        h0 = self.in_conv(x)
        h1 = self.d1(h0, t_emb)
        h2 = self.d2(self.down1(h1), t_emb)
        h3 = self.d3(self.down2(h2), t_emb)
        m = self.m1(self.down3(h3), t_emb)
        m = self.attn(m)
        m = self.m2(m, t_emb)
        u3 = self.u3(torch.cat([self.up3(m), h3], dim=1), t_emb)
        u2 = self.u2(torch.cat([self.up2(u3), h2], dim=1), t_emb)
        u1 = self.u1(torch.cat([self.up1(u2), h1], dim=1), t_emb)
        return self.out_conv(F.silu(self.out_norm(u1)))
