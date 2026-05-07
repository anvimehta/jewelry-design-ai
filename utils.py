"""Shared helpers: device picking, seeding, and saving image grids."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_grid(images_in_01: torch.Tensor, path: Path, nrow: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(images_in_01.clamp(0, 1), nrow=nrow, padding=2)
    save_image(grid, path)
