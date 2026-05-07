"""Necklace image dataset with the heavy augmentation that the diffusion model needs.

With ~470 training images, every image is seen ~thousands of times during training,
so without strong augmentation the UNet just memorizes the training set instead of
learning a distribution. Random resized crops + flip + small rotation + color jitter
give us enough effective variety from a tiny corpus.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class NecklaceDataset(Dataset):
    def __init__(self, root: str, image_size: int = 64):
        self.paths = sorted(
            p for p in Path(root).iterdir()
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        )
        if not self.paths:
            raise FileNotFoundError(f'No images in {root}')
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15, fill=255),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(Image.open(self.paths[idx]).convert('RGB'))
