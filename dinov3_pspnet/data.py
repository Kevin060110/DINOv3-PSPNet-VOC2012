from __future__ import annotations

import random

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class JointTransform:
    """Apply aligned geometry to image and categorical mask."""

    def __init__(self, size: int, train: bool):
        if size % 16:
            raise ValueError("image_size must be divisible by the DINOv3 patch size (16)")
        self.size = size
        self.train = train

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, [self.size, self.size], InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, [self.size, self.size], InterpolationMode.NEAREST)
        if self.train and random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        image = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        # PIL conversion preserves VOC's ignore label 255.
        mask = torch.as_tensor(__import__("numpy").array(mask), dtype=torch.long)
        return image, mask


def make_loader(
    root: str,
    split: str,
    size: int,
    batch_size: int,
    workers: int,
    download: bool = False,
) -> DataLoader:
    dataset = VOCSegmentation(
        root=root,
        year="2012",
        image_set=split,
        download=download,
        transforms=JointTransform(size, train=split == "train"),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )

