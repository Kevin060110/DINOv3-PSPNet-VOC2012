from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .model import DINOv3PSPNet, load_dinov3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args) -> DINOv3PSPNet:
    backbone = load_dinov3(args.dinov3_repo, args.weights, pretrained=True)
    return DINOv3PSPNet(backbone, freeze_backbone=not args.finetune_backbone)


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, best_miou: float, args) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
         "best_miou": best_miou, "args": vars(args)}, path
    )


def load_checkpoint(path: str | Path, model, device, optimizer=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def write_json(path: str | Path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

