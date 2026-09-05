from __future__ import annotations

import torch
from torch import nn
from tqdm import tqdm

from .metrics import SegmentationMetrics


def run_epoch(model, loader, device, criterion, optimizer=None, amp: bool = True):
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics()
    total_loss = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    for images, masks in tqdm(loader, desc="train" if training else "val", leave=False):
        images, masks = images.to(device), masks.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type, enabled=amp and device.type == "cuda"
        ):
            logits = model(images)
            loss = criterion(logits, masks)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * images.shape[0]
        metrics.update(logits, masks)
    result = metrics.compute()
    result["loss"] = total_loss / len(loader.dataset)
    return result


def make_criterion():
    return nn.CrossEntropyLoss(ignore_index=255)

