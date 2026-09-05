from __future__ import annotations

import torch
from torch import Tensor


class SegmentationMetrics:
    def __init__(self, num_classes: int = 21, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    @torch.no_grad()
    def update(self, logits_or_prediction: Tensor, target: Tensor) -> None:
        prediction = logits_or_prediction.argmax(1) if logits_or_prediction.ndim == 4 else logits_or_prediction
        prediction, target = prediction.cpu().long(), target.cpu().long()
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        indices = self.num_classes * target[valid] + prediction[valid]
        self.confusion += torch.bincount(indices, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> dict[str, object]:
        matrix = self.confusion.float()
        intersection = matrix.diag()
        union = matrix.sum(1) + matrix.sum(0) - intersection
        present = union > 0
        iou = torch.full((self.num_classes,), float("nan"))
        iou[present] = intersection[present] / union[present]
        pixel_accuracy = intersection.sum() / matrix.sum().clamp_min(1)
        return {"miou": iou[present].mean().item(), "pixel_accuracy": pixel_accuracy.item(), "iou": iou.tolist()}

