from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    """The pyramid pooling module from PSPNet."""

    def __init__(self, in_channels: int, out_channels: int = 96, bins=(1, 2, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                # GroupNorm also works for the 1x1 branch with batch size one.
                nn.GroupNorm(1, out_channels),
                nn.ReLU(inplace=True),
            )
            for size in bins
        )

    def forward(self, x: Tensor) -> Tensor:
        spatial_size = x.shape[-2:]
        pooled = [x]
        for branch in self.branches:
            y = branch(x)
            pooled.append(F.interpolate(y, spatial_size, mode="bilinear", align_corners=False))
        return torch.cat(pooled, dim=1)


class PSPHead(nn.Module):
    def __init__(self, in_channels: int = 384, num_classes: int = 21, ppm_channels: int = 96):
        super().__init__()
        self.ppm = PyramidPoolingModule(in_channels, ppm_channels)
        merged_channels = in_channels + 4 * ppm_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(merged_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.ppm(x))


def load_dinov3(repo: str, weights: str | None, pretrained: bool = True) -> nn.Module:
    """Load the official ViT-S/16 through the local DINOv3 torch hub repo."""
    repo_path = Path(repo).expanduser()
    if not repo_path.joinpath("hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv3 repo must contain hubconf.py: {repo_path}")
    if pretrained and not weights:
        raise ValueError("Pretrained DINOv3 requires --weights (local file or authorized URL).")
    # The official loader converts even a local path to a file URL and copies it
    # through TORCH_HOME. Loading the state dict ourselves avoids that unnecessary
    # copy and works in restricted/Windows environments.
    local_weights = Path(weights).expanduser() if weights else None
    if local_weights and local_weights.is_file():
        model = torch.hub.load(
            str(repo_path), "dinov3_vits16", source="local", pretrained=False
        )
        state_dict = torch.load(local_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return model
    kwargs: dict[str, Any] = {"pretrained": pretrained}
    if weights:
        kwargs["weights"] = weights
    return torch.hub.load(str(repo_path), "dinov3_vits16", source="local", **kwargs)


class DINOv3PSPNet(nn.Module):
    """DINOv3 ViT-S/16 backbone followed by a PSPNet segmentation head."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 21,
        embed_dim: int = 384,
        patch_size: int = 16,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = PSPHead(embed_dim, num_classes)
        self.patch_size = patch_size
        self.freeze_backbone = freeze_backbone
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.freeze_backbone = not trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _patch_tokens(self, images: Tensor) -> Tensor:
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            features = self.backbone.forward_features(images)
        if isinstance(features, dict):
            for key in ("x_norm_patchtokens", "x_patchtokens", "patch_tokens"):
                if key in features:
                    return features[key]
            raise KeyError(f"No patch-token key in backbone output: {tuple(features)}")
        if isinstance(features, Tensor) and features.ndim == 3:
            return features
        raise TypeError("DINOv3 forward_features must return a token tensor or feature dictionary")

    def forward(self, images: Tensor) -> Tensor:
        input_size = images.shape[-2:]
        if input_size[0] % self.patch_size or input_size[1] % self.patch_size:
            raise ValueError(f"Input H and W must be divisible by {self.patch_size}; got {input_size}")
        tokens = self._patch_tokens(images)
        grid_h, grid_w = input_size[0] // self.patch_size, input_size[1] // self.patch_size
        if tokens.shape[1] != grid_h * grid_w:
            raise ValueError(
                f"Expected {grid_h * grid_w} patch tokens for {input_size}, got {tokens.shape[1]}"
            )
        feature_map = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_h, grid_w)
        logits = self.head(feature_map)
        return F.interpolate(logits, input_size, mode="bilinear", align_corners=False)
