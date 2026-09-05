import torch
from torch import nn

from dinov3_pspnet.metrics import SegmentationMetrics
from dinov3_pspnet.model import DINOv3PSPNet


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 384, 16, stride=16)

    def forward_features(self, x):
        features = self.projection(x).flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": features}


def test_model_restores_input_resolution():
    model = DINOv3PSPNet(FakeBackbone())
    output = model(torch.randn(2, 3, 64, 80))
    assert output.shape == (2, 21, 64, 80)
    assert not any(p.requires_grad for p in model.backbone.parameters())


def test_metrics_ignore_255_and_compute_perfect_miou():
    target = torch.tensor([[[0, 1], [255, 1]]])
    prediction = torch.tensor([[[0, 1], [2, 1]]])
    metric = SegmentationMetrics(num_classes=3)
    metric.update(prediction, target)
    result = metric.compute()
    assert result["miou"] == 1.0
    assert result["pixel_accuracy"] == 1.0

