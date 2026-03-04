"""RepVGG block implementation."""
from __future__ import annotations

import torch
from torch import nn


class RepVGGBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.id_bn = nn.BatchNorm2d(in_ch) if (in_ch == out_ch and stride == 1) else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.branch3(x) + self.branch1(x)
        if self.id_bn is not None:
            y = y + self.id_bn(x)
        return self.act(y)

    def reparameterize_for_inference(self) -> None:
        """Placeholder API for branch folding compatibility."""
        return None
