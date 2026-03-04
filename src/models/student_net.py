"""Student network."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.models.moment_decoder import MomentDecoder
from src.models.repvgg_blocks import RepVGGBlock


class StudentNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.e1 = nn.Sequential(RepVGGBlock(9, base), RepVGGBlock(base, base))
        self.e2 = nn.Sequential(RepVGGBlock(base, base * 2, stride=2), RepVGGBlock(base * 2, base * 2))
        self.e3 = nn.Sequential(RepVGGBlock(base * 2, base * 4, stride=2), RepVGGBlock(base * 4, base * 4))
        self.bottleneck = RepVGGBlock(base * 4, base * 4)
        self.d2 = RepVGGBlock(base * 4 + base * 2, base * 2)
        self.d1 = RepVGGBlock(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)
        self.decoder = MomentDecoder()

    def forward(self, x: torch.Tensor, return_logits: bool = True, return_params: bool = True) -> dict[str, torch.Tensor]:
        f1 = self.e1(x)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        b = self.bottleneck(f3)
        u2 = F.interpolate(b, scale_factor=2.0, mode="bilinear", align_corners=False)
        u2 = self.d2(torch.cat([u2, f2], dim=1))
        u1 = F.interpolate(u2, scale_factor=2.0, mode="bilinear", align_corners=False)
        u1 = self.d1(torch.cat([u1, f1], dim=1))
        logits_full = self.head(u1)
        logits = F.interpolate(logits_full, size=(72, 128), mode="bilinear", align_corners=False)

        out: dict[str, torch.Tensor] = {}
        if return_logits:
            out["logits"] = logits
        if return_params:
            params = self.decoder(logits)
            out.update({"mu_xy": params["mu_xy"], "dir_xy": params["dir_xy"], "l": params["l"], "cov": params["cov"], "prob": params["prob"]})
        return out
