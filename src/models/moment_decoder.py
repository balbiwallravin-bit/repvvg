"""Moments decoder from heatmap logits."""
from __future__ import annotations

import torch
from torch import nn


class MomentDecoder(nn.Module):
    def __init__(self, h: int = 72, w: int = 128, stride: int = 4, gamma: float = 2.0, k_len: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.h = h
        self.w = w
        self.stride = stride
        self.gamma = gamma
        self.k_len = k_len
        self.eps = eps
        gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
        self.register_buffer("grid_x", gx[None, None])
        self.register_buffer("grid_y", gy[None, None])

    def prob(self, logits: torch.Tensor) -> torch.Tensor:
        h = torch.sigmoid(logits)
        p = (h + self.eps).pow(self.gamma)
        return p / (p.sum(dim=(-1, -2), keepdim=True) + self.eps)

    def forward(self, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        p = self.prob(logits)
        mx = (p * self.grid_x).sum(dim=(-1, -2))
        my = (p * self.grid_y).sum(dim=(-1, -2))
        dx = self.grid_x - mx[..., None, None]
        dy = self.grid_y - my[..., None, None]
        sxx = (p * dx * dx).sum(dim=(-1, -2))
        sxy = (p * dx * dy).sum(dim=(-1, -2))
        syy = (p * dy * dy).sum(dim=(-1, -2))
        t = sxx - syy
        u = torch.sqrt(t * t + 4.0 * sxy * sxy + self.eps)
        vx = 2.0 * sxy
        vy = syy - sxx + u
        vn = torch.sqrt(vx * vx + vy * vy + self.eps)
        dir_x = vx / (vn + self.eps)
        dir_y = vy / (vn + self.eps)
        lam = 0.5 * (sxx + syy + u)
        sigma = torch.sqrt(lam + self.eps)

        mu_img = torch.stack([(mx + 0.5) * self.stride, (my + 0.5) * self.stride], dim=-1)
        dir_xy = torch.stack([dir_x, dir_y], dim=-1)
        l_img = self.k_len * sigma * self.stride
        return {
            "prob": p,
            "mu_xy": mu_img,
            "dir_xy": dir_xy,
            "l": l_img,
            "cov": torch.stack([sxx, sxy, syy], dim=-1),
        }
