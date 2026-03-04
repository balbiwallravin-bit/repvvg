from __future__ import annotations

import math

import torch

from src.models.moment_decoder import MomentDecoder


def test_moment_decoder_direction() -> None:
    h, w = 72, 128
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    theta = math.radians(30)
    line = xx.float() * math.cos(theta) + yy.float() * math.sin(theta)
    hm = torch.exp(-((line - 40.0) ** 2) / 8.0)[None, None]
    logits = torch.logit(hm.clamp(1e-4, 1 - 1e-4))

    dec = MomentDecoder(h=h, w=w)
    out = dec(logits)
    d = out["dir_xy"][0, 0]
    pred = math.degrees(math.atan2(float(d[1]), float(d[0])))
    target = 30.0
    alt = (pred + 180.0) % 360.0
    err = min(abs(pred - target), abs(alt - target))
    assert err < 10.0
