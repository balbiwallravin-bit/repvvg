from __future__ import annotations

import math

import torch

from src.models.moment_decoder import MomentDecoder


def test_moment_decoder_direction() -> None:
    h, w = 72, 128
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    theta = math.radians(30.0)
    cx, cy = 64.0, 36.0
    x = xx.float() - cx
    y = yy.float() - cy
    # rotate coordinates: xp along major axis direction theta
    xp = x * math.cos(theta) + y * math.sin(theta)
    yp = -x * math.sin(theta) + y * math.cos(theta)

    # elongated gaussian with major axis along xp
    hm = torch.exp(-0.5 * ((xp / 18.0) ** 2 + (yp / 2.0) ** 2))[None, None]
    logits = torch.logit(hm.clamp(1e-4, 1 - 1e-4))

    dec = MomentDecoder(h=h, w=w)
    out = dec(logits)
    d = out["dir_xy"][0, 0]
    pred = math.degrees(math.atan2(float(d[1]), float(d[0])))
    target = 30.0
    alt = (pred + 180.0) % 360.0
    err = min(abs(pred - target), abs(alt - target))
    assert err < 10.0


def test_moment_decoder_state_dict_roundtrip() -> None:
    dec = MomentDecoder()
    state = dec.state_dict()

    other = MomentDecoder()
    other.load_state_dict(state)

    assert torch.equal(other.grid_x, dec.grid_x)
    assert torch.equal(other.grid_y, dec.grid_y)


from src.models.student_net import StudentNet


def test_student_net_visi_head_output() -> None:
    m = StudentNet().eval()
    x = torch.randn(2, 9, 288, 512)
    out = m(x, return_logits=True, return_params=False)
    assert out["visi_logit"].shape == (2, 1)
    assert out["visi_prob"].shape == (2, 1)
