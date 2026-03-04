"""Knowledge distillation losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.moment_decoder import MomentDecoder


def _kl_pt_ps(pt: torch.Tensor, ps: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (pt * ((pt + eps).log() - (ps + eps).log())).sum(dim=(-1, -2, -3))


def sobel_grad(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=x.device, dtype=x.dtype)
    ky = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def kd_total_loss(
    logits_s: torch.Tensor,
    hm_t: torch.Tensor,
    score: torch.Tensor,
    decoder: MomentDecoder,
    a: float = 1.0,
    b: float = 10.0,
    c: float = 1.0,
    d: float = 0.5,
    use_grad: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pt = decoder.prob(hm_t.float().clamp(0, 1).logit(eps=1e-4))
    ps = decoder.prob(logits_s)
    mom_t = decoder(hm_t.float().clamp(0, 1).logit(eps=1e-4))
    mom_s = decoder(logits_s)

    l_kl = _kl_pt_ps(pt, ps)
    l_mu = F.smooth_l1_loss(mom_s["mu_xy"], mom_t["mu_xy"], reduction="none").mean(dim=-1)
    l_sigma = F.smooth_l1_loss(mom_s["cov"], mom_t["cov"], reduction="none").mean(dim=-1)
    w_sigma = (mom_t["l"] > 3.0).float() * 1.0 + (mom_t["l"] <= 3.0).float() * 0.2

    l_grad = torch.zeros_like(l_kl)
    if use_grad:
        l_grad = F.l1_loss(sobel_grad(ps), sobel_grad(pt), reduction="none").mean(dim=(-1, -2, -3))

    w = score.clamp(0, 1)
    total = w * (a * l_kl + b * l_mu + c * w_sigma * l_sigma + d * l_grad)
    loss = total.mean()
    details = {
        "loss": loss.detach(),
        "l_kl": l_kl.mean().detach(),
        "l_mu": l_mu.mean().detach(),
        "l_sigma": l_sigma.mean().detach(),
        "l_grad": l_grad.mean().detach(),
        "mu_err_px": (mom_s["mu_xy"] - mom_t["mu_xy"]).abs().mean().detach(),
    }
    return loss, details
