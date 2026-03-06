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
    visi_t: torch.Tensor | None = None,
    visi_logit_s: torch.Tensor | None = None,
    a: float = 1.0,
    b: float = 10.0,
    c: float = 1.0,
    d: float = 0.5,
    e: float = 1.0,
    neg_hm_scale: float = 0.1,
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

    if visi_t is None:
        visi_t = (score >= 0.25).float().view(-1)
    else:
        visi_t = visi_t.float().view(-1)

    score_w = score.clamp(0, 1).view(-1)
    hm_w = score_w * (visi_t + (1.0 - visi_t) * neg_hm_scale)
    total = hm_w * (a * l_kl + b * l_mu + c * w_sigma * l_sigma + d * l_grad)

    l_vis = torch.zeros_like(total)
    if visi_logit_s is not None:
        l_vis = F.binary_cross_entropy_with_logits(visi_logit_s.view(-1), visi_t, reduction="none")
        total = total + e * l_vis

    loss = total.mean()

    pred_vis = torch.zeros_like(visi_t)
    if visi_logit_s is not None:
        pred_vis = (torch.sigmoid(visi_logit_s.view(-1)) >= 0.5).float()

    vis_mask = visi_t > 0.5
    mu_err_visible = (mom_s["mu_xy"] - mom_t["mu_xy"]).abs().mean(dim=(-1, -2))
    mu_err_visible = mu_err_visible[vis_mask].mean() if vis_mask.any() else torch.tensor(0.0, device=logits_s.device)

    details = {
        "loss": loss.detach(),
        "l_kl": l_kl.mean().detach(),
        "l_mu": l_mu.mean().detach(),
        "l_sigma": l_sigma.mean().detach(),
        "l_grad": l_grad.mean().detach(),
        "l_vis": l_vis.mean().detach() if visi_logit_s is not None else torch.tensor(0.0, device=logits_s.device),
        "mu_err_px": (mom_s["mu_xy"] - mom_t["mu_xy"]).abs().mean().detach(),
        "mu_err_px_visible": mu_err_visible.detach(),
        "visi_acc": (pred_vis == visi_t).float().mean().detach() if visi_logit_s is not None else torch.tensor(0.0, device=logits_s.device),
    }
    return loss, details
