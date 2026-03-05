"""Simple eval script with visible/no-ball split metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.datasets.frame_window_dataset import FrameWindowDataset, kd_collate
from src.datasets.ready_index import parse_index
from src.losses.kd_losses import kd_total_loss
from src.models.student_net import StudentNet


def _resolve_ckpt_path(ckpt_arg: str) -> Path:
    p = Path(ckpt_arg)
    tried: list[Path] = []

    if p.is_file():
        return p

    if p.is_dir():
        cands = [p / "checkpoints" / "best.pt", p / "checkpoints" / "last.pt", p / "best.pt", p / "last.pt"]
        for c in cands:
            tried.append(c)
            if c.is_file():
                return c
        raise FileNotFoundError(f"No checkpoint found under directory: {p}; tried: {[str(x) for x in tried]}")

    # File path not found: try sibling last.pt fallback for best.pt typo/absence
    tried.append(p)
    if p.name == "best.pt":
        fallback = p.with_name("last.pt")
        tried.append(fallback)
        if fallback.is_file():
            return fallback

    raise FileNotFoundError(f"Checkpoint not found: {p}; tried: {[str(x) for x in tried]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--ready_root", required=True)
    ap.add_argument("--pseudo_root", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint file or run directory")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_batches", type=int, default=0, help="0 means evaluate all batches")
    ap.add_argument("--log_every", type=int, default=10, help="print progress every N processed batches")

    ap.add_argument("--roi_enable", type=int, default=1)
    ap.add_argument("--roi_ref_w", type=int, default=1920)
    ap.add_argument("--roi_ref_h", type=int, default=1080)
    ap.add_argument("--roi_x0", type=int, default=367)
    ap.add_argument("--roi_y0", type=int, default=100)
    ap.add_argument("--roi_x1", type=int, default=1760)
    ap.add_argument("--roi_y1", type=int, default=884)

    ap.add_argument("--visi_thr", type=float, default=0.25)
    ap.add_argument("--vis_conf_thr", type=float, default=0.5)
    ap.add_argument("--neg_hm_scale", type=float, default=0.1)
    args = ap.parse_args()

    specs = parse_index(args.index, args.ready_root)
    ds = FrameWindowDataset(
        specs,
        args.pseudo_root,
        roi_enable=args.roi_enable,
        roi_ref_w=args.roi_ref_w,
        roi_ref_h=args.roi_ref_h,
        roi_x0=args.roi_x0,
        roi_y0=args.roi_y0,
        roi_x1=args.roi_x1,
        roi_y1=args.roi_y1,
        visi_thr=args.visi_thr,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=kd_collate)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = StudentNet().to(dev)

    def _load_ckpt(path: str) -> dict[str, Any]:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    ckpt_path = _resolve_ckpt_path(args.ckpt)
    print({"ckpt": str(ckpt_path)})
    m.load_state_dict(_load_ckpt(str(ckpt_path))["model"])
    m.eval()

    vals: list[float] = []
    seen = 0

    n_pos = 0
    n_neg = 0
    fp = 0
    fn = 0
    mu_err_visible_sum = 0.0

    with torch.no_grad():
        for batch_idx, b in enumerate(dl, start=1):
            if args.max_batches > 0 and seen >= args.max_batches:
                break
            if not b:
                continue

            x = b["x"].to(dev)
            hm_t = b["hm_t"].to(dev)
            score = b["score"].to(dev)
            visi = b["visi"].to(dev)

            o = m(x, return_params=False)
            _, d = kd_total_loss(o["logits"], hm_t, score, m.decoder, visi_t=visi, visi_logit_s=o["visi_logit"], neg_hm_scale=args.neg_hm_scale)
            vals.append(float(d["loss"]))
            seen += 1

            vis_prob = torch.sigmoid(o["visi_logit"].view(-1))
            vis_pred = (vis_prob >= args.vis_conf_thr).float()
            vis_true = visi.view(-1)

            n_pos += int((vis_true > 0.5).sum().item())
            n_neg += int((vis_true <= 0.5).sum().item())
            fp += int(((vis_true <= 0.5) & (vis_pred > 0.5)).sum().item())
            fn += int(((vis_true > 0.5) & (vis_pred <= 0.5)).sum().item())
            mu_err_visible_sum += float(d["mu_err_px_visible"]) * max(1, int((vis_true > 0.5).sum().item()))

            if args.log_every > 0 and (seen % args.log_every == 0):
                print({"progress": f"{seen} batches", "avg_loss": sum(vals) / len(vals), "loader_batch": batch_idx})

    out = {
        "loss": sum(vals) / max(1, len(vals)),
        "fp_rate": fp / max(1, n_neg),
        "fn_rate": fn / max(1, n_pos),
        "mu_err_px_visible": mu_err_visible_sum / max(1, n_pos),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
    print(out)


if __name__ == "__main__":
    main()
