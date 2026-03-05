"""Simple eval script."""
from __future__ import annotations

import argparse
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.datasets.frame_window_dataset import FrameWindowDataset, kd_collate
from src.datasets.ready_index import parse_index
from src.losses.kd_losses import kd_total_loss
from src.models.student_net import StudentNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--ready_root", required=True)
    ap.add_argument("--pseudo_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_batches", type=int, default=0, help="0 means evaluate all batches")
    ap.add_argument("--log_every", type=int, default=10, help="print progress every N processed batches")
    args = ap.parse_args()

    specs = parse_index(args.index, args.ready_root)
    ds = FrameWindowDataset(specs, args.pseudo_root)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=kd_collate)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = StudentNet().to(dev)

    def _load_ckpt(path: str) -> dict[str, Any]:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    m.load_state_dict(_load_ckpt(args.ckpt)["model"])
    m.eval()

    vals = []
    seen = 0
    with torch.no_grad():
        for batch_idx, b in enumerate(dl, start=1):
            if args.max_batches > 0 and seen >= args.max_batches:
                break
            if not b:
                continue
            o = m(b["x"].to(dev), return_params=False)
            _, d = kd_total_loss(o["logits"], b["hm_t"].to(dev), b["score"].to(dev), m.decoder)
            vals.append(float(d["loss"]))
            seen += 1
            if args.log_every > 0 and (seen % args.log_every == 0):
                print({"progress": f"{seen} batches", "avg_loss": sum(vals) / len(vals), "loader_batch": batch_idx})
    print({"loss": sum(vals) / max(1, len(vals))})


if __name__ == "__main__":
    main()
