"""Simple eval script."""
from __future__ import annotations

import argparse
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
    args = ap.parse_args()

    specs = parse_index(args.index, args.ready_root)
    ds = FrameWindowDataset(specs, args.pseudo_root)
    dl = DataLoader(ds, batch_size=16, collate_fn=kd_collate)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = StudentNet().to(dev)
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    m.eval()

    vals = []
    with torch.no_grad():
        for b in dl:
            if not b:
                continue
            o = m(b["x"].to(dev), return_params=False)
            _, d = kd_total_loss(o["logits"], b["hm_t"].to(dev), b["score"].to(dev), m.decoder)
            vals.append(float(d["loss"]))
    print({"loss": sum(vals) / max(1, len(vals))})


if __name__ == "__main__":
    main()
