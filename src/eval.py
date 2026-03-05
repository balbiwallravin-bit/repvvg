"""Simple eval script."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.datasets.frame_window_dataset import FrameWindowDataset, kd_collate
from src.datasets.ready_index import parse_index
from src.losses.kd_losses import kd_total_loss
from src.models.student_net import StudentNet
from src.utils.checkpoint import load_model_state_dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--ready_root", required=True)
    ap.add_argument("--pseudo_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    specs = parse_index(args.index, args.ready_root)
    ds = FrameWindowDataset(specs, args.pseudo_root)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=kd_collate)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = StudentNet().to(dev)
    print(f"[eval] loading checkpoint: {ckpt_path}")
    m.load_state_dict(load_model_state_dict(str(ckpt_path)))
    print("[eval] checkpoint loaded")
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
