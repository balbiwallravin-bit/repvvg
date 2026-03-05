"""Simple eval script."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    ap.add_argument("--max_batches", type=int, default=0, help=">0 to stop early for debugging")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("[eval] parsing index...")
    specs = parse_index(args.index, args.ready_root)
    print(f"[eval] parsed samples: {len(specs)}")

    ds = FrameWindowDataset(specs, args.pseudo_root)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=kd_collate)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = StudentNet().to(dev)
    print(f"[eval] device: {dev}")

    print(f"[eval] loading checkpoint: {ckpt_path}")
    m.load_state_dict(load_model_state_dict(str(ckpt_path)))
    print("[eval] checkpoint loaded")
    m.eval()

    vals: list[float] = []
    valid_batches = 0
    empty_batches = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        pbar = tqdm(dl, total=len(dl), desc="eval", dynamic_ncols=True)
        for step, b in enumerate(pbar, start=1):
            if args.max_batches > 0 and step > args.max_batches:
                break
            if not b:
                empty_batches += 1
                continue
            o = m(b["x"].to(dev), return_params=False)
            _, d = kd_total_loss(o["logits"], b["hm_t"].to(dev), b["score"].to(dev), m.decoder)
            loss_v = float(d["loss"])
            vals.append(loss_v)
            valid_batches += 1

            if step % max(1, args.log_every) == 0:
                elapsed = time.perf_counter() - t0
                avg = sum(vals) / max(1, len(vals))
                print(
                    f"[eval] step={step}/{len(dl)} valid={valid_batches} empty={empty_batches} "
                    f"avg_loss={avg:.6f} elapsed_s={elapsed:.1f}"
                )

    elapsed = time.perf_counter() - t0
    out = {
        "loss": sum(vals) / max(1, len(vals)),
        "valid_batches": valid_batches,
        "empty_batches": empty_batches,
        "elapsed_s": round(elapsed, 3),
    }
    print(out)


if __name__ == "__main__":
    main()
