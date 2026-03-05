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
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--pin_memory", type=int, default=1)
    ap.add_argument("--amp", type=int, default=1)
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
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": kd_collate,
        "pin_memory": bool(args.pin_memory),
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    dl = DataLoader(ds, **loader_kwargs)

    use_cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    m = StudentNet().to(dev)
    print(f"[eval] device: {dev}")

    print(f"[eval] loading checkpoint: {ckpt_path}")
    m.load_state_dict(load_model_state_dict(str(ckpt_path)))
    print("[eval] checkpoint loaded")
    m.eval()

    vals: list[float] = []
    valid_batches = 0
    empty_batches = 0
    total_batches = len(dl)
    t0 = time.perf_counter()

    with torch.no_grad():
        pbar = tqdm(dl, total=total_batches, desc="eval", dynamic_ncols=True)
        for step, b in enumerate(pbar, start=1):
            if args.max_batches > 0 and step > args.max_batches:
                break
            if not b:
                empty_batches += 1
                continue

            x = b["x"].to(dev, non_blocking=use_cuda)
            hm_t = b["hm_t"].to(dev, non_blocking=use_cuda)
            score = b["score"].to(dev, non_blocking=use_cuda)

            with torch.amp.autocast(device_type="cuda", enabled=bool(args.amp and use_cuda)):
                o = m(x, return_params=False)
                _, d = kd_total_loss(o["logits"], hm_t, score, m.decoder)

            loss_v = float(d["loss"])
            vals.append(loss_v)
            valid_batches += 1

            if step % max(1, args.log_every) == 0:
                elapsed = time.perf_counter() - t0
                avg = sum(vals) / max(1, len(vals))
                bps = step / max(1e-6, elapsed)
                eta = max(0.0, (total_batches - step) / max(1e-6, bps))
                print(
                    f"[eval] step={step}/{total_batches} valid={valid_batches} empty={empty_batches} "
                    f"avg_loss={avg:.6f} bps={bps:.2f} eta_s={eta:.1f} elapsed_s={elapsed:.1f}"
                )

    elapsed = time.perf_counter() - t0
    out = {
        "loss": sum(vals) / max(1, len(vals)),
        "valid_batches": valid_batches,
        "empty_batches": empty_batches,
        "elapsed_s": round(elapsed, 3),
        "batches_per_sec": round(valid_batches / max(1e-6, elapsed), 3),
    }
    print(out)


if __name__ == "__main__":
    main()
