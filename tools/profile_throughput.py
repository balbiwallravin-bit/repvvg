from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import time

import numpy as np
import torch

from src.datasets.frame_window_dataset import FrameWindowDataset, kd_collate
from src.datasets.ready_index import parse_index
from src.models.student_net import StudentNet


def _bench_tensor(model: StudentNet, x: torch.Tensor, iters: int) -> dict[str, float]:
    times: list[float] = []
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, return_params=False)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x, return_params=False)
            if x.device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {
        "fps": float(x.shape[0] * 1000.0 / t.mean()),
        "p50_ms": float(np.percentile(t, 50)),
        "p99_ms": float(np.percentile(t, 99)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--index", default="")
    ap.add_argument("--ready_root", default="")
    ap.add_argument("--pseudo_root", default="")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudentNet().to(dev).eval()

    # mode A: random synthetic input benchmark
    if not args.index:
        x = torch.randn(args.batch_size, 9, 288, 512, device=dev)
        print({"mode": "random", **_bench_tensor(model, x, args.iters)})
        return

    # mode B: real dataset benchmark (decode/preprocess + forward)
    if not args.ready_root or not args.pseudo_root:
        raise ValueError("When --index is set, --ready_root and --pseudo_root are required")

    specs = parse_index(args.index, args.ready_root)
    if not specs:
        raise RuntimeError("No valid samples from index for profiling")
    specs = specs[: min(len(specs), args.samples)]
    ds = FrameWindowDataset(specs, args.pseudo_root, strict=0)

    lat_ms: list[float] = []
    n = 0
    with torch.no_grad():
        for i in range(0, len(ds), args.batch_size):
            batch_items = [ds[j] for j in range(i, min(i + args.batch_size, len(ds)))]
            b = kd_collate(batch_items)
            if not b:
                continue
            x = b["x"].to(dev, non_blocking=True)
            t0 = time.perf_counter()
            _ = model(x, return_params=False)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            lat_ms.append((time.perf_counter() - t0) * 1000)
            n += x.shape[0]

    t = np.array(lat_ms)
    print(
        {
            "mode": "real_images",
            "samples": int(n),
            "fps": float(n * 1000.0 / t.sum()) if t.size > 0 else 0.0,
            "p50_batch_ms": float(np.percentile(t, 50)) if t.size > 0 else 0.0,
            "p99_batch_ms": float(np.percentile(t, 99)) if t.size > 0 else 0.0,
        }
    )


if __name__ == "__main__":
    main()
