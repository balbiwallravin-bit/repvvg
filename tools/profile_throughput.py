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

from src.models.student_net import StudentNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = StudentNet().to(dev).eval()
    x = torch.randn(args.batch_size, 9, 288, 512, device=dev)

    times = []
    with torch.no_grad():
        for _ in range(10):
            _ = m(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(args.iters):
            t0 = time.perf_counter()
            _ = m(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    fps = args.batch_size * 1000.0 / t.mean()
    print({"fps": float(fps), "p50_ms": float(np.percentile(t, 50)), "p99_ms": float(np.percentile(t, 99))})


if __name__ == "__main__":
    main()
