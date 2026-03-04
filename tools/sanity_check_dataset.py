from __future__ import annotations


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import random

from src.datasets.frame_window_dataset import FrameWindowDataset
from src.datasets.ready_index import parse_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--ready_root", required=True)
    ap.add_argument("--pseudo_root", required=True)
    ap.add_argument("--samples", type=int, default=200)
    args = ap.parse_args()

    specs = parse_index(args.index, args.ready_root)
    ds = FrameWindowDataset(specs, args.pseudo_root, strict=0)
    n = min(args.samples, len(ds))
    idxs = random.sample(range(len(ds)), n) if n > 0 else []
    good = 0
    for i in idxs:
        item = ds[i]
        if item is not None and item["x"].shape == (9, 288, 512) and item["hm_t"].shape == (1, 72, 128):
            good += 1
    print({
        "checked": n,
        "good": good,
        "missing_npz": len(ds.badcases["missing_npz"]),
        "missing_frames": len(ds.badcases["missing_frames"]),
    })


if __name__ == "__main__":
    main()
