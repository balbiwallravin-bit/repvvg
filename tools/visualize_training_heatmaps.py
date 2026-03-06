from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.datasets.frame_window_dataset import FrameWindowDataset
from src.datasets.ready_index import parse_index


def _overlay(frame_rgb: np.ndarray, hm: np.ndarray) -> np.ndarray:
    hm = np.ascontiguousarray(hm, dtype=np.float32)
    hm_up = cv2.resize(hm, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap((np.clip(hm_up, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_rgb, 0.6, cv2.cvtColor(color, cv2.COLOR_BGR2RGB), 0.4, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract and visualize training heatmaps.")
    ap.add_argument("--index", required=True)
    ap.add_argument("--ready_root", required=True)
    ap.add_argument("--pseudo_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=int, default=0, help="start index when mode=sequential")
    ap.add_argument("--mode", choices=["random", "sequential"], default="random")
    args = ap.parse_args()

    specs = parse_index(args.index, args.ready_root)
    ds = FrameWindowDataset(specs, args.pseudo_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(args.num_samples, len(ds))
    if args.mode == "random":
        rng = random.Random(args.seed)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[:n]
    else:
        idxs = list(range(args.start, min(args.start + n, len(ds))))

    saved = 0
    for rank, idx in enumerate(idxs, start=1):
        item = ds[idx]
        if item is None:
            continue

        frames = item["x"].numpy().reshape(3, 3, 288, 512).transpose(0, 2, 3, 1)
        frames = [(f * 255).astype(np.uint8) for f in frames]
        mid = frames[1]

        hm = item["hm_t"][0].numpy()
        hm_vis = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
        hm_vis = cv2.resize(hm_vis, (512, 288), interpolation=cv2.INTER_NEAREST)
        hm_vis = cv2.cvtColor(hm_vis, cv2.COLOR_GRAY2RGB)
        ov = _overlay(mid, hm)
        strip = np.concatenate([frames[0], mid, frames[2], hm_vis, ov], axis=1)

        p = out_dir / f"sample_{rank:03d}_idx_{idx}.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        saved += 1
        print(f"[progress] {saved}/{n} -> {p}")

    print({"saved": saved, "requested": n, "out_dir": str(out_dir)})


if __name__ == "__main__":
    main()
