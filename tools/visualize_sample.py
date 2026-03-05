from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import random
import cv2
import numpy as np
import torch

from src.datasets.frame_window_dataset import FrameWindowDataset
from src.datasets.ready_index import parse_index
from src.models.student_net import StudentNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--ready_root", required=True)
    ap.add_argument("--pseudo_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_id", type=int, default=-1)
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()

    specs = parse_index(args.index, args.ready_root)
    ds = FrameWindowDataset(specs, args.pseudo_root)
    idx = args.sample_id if 0 <= args.sample_id < len(ds) else random.randrange(len(ds))
    item = ds[idx]
    if item is None:
        raise RuntimeError("selected sample invalid")

    model = StudentNet().eval()
    if args.ckpt:
        try:
            ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    with torch.no_grad():
        out = model(item["x"].unsqueeze(0))
    hm_s = out["prob"][0, 0].numpy()
    hm_t = item["hm_t"][0].numpy()

    frames = item["x"].numpy().reshape(3, 3, 288, 512).transpose(0, 2, 3, 1)
    frames = [(f * 255).astype(np.uint8) for f in frames]
    strip = np.concatenate(frames, axis=1)

    def heat_overlay(base: np.ndarray, hm: np.ndarray) -> np.ndarray:
        hm_up = cv2.resize(hm, (512, 288))
        color = cv2.applyColorMap((np.clip(hm_up, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(base, 0.6, cv2.cvtColor(color, cv2.COLOR_BGR2RGB), 0.4, 0)

    ov_t = heat_overlay(frames[1], hm_t)
    ov_s = heat_overlay(frames[1], hm_s)
    canvas = np.concatenate([strip, ov_t, ov_s], axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"sample_{idx}.jpg"
    cv2.imwrite(str(p), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(str(p))


if __name__ == "__main__":
    main()
