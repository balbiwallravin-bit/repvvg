from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from src.models.student_net import StudentNet


def _load_ckpt(path: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_start_ts(video_name: str) -> str:
    m = re.search(r"_S(\d{14})_", video_name)
    if not m:
        raise ValueError(f"cannot parse start timestamp from filename: {video_name}")
    return m.group(1)


def _preprocess_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (288, 512):
        rgb = cv2.resize(rgb, (512, 288), interpolation=cv2.INTER_LINEAR)
    return rgb.astype(np.float32) / 255.0


def _draw_result(frame_bgr: np.ndarray, mu_xy: np.ndarray, dir_xy: np.ndarray, length_px: float, score: float) -> np.ndarray:
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    sx = w / 512.0
    sy = h / 288.0

    cx = float(mu_xy[0] * sx)
    cy = float(mu_xy[1] * sy)

    dx = float(dir_xy[0])
    dy = float(dir_xy[1])
    n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    dx /= n
    dy /= n

    half = max(float(length_px), 4.0) * 0.5
    ex = dx * half * sx
    ey = dy * half * sy

    p1 = (int(round(cx - ex)), int(round(cy - ey)))
    p2 = (int(round(cx + ex)), int(round(cy + ey)))
    c = (0, 255, 0)
    cv2.line(vis, p1, p2, c, 2, lineType=cv2.LINE_AA)
    cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.putText(vis, f"score={score:.3f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    return vis


def track_one_video(video_path: Path, ckpt_path: Path, out_root: Path, device: torch.device, log_every: int = 30) -> Path:
    start_ts = _parse_start_ts(video_path.name)
    out_dir = out_root / start_ts
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}_tracked.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    model = StudentNet().to(device).eval()
    model.load_state_dict(_load_ckpt(str(ckpt_path))["model"])

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    if not frames:
        cap.release()
        writer.release()
        raise RuntimeError(f"empty video: {video_path}")

    print(f"[track] {video_path} total_frames={len(frames)} -> {out_path}")

    with torch.no_grad():
        for i in range(len(frames)):
            left = frames[max(i - 1, 0)]
            mid = frames[i]
            right = frames[min(i + 1, len(frames) - 1)]

            x = np.concatenate([
                _preprocess_rgb(left).transpose(2, 0, 1),
                _preprocess_rgb(mid).transpose(2, 0, 1),
                _preprocess_rgb(right).transpose(2, 0, 1),
            ], axis=0)
            xt = torch.from_numpy(x).unsqueeze(0).to(device)
            out = model(xt)
            mu_xy = out["mu_xy"][0, 0].cpu().numpy()
            dir_xy = out["dir_xy"][0, 0].cpu().numpy()
            length_px = float(out["l"][0, 0].cpu().item())
            score = float(out["prob"][0, 0].max().cpu().item())

            vis = _draw_result(mid, mu_xy, dir_xy, length_px, score)
            writer.write(vis)

            if log_every > 0 and ((i + 1) % log_every == 0 or i + 1 == len(frames)):
                pct = 100.0 * (i + 1) / len(frames)
                print(f"[progress] {video_path.name}: {i + 1}/{len(frames)} ({pct:.1f}%)")

    cap.release()
    writer.release()
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run StudentNet tracking on videos and export tracked videos.")
    ap.add_argument("--videos", nargs="+", required=True, help="input video paths")
    ap.add_argument("--ckpt", required=True, help="checkpoint path")
    ap.add_argument("--out_root", default="/home/lht/blurtrack/outputs", help="output root")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log_every", type=int, default=30)
    args = ap.parse_args()

    device = torch.device(args.device)
    out_root = Path(args.out_root)

    outputs: list[Path] = []
    for vp in args.videos:
        p = Path(vp)
        outputs.append(track_one_video(p, Path(args.ckpt), out_root, device, log_every=args.log_every))

    print("[done] outputs:")
    for op in outputs:
        print(str(op))


if __name__ == "__main__":
    main()
