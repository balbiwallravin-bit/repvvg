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

# Default ROI in original 1920x1080 coordinate system.
ROI_X0 = 367
ROI_Y0 = 100
ROI_X1 = 1760
ROI_Y1 = 884


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


def _model_to_frame_xy(mu_xy_model: np.ndarray, w: int, h: int) -> np.ndarray:
    sx = w / 512.0
    sy = h / 288.0
    return np.array([float(mu_xy_model[0]) * sx, float(mu_xy_model[1]) * sy], dtype=np.float32)


def _center_weight(point_xy: np.ndarray, roi: tuple[int, int, int, int], sigma_scale: float = 0.35) -> float:
    x0, y0, x1, y1 = roi
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    hw = max((x1 - x0) * 0.5, 1.0)
    hh = max((y1 - y0) * 0.5, 1.0)

    nx = (float(point_xy[0]) - cx) / hw
    ny = (float(point_xy[1]) - cy) / hh
    d2 = nx * nx + ny * ny
    sigma2 = max(sigma_scale * sigma_scale, 1e-6)
    return float(np.exp(-0.5 * d2 / sigma2))


def _in_roi(point_xy: np.ndarray, roi: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = roi
    x, y = float(point_xy[0]), float(point_xy[1])
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def _draw_result(
    frame_bgr: np.ndarray,
    mu_xy_model: np.ndarray,
    dir_xy: np.ndarray,
    length_px_model: float,
    score_raw: float,
    score_roi: float,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    sx = w / 512.0
    sy = h / 288.0

    center_xy = _model_to_frame_xy(mu_xy_model, w, h)
    cx = float(center_xy[0])
    cy = float(center_xy[1])

    dx = float(dir_xy[0])
    dy = float(dir_xy[1])
    n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    dx /= n
    dy /= n

    half = max(float(length_px_model), 4.0) * 0.5
    ex = dx * half * sx
    ey = dy * half * sy

    p1 = (int(round(cx - ex)), int(round(cy - ey)))
    p2 = (int(round(cx + ex)), int(round(cy + ey)))
    c = (0, 255, 0)
    cv2.line(vis, p1, p2, c, 2, lineType=cv2.LINE_AA)
    cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    x0, y0, x1, y1 = roi
    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 128, 0), 2, lineType=cv2.LINE_AA)

    cv2.putText(vis, f"score_raw={score_raw:.3f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, f"score_roi={score_roi:.3f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def track_one_video(
    video_path: Path,
    ckpt_path: Path,
    out_root: Path,
    device: torch.device,
    roi: tuple[int, int, int, int],
    smooth_alpha: float,
    center_boost: float,
    log_every: int = 30,
) -> Path:
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

    smoothed_mu_model: np.ndarray | None = None

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

            mu_model = out["mu_xy"][0, 0].cpu().numpy().astype(np.float32)
            if smoothed_mu_model is None:
                smoothed_mu_model = mu_model
            else:
                smoothed_mu_model = smooth_alpha * mu_model + (1.0 - smooth_alpha) * smoothed_mu_model

            dir_xy = out["dir_xy"][0, 0].cpu().numpy()
            length_px = float(out["l"][0, 0].cpu().item())
            score_raw = float(out["prob"][0, 0].max().cpu().item())

            center_xy = _model_to_frame_xy(smoothed_mu_model, w, h)
            in_roi = _in_roi(center_xy, roi)
            c_weight = _center_weight(center_xy, roi)
            roi_gate = 1.0 if in_roi else 0.05
            score_roi = score_raw * (roi_gate * (1.0 + center_boost * c_weight))

            vis = _draw_result(mid, smoothed_mu_model, dir_xy, length_px, score_raw, score_roi, roi)
            writer.write(vis)

            if log_every > 0 and ((i + 1) % log_every == 0 or i + 1 == len(frames)):
                pct = 100.0 * (i + 1) / len(frames)
                print(f"[progress] {video_path.name}: {i + 1}/{len(frames)} ({pct:.1f}%) score_roi={score_roi:.3f}")

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
    ap.add_argument("--smooth_alpha", type=float, default=0.35, help="EMA alpha for center smoothing")
    ap.add_argument("--center_boost", type=float, default=1.0, help="extra score multiplier near ROI center")
    ap.add_argument("--roi_x0", type=int, default=ROI_X0)
    ap.add_argument("--roi_y0", type=int, default=ROI_Y0)
    ap.add_argument("--roi_x1", type=int, default=ROI_X1)
    ap.add_argument("--roi_y1", type=int, default=ROI_Y1)
    args = ap.parse_args()

    device = torch.device(args.device)
    out_root = Path(args.out_root)
    roi = (args.roi_x0, args.roi_y0, args.roi_x1, args.roi_y1)

    outputs: list[Path] = []
    for vp in args.videos:
        p = Path(vp)
        outputs.append(
            track_one_video(
                p,
                Path(args.ckpt),
                out_root,
                device,
                roi=roi,
                smooth_alpha=args.smooth_alpha,
                center_boost=args.center_boost,
                log_every=args.log_every,
            )
        )

    print("[done] outputs:")
    for op in outputs:
        print(str(op))


if __name__ == "__main__":
    main()
