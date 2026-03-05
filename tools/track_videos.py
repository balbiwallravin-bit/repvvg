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

# Default ROI in reference 1920x1080 coordinate system.
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


def _frame_to_model_xy(point_xy: np.ndarray, w: int, h: int) -> np.ndarray:
    sx = 512.0 / max(float(w), 1.0)
    sy = 288.0 / max(float(h), 1.0)
    return np.array([float(point_xy[0]) * sx, float(point_xy[1]) * sy], dtype=np.float32)


def _scale_roi_to_frame(roi_ref: tuple[int, int, int, int], ref_size: tuple[int, int], frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
    rx0, ry0, rx1, ry1 = roi_ref
    ref_w, ref_h = ref_size
    fw, fh = frame_size
    sx = fw / max(float(ref_w), 1.0)
    sy = fh / max(float(ref_h), 1.0)

    x0 = int(round(rx0 * sx))
    y0 = int(round(ry0 * sy))
    x1 = int(round(rx1 * sx))
    y1 = int(round(ry1 * sy))
    x0 = max(0, min(fw - 1, x0))
    x1 = max(0, min(fw - 1, x1))
    y0 = max(0, min(fh - 1, y0))
    y1 = max(0, min(fh - 1, y1))
    if x1 <= x0:
        x1 = min(fw - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(fh - 1, y0 + 1)
    return (x0, y0, x1, y1)


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


def _clip_to_roi(point_xy: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    return np.array([
        float(np.clip(point_xy[0], x0, x1)),
        float(np.clip(point_xy[1], y0, y1)),
    ], dtype=np.float32)


def _roi_masked_mu_from_prob(prob_hw: np.ndarray, roi_frame: tuple[int, int, int, int], frame_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Return model-space mu_xy from ROI-masked heatmap and masked max score."""
    h_hm, w_hm = prob_hw.shape
    fw, fh = frame_size
    x0, y0, x1, y1 = roi_frame

    sx = w_hm / max(float(fw), 1.0)
    sy = h_hm / max(float(fh), 1.0)
    hx0 = int(np.clip(np.floor(x0 * sx), 0, w_hm - 1))
    hx1 = int(np.clip(np.ceil(x1 * sx), 0, w_hm - 1))
    hy0 = int(np.clip(np.floor(y0 * sy), 0, h_hm - 1))
    hy1 = int(np.clip(np.ceil(y1 * sy), 0, h_hm - 1))

    mask = np.zeros_like(prob_hw, dtype=np.float32)
    mask[hy0 : hy1 + 1, hx0 : hx1 + 1] = 1.0
    p = prob_hw.astype(np.float32) * mask
    s = float(p.sum())
    if s <= 1e-8:
        p = prob_hw.astype(np.float32)
        s = float(p.sum())
    p /= max(s, 1e-8)

    yy, xx = np.meshgrid(np.arange(h_hm, dtype=np.float32), np.arange(w_hm, dtype=np.float32), indexing="ij")
    mx = float((p * xx).sum())
    my = float((p * yy).sum())

    # heatmap index -> model 512x288 coords (stride=4 with +0.5 cell center)
    mu_model = np.array([(mx + 0.5) * 4.0, (my + 0.5) * 4.0], dtype=np.float32)
    score_masked = float(p.max())
    return mu_model, score_masked


def _draw_result(
    frame_bgr: np.ndarray,
    mu_xy_model: np.ndarray,
    dir_xy: np.ndarray,
    length_px_model: float,
    score_raw: float,
    score_roi: float,
    roi: tuple[int, int, int, int],
    draw_track: bool,
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

    x0, y0, x1, y1 = roi
    p1 = (int(round(np.clip(cx - ex, x0, x1))), int(round(np.clip(cy - ey, y0, y1))))
    p2 = (int(round(np.clip(cx + ex, x0, x1))), int(round(np.clip(cy + ey, y0, y1))))
    c = (0, 255, 0)
    if draw_track:
        cv2.line(vis, p1, p2, c, 2, lineType=cv2.LINE_AA)
        cv2.circle(vis, (int(round(np.clip(cx, x0, x1))), int(round(np.clip(cy, y0, y1)))), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 128, 0), 2, lineType=cv2.LINE_AA)

    cv2.putText(vis, f"score_raw={score_raw:.3f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, f"score_roi={score_roi:.3f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"visible={1 if draw_track else 0}", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2, cv2.LINE_AA)
    return vis


def track_one_video(
    video_path: Path,
    ckpt_path: Path,
    out_root: Path,
    device: torch.device,
    roi_ref: tuple[int, int, int, int],
    roi_ref_size: tuple[int, int],
    smooth_alpha: float,
    center_boost: float,
    log_every: int = 30,
    min_score_roi: float = 0.12,
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
    roi_frame = _scale_roi_to_frame(roi_ref, roi_ref_size, (w, h))

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

    print(f"[track] {video_path} total_frames={len(frames)} roi_frame={roi_frame} -> {out_path}")

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

            prob = out["prob"][0, 0].cpu().numpy()
            mu_model, score_masked = _roi_masked_mu_from_prob(prob, roi_frame, (w, h))

            if smoothed_mu_model is None:
                smoothed_mu_model = mu_model
            else:
                smoothed_mu_model = smooth_alpha * mu_model + (1.0 - smooth_alpha) * smoothed_mu_model

            smoothed_xy_frame = _model_to_frame_xy(smoothed_mu_model, w, h)
            smoothed_xy_frame = _clip_to_roi(smoothed_xy_frame, roi_frame)
            smoothed_mu_model = _frame_to_model_xy(smoothed_xy_frame, w, h)

            dir_xy = out["dir_xy"][0, 0].cpu().numpy()
            length_px = float(out["l"][0, 0].cpu().item())
            score_raw = float(out["prob"][0, 0].max().cpu().item())

            c_weight = _center_weight(smoothed_xy_frame, roi_frame)
            in_roi = _in_roi(smoothed_xy_frame, roi_frame)
            roi_gate = 1.0 if in_roi else 0.05
            score_roi = score_masked * (roi_gate * (1.0 + center_boost * c_weight))

            draw_track = score_roi >= min_score_roi
            vis = _draw_result(mid, smoothed_mu_model, dir_xy, length_px, score_raw, score_roi, roi_frame, draw_track=draw_track)
            writer.write(vis)

            if log_every > 0 and ((i + 1) % log_every == 0 or i + 1 == len(frames)):
                pct = 100.0 * (i + 1) / len(frames)
                print(f"[progress] {video_path.name}: {i + 1}/{len(frames)} ({pct:.1f}%) score_roi={score_roi:.3f} visible={int(draw_track)}")

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
    ap.add_argument("--roi_ref_w", type=int, default=1920, help="ROI reference width")
    ap.add_argument("--roi_ref_h", type=int, default=1080, help="ROI reference height")
    ap.add_argument("--roi_x0", type=int, default=ROI_X0)
    ap.add_argument("--roi_y0", type=int, default=ROI_Y0)
    ap.add_argument("--roi_x1", type=int, default=ROI_X1)
    ap.add_argument("--roi_y1", type=int, default=ROI_Y1)
    ap.add_argument("--min_score_roi", type=float, default=0.12, help="hide track drawing when score_roi is below threshold")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_root = Path(args.out_root)
    roi_ref = (args.roi_x0, args.roi_y0, args.roi_x1, args.roi_y1)
    roi_ref_size = (args.roi_ref_w, args.roi_ref_h)

    outputs: list[Path] = []
    for vp in args.videos:
        p = Path(vp)
        outputs.append(
            track_one_video(
                p,
                Path(args.ckpt),
                out_root,
                device,
                roi_ref=roi_ref,
                roi_ref_size=roi_ref_size,
                smooth_alpha=args.smooth_alpha,
                center_boost=args.center_boost,
                log_every=args.log_every,
                min_score_roi=args.min_score_roi,
            )
        )

    print("[done] outputs:")
    for op in outputs:
        print(str(op))


if __name__ == "__main__":
    main()
