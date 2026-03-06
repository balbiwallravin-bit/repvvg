"""ROI helpers for masking in image/heatmap spaces."""
from __future__ import annotations

import numpy as np


def scale_roi(
    roi_xyxy: tuple[int, int, int, int],
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi_xyxy
    sw, sh = src_size
    dw, dh = dst_size
    sx = dw / max(float(sw), 1.0)
    sy = dh / max(float(sh), 1.0)
    rx0 = int(round(x0 * sx))
    ry0 = int(round(y0 * sy))
    rx1 = int(round(x1 * sx))
    ry1 = int(round(y1 * sy))
    rx0 = max(0, min(dw - 1, rx0))
    rx1 = max(0, min(dw - 1, rx1))
    ry0 = max(0, min(dh - 1, ry0))
    ry1 = max(0, min(dh - 1, ry1))
    if rx1 <= rx0:
        rx1 = min(dw - 1, rx0 + 1)
    if ry1 <= ry0:
        ry1 = min(dh - 1, ry0 + 1)
    return rx0, ry0, rx1, ry1


def mask_hw(arr: np.ndarray, roi_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """Mask outside ROI in HxW or CxHxW arrays."""
    x0, y0, x1, y1 = roi_xyxy
    out = np.zeros_like(arr)
    if arr.ndim == 2:
        out[y0 : y1 + 1, x0 : x1 + 1] = arr[y0 : y1 + 1, x0 : x1 + 1]
    elif arr.ndim == 3:
        out[:, y0 : y1 + 1, x0 : x1 + 1] = arr[:, y0 : y1 + 1, x0 : x1 + 1]
    else:
        raise ValueError(f"unsupported array ndim: {arr.ndim}")
    return out
