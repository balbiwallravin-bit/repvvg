"""NPZ pseudo heatmap loading."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_npz_hm(npz_path: str, strict: int = 0) -> tuple[np.ndarray, float, float]:
    """Load pseudo labels and normalize hm to [1,72,128] float16."""
    p = Path(npz_path)
    if not p.exists():
        if strict:
            raise FileNotFoundError(npz_path)
        raise ValueError(f"missing_npz:{npz_path}")

    try:
        data = np.load(npz_path)
        hm = data["hm"]
        score = float(data.get("score", 1.0))
        score_raw = float(data.get("score_raw", score))
    except Exception as e:
        if strict:
            raise
        raise ValueError(f"bad_npz:{npz_path}:{e}")

    hm = np.asarray(hm, dtype=np.float32)
    if hm.ndim == 3:
        hm = hm[0]
    if hm.shape == (288, 512):
        hm = cv2.resize(hm, (128, 72), interpolation=cv2.INTER_AREA)
    elif hm.shape != (72, 128):
        raise ValueError(f"unsupported_hm_shape:{hm.shape}")
    hm = hm[None, :, :].astype(np.float16)
    return hm, score, score_raw
