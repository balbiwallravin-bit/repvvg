"""Image transforms."""
from __future__ import annotations

import cv2
import numpy as np


def read_rgb_288x512(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (288, 512):
        img = cv2.resize(img, (512, 288), interpolation=cv2.INTER_LINEAR)
    return (img.astype(np.float32) / 255.0)
