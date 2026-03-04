"""Frame window dataset for KD."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.npz_pseudo import load_npz_hm
from src.datasets.ready_index import SampleSpec
from src.datasets.transforms import read_rgb_288x512


class FrameWindowDataset(Dataset):
    def __init__(self, specs: list[SampleSpec], pseudo_root: str, strict: int = 0):
        self.specs = specs
        self.pseudo_root = Path(pseudo_root)
        self.strict = strict
        self.badcases: dict[str, list[str]] = {"missing_frames": [], "missing_npz": []}

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        spec = self.specs[idx]
        try:
            imgs = [read_rgb_288x512(p) for p in spec.frame_paths]
            x = torch.from_numpy(np.concatenate([i.transpose(2, 0, 1) for i in imgs], axis=0)).float()
        except Exception:
            self.badcases["missing_frames"].append("|".join(spec.frame_paths))
            if self.strict:
                raise
            return None
        npz_path = str(self.pseudo_root / spec.segment / f"{spec.target_id}.npz")
        try:
            hm, score, score_raw = load_npz_hm(npz_path, strict=self.strict)
        except Exception:
            self.badcases["missing_npz"].append(npz_path)
            if self.strict:
                raise
            return None

        return {
            "x": x,
            "hm_t": torch.from_numpy(hm),
            "score": torch.tensor(float(max(0.0, min(1.0, score))), dtype=torch.float32),
            "score_raw": torch.tensor(float(score_raw), dtype=torch.float32),
            "meta": {"segment": spec.segment, "target_id": spec.target_id, "frame_paths": spec.frame_paths},
        }


def kd_collate(batch: list[dict[str, Any] | None]) -> dict[str, Any]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "hm_t": torch.stack([b["hm_t"] for b in batch]),
        "score": torch.stack([b["score"] for b in batch]),
        "score_raw": torch.stack([b["score_raw"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }
