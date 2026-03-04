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


def _resolve_frame_path(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)

    parent = p.parent
    stem = p.stem
    suffixes = [p.suffix, ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    suffixes = [s for i, s in enumerate(suffixes) if s and s not in suffixes[:i]]

    # numeric fallback: 000254 -> 254 and vice versa
    cand_stems = [stem]
    if stem.isdigit():
        raw = str(int(stem))
        z6 = stem.zfill(6)
        cand_stems.extend([raw, z6])

    for cs in cand_stems:
        for suf in suffixes:
            cp = parent / f"{cs}{suf}"
            if cp.exists():
                return str(cp)

    # final fuzzy fallback: any file with same numeric token
    if stem.isdigit():
        token = str(int(stem))
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            for fp in parent.glob(ext):
                if fp.stem.isdigit() and str(int(fp.stem)) == token:
                    return str(fp)

    raise FileNotFoundError(path)


def _candidate_npz_paths(pseudo_root: Path, segment: str, target_id: str) -> list[str]:
    ids = [target_id]
    if target_id.isdigit():
        ids.extend([str(int(target_id)), target_id.zfill(6)])
    # dedupe preserve order
    uniq: list[str] = []
    for x in ids:
        if x not in uniq:
            uniq.append(x)
    return [str(pseudo_root / segment / f"{x}.npz") for x in uniq]


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
            resolved = [_resolve_frame_path(p) for p in spec.frame_paths]
            imgs = [read_rgb_288x512(p) for p in resolved]
            x = torch.from_numpy(np.concatenate([i.transpose(2, 0, 1) for i in imgs], axis=0)).float()
        except Exception:
            self.badcases["missing_frames"].append("|".join(spec.frame_paths))
            if self.strict:
                raise
            return None

        hm = None
        score = 1.0
        score_raw = 1.0
        npz_err = None
        used_npz = None
        for npz_path in _candidate_npz_paths(self.pseudo_root, spec.segment, spec.target_id):
            try:
                hm, score, score_raw = load_npz_hm(npz_path, strict=self.strict)
                used_npz = npz_path
                break
            except Exception as e:
                npz_err = e
                continue

        if hm is None:
            self.badcases["missing_npz"].append(str(npz_err) if npz_err is not None else f"{spec.segment}:{spec.target_id}")
            if self.strict:
                raise ValueError(f"npz not found for {spec.segment}/{spec.target_id}")
            return None

        return {
            "x": x,
            "hm_t": torch.from_numpy(hm),
            "score": torch.tensor(float(max(0.0, min(1.0, score))), dtype=torch.float32),
            "score_raw": torch.tensor(float(score_raw), dtype=torch.float32),
            "meta": {
                "segment": spec.segment,
                "target_id": spec.target_id,
                "frame_paths": spec.frame_paths,
                "resolved_frame_paths": resolved,
                "npz_path": used_npz,
            },
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
