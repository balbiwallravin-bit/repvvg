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
from src.utils.roi import mask_hw, scale_roi


def _resolve_frame_path(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)

    parent = p.parent
    stem = p.stem
    suffixes = [p.suffix, ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    suffixes = [s for i, s in enumerate(suffixes) if s and s not in suffixes[:i]]

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

    if stem.isdigit():
        token = str(int(stem))
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            for fp in parent.glob(ext):
                if fp.stem.isdigit() and str(int(fp.stem)) == token:
                    return str(fp)

    raise FileNotFoundError(path)


def _candidate_npz_paths(pseudo_root: Path, segment: str, target_id: str) -> list[str]:
    ids: list[str] = [target_id]
    if target_id.isdigit():
        raw = str(int(target_id))
        ids.append(raw)
        for w in (3, 4, 5, 6):
            ids.append(raw.zfill(w))

    uniq: list[str] = []
    for x in ids:
        if x not in uniq:
            uniq.append(x)
    return [str(pseudo_root / segment / f"{x}.npz") for x in uniq]


class FrameWindowDataset(Dataset):
    def __init__(
        self,
        specs: list[SampleSpec],
        pseudo_root: str,
        strict: int = 0,
        roi_enable: int = 1,
        roi_ref_w: int = 1920,
        roi_ref_h: int = 1080,
        roi_x0: int = 367,
        roi_y0: int = 100,
        roi_x1: int = 1760,
        roi_y1: int = 884,
        visi_thr: float = 0.25,
        hard_neg_ratio: float = 0.0,
    ):
        self.specs = specs
        self.pseudo_root = Path(pseudo_root)
        self.strict = strict
        self.badcases: dict[str, list[str]] = {"missing_frames": [], "missing_npz": []}

        self.roi_enable = bool(roi_enable)
        self.roi_ref = (roi_x0, roi_y0, roi_x1, roi_y1)
        self.roi_ref_size = (roi_ref_w, roi_ref_h)
        self.roi_img = scale_roi(self.roi_ref, self.roi_ref_size, (512, 288))
        self.roi_hm = scale_roi(self.roi_ref, self.roi_ref_size, (128, 72))

        self.visi_thr = float(visi_thr)
        self.hard_neg_ratio = max(0.0, float(hard_neg_ratio))
        self.indices = list(range(len(self.specs)))
        if self.hard_neg_ratio > 0:
            self._append_hard_negatives()

    def _append_hard_negatives(self) -> None:
        low: list[tuple[float, int]] = []
        for i, spec in enumerate(self.specs):
            score = 1.0
            for npz_path in _candidate_npz_paths(self.pseudo_root, spec.segment, spec.target_id):
                try:
                    _, score, _ = load_npz_hm(npz_path, strict=0)
                    break
                except Exception:
                    continue
            low.append((float(score), i))
        low.sort(key=lambda x: x[0])
        n_pick = int(len(low) * min(self.hard_neg_ratio, 1.0))
        self.indices.extend([idx for _, idx in low[:n_pick]])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        spec = self.specs[self.indices[idx]]
        try:
            resolved = [_resolve_frame_path(p) for p in spec.frame_paths]
            imgs = [read_rgb_288x512(p) for p in resolved]
            x = np.concatenate([i.transpose(2, 0, 1) for i in imgs], axis=0).astype(np.float32)
            if self.roi_enable:
                x = mask_hw(x, self.roi_img)
            x_t = torch.from_numpy(x).float()
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

        hm = hm.astype(np.float32)
        if self.roi_enable:
            hm = mask_hw(hm, self.roi_hm)
        visi = 1.0 if float(score) >= self.visi_thr else 0.0

        return {
            "x": x_t,
            "hm_t": torch.from_numpy(hm.astype(np.float16)),
            "score": torch.tensor(float(max(0.0, min(1.0, score))), dtype=torch.float32),
            "score_raw": torch.tensor(float(score_raw), dtype=torch.float32),
            "visi": torch.tensor(visi, dtype=torch.float32),
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
        "visi": torch.stack([b["visi"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }
