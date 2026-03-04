"""Robust parser for index_train_ready.jsonl."""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SampleSpec:
    segment: str
    frame_paths: tuple[str, str, str]
    target_frame_path: str
    target_id: str


def _to_frame_id(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, int):
        return f"{v:06d}"
    s = str(v)
    stem = Path(s).stem
    m = re.search(r"(\d+)$", stem)
    if not m:
        return None
    return m.group(1).zfill(6)


def _segment_from_path(path: str) -> str | None:
    m = re.search(r"video_maked_ready/([^/]+)/frames_roi", path)
    return m.group(1) if m else None


def parse_index(jsonl_path: str, ready_root: str) -> list[SampleSpec]:
    """Parse JSONL index with adaptive schema handling."""
    specs: list[SampleSpec] = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            segment = obj.get("segment")
            frames = obj.get("frames")
            if not segment and isinstance(frames, list) and frames:
                segment = _segment_from_path(str(frames[0]))
            if not segment:
                warnings.warn("skip line without segment")
                continue

            ids: list[str] = []
            if all(k in obj for k in ["t0", "t1", "t2"]):
                ids = [_to_frame_id(obj.get("t0")), _to_frame_id(obj.get("t1")), _to_frame_id(obj.get("t2"))]  # type: ignore[list-item]
                if any(x is None for x in ids):
                    warnings.warn("skip line with invalid explicit frame ids")
                    continue
                target_id = _to_frame_id(obj.get("target"))
                if target_id is None:
                    target_id = ids[1]
            elif isinstance(frames, list) and len(frames) >= 3:
                ids = [_to_frame_id(p) for p in frames[:3]]  # type: ignore[list-item]
                if any(x is None for x in ids):
                    continue
                t = obj.get("target", 1)
                if isinstance(t, int):
                    t_idx = max(0, min(2, t))
                    target_id = ids[t_idx]
                else:
                    target_id = _to_frame_id(t) or ids[1]
            else:
                continue

            frame_paths = tuple(str(Path(ready_root) / segment / "frames_roi" / f"{fid}.jpg") for fid in ids)
            target_path = str(Path(ready_root) / segment / "frames_roi" / f"{target_id}.jpg")
            specs.append(SampleSpec(segment=segment, frame_paths=frame_paths, target_frame_path=target_path, target_id=target_id))
        except Exception:
            continue
    return specs
