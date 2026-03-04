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
    """Normalized sample spec for KD training."""

    segment: str
    frame_paths: tuple[str, str, str]
    target_frame_path: str
    target_id: str


def _to_frame_id(v: Any) -> str | None:
    """Convert frame-like value to zero-padded numeric frame id."""
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
    """Extract segment from common ready/raw frame path formats."""
    # preferred format:
    # .../video_maked_ready/{segment}/frames_roi/000123.jpg
    # fallback raw format:
    # .../video_maked/{segment}/frames_roi/000123.jpg
    m = re.search(r"/(?:video_maked_ready|video_maked)/([^/]+)/frames_roi(?:/|$)", path)
    if m:
        return m.group(1)

    p = Path(path)
    # generic fallback: .../{segment}/frames_roi/<file>
    parts = p.parts
    if "frames_roi" in parts:
        idx = parts.index("frames_roi")
        if idx >= 1:
            return parts[idx - 1]
    return None


def _segment_from_obj(obj: dict[str, Any]) -> str | None:
    for key in ["segment", "seg", "scene", "clip", "video"]:
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def parse_index(jsonl_path: str, ready_root: str) -> list[SampleSpec]:
    """Parse JSONL index with adaptive schema handling.

    Supports both explicit-frame schema and path-list schema.
    Invalid lines are skipped with warnings.
    """
    specs: list[SampleSpec] = []
    lines = Path(jsonl_path).read_text(encoding="utf-8").splitlines()

    for ln, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                warnings.warn(f"skip line {ln}: not a JSON object")
                continue

            frames = obj.get("frames")
            segment = _segment_from_obj(obj)
            if not segment and isinstance(frames, list) and frames:
                segment = _segment_from_path(str(frames[0]))
            if not segment:
                # probe more path-like keys
                for path_key in ["frame_path", "target_frame_path", "img_path", "path"]:
                    pv = obj.get(path_key)
                    if isinstance(pv, str):
                        segment = _segment_from_path(pv)
                        if segment:
                            break
            if not segment:
                warnings.warn(f"skip line {ln}: missing segment")
                continue

            ids: list[str] = []
            target_id: str | None = None

            if all(k in obj for k in ["t0", "t1", "t2"]):
                ids = [_to_frame_id(obj.get("t0")), _to_frame_id(obj.get("t1")), _to_frame_id(obj.get("t2"))]  # type: ignore[list-item]
                if any(x is None for x in ids):
                    warnings.warn(f"skip line {ln}: invalid explicit frame ids")
                    continue
                target_id = _to_frame_id(obj.get("target")) or ids[1]

            elif isinstance(frames, list) and len(frames) >= 3:
                ids = [_to_frame_id(p) for p in frames[:3]]  # type: ignore[list-item]
                if any(x is None for x in ids):
                    warnings.warn(f"skip line {ln}: invalid frame id in frames list")
                    continue

                t = obj.get("target", 1)
                if isinstance(t, int):
                    t_idx = max(0, min(2, t))
                    target_id = ids[t_idx]
                else:
                    target_id = _to_frame_id(t) or ids[1]

            else:
                warnings.warn(f"skip line {ln}: no supported frame fields")
                continue

            frame_paths = tuple(str(Path(ready_root) / segment / "frames_roi" / f"{fid}.jpg") for fid in ids)
            target_path = str(Path(ready_root) / segment / "frames_roi" / f"{target_id}.jpg")
            specs.append(
                SampleSpec(
                    segment=segment,
                    frame_paths=frame_paths,
                    target_frame_path=target_path,
                    target_id=target_id,
                )
            )
        except Exception as exc:
            warnings.warn(f"skip line {ln}: parse exception: {exc}")

    return specs
