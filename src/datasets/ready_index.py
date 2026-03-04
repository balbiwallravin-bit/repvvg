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
    s = str(v).strip()
    if not s:
        return None
    stem = Path(s).stem
    m = re.search(r"(\d+)$", stem)
    if not m:
        return None
    return m.group(1).zfill(6)


def _segment_from_path(path: str) -> str | None:
    """Extract segment from common ready/raw frame path formats."""
    m = re.search(r"/(?:video_maked_ready|video_maked)/([^/]+)/frames_roi(?:/|$)", path)
    if m:
        return m.group(1)

    p = Path(path)
    parts = p.parts
    if "frames_roi" in parts:
        idx = parts.index("frames_roi")
        if idx >= 1:
            return parts[idx - 1]
    return None


def _collect_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            out.extend(_collect_strings(x))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect_strings(v))
    return out


def _collect_paths_from_obj(obj: dict[str, Any]) -> list[str]:
    all_strings = _collect_strings(obj)
    return [s for s in all_strings if "/frames_roi/" in s or s.endswith(".jpg") or s.endswith(".png")]


def _segment_from_obj(obj: dict[str, Any]) -> str | None:
    for key in ["segment", "seg", "scene", "clip", "video", "segment_id", "seq", "sequence"]:
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _ids_from_obj(obj: dict[str, Any], candidate_paths: list[str]) -> tuple[list[str], str | None]:
    # explicit t0/t1/t2 style
    explicit_key_sets = [
        ("t0", "t1", "t2"),
        ("tm2", "tm1", "t"),
        ("prev2", "prev1", "curr"),
        ("frame0", "frame1", "frame2"),
    ]
    for k0, k1, k2 in explicit_key_sets:
        if all(k in obj for k in [k0, k1, k2]):
            ids = [_to_frame_id(obj.get(k0)), _to_frame_id(obj.get(k1)), _to_frame_id(obj.get(k2))]  # type: ignore[list-item]
            if any(i is None for i in ids):
                return [], None
            target_id = _to_frame_id(obj.get("target")) or _to_frame_id(obj.get("target_id")) or ids[1]
            return ids, target_id

    # any list-like frame container
    list_keys = ["frames", "frame_paths", "imgs", "images", "window", "triplet"]
    for lk in list_keys:
        v = obj.get(lk)
        if isinstance(v, list) and len(v) >= 3:
            ids = [_to_frame_id(x) for x in v[:3]]  # type: ignore[list-item]
            if any(i is None for i in ids):
                continue
            t = obj.get("target", obj.get("target_idx", obj.get("label", 1)))
            if isinstance(t, int):
                target = ids[max(0, min(2, t))]
            else:
                target = _to_frame_id(t) or ids[1]
            return ids, target

    # fallback: use first 3 path-like strings found recursively
    path_ids = [_to_frame_id(p) for p in candidate_paths if _to_frame_id(p) is not None]
    if len(path_ids) >= 3:
        ids = path_ids[:3]
        t = obj.get("target", obj.get("target_idx", 1))
        if isinstance(t, int):
            target = ids[max(0, min(2, t))]
        else:
            target = _to_frame_id(t) or ids[1]
        return ids, target

    return [], None


def parse_index(jsonl_path: str, ready_root: str) -> list[SampleSpec]:
    """Parse JSONL index with adaptive schema handling."""
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

            candidate_paths = _collect_paths_from_obj(obj)
            segment = _segment_from_obj(obj)
            if not segment:
                for p in candidate_paths:
                    segment = _segment_from_path(p)
                    if segment:
                        break

            if not segment:
                warnings.warn(f"skip line {ln}: missing segment")
                continue

            ids, target_id = _ids_from_obj(obj, candidate_paths)
            if len(ids) != 3 or target_id is None:
                warnings.warn(f"skip line {ln}: cannot resolve frame ids")
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
