"""I/O helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def write_lines(path: str | Path, lines: Iterable[str]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
