from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.datasets.frame_window_dataset import FrameWindowDataset
from src.datasets.ready_index import parse_index


def test_dataset_mapping(tmp_path: Path) -> None:
    ready = tmp_path / "video_maked_ready"
    pseudo = tmp_path / "pseudo"
    seg = "S001"
    fr = ready / seg / "frames_roi"
    npz_dir = pseudo / seg
    fr.mkdir(parents=True)
    npz_dir.mkdir(parents=True)

    for fid in ["000123", "000124", "000125"]:
        img = np.zeros((288, 512, 3), dtype=np.uint8)
        cv2.imwrite(str(fr / f"{fid}.jpg"), img)
    np.savez(npz_dir / "000124.npz", hm=np.zeros((72, 128), dtype=np.float32), score=0.8, score_raw=0.9)

    index = tmp_path / "index.jsonl"
    rec = {
        "frames": [
            str(fr / "000123.jpg"),
            str(fr / "000124.jpg"),
            str(fr / "000125.jpg"),
        ],
        "target": 1,
    }
    index.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    specs = parse_index(str(index), str(ready))
    assert len(specs) == 1
    assert specs[0].target_id == "000124"

    ds = FrameWindowDataset(specs, str(pseudo))
    item = ds[0]
    assert item is not None
    assert item["x"].shape == (9, 288, 512)
    assert item["hm_t"].shape == (1, 72, 128)
