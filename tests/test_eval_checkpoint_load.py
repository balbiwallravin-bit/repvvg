from __future__ import annotations

from pathlib import Path

import torch

from src.utils.checkpoint import load_model_state_dict


def test_load_model_state_dict_from_training_checkpoint(tmp_path: Path) -> None:
    ckpt = {
        "model": {
            "module.backbone.weight": torch.ones(2, 2),
            "module.backbone.bias": torch.zeros(2),
        },
        "opt": {"lr": 1e-3},
    }
    ckpt_path = tmp_path / "train_ckpt.pt"
    torch.save(ckpt, ckpt_path)

    state = load_model_state_dict(str(ckpt_path))
    assert "backbone.weight" in state
    assert "backbone.bias" in state


def test_load_model_state_dict_from_plain_state_dict(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "plain.pt"
    torch.save({"decoder.weight": torch.randn(1, 1)}, ckpt_path)

    state = load_model_state_dict(str(ckpt_path))
    assert "decoder.weight" in state
