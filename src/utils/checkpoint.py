"""Checkpoint loading helpers."""
from __future__ import annotations

import torch


def load_model_state_dict(ckpt_path: str) -> dict[str, torch.Tensor]:
    """Load model weights from checkpoint file.

    Supports both plain state_dict checkpoints and training checkpoints
    with a top-level "model" key.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)!r}")

    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a valid state_dict")

    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    return state
