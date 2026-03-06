from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    mod_path = Path(__file__).resolve().parents[1] / "tools" / "report_training.py"
    spec = importlib.util.spec_from_file_location("report_training", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_report_contains_key_sections() -> None:
    mod = _load_module()
    rows = [
        {"step": 0, "epoch": 0, "loss": 10.0, "mu_err_px_visible": 5.0, "visi_acc": 0.4, "l_kl": 1.0, "l_mu": 2.0, "l_sigma": 3.0, "l_vis": 0.8, "mu_err_px": 6.0, "mean_score": 0.2, "mean_visi": 0.5},
        {"step": 10, "epoch": 1, "loss": 8.0, "mu_err_px_visible": 4.0, "visi_acc": 0.6, "l_kl": 0.9, "l_mu": 1.8, "l_sigma": 2.5, "l_vis": 0.6, "mu_err_px": 5.0, "mean_score": 0.3, "mean_visi": 0.6},
    ]
    rep = mod.build_report(rows)
    assert "# Training Report" in rep
    assert "| metric | first | last |" in rep
    assert "| loss |" in rep
    assert "## Epoch-level means" in rep
