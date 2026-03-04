"""Logging setup."""
from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(log_path: str | Path) -> logging.Logger:
    logger = logging.getLogger("blurtrack_kd")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
