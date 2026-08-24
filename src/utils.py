"""Shared helper: seeding, config loading, device resolution, logging setup."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict[str, Any]) -> torch.device:
    """Resolve device strictly from config — no auto-detection.

    Config picks the device explicitly (`device: cpu|mps|cuda`) so the same
    yaml never silently trains on the wrong hardware when moved between
    the Mac dev machine and the GPU server.
    """
    device_str = cfg["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configs requests device: cuda but torch.cuda.is_available() is False")
    if device_str == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("configs requests device: mps but torch.backends.mps.is_available() is False")
    return torch.device(device_str)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
