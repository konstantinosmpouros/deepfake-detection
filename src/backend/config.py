"""Backend configuration: repo paths, device selection, prediction storage.

Importing this module also puts ``notebooks/`` on ``sys.path`` so the app can
reuse the EXACT model builders / transforms the notebooks trained with
(`from utils import datasets, models, training, embed`). This keeps the app and
the research code in lock-step — no duplicated architecture definitions.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# This file lives at src/backend/config.py
BACKEND_DIR = Path(__file__).resolve().parent          # .../src/backend
REPO_ROOT = BACKEND_DIR.parents[1]                     # .../deepfake-detection
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
ARTIFACTS_DIR = NOTEBOOKS_DIR / "artifacts"            # per-pipeline models/ live here
DATA_DIR = REPO_ROOT / "data"
AIR_DIR = DATA_DIR / "ai-real-images"                  # holds norm_stats.json (dataset norm)
PREDICTIONS_DIR = BACKEND_DIR / "predictions"          # <pipeline>/<timestamp>/{image, prediction.json}

# Make `from utils import ...` resolve to notebooks/utils (the notebooks do the same).
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))


def get_device() -> torch.device:
    """Resolve the inference device. Set DF_DEVICE=cpu to keep off the GPU
    (e.g. while models are training there)."""
    forced = os.environ.get("DF_DEVICE")
    if forced:
        return torch.device(forced)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Canonical pipeline *folder* names (CLAUDE.md). Selectable keys may differ
# (cnn-finetune exposes one key per backbone) — see pipelines/__init__.py.
PIPELINE_NAMES = [
    "cnn-scratch",
    "cnn-residual",
    "cnn-finetune",
    "vit-lora",
    "clip-probe",
    "two-stream",
]
