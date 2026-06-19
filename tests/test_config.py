"""Tests for src/backend/config.py — repo wiring + device selection.

Importing config pulls in torch (so the app and notebooks share one stack) and
puts notebooks/ on sys.path. The DF_DEVICE override is the knob that keeps the
app off the GPU while models train there.
"""
from __future__ import annotations

import sys

import torch

from src.backend import config as cfg


def test_repo_root_and_dirs_are_consistent():
    assert (cfg.REPO_ROOT / "CLAUDE.md").exists()
    assert cfg.NOTEBOOKS_DIR == cfg.REPO_ROOT / "notebooks"
    assert cfg.ARTIFACTS_DIR == cfg.NOTEBOOKS_DIR / "artifacts"
    assert cfg.AIR_DIR == cfg.DATA_DIR / "ai-real-images"
    # predictions live UNDER the backend (project decision), not at repo root.
    assert cfg.PREDICTIONS_DIR == cfg.BACKEND_DIR / "predictions"
    assert cfg.BACKEND_DIR.name == "backend"


def test_importing_config_puts_notebooks_on_path():
    assert str(cfg.NOTEBOOKS_DIR) in sys.path     # enables `from utils import ...`


def test_pipeline_names_match_claude_md():
    assert cfg.PIPELINE_NAMES == [
        "cnn-scratch", "cnn-residual", "cnn-finetune",
        "vit-lora", "clip-probe", "two-stream",
    ]


def test_df_device_override(monkeypatch):
    monkeypatch.setenv("DF_DEVICE", "cpu")
    assert cfg.get_device() == torch.device("cpu")


def test_get_device_default_is_valid(monkeypatch):
    monkeypatch.delenv("DF_DEVICE", raising=False)
    dev = cfg.get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {"cpu", "cuda"}
