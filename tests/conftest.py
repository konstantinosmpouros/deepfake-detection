"""Shared pytest fixtures + path wiring for the deepfake-detection test suite.

The notebooks import their helpers as ``from utils import metrics`` (with
``notebooks/`` on ``sys.path``) and the app imports as ``from src.backend import
schemas`` (with the repo root on ``sys.path``). We replicate both here so tests
exercise the *exact* modules the project uses — no copies, no shims.

Markers (see pytest.ini):
  * ``download`` — needs to fetch pretrained weights over the network; skipped
    unless ``--run-download`` is passed.
  * ``gpu``      — needs a CUDA device; skipped automatically when none is present.
  * ``slow``     — builds real (timm/peft) architectures on CPU; runnable, just heavier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---- make `utils.*` and `src.backend.*` importable -------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
for p in (REPO_ROOT, NOTEBOOKS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---- CLI option + skip logic ----------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--run-download",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.download (fetch pretrained weights)",
    )


def pytest_collection_modifyitems(config, items):
    run_download = config.getoption("--run-download")
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    skip_download = pytest.mark.skip(reason="needs network weights; pass --run-download")
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "download" in item.keywords and not run_download:
            item.add_marker(skip_download)
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)


# ---- shared fixtures -------------------------------------------------------
@pytest.fixture
def rng():
    """Deterministic numpy Generator (no global-seed side effects)."""
    return np.random.default_rng(0)


@pytest.fixture
def separable_probs(rng):
    """A perfectly separable binary problem: (y_true, y_prob) with AUC == 1.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.12, 0.30, 0.70, 0.88, 0.95])
    return y_true, y_prob


@pytest.fixture
def tiny_image_factory(tmp_path):
    """Factory writing a solid-colour RGB PNG to tmp and returning its path."""
    from PIL import Image

    def _make(name="img.png", color=(127, 64, 200), size=(48, 48), mode="RGB"):
        img = Image.new(mode, size, color if mode == "RGB" else color[0])
        path = tmp_path / name
        img.save(path)
        return path

    return _make
