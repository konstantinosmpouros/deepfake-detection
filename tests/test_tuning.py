"""Tests for utils/tuning.py — Optuna scaffolding + focal loss + artifact saving.

All CPU. The study runs a trivial analytic objective (no models, no data) so it
finishes in milliseconds while still exercising the real SQLite-backed study and
the save_study_artifacts() JSON layout the notebooks commit.
"""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import tuning as Tu


# ---- focal loss ------------------------------------------------------------
def test_focal_gamma_zero_equals_bce_mean():
    torch.manual_seed(0)
    logits = torch.randn(64)
    targets = (torch.rand(64) > 0.5).float()
    focal0 = Tu.FocalLoss(gamma=0.0)(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    assert focal0.item() == pytest.approx(bce.item(), rel=1e-5)


def test_focal_downweights_easy_examples():
    # A confident-correct example should contribute far less under focal than BCE.
    logits = torch.tensor([6.0])           # very confident
    targets = torch.tensor([1.0])          # ...and correct
    focal = Tu.FocalLoss(gamma=2.0)(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    assert focal.item() < bce.item()
    assert focal.item() >= 0.0


def test_make_loss_dispatch():
    assert isinstance(Tu.make_loss("bce"), nn.BCEWithLogitsLoss)
    assert isinstance(Tu.make_loss("focal", focal_gamma=1.5), Tu.FocalLoss)
    assert Tu.make_loss("focal", focal_gamma=1.5).gamma == 1.5
    # Unknown name falls back to BCE (never crashes a search).
    assert isinstance(Tu.make_loss("something-else"), nn.BCEWithLogitsLoss)


# ---- study lifecycle + artifacts ------------------------------------------
def _run_tiny_study(storage_dir, n_trials=8):
    study = Tu.make_study("unittest-study", storage_dir, seed=42)

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        y = trial.suggest_categorical("y", ["a", "b"])
        bonus = 0.1 if y == "a" else 0.0
        return -((x - 0.7) ** 2) + bonus       # maximized near x=0.7, y="a"

    study.optimize(objective, n_trials=n_trials)
    return study


def test_make_study_is_resumable(tmp_path):
    s1 = _run_tiny_study(tmp_path, n_trials=4)
    n_after_first = len(s1.trials)
    # Re-open the SAME study name/storage -> trials persist (load_if_exists).
    s2 = Tu.make_study("unittest-study", tmp_path, seed=42)
    assert len(s2.trials) == n_after_first
    assert s2.best_value is not None


def test_save_study_artifacts_writes_expected_json(tmp_path):
    study = _run_tiny_study(tmp_path / "db")
    fig_dir = tmp_path / "figures"
    metrics_dir = tmp_path / "metrics"
    search_space = {"x": "uniform(0,1)", "y": "categorical(a,b)"}
    dump = Tu.save_study_artifacts(study, search_space, fig_dir, metrics_dir)

    # best_params.json — the small committed winner.
    best = json.loads((metrics_dir / "best_params.json").read_text())
    assert set(best) == {"x", "y"}

    # optuna_trials.json — full search record.
    trials = json.loads((metrics_dir / "optuna_trials.json").read_text())
    assert trials["study"] == "unittest-study"
    assert trials["metric"] == "val_auc"
    assert trials["search_space"] == search_space
    assert trials["n_trials"] == len(study.trials)
    assert len(trials["trials"]) == len(study.trials)
    one = trials["trials"][0]
    assert {"number", "state", "value", "params", "intermediate", "duration_s"} <= set(one)

    # Returned dump mirrors the file.
    assert dump["best_value"] == study.best_value


def test_load_best_params_merges_defaults(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "best_params.json").write_text(json.dumps({"lr": 0.01, "feat": 512}))
    merged = Tu.load_best_params(metrics_dir, defaults={"lr": 0.1, "p_drop": 0.3})
    assert merged == {"lr": 0.01, "p_drop": 0.3, "feat": 512}   # saved overrides defaults


def test_load_best_params_without_file_returns_defaults(tmp_path):
    defaults = {"lr": 0.1, "feat": 256}
    assert Tu.load_best_params(tmp_path, defaults=defaults) == defaults


def test_report_or_prune_reports_without_pruning(tmp_path):
    # n_startup_trials=5 so the very first trial is never pruned -> no raise.
    study = Tu.make_study("prune-study", tmp_path, seed=1)
    captured = {}

    def objective(trial):
        for step in range(3):
            Tu.report_or_prune(trial, step, value=0.5 + 0.1 * step)
        captured["ran"] = True
        return 0.9

    study.optimize(objective, n_trials=1)
    assert captured.get("ran") is True
    assert study.best_value == pytest.approx(0.9)
