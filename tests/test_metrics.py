"""Tests for utils/metrics.py — the shared real-vs-fake metric functions.

Pure functions (sklearn under the hood); no GPU, no model, no disk except the
one save_metrics round-trip on tmp_path.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from utils import metrics as Me

ALL_SCALARS = ("accuracy", "f1_macro", "precision", "recall",
               "auc_roc", "pr_auc", "mcc", "brier")


def test_classification_metrics_perfect_separation(separable_probs):
    y_true, y_prob = separable_probs
    m = Me.classification_metrics(y_true, y_prob, threshold=0.5)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["auc_roc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["mcc"] == pytest.approx(1.0)
    # Brier is low (probabilities close to the 0/1 targets) but not necessarily 0.
    assert 0.0 <= m["brier"] < 0.1


def test_confusion_matrix_orientation():
    # 2 real (0) + 2 fake (1); predict everything fake at threshold 0.5.
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.9, 0.9, 0.9, 0.9])
    m = Me.classification_metrics(y_true, y_prob, threshold=0.5)
    cm = m["confusion_matrix"]                       # [[TN, FP], [FN, TP]]
    assert cm == [[0, 2], [0, 2]]
    assert m["n"] == 4 and m["n_fake"] == 2 and m["n_real"] == 2


def test_threshold_free_metrics_are_threshold_invariant(separable_probs):
    y_true, y_prob = separable_probs
    a = Me.classification_metrics(y_true, y_prob, threshold=0.2)
    b = Me.classification_metrics(y_true, y_prob, threshold=0.8)
    # AUC / PR-AUC / Brier depend only on probabilities, not the threshold.
    assert a["auc_roc"] == pytest.approx(b["auc_roc"])
    assert a["pr_auc"] == pytest.approx(b["pr_auc"])
    assert a["brier"] == pytest.approx(b["brier"])
    # ...but the decision threshold changes the hard-label metrics.
    assert a["recall"] >= b["recall"]


def test_single_class_auc_is_nan_but_does_not_raise():
    y_true = np.zeros(5, dtype=int)                  # only the "real" class present
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    m = Me.classification_metrics(y_true, y_prob)
    assert math.isnan(m["auc_roc"])
    assert math.isnan(m["pr_auc"])
    assert m["accuracy"] == pytest.approx(1.0)       # all predicted real at 0.5


def test_metrics_in_valid_ranges(rng):
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.random(200)
    m = Me.classification_metrics(y_true, y_prob)
    for k in ("accuracy", "f1_macro", "precision", "recall", "auc_roc", "pr_auc", "brier"):
        assert 0.0 <= m[k] <= 1.0
    assert -1.0 <= m["mcc"] <= 1.0


def test_best_f1_threshold(separable_probs):
    y_true, y_prob = separable_probs
    out = Me.best_f1_threshold(y_true, y_prob)
    assert set(out) == {"threshold", "f1_macro"}
    assert 0.0 < out["threshold"] < 1.0
    assert out["f1_macro"] == pytest.approx(1.0)     # separable -> perfect F1 exists


def test_save_metrics_roundtrip_and_numpy_coercion(tmp_path, rng):
    y_true = rng.integers(0, 2, size=50)
    y_prob = rng.random(50)
    m = Me.classification_metrics(y_true, y_prob)
    # Inject numpy scalar / array types that vanilla json.dump would choke on.
    m["np_float"] = np.float64(0.5)
    m["np_int"] = np.int64(7)
    m["np_arr"] = np.array([1, 2, 3])
    path = Me.save_metrics(m, tmp_path / "sub" / "metrics.json")
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["np_float"] == 0.5
    assert loaded["np_int"] == 7
    assert loaded["np_arr"] == [1, 2, 3]
    assert loaded["confusion_matrix"] == m["confusion_matrix"]


def test_summary_table_drops_confusion_matrix(separable_probs):
    y_true, y_prob = separable_probs
    m = Me.classification_metrics(y_true, y_prob)
    df = Me.summary_table(m)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "confusion_matrix" not in df.columns
    for k in ALL_SCALARS:
        assert k in df.columns
