"""Binary real-vs-fake metrics. fake=1 / real=0, inputs are p_fake in [0,1].

sklearn is used for metrics only (project rule). Pure functions; the one I/O
helper (`save_metrics`) is the only thing that touches disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """All §3.3 metrics in a flat JSON-serializable dict.

    Threshold-free metrics (auc_roc, pr_auc, brier) use y_prob; the rest use the
    0/1 decision at `threshold`. Single-class slices -> AUC/PR-AUC = NaN (guarded).
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel().astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:                       # only one class present
        auc, pr_auc = float("nan"), float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "n": int(y_true.size), "n_fake": int(y_true.sum()), "n_real": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auc_roc": auc,
        "pr_auc": pr_auc,
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": cm.tolist(),     # [[TN, FP], [FN, TP]]
    }


def best_f1_threshold(y_true, y_prob) -> dict:
    """Threshold maximizing macro-F1 (tune on VALIDATION, never test)."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel().astype(float)
    _, _, thr = precision_recall_curve(y_true, y_prob)
    candidates = np.unique(np.clip(thr, 1e-4, 1 - 1e-4))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return {"threshold": best_t, "f1_macro": float(best_f1)}


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_metrics(metrics: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(metrics), indent=2))
    return path


def summary_table(metrics: dict) -> pd.DataFrame:
    """One-row DataFrame of the scalar metrics (drops the confusion matrix)."""
    row = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    return pd.DataFrame([row])
