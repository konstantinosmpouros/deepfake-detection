"""Plotting helpers reused across pipelines. Each returns a matplotlib Figure;
the notebook saves it (keeps save paths visible per the conventions).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_training_curves(history: dict):
    """history: dict of lists, e.g. train_loss/val_loss/val_auc/val_acc. Plots what's present."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, 1 + max((len(v) for v in history.values()), default=0))
    for key, ax in [("loss", ax1), ("auc", ax2)]:
        for name, ys in history.items():
            if key in name:
                ax.plot(range(1, len(ys) + 1), ys, marker=".", label=name)
        ax.set_xlabel("epoch"); ax.set_title(key); ax.legend()
    if "val_loss" in history and history["val_loss"]:
        best = int(np.argmin(history["val_loss"])) + 1
        ax1.axvline(best, color="grey", ls="--", lw=1)
    fig.tight_layout()
    return fig


def plot_confusion(cm, labels=("real", "fake")):
    """cm is [[TN,FP],[FN,TP]]; annotate counts + row-normalized percentages."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm, dtype=float)
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{int(cm[i, j])}\n{pct[i, j]:.1%}"
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig


def plot_roc_pr(y_true, y_prob):
    """ROC (with chance line) and PR (with prevalence baseline)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true).ravel(); y_prob = np.asarray(y_prob).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.4f}")
    ax1.plot([0, 1], [0, 1], "--", color="grey")
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC"); ax1.legend()
    ax2.plot(rec, prec, label=f"AP={average_precision_score(y_true, y_prob):.4f}")
    ax2.axhline(y_true.mean(), ls="--", color="grey", label="prevalence")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("Precision-Recall"); ax2.legend()
    fig.tight_layout()
    return fig


def plot_reliability(y_true, y_prob, n_bins: int = 10):
    """Calibration diagram: mean predicted p_fake vs observed fake-fraction per bin."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import brier_score_loss

    y_true = np.asarray(y_true).ravel(); y_prob = np.asarray(y_prob).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            xs.append(y_prob[m].mean()); ys.append(y_true[m].mean()); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(xs, ys, marker="o", label="model")
    ax.set_xlabel("mean predicted p_fake"); ax.set_ylabel("observed fake fraction")
    ax.set_title(f"Reliability (Brier={brier_score_loss(y_true, y_prob):.4f})"); ax.legend()
    fig.tight_layout()
    return fig


def plot_per_generator_bar(df, ref_acc: float | None = None):
    """df with columns ['generator','accuracy','n']; sorted ascending, optional ref line."""
    import matplotlib.pyplot as plt

    d = df.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(d["generator"], d["accuracy"], color="#c44e52")
    for bar, n in zip(bars, d["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"n={int(n)}",
                ha="center", va="bottom", fontsize=8)
    if ref_acc is not None:
        ax.axhline(ref_acc, ls="--", color="#4c72b0", label=f"in-dist acc={ref_acc:.3f}")
        ax.legend()
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy"); ax.set_title("Cross-generator (OOD) accuracy")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig
