"""Optuna hyperparameter-search helpers (notebooks 06-13).

The mechanics live here; each notebook keeps its `objective` (the visible
`trial.suggest_*` search space). Studies maximize **validation AUC** with a TPE
sampler + MedianPruner and **SQLite storage** (resumable). Saving strategy:
  artifacts/<pipeline>/tuning/<pipeline>.db        # full study, all trials, resumable
  artifacts/<pipeline>/metrics/best_params.json    # the winner (small, committed)
  artifacts/<pipeline>/metrics/optuna_trials.json  # search space + EVERY trial
  artifacts/<pipeline>/figures/optuna_*.png        # history / importances / parallel
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss on logits (Lin et al. 2017). Works with soft targets."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.where(targets >= 0.5, p, 1 - p)
        return (((1 - pt) ** self.gamma) * ce).mean()


def make_loss(name: str = "bce", focal_gamma: float = 2.0):
    """'bce' -> BCEWithLogitsLoss, 'focal' -> FocalLoss(gamma).

    Label smoothing is applied upstream by `training.train_one_epoch(label_smooth=...)`,
    so it composes with either loss.
    """
    return FocalLoss(focal_gamma) if name == "focal" else nn.BCEWithLogitsLoss()


def make_study(study_name: str, storage_dir, seed: int = 42):
    """TPE + MedianPruner study backed by SQLite (resumable; load_if_exists)."""
    import optuna

    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(storage_dir / (study_name + '.db')).as_posix()}"
    return optuna.create_study(
        direction="maximize", study_name=study_name, storage=storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )


def report_or_prune(trial, step: int, value: float) -> None:
    """Report an intermediate val AUC; raise TrialPruned if the pruner says so."""
    import optuna

    trial.report(value, step)
    if trial.should_prune():
        raise optuna.TrialPruned()


def quick_train_eval(model, train_loader, val_loader, device, *, lr, weight_decay,
                     epochs, trial=None, loss_fn=None, label_smooth=0.0, grad_clip=1.0):
    """Train a SINGLE-output model `epochs` and return best val AUC (with pruning).

    Only requires-grad params are optimized (LoRA-friendly). Uses the shared
    train_one_epoch/evaluate so search and final training share one code path.
    """
    from utils import metrics as Me
    from utils import training as T

    loss_fn = loss_fn or nn.BCEWithLogitsLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    spe = len(train_loader)
    sched = T.build_cosine_with_warmup(opt, total_steps=epochs * spe, warmup_steps=max(1, spe // 2))
    best = 0.0
    for ep in range(epochs):
        T.train_one_epoch(model, train_loader, opt, loss_fn, device, scheduler=sched,
                          grad_clip=grad_clip, label_smooth=label_smooth)
        yv, pv, _ = T.evaluate(model, val_loader, device)
        auc = Me.classification_metrics(yv, pv)["auc_roc"]
        best = max(best, auc)
        if trial is not None:
            report_or_prune(trial, ep, auc)
    return best


def cleanup(*objs) -> None:
    """Drop references + empty CUDA cache between trials (avoids cross-trial OOM)."""
    import gc

    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_study_artifacts(study, search_space: dict, fig_dir, metrics_dir) -> dict:
    """Persist best_params.json, optuna_trials.json (space + every trial), and figures."""
    fig_dir, metrics_dir = Path(fig_dir), Path(metrics_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "best_params.json").write_text(json.dumps(study.best_params, indent=2))

    trials = [{
        "number": t.number,
        "state": t.state.name,
        "value": t.value,
        "params": t.params,
        "intermediate": {str(k): v for k, v in t.intermediate_values.items()},
        "duration_s": (t.duration.total_seconds() if t.duration else None),
    } for t in study.trials]
    dump = {
        "study": study.study_name, "direction": "maximize", "metric": "val_auc",
        "sampler": "TPESampler", "pruner": "MedianPruner",
        "n_trials": len(study.trials),
        "n_complete": sum(t.state.name == "COMPLETE" for t in study.trials),
        "n_pruned": sum(t.state.name == "PRUNED" for t in study.trials),
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "search_space": search_space,
        "trials": trials,
    }
    (metrics_dir / "optuna_trials.json").write_text(json.dumps(dump, indent=2))

    try:
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import (
            plot_optimization_history, plot_parallel_coordinate, plot_param_importances)
        for fn, name in [(plot_optimization_history, "optuna_history"),
                         (plot_param_importances, "optuna_importances"),
                         (plot_parallel_coordinate, "optuna_parallel")]:
            try:
                ax = fn(study)
                ax.figure.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
                plt.close(ax.figure)
            except Exception as e:
                print(f"  (skip {name}: {e})")
    except Exception as e:
        print("  (optuna viz unavailable:", e, ")")
    return dump


def load_best_params(metrics_dir, defaults: dict | None = None) -> dict:
    """Merge defaults with a saved best_params.json (if present)."""
    out = dict(defaults or {})
    p = Path(metrics_dir) / "best_params.json"
    if p.exists():
        out.update(json.loads(p.read_text()))
    return out
