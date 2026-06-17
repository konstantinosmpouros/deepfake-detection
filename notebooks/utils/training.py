"""Small, composable training pieces. The epoch loop stays VISIBLE in the notebook.

Everything assumes bf16 autocast on CUDA (no GradScaler — bf16 keeps fp32 range)
and channels_last batches. Models output a single logit; loss is BCEWithLogitsLoss.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch


def smooth_binary_targets(y: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    """Label smoothing for BCE: 1 -> 1-eps/2, 0 -> eps/2 (mild; heavy smoothing blunts artifacts)."""
    return y * (1.0 - eps) + 0.5 * eps


def _move(x, y, device):
    # channels_last only applies to 4D image tensors; embeddings (2D) move plainly.
    if x.dim() == 4:
        x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
    else:
        x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True).float()
    return x, y


def train_one_epoch(model, loader, optimizer, loss_fn, device, scheduler=None,
                    grad_clip: float | None = 1.0, accum_steps: int = 1,
                    ema=None, label_smooth: float = 0.0,
                    amp_dtype=torch.bfloat16) -> dict:
    """One epoch. Per-batch scheduler stepping. Returns {'loss', 'lr'}."""
    model.train()
    use_amp = device.type == "cuda"
    total, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for it, (x, y) in enumerate(loader):
        x, y = _move(x, y, device)
        target = smooth_binary_targets(y, label_smooth) if label_smooth > 0 else y
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            if logits.ndim > 1:
                logits = logits.squeeze(1)
            loss = loss_fn(logits, target)
        (loss / accum_steps).backward()
        if (it + 1) % accum_steps == 0:
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            if scheduler is not None:
                scheduler.step()
        total += loss.item() * y.size(0)
        n += y.size(0)
    return {"loss": total / max(n, 1), "lr": optimizer.param_groups[0]["lr"]}


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None, amp_dtype=torch.bfloat16):
    """Return (y_true, y_prob, loss). y_prob = sigmoid(logits); loss is mean BCE or None."""
    model.eval()
    use_amp = device.type == "cuda"
    probs, trues, total, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = _move(x, y, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            if logits.ndim > 1:
                logits = logits.squeeze(1)
            if loss_fn is not None:
                total += loss_fn(logits, y).item() * y.size(0)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        trues.append(y.cpu().numpy())
        n += y.size(0)
    y_true = np.concatenate(trues).astype(int)
    y_prob = np.concatenate(probs)
    return y_true, y_prob, (total / max(n, 1) if loss_fn is not None else None)


class EarlyStopper:
    """Track best metric; .step(value) -> (improved, should_stop). mode 'max' or 'min'."""

    def __init__(self, mode: str = "max", patience: int = 7, min_delta: float = 1e-3):
        self.mode, self.patience, self.min_delta = mode, patience, min_delta
        self.best = -math.inf if mode == "max" else math.inf
        self.bad = 0

    def _better(self, v):
        return (v > self.best + self.min_delta) if self.mode == "max" else (v < self.best - self.min_delta)

    def step(self, value: float):
        if self._better(value):
            self.best, self.bad = value, 0
            return True, False
        self.bad += 1
        return False, self.bad >= self.patience


class EMA:
    """Exponential moving average of model params (decay~0.999). Apply before eval."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for sb, b in zip(self.shadow.buffers(), model.buffers()):
            sb.copy_(b)

    def copy_to(self, model):
        """Swap EMA weights into `model` (keeps a backup for restore)."""
        self._backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow.state_dict())

    def restore(self, model):
        if self._backup is not None:
            model.load_state_dict(self._backup)
            self._backup = None

    def state_dict(self):
        return self.shadow.state_dict()


def build_cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int,
                             min_lr_ratio: float = 0.01):
    """Per-BATCH LR schedule: linear warmup then cosine decay to min_lr_ratio*lr."""
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cos = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        return min_lr_ratio + (1 - min_lr_ratio) * cos
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None,
                    best_metric=None, ema=None, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model": model.state_dict(), "epoch": epoch, "best_metric": best_metric}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def load_ema_weights(path, model, map_location="cpu") -> None:
    """Load the EMA weights from a checkpoint into `model` (for eval/inference)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["ema"] if "ema" in ckpt else ckpt["model"])


# ---- Slim, shareable weights (GitHub-friendly) -----------------------------
# Save ONLY the weights needed to reconstruct the inference model — no optimizer
# state. For pipelines with a frozen, re-downloadable backbone (vit-lora,
# clip-probe) save just the TRAINED part; the teammate rebuilds the architecture
# (which downloads the frozen weights) and attaches these.

def trained_state_dict(model) -> dict:
    """State-dict subset for parameters that require grad (the trained part only)."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in keep}


def save_weights(path, state_dict, meta: dict | None = None) -> None:
    """Save a slim weights file: {'state_dict': cpu tensors, 'meta': {...}}."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
                "meta": meta or {}}, path)


def load_weights(path, model, strict: bool = True, map_location="cpu") -> dict:
    """Load a slim weights file into `model`; returns the meta dict. Use strict=False
    when loading a trained-only subset onto a freshly built (frozen+head) model."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=strict)
    return ckpt.get("meta", {})
