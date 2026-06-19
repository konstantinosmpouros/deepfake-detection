"""Shared evaluation harness for the `eval-*` notebooks.

Reconstructs each TRAINED pipeline from its committed `best_params.json` (the
Optuna-tuned architecture) + weights, and provides the three evaluation
protocols: in-distribution (ai-real-images test), cross-generator (tiny-genimage),
and robustness (perturbed test). Heavy logic lives here so the eval notebooks
stay readable.

NOTE: this is a *new* module — none of the training notebooks import it, so adding
it never affects a run in progress. Each pipeline is rebuilt from `best_params.json`
because the Optuna-tuned width/depth/rank differ from the build defaults.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2

from utils import datasets as D
from utils import metrics as Me
from utils import models as M
from utils import training as T
from utils.paths import repo_paths

NB = Path(__file__).resolve().parents[1]          # notebooks/
ART = NB / "artifacts"
_P = repo_paths(NB)
DATA = _P["data"]
AIR = DATA / "ai-real-images"
SPLIT = AIR / "manifest_split.csv"
TINY = DATA / "tiny-genimage" / "manifest_clean.csv"

GEN_MAP = {
    "imagenet_ai_0419_biggan": "biggan", "imagenet_ai_0419_vqdm": "vqdm",
    "imagenet_ai_0424_sdv5": "sdv5", "imagenet_ai_0424_wukong": "wukong",
    "imagenet_ai_0508_adm": "adm", "imagenet_glide": "glide", "imagenet_midjourney": "midjourney",
}

# name -> spec. family: image | clip | patch | dire. multi: has per-stream forward_all.
SPECS = [
    ("cnn-scratch",    dict(family="image", size=128, norm="dataset", multi=False)),
    ("cnn-residual",   dict(family="image", size=128, norm="dataset", multi=False)),
    ("cnn-finetune",   dict(family="image", size=224, norm="imagenet", multi=False)),
    ("vit-lora",       dict(family="image", size=224, norm="imagenet", multi=False)),
    ("clip-probe",     dict(family="clip",  size=224, norm="clip",     multi=False)),
    ("two-stream",     dict(family="image", size=128, norm="dataset",  multi=True)),
    ("freqcross",      dict(family="image", size=128, norm="dataset",  multi=True)),
    ("srm-noise",      dict(family="image", size=128, norm="dataset",  multi=False)),
    ("patch-ensemble", dict(family="patch", size=224, norm="imagenet", multi=False)),
    ("dire-recon",     dict(family="dire",  size=224, norm="imagenet", multi=False)),
]
SPEC = dict(SPECS)
ORDER = [n for n, _ in SPECS]
IMAGE_FAMILY = [n for n, s in SPECS if s["family"] == "image"]

# Robustness sweeps: perturbation -> increasing-strength levels.
PERTURBATIONS = {
    "jpeg_quality": [100, 90, 80, 70, 60],
    "gaussian_blur_sigma": [0.0, 0.5, 1.0, 1.5, 2.0],
    "downsample_scale": [1.0, 0.75, 0.5, 0.35, 0.25],
    "gaussian_noise_std": [0.0, 0.02, 0.05, 0.1, 0.15],
}


# ---- artifact readers ------------------------------------------------------

def _read(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def best_params(name) -> dict:
    return _read(ART / name / "metrics" / "best_params.json") or {}


def metrics(name) -> dict | None:
    return _read(ART / name / "metrics" / "metrics.json")


def optuna_trials(name) -> dict | None:
    return _read(ART / name / "metrics" / "optuna_trials.json")


def available() -> list[str]:
    """Pipelines that have a metrics.json AND a model checkpoint on disk."""
    out = []
    for n in ORDER:
        mdir = ART / n / "models"
        has_w = mdir.exists() and (list(mdir.glob("best*.pt")))
        if metrics(n) and has_w:
            out.append(n)
    return out


# ---- label helpers (re-derive the orderings the training notebooks used) ---

def split_labels(split: str) -> np.ndarray:
    """Labels for ai-real-images <split>, ordered by cache_idx (matches clip_emb_*.npy)."""
    df = pd.read_csv(SPLIT); df = df[df["keep"]]
    sub = df[df["split_final"] == split].sort_values("cache_idx")
    return (sub["label"].values == "fake").astype(np.float32)


def _dire_subsample(df_split, n, seed=42):
    """Replicate the dire-recon notebook's stratified subsample (same seed/order)."""
    per = n // 2
    parts = [df_split[df_split["label"] == lab].sample(min(per, int((df_split["label"] == lab).sum())), random_state=seed)
             for lab in ["real", "fake"]]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# ---- model reconstruction (from best_params) -------------------------------

def load_model(name: str, device):
    """Rebuild the architecture from best_params.json and load its trained weights."""
    bp = best_params(name)
    m = metrics(name) or {}
    mdir = ART / name / "models"
    if name == "cnn-scratch":
        model = M.build_cnn_scratch(p_drop=bp.get("p_drop", 0.3))
        T.load_checkpoint(mdir / "best.pt", model, map_location=device)
    elif name == "cnn-residual":
        model = M.build_cnn_residual(attention="se", se_reduction=16, p_drop=bp.get("p_drop", 0.3))
        T.load_ema_weights(mdir / "best.pt", model, map_location=device)
    elif name == "cnn-finetune":
        bb = bp.get("backbone") or m.get("backbone", "efficientnet_b0")
        model = M.build_cnn_finetune(bb, pretrained=False, p_drop=bp.get("p_drop", 0.3))
        T.load_weights(mdir / f"best_{bb}.pt", model, map_location=device)
    elif name == "vit-lora":
        model = M.build_vit_lora(r=bp.get("r", 16), lora_alpha=bp.get("lora_alpha", 16),
                                 lora_dropout=bp.get("lora_dropout", 0.05), p_drop=bp.get("p_drop", 0.1))
        T.load_weights(mdir / "best.pt", model, strict=False, map_location=device)
        model = model.merge_and_unload()
    elif name == "clip-probe":
        emb_dim = int(m.get("embedding_dim", 512))
        model = M.build_mlp_head(emb_dim, hidden=bp.get("hidden", 256), n_layers=bp.get("n_layers", 1), p_drop=bp.get("p_drop", 0.3))
        T.load_checkpoint(mdir / "best.pt", model, map_location=device)
    elif name == "two-stream":
        model = M.build_two_stream(feat=bp.get("feat", 256), p_drop=bp.get("p_drop", 0.3))
        T.load_checkpoint(mdir / "best.pt", model, map_location=device)
    elif name == "freqcross":
        model = M.build_freqcross(size=SPEC[name]["size"], feat=bp.get("feat", 256),
                                  n_radial=bp.get("n_radial", 64), p_drop=bp.get("p_drop", 0.3))
        T.load_checkpoint(mdir / "best.pt", model, map_location=device)
    elif name == "srm-noise":
        model = M.build_srm_cnn(feat=bp.get("feat", 256), bayar_ch=bp.get("bayar_ch", 3), p_drop=bp.get("p_drop", 0.3))
        T.load_checkpoint(mdir / "best.pt", model, map_location=device)
    elif name == "patch-ensemble":
        bb = bp.get("backbone", "efficientnet_b0")
        model = M.build_patch_ensemble(bb, pretrained=False, mil_hidden=bp.get("mil_hidden", 128), p_drop=bp.get("p_drop", 0.3))
        T.load_weights(mdir / "best.pt", model, map_location=device)
    elif name == "dire-recon":
        bb = bp.get("backbone", "efficientnet_b0")
        model = M.build_cnn_finetune(bb, pretrained=False, p_drop=bp.get("p_drop", 0.3))
        T.load_weights(mdir / "best.pt", model, map_location=device)
    else:
        raise KeyError(name)
    return model.to(device).eval()


# ---- in-distribution + OOD probabilities (family-aware) --------------------

@torch.no_grad()
def _emb_probs(model, emb, device, bs=4096):
    model.eval(); out = []
    for i in range(0, len(emb), bs):
        x = torch.from_numpy(np.asarray(emb[i:i + bs])).to(device)
        logit = model(x)
        out.append(torch.sigmoid(logit).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def _arr_probs(model, arr, device, mean, std, size, bs=256):
    """Probs over a uint8 (N,H,W,3) array via the eval transform (for DIRE maps)."""
    tf = D.build_eval_tf(size, mean, std); out = []
    for i in range(0, len(arr), bs):
        xb = torch.stack([tf(torch.from_numpy(np.array(a)).permute(2, 0, 1).contiguous()) for a in arr[i:i + bs]]).to(device)
        logit = model(xb)
        if logit.ndim > 1:
            logit = logit.squeeze(1)
        out.append(torch.sigmoid(logit).float().cpu().numpy())
    return np.concatenate(out)


def indist_probs(name, model, device, batch_size=256, num_workers=0):
    """Return (y_true, y_prob) on the in-distribution test set."""
    sp = SPEC[name]
    if sp["family"] == "image":
        ld = D.make_loaders(SPLIT, working_size=sp["size"], batch_size=batch_size, num_workers=num_workers, norm=sp["norm"])
        y, p, _ = T.evaluate(model, ld["test"], device)
        return y, p
    if sp["family"] == "clip":
        emb = np.load(AIR / "clip_emb_test.npy")
        return split_labels("test").astype(int), _emb_probs(model, emb, device)
    if sp["family"] == "patch":
        K = best_params(name).get("K", 4)
        ld = D.make_patch_loaders(SPLIT, patch=sp["size"], k=K, batch_size=16, num_workers=num_workers)
        y, p, _ = T.evaluate(model, ld["test"], device)
        return y, p
    if sp["family"] == "dire":
        arr = np.load(AIR / "dire_test.npy")
        df = pd.read_csv(SPLIT); df = df[df["keep"]]; df = df[df["split_final"] == "test"]
        sub = _dire_subsample(df, len(arr))
        y = (sub["label"].values == "fake").astype(int)
        mean, std = D.IMAGENET_MEAN, D.IMAGENET_STD
        return y, _arr_probs(model, arr, device, mean, std, sp["size"])
    raise KeyError(name)


def ood_frame(name, model, device, batch_size=256, num_workers=0) -> pd.DataFrame:
    """Return a DataFrame with columns [generator, y_true, p_fake] over tiny-genimage."""
    sp = SPEC[name]
    if sp["family"] == "image":
        mean, std = D.resolve_stats(sp["norm"], AIR)
        ld, df = D.make_ood_loader(TINY, sp["size"], batch_size, mean, std, num_workers=num_workers)
        y, p, _ = T.evaluate(model, ld, device)
    elif sp["family"] == "clip":
        df = pd.read_csv(TINY); df = df[df["keep"]].reset_index(drop=True)
        emb = np.load(DATA / "tiny-genimage" / "clip_emb.npy")
        y = (df["label"].values == "fake").astype(int); p = _emb_probs(model, emb, device)
    elif sp["family"] == "patch":
        K = best_params(name).get("K", 4)
        ld, df = D.make_patch_ood_loader(TINY, patch=sp["size"], k=K, batch_size=16, num_workers=num_workers)
        y, p, _ = T.evaluate(model, ld, device)
    elif sp["family"] == "dire":
        arr = np.load(DATA / "tiny-genimage" / "dire_ood.npy")
        df = pd.read_csv(TINY); df = df[df["keep"]]
        df = _dire_subsample(df, len(arr))
        y = (df["label"].values == "fake").astype(int)
        p = _arr_probs(model, arr, device, D.IMAGENET_MEAN, D.IMAGENET_STD, sp["size"])
    else:
        raise KeyError(name)
    out = df.copy().reset_index(drop=True)
    out["y_true"] = y; out["p_fake"] = p
    out["generator"] = out["source"].map(GEN_MAP).fillna(out["source"])
    return out[["generator", "y_true", "p_fake"]]


# ---- robustness (image family) ---------------------------------------------

def _perturb_tf(kind, level, size, mean, std):
    """Eval transform that applies one perturbation (input: uint8 CHW tensor)."""
    pre, post_extra = [], []
    if kind == "jpeg_quality":
        pre = [v2.JPEG(quality=(int(level), int(level)))]
    elif kind == "gaussian_blur_sigma" and level > 0:
        pre = [v2.GaussianBlur(kernel_size=7, sigma=(float(level), float(level)))]
    elif kind == "downsample_scale" and level < 1.0:
        small = max(8, int(round(size * float(level))))
        pre = [v2.Resize(small, antialias=True)]
    resize = [v2.Resize(size, antialias=True), v2.CenterCrop(size), v2.ToDtype(torch.float32, scale=True)]
    if kind == "gaussian_noise_std" and level > 0:
        s = float(level)
        post_extra = [v2.Lambda(lambda x: (x + torch.randn_like(x) * s).clamp(0, 1))]
    norm = [v2.Normalize(mean=list(mean), std=list(std))]
    return v2.Compose(pre + resize + post_extra + norm)


def robustness_loader(name, kind, level, subsample=2000, batch_size=128, num_workers=0, seed=42):
    """Perturbed in-distribution test loader for an IMAGE-family pipeline."""
    assert SPEC[name]["family"] == "image", f"{name} is not an image-family pipeline"
    df = pd.read_csv(SPLIT); df = df[df["keep"]]; df = df[df["split_final"] == "test"].reset_index(drop=True)
    if subsample and subsample < len(df):
        df = df.sample(subsample, random_state=seed).reset_index(drop=True)
    sp = SPEC[name]; mean, std = D.resolve_stats(sp["norm"], AIR)
    tf = _perturb_tf(kind, level, sp["size"], mean, std)
    ds = D.DeepfakeDataset(df, tf, source="files")
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def robustness_point(name, model, kind, level, device, **kw):
    """Accuracy + AUC of a pipeline at one perturbation level."""
    ld = robustness_loader(name, kind, level, **kw)
    y, p, _ = T.evaluate(model, ld, device)
    mm = Me.classification_metrics(y, p)
    return {"accuracy": mm["accuracy"], "auc_roc": mm["auc_roc"]}
