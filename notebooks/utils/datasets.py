"""Data pipeline: 256px disk cache, transforms, a picklable Dataset, and loaders.

Design (see plan): images are pre-decoded once into a per-split uint8 memmap at
256x256 (Resize-shorter-side + CenterCrop) which both (a) avoids re-decoding the
~48GB of JPEGs every epoch and (b) neutralizes the real-vs-fake resolution shortcut.
The Dataset stores only paths/indices and opens the memmap lazily per worker so it
stays picklable (Windows spawn safe). Heavy logic is here; notebooks stay readable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated files (02_cleaning flags them)

# ---- Constants -------------------------------------------------------------

CACHE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_MEAN = (0.5, 0.5, 0.5)   # fallback until norm_stats.json exists
DEFAULT_STD = (0.25, 0.25, 0.25)
LABEL_TO_INT = {"real": 0.0, "fake": 1.0}  # fake = positive class (p_fake)


# ---- Cache build + normalization stats ------------------------------------

def _square_resize(img: Image.Image, size: int) -> Image.Image:
    """Resize shorter side to `size` (bilinear) then center-crop `size`x`size`."""
    w, h = img.size
    short = min(w, h)
    nw, nh = round(w * size / short), round(h * size / short)
    img = img.resize((nw, nh), Image.Resampling.BILINEAR)
    left, top = (nw - size) // 2, (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def build_cache(df_subset: pd.DataFrame, out_path, size: int = CACHE_SIZE, force: bool = False) -> int:
    """Decode each row's image into a uint8 (N,size,size,3) memmap at `out_path`.

    Idempotent: if the file already exists with the right shape, returns early.
    Low-RAM (one image at a time). Row order == df_subset order == cache index.
    """
    from tqdm.auto import tqdm

    out_path = Path(out_path)
    n = len(df_subset)
    if out_path.exists() and not force:
        existing = np.load(out_path, mmap_mode="r")
        if existing.shape == (n, size, size, 3):
            del existing
            return n
        del existing
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(n, size, size, 3))
    for i, fp in enumerate(tqdm(df_subset["filepath"].tolist(), desc=out_path.name)):
        img = _square_resize(Image.open(fp).convert("RGB"), size)
        mm[i] = np.asarray(img, dtype=np.uint8)
    mm.flush()
    del mm
    return n


def compute_norm_stats(memmap, chunk: int = 2000) -> tuple[list[float], list[float]]:
    """Streaming per-channel mean/std over [0,1] pixels of a uint8 (N,H,W,3) memmap."""
    n = len(memmap)
    s = np.zeros(3, dtype=np.float64)
    ss = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, n, chunk):
        block = np.asarray(memmap[start:start + chunk], dtype=np.float64) / 255.0
        block = block.reshape(-1, 3)
        s += block.sum(0)
        ss += (block ** 2).sum(0)
        count += block.shape[0]
    mean = s / count
    std = np.sqrt(np.maximum(ss / count - mean ** 2, 1e-12))
    return mean.tolist(), std.tolist()


def resolve_stats(norm: str, data_dir) -> tuple[tuple, tuple]:
    """Return (mean, std): ImageNet stats for 'imagenet', else dataset stats from norm_stats.json."""
    if norm == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    p = Path(data_dir) / "norm_stats.json"
    if p.exists():
        d = json.loads(p.read_text())
        return tuple(d["mean"]), tuple(d["std"])
    return DEFAULT_MEAN, DEFAULT_STD


def denormalize(t: torch.Tensor, mean, std) -> torch.Tensor:
    """Invert Normalize for display; returns a CHW tensor clamped to [0,1]."""
    mean = torch.tensor(mean, device=t.device).view(-1, 1, 1)
    std = torch.tensor(std, device=t.device).view(-1, 1, 1)
    return (t * std + mean).clamp(0, 1)


# ---- Transforms (v2) -------------------------------------------------------

def build_train_tf(size: int, mean, std) -> v2.Compose:
    """Light augmentation only (heavy aug would wash out generative artifacts)."""
    return v2.Compose([
        v2.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.RandomHorizontalFlip(0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(mean), std=list(std)),
    ])


def build_eval_tf(size: int, mean, std) -> v2.Compose:
    """Deterministic: resize shorter side to `size` + center crop + normalize."""
    return v2.Compose([
        v2.Resize(size, antialias=True),
        v2.CenterCrop(size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(mean), std=list(std)),
    ])


# ---- Dataset (picklable; lazy per-worker memmap) ---------------------------

class DeepfakeDataset(Dataset):
    """Returns (CHW float tensor, label float). label: fake=1.0 / real=0.0.

    source='cache' reads row `cache_idx` from a uint8 memmap (fast); source='files'
    decodes `filepath` on the fly (for OOD / app). The memmap is opened lazily on
    first access so the dataset pickles cleanly across DataLoader workers.
    """

    def __init__(self, df: pd.DataFrame, transform, source: str = "cache",
                 cache_path=None, label_col: str = "label"):
        self.transform = transform
        self.source = source
        self.cache_path = str(cache_path) if cache_path is not None else None
        self.labels = df[label_col].map(LABEL_TO_INT).to_numpy(dtype=np.float32)
        if source == "cache":
            assert "cache_idx" in df.columns, "cache source requires a 'cache_idx' column"
            self.cache_idx = df["cache_idx"].to_numpy(dtype=np.int64)
            self.filepaths = None
        else:
            self.filepaths = df["filepath"].tolist()
            self.cache_idx = None
        self._mm = None

    def __len__(self) -> int:
        return len(self.labels)

    def _ensure_mm(self):
        if self._mm is None:
            self._mm = np.load(self.cache_path, mmap_mode="r")

    def __getitem__(self, i):
        if self.source == "cache":
            self._ensure_mm()
            arr = np.asarray(self._mm[self.cache_idx[i]])          # HWC uint8
        else:
            arr = np.asarray(Image.open(self.filepaths[i]).convert("RGB"), dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()    # CHW uint8
        return self.transform(t), self.labels[i]


# ---- Loaders ---------------------------------------------------------------

def _dl_kwargs(num_workers: int) -> dict:
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return kw


def make_loaders(manifest_split_path, working_size: int, batch_size: int,
                 num_workers: int = 0, norm: str = "dataset", cache_dir=None,
                 eval_batch_size: int | None = None) -> dict:
    """Build train/val/test DataLoaders from manifest_split.csv + the 256px cache.

    norm: 'dataset' (from norm_stats.json) for from-scratch CNNs, 'imagenet' for
    pretrained backbones. Returns {'train','val','test'}.
    """
    manifest_split_path = Path(manifest_split_path)
    data_dir = manifest_split_path.parent
    cache_dir = Path(cache_dir) if cache_dir else (data_dir / "cache")
    df = pd.read_csv(manifest_split_path)
    df = df[df["keep"]].copy()
    mean, std = resolve_stats(norm, data_dir)
    train_tf, eval_tf = build_train_tf(working_size, mean, std), build_eval_tf(working_size, mean, std)
    eval_bs = eval_batch_size or batch_size * 2
    kw = _dl_kwargs(num_workers)

    loaders = {}
    for split, tf, shuffle, bs in [("train", train_tf, True, batch_size),
                                   ("val", eval_tf, False, eval_bs),
                                   ("test", eval_tf, False, eval_bs)]:
        sub = df[df["split_final"] == split].reset_index(drop=True)
        cache_path = cache_dir / f"cache_{split}_{CACHE_SIZE}.npy"
        ds = DeepfakeDataset(sub, tf, source="cache", cache_path=cache_path)
        loaders[split] = DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False, **kw)
    return loaders


def make_ood_loader(tiny_manifest_path, working_size: int, batch_size: int,
                    mean, std, num_workers: int = 0) -> tuple[DataLoader, pd.DataFrame]:
    """OOD loader over ALL of tiny-genimage (source='files'); returns (loader, df).

    Pass the SAME (mean, std) the model was trained with (use resolve_stats).
    The returned df (keep==True, reset index) aligns row-for-row with loader order
    so per-generator grouping by `source` is straightforward.
    """
    df = pd.read_csv(tiny_manifest_path)
    if "keep" in df.columns:
        df = df[df["keep"]].copy()
    df = df.reset_index(drop=True)
    eval_tf = build_eval_tf(working_size, mean, std)
    ds = DeepfakeDataset(df, eval_tf, source="files")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, **_dl_kwargs(num_workers))
    return loader, df
