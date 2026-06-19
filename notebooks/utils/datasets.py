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


# ---- Patch-bag dataset (for the patch-ensemble pipeline, notebook 14) -------
# This pipeline deliberately works at NATIVE resolution (not the 256 cache), so
# the high-frequency generative artifacts that resizing destroys are preserved.
# Each item is a bag of K patches (multiple-instance learning).

class PatchBagDataset(Dataset):
    """Return (K, 3, patch, patch) float tensor + label, read from the original file.

    train=True  -> K random native-resolution crops (small images upscaled to fit).
    train=False -> K deterministic corner crops (center added when K==5).
    """

    def __init__(self, df: pd.DataFrame, patch: int = 224, k: int = 4, train: bool = True,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD, label_col: str = "label"):
        self.files = df["filepath"].tolist()
        self.labels = df[label_col].map(LABEL_TO_INT).to_numpy(dtype=np.float32)
        self.patch, self.k, self.train = patch, k, train
        self.mean = torch.tensor(list(mean)).view(3, 1, 1)
        self.std = torch.tensor(list(std)).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.files)

    def _norm(self, pil_patch) -> torch.Tensor:
        arr = np.array(pil_patch, dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return (t - self.mean) / self.std

    def _crops(self, img):
        p = self.patch
        w, h = img.size
        if min(w, h) < p:                                         # upscale small images to fit
            short = min(w, h)
            nw, nh = round(w * p / short), round(h * p / short)
            img = img.resize((nw, nh), Image.Resampling.BILINEAR)
            w, h = nw, nh
        if self.train:
            out = []
            for _ in range(self.k):
                left = int(torch.randint(0, w - p + 1, (1,)).item())
                top = int(torch.randint(0, h - p + 1, (1,)).item())
                out.append(img.crop((left, top, left + p, top + p)))
            return out
        coords = [(0, 0), (w - p, 0), (0, h - p), (w - p, h - p)]
        if self.k == 5:
            coords.append(((w - p) // 2, (h - p) // 2))
        return [img.crop((l, t, l + p, t + p)) for (l, t) in coords[:self.k]]

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        x = torch.stack([self._norm(c) for c in self._crops(img)])  # (K,3,p,p)
        return x, self.labels[i]


def make_patch_loaders(manifest_split_path, patch: int = 224, k: int = 4, batch_size: int = 16,
                       num_workers: int = 4, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> dict:
    """train/val/test bag-of-patch loaders from manifest_split.csv (reads original files)."""
    df = pd.read_csv(manifest_split_path)
    df = df[df["keep"]].copy()
    kw = _dl_kwargs(num_workers)
    loaders = {}
    for split, train, shuffle in [("train", True, True), ("val", False, False), ("test", False, False)]:
        sub = df[df["split_final"] == split].reset_index(drop=True)
        ds = PatchBagDataset(sub, patch=patch, k=k, train=train, mean=mean, std=std)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, **kw)
    return loaders


def make_patch_ood_loader(tiny_manifest_path, patch: int = 224, k: int = 4, batch_size: int = 16,
                          num_workers: int = 4, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Bag-of-patch OOD loader over all of tiny-genimage; returns (loader, df)."""
    df = pd.read_csv(tiny_manifest_path)
    if "keep" in df.columns:
        df = df[df["keep"]].copy()
    df = df.reset_index(drop=True)
    ds = PatchBagDataset(df, patch=patch, k=k, train=False, mean=mean, std=std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, **_dl_kwargs(num_workers))
    return loader, df
