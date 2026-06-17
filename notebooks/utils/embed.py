"""Image embedding helpers for EDA visualization (t-SNE / UMAP / PCA).

Two backends:
- ``clip``   : frozen CLIP image encoder (semantic features; previews the
               `clip-probe` pipeline). Downloads weights once.
- ``pixels`` : flattened down-scaled RGB pixels (no download, low-level baseline).

These produce feature matrices only; the dimensionality reduction and plotting
stay visible in the notebook.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate corrupt files (02_cleaning flags them)


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def embed_uint8(model, device, arrays, size: int = 224,
                mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, batch_size: int = 256) -> np.ndarray:
    """Encode a uint8 (N,H,W,3) array/memmap with a CLIP model (GPU resize+normalize, no decode).

    Fast path for embedding the pre-decoded 256px cache. Returns L2-normalized (N,D).
    """
    import torch.nn.functional as F

    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(arrays), batch_size):
            b = np.asarray(arrays[i:i + batch_size])
            x = torch.from_numpy(b).to(device).permute(0, 3, 1, 2).float() / 255.0
            x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
            x = (x - m) / s
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.float().cpu().numpy())
    return np.concatenate(feats)


def load_clip(model_name: str = "ViT-B-32", pretrained: str = "openai", device: str | None = None):
    """Load a frozen CLIP model + its eval preprocessing. Returns (model, preprocess, device)."""
    import open_clip

    device = device or get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    return model.to(device).eval(), preprocess, device


def embed_paths(paths, model, preprocess, device, batch_size: int = 64) -> np.ndarray:
    """Encode image paths into L2-normalized CLIP embeddings, shape ``(N, D)``."""
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(device)
            f = model.encode_image(imgs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.float().cpu().numpy())
    return np.concatenate(feats)


def pixel_features(paths, size: int = 32) -> np.ndarray:
    """Flattened down-scaled RGB pixels, shape ``(N, size*size*3)`` — a no-download baseline."""
    return np.stack([
        np.asarray(Image.open(p).convert("RGB").resize((size, size))).reshape(-1)
        for p in paths
    ]).astype(np.float32)
