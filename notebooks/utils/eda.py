"""Primitives for exploratory data analysis: image I/O, basic stats, and frequency.

Small, single-purpose helpers. The EDA notebook does the sampling, looping, and
plotting visibly; these only remove repetition.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageFile

# Tolerate truncated/corrupt files so EDA doesn't crash on a bad image.
# (02_cleaning flags these properly; here we just want to keep reading.)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Luminance weights (Rec. 601) for RGB -> grayscale.
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def read_rgb(path, size: int | None = None) -> np.ndarray:
    """Load an image as an ``H×W×3`` uint8 RGB array, optionally resized to ``size×size``."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img)


def image_meta(path) -> dict:
    """Return basic metadata (width, height, mode, file size in bytes) for one image."""
    with Image.open(path) as im:
        w, h = im.size
        mode = im.mode
    return {"width": w, "height": h, "mode": mode, "bytes": os.path.getsize(path)}


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Luminance grayscale (float32) from an RGB array."""
    return rgb.astype(np.float32) @ LUMA


def log_power_spectrum(gray: np.ndarray) -> np.ndarray:
    """Centered log power spectrum ``log(1 + |F|^2)`` of a 2D grayscale image."""
    f = np.fft.fftshift(np.fft.fft2(gray))
    return np.log1p(np.abs(f) ** 2)


def azimuthal_average(img2d: np.ndarray) -> np.ndarray:
    """Radial (azimuthally-averaged) profile of a 2D array, indexed by integer radius."""
    h, w = img2d.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y, x = np.indices((h, w))
    r = np.hypot(x - cx, y - cy).astype(int)
    totals = np.bincount(r.ravel(), weights=img2d.ravel())
    counts = np.bincount(r.ravel())
    return totals / np.maximum(counts, 1)
