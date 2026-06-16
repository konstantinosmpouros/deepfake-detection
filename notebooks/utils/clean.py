"""Data-integrity helpers for the cleaning notebook: corrupt detection + hashing.

One disk read per image yields: readability (strict — truncated files raise),
a content hash (exact duplicates), and a perceptual hash (near-duplicates).
The notebook does the grouping/leakage logic visibly; these are the primitives.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from scipy.fftpack import dct

# Decode very large images instead of raising a DecompressionBomb error;
# we flag oversized images by their dimensions in the notebook instead.
Image.MAX_IMAGE_PIXELS = None


def _phash(img: Image.Image, img_size: int = 32, hash_size: int = 8) -> str:
    """64-bit DCT perceptual hash as a 16-char hex string."""
    g = img.convert("L").resize((img_size, img_size), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = d[:hash_size, :hash_size].flatten()
    med = np.median(low[1:])           # exclude the DC term
    bits = low > med
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:016x}"


def scan_image(path) -> dict:
    """One read per file -> integrity + hashes + dims. Never raises.

    Returns ``{readable, error, sha1, phash, width, height, mode}``. A truncated
    or corrupt file comes back with ``readable=False`` and the error message.
    """
    rec = {"readable": True, "error": "", "sha1": None, "phash": None,
           "width": None, "height": None, "mode": None}
    try:
        data = Path(path).read_bytes()
        rec["sha1"] = hashlib.sha1(data).hexdigest()
        prev = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = False     # strict: catch truncated files
        try:
            im = Image.open(io.BytesIO(data))
            rec["width"], rec["height"], rec["mode"] = im.width, im.height, im.mode
            im.draft("L", (64, 64))                 # fast, low-memory decode (JPEG)
            im.load()                               # force decode -> raises if truncated
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev
        rec["phash"] = _phash(im)
    except Exception as e:
        rec["readable"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def hamming(h1: str, h2: str) -> int:
    """Hamming distance between two 64-bit pHash hex strings."""
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")
