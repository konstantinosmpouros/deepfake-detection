"""Dataset helpers: download, root resolution, and layout-agnostic manifest building.

Supports datasets with different folder layouts by inferring ``label`` / ``split`` /
``source`` from folder names rather than hardcoding one structure. Verified on:

- ``ai-real-images``: ``<split>/<fake|real>/*.jpg``
- ``tiny-genimage`` : ``<generator>/<train|val>/<ai|nature>/*.png``

Each function does one thing; the data-collection notebook wires them together so
the logic stays visible there.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---- Dataset facts ---------------------------------------------------------

# Known Kaggle slugs by short dataset name (the name is the folder/manifest key).
KAGGLE_SLUGS = {
    "ai-real-images": "tristanzhang32/ai-generated-images-vs-real-images",
    "tiny-genimage": "yangsangtai/tiny-genimage",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Folder-name (lowercased) -> canonical label / split.
LABEL_SYNONYMS = {
    "real": "real", "reals": "real", "nature": "real", "natural": "real", "authentic": "real",
    "fake": "fake", "fakes": "fake", "ai": "fake", "generated": "fake", "synthetic": "fake", "gan": "fake",
}
SPLIT_SYNONYMS = {
    "train": "train", "training": "train",
    "test": "test", "testing": "test",
    "val": "val", "valid": "val", "validation": "val",
}


# ---- Download --------------------------------------------------------------

def download_kaggle_dataset(slug: str) -> Path:
    """Download a Kaggle dataset via ``kagglehub`` and return its local path.

    Requires Kaggle credentials (``~/.kaggle/kaggle.json`` or the
    ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` env vars). Files land in the kagglehub
    cache; we reference them there rather than copying them.
    """
    import kagglehub  # imported lazily so the module loads without it installed

    return Path(kagglehub.dataset_download(slug))


def find_dataset_root(search_dir: Path | str) -> Path:
    """Descend through single-folder wrappers to the directory that holds the data.

    kagglehub sometimes nests the payload one level deep. We step into a folder
    only while it contains exactly one subdirectory and no files, so we stop at
    the real root (which has the split/label/generator folders).
    """
    current = Path(search_dir)
    while True:
        entries = [p for p in current.iterdir() if not p.name.startswith(".")]
        subdirs = [p for p in entries if p.is_dir()]
        files = [p for p in entries if p.is_file()]
        if len(subdirs) == 1 and not files:
            current = subdirs[0]
        else:
            return current


# ---- Layout inference ------------------------------------------------------

def _match_synonym(parts: list[str], mapping: dict[str, str]) -> str | None:
    """Return the canonical value for the first path part that matches ``mapping``."""
    for part in parts:
        if part.lower() in mapping:
            return mapping[part.lower()]
    return None


def _infer_source(parts: list[str]) -> str | None:
    """The first dir part that is neither a split nor a label (e.g. a generator name)."""
    for part in parts:
        low = part.lower()
        if low in SPLIT_SYNONYMS or low in LABEL_SYNONYMS:
            continue
        return part  # keep original case, e.g. 'imagenet_ai_0419_biggan'
    return None


def iter_images(root: Path | str):
    """Yield every image file under ``root`` (recursively)."""
    root = Path(root)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


# ---- Manifest --------------------------------------------------------------

def build_manifest(root: Path | str) -> pd.DataFrame:
    """Walk the dataset and return one row per image, inferring fields from folders.

    Columns: ``filepath, filename, label, split, source``. ``label`` is
    ``real`` / ``fake`` (or ``None`` if a folder name didn't match a known
    synonym), ``split`` is ``train`` / ``test`` / ``val`` / ``None``, and
    ``source`` captures a per-generator folder when present (else ``None``).
    """
    root = Path(root).resolve()
    rows = []
    for p in iter_images(root):
        dir_parts = list(p.relative_to(root).parts[:-1])  # drop the filename
        rows.append({
            "filepath": str(p),
            "filename": p.name,
            "label": _match_synonym(dir_parts, LABEL_SYNONYMS),
            "split": _match_synonym(dir_parts, SPLIT_SYNONYMS),
            "source": _infer_source(dir_parts),
        })
    return pd.DataFrame(rows)


def summarize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Counts grouped by the present structural columns (source/split/label)."""
    cols = [c for c in ("source", "split", "label") if c in df.columns]
    return df.groupby(cols, dropna=False).size().reset_index(name="count")
