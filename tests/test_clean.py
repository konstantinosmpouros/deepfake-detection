"""Tests for utils/clean.py — corrupt detection + exact/perceptual hashing.

Uses small PIL images written to tmp; no GPU, no network. These are the
primitives the cleaning notebook builds dedup/leakage logic on top of.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from utils import clean as C


def _save(tmp_path, name, arr):
    """Write an HxWx3 uint8 array as PNG and return the path."""
    path = tmp_path / name
    Image.fromarray(arr.astype(np.uint8), "RGB").save(path)
    return path


def test_scan_identical_files_match_on_both_hashes(tmp_path, rng):
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    a = _save(tmp_path, "a.png", arr)
    b = _save(tmp_path, "b.png", arr.copy())
    ra, rb = C.scan_image(a), C.scan_image(b)
    assert ra["readable"] and rb["readable"]
    assert ra["sha1"] == rb["sha1"]                 # byte-identical PNGs
    assert ra["phash"] == rb["phash"]
    assert C.hamming(ra["phash"], rb["phash"]) == 0
    assert ra["width"] == 64 and ra["height"] == 64 and ra["mode"] == "RGB"


def test_scan_reports_dimensions_and_mode(tmp_path):
    arr = np.zeros((30, 50, 3), dtype=np.uint8)
    rec = C.scan_image(_save(tmp_path, "dims.png", arr))
    assert rec["width"] == 50 and rec["height"] == 30


def test_near_duplicate_has_small_hamming_distance(tmp_path, rng):
    base = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    nudged = base.copy()
    nudged[0, 0] = (nudged[0, 0].astype(int) + 3) % 256   # imperceptible tweak
    r1 = C.scan_image(_save(tmp_path, "base.png", base))
    r2 = C.scan_image(_save(tmp_path, "nudged.png", nudged))
    # Different bytes (sha1 differs) but perceptually close (small pHash distance).
    assert r1["sha1"] != r2["sha1"]
    assert C.hamming(r1["phash"], r2["phash"]) <= 4


def test_distinct_images_differ(tmp_path, rng):
    black = np.zeros((64, 64, 3), dtype=np.uint8)
    noise = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    r1 = C.scan_image(_save(tmp_path, "black.png", black))
    r2 = C.scan_image(_save(tmp_path, "noise.png", noise))
    assert r1["sha1"] != r2["sha1"]


def test_corrupt_file_is_flagged_not_raised(tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"\xff\xd8\xff" + b"this is not a real jpeg payload" * 3)
    rec = C.scan_image(bad)
    assert rec["readable"] is False
    assert rec["error"]                              # non-empty message
    assert rec["phash"] is None


def test_missing_file_is_flagged_not_raised(tmp_path):
    rec = C.scan_image(tmp_path / "does-not-exist.png")
    assert rec["readable"] is False
    assert rec["error"]


@pytest.mark.parametrize("h1,h2,expected", [
    ("0000000000000000", "0000000000000000", 0),
    ("0000000000000000", "000000000000000f", 4),
    ("ffffffffffffffff", "0000000000000000", 64),
])
def test_hamming_known_values(h1, h2, expected):
    assert C.hamming(h1, h2) == expected
