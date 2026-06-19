"""Tests for src/backend/pipelines/__init__.py — the selector-key registry.

Marked `slow` because importing the registry pulls in every pipeline module
(and torch). No models are built here: list_pipelines()/__init__ only read
metadata + check weight-file availability on disk, so this stays GPU-free.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

EXPECTED_KEYS = {
    "cnn-scratch", "cnn-residual",
    "cnn-finetune-efficientnet_b0", "cnn-finetune-resnet50",
    "vit-lora", "clip-probe", "two-stream",
}


def test_all_keys_present():
    from src.backend import pipelines as PL

    assert set(PL.all_keys()) == EXPECTED_KEYS
    assert len(PL.all_keys()) == 7        # cnn-finetune contributes two keys


def test_list_pipelines_metadata_shape():
    from src.backend import pipelines as PL

    rows = PL.list_pipelines()
    assert len(rows) == 7
    required = {"key", "label", "pipeline", "available", "downloads_backbone"}
    for row in rows:
        assert required <= set(row)
        assert isinstance(row["available"], bool)
        assert isinstance(row["downloads_backbone"], bool)


def test_both_finetune_keys_map_to_one_folder():
    from src.backend import pipelines as PL

    rows = {r["key"]: r for r in PL.list_pipelines()}
    assert rows["cnn-finetune-efficientnet_b0"]["pipeline"] == "cnn-finetune"
    assert rows["cnn-finetune-resnet50"]["pipeline"] == "cnn-finetune"


def test_get_pipeline_unknown_key_raises():
    from src.backend import pipelines as PL

    with pytest.raises(KeyError):
        PL.get_pipeline("not-a-real-pipeline")
