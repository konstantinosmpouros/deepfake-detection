"""Tests for src/backend/schemas.py — the shared prediction/response contract.

Pure pydantic models (no torch). These guard the JSON shape the Streamlit UI
relies on: every prediction carries `final` + a `components` list.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.backend import schemas as S


def test_prediction_response_roundtrip():
    pr = S.PredictionResponse(
        pipeline="two-stream",
        key="two-stream",
        timestamp="2026-06-19T12:34:56Z",
        image="upload.png",
        final=S.Decision(label="fake", p_fake=0.93),
        components=[
            S.Component(name="spatial", label="fake", p_fake=0.88),
            S.Component(name="frequency", label="fake", p_fake=0.95),
        ],
        saved_to="predictions/two-stream/2026-06-19T12-34-56Z/",
    )
    dumped = pr.model_dump()
    assert dumped["final"]["label"] == "fake"
    assert dumped["final"]["p_fake"] == 0.93
    assert [c["name"] for c in dumped["components"]] == ["spatial", "frequency"]


def test_single_component_pipeline_still_has_components_list():
    pr = S.PredictionResponse(
        pipeline="cnn-scratch",
        key="cnn-scratch",
        timestamp="2026-06-19T00:00:00Z",
        image="x.jpg",
        final=S.Decision(label="real", p_fake=0.12),
        components=[S.Component(name="cnn-scratch", label="real", p_fake=0.12)],
        saved_to="predictions/cnn-scratch/ts/",
    )
    assert len(pr.components) == 1
    assert pr.components[0].name == "cnn-scratch"


def test_component_extends_decision():
    c = S.Component(name="freq", label="fake", p_fake=0.7)
    assert isinstance(c, S.Decision)
    assert c.name == "freq" and c.label == "fake"


def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        S.Decision(label="fake")          # p_fake missing


def test_status_response_shape():
    sr = S.StatusResponse(
        resident="vit-lora",
        busy=False,
        device="cuda",
        pipelines=[
            S.PipelineInfo(key="vit-lora", label="ViT + LoRA", pipeline="vit-lora",
                           available=True, downloads_backbone=True, state="warm"),
            S.PipelineInfo(key="cnn-scratch", label="CNN scratch", pipeline="cnn-scratch",
                           available=True, downloads_backbone=False, state="cold"),
        ],
    )
    assert sr.resident == "vit-lora"
    assert {p.state for p in sr.pipelines} == {"warm", "cold"}


def test_explain_response_optional_fields_default_none():
    er = S.ExplainResponse(available=False, method=None, overlay_png_b64=None)
    assert er.reason is None                  # optional, defaults to None
    er2 = S.ExplainResponse(available=True, method="grad-cam",
                            overlay_png_b64="iVBORw0KGgo=", reason=None)
    assert er2.method == "grad-cam"
