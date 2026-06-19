"""Pydantic response models — the shared schema every pipeline conforms to."""
from __future__ import annotations

from pydantic import BaseModel


class PipelineInfo(BaseModel):
    key: str
    label: str
    pipeline: str
    available: bool          # weights file present on disk
    downloads_backbone: bool
    state: str               # "cold" | "warming" | "warm"


class StatusResponse(BaseModel):
    resident: str | None     # key of the warm pipeline, if any
    busy: bool               # a warm-up is currently in progress
    device: str
    pipelines: list[PipelineInfo]


class SelectRequest(BaseModel):
    key: str


class SelectResponse(BaseModel):
    accepted: bool
    state: str
    detail: str | None = None


class Decision(BaseModel):
    label: str               # "fake" | "real"
    p_fake: float


class Component(Decision):
    name: str


class PredictionResponse(BaseModel):
    pipeline: str            # artifact/predictions folder name
    key: str                 # selector key (distinguishes finetune backbones)
    timestamp: str           # ISO-8601 UTC
    image: str               # original filename
    final: Decision
    components: list[Component]
    saved_to: str            # predictions/<pipeline>/<timestamp>/


class ExplainResponse(BaseModel):
    available: bool                   # False for pipelines without a spatial map (clip-probe)
    method: str | None                # "grad-cam" | "attention-rollout" | None
    overlay_png_b64: str | None       # base64-encoded PNG overlay, if available
    reason: str | None = None         # why it's unavailable, when applicable
