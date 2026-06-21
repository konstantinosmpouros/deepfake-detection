"""Pipeline registry: selector key -> factory(device) -> BasePipeline.

`manager.py` and `main.py` only ever call `get_pipeline(key, device)` and
`list_pipelines()`, so nothing outside this package knows pipeline internals.
cnn-finetune exposes one key per backbone (both write to the 'cnn-finetune'
folder). All ten developed pipelines are registered here; each adapter rebuilds
its architecture from the committed best_params.json (via eval_protocols), so the
app matches the report exactly.
"""
from __future__ import annotations

from .base import BasePipeline
from .cnn_scratch import CNNScratchPipeline
from .cnn_residual import CNNResidualPipeline
from .cnn_finetune import CNNFinetunePipeline
from .vit_lora import VitLoraPipeline
from .clip_probe import ClipProbePipeline
from .two_stream import TwoStreamPipeline
from .freqcross import FreqCrossPipeline
from .srm_noise import SrmNoisePipeline
from .patch_ensemble import PatchEnsemblePipeline
from .dire_recon import DireReconPipeline

# Ordered: key -> factory. device defaults to None for cheap metadata listing
# (the architecture is only built in warmup(), not in __init__).
REGISTRY = {
    "cnn-scratch": lambda device=None: CNNScratchPipeline(device),
    "cnn-residual": lambda device=None: CNNResidualPipeline(device),
    "cnn-finetune-efficientnet_b0": lambda device=None: CNNFinetunePipeline(device, "efficientnet_b0"),
    "cnn-finetune-resnet50": lambda device=None: CNNFinetunePipeline(device, "resnet50"),
    "vit-lora": lambda device=None: VitLoraPipeline(device),
    "clip-probe": lambda device=None: ClipProbePipeline(device),
    "two-stream": lambda device=None: TwoStreamPipeline(device),
    "freqcross": lambda device=None: FreqCrossPipeline(device),
    "srm-noise": lambda device=None: SrmNoisePipeline(device),
    "patch-ensemble": lambda device=None: PatchEnsemblePipeline(device),
    "dire-recon": lambda device=None: DireReconPipeline(device),
}


def all_keys() -> list[str]:
    return list(REGISTRY.keys())


def get_pipeline(key: str, device=None) -> BasePipeline:
    if key not in REGISTRY:
        raise KeyError(f"unknown pipeline key: {key!r}")
    return REGISTRY[key](device)


def list_pipelines() -> list[dict]:
    """Lightweight metadata for every pipeline (no models built)."""
    out = []
    for key in REGISTRY:
        p = REGISTRY[key]()
        out.append({
            "key": p.key,
            "label": p.label,
            "pipeline": p.pipeline,
            "available": p.is_available(),
            "downloads_backbone": p.downloads_backbone,
        })
    return out
