"""cnn-scratch — small from-scratch CNN baseline (128px, dataset norm)."""
from __future__ import annotations

from .base import BasePipeline, M


class CNNScratchPipeline(BasePipeline):
    key = "cnn-scratch"
    label = "cnn-scratch"
    working_size = 128
    norm = "dataset"

    def build(self):
        return M.build_cnn_scratch(p_drop=0.3)

    def target_layers(self):
        return [self.model.features[-1]]   # last conv block
