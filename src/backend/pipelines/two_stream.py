"""two-stream — RGB + frequency fusion (128px, dataset norm). Multi-component.

Reports the fused p_fake plus per-stream (spatial / frequency) p_fake via the
model's forward_all().
"""
from __future__ import annotations

import torch

from .base import BasePipeline, M


class TwoStreamPipeline(BasePipeline):
    key = "two-stream"
    label = "two-stream"
    working_size = 128
    norm = "dataset"

    def build(self):
        return M.build_two_stream(feat=256, p_drop=0.3)

    def target_layers(self):
        # Grad-CAM w.r.t. the fused decision, localized on the spatial branch.
        return [self.model.spatial.features[-1]]

    def forward_components(self, x):
        fused, spatial, freq = self.model.forward_all(x)
        return {
            "final": torch.sigmoid(self._squeeze(fused)).item(),
            "components": {
                "spatial": torch.sigmoid(self._squeeze(spatial)).item(),
                "frequency": torch.sigmoid(self._squeeze(freq)).item(),
            },
        }
