"""freqcross — RGB + FFT + radial-spectrum, attention fusion (128px, dataset norm).

Multi-component: reports the fused p_fake plus per-branch (spatial / frequency /
radial) p_fake via the model's forward_all(). The tuned architecture
(feat / n_radial) is rebuilt from best_params.json by eval_protocols.load_model.
"""
from __future__ import annotations

import torch

from .base import BasePipeline


class FreqCrossPipeline(BasePipeline):
    key = "freqcross"
    label = "freqcross"
    working_size = 128
    norm = "dataset"

    def target_layers(self):
        # Grad-CAM w.r.t. the fused decision, localized on the spatial branch.
        return [self.model.spatial.features[-1]]

    def forward_components(self, x):
        fused, spatial, freq, radial = self.model.forward_all(x)

        def p(t):
            return torch.sigmoid(self._squeeze(t)).item()

        return {
            "final": p(fused),
            "components": {"spatial": p(spatial), "frequency": p(freq), "radial": p(radial)},
        }
