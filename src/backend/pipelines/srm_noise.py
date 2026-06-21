"""srm-noise — SRM + Bayar noise-residual detector (128px, dataset norm).

Single-component. The fixed SRM high-pass bank and the constrained Bayar conv
suppress content and surface the residual fingerprint; a small CNN classifies it.
Architecture (feat / bayar_ch) is rebuilt from best_params.json.
"""
from __future__ import annotations

from .base import BasePipeline


class SrmNoisePipeline(BasePipeline):
    key = "srm-noise"
    label = "srm-noise"
    working_size = 128
    norm = "dataset"

    def target_layers(self):
        # Grad-CAM on the residual feature CNN (content already suppressed upstream).
        return [self.model.body.features[-1]]
