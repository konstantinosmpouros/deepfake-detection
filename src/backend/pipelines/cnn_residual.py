"""cnn-residual — custom residual CNN + SE attention, trained with EMA.

The committed checkpoint stores the EMA weights under the 'ema' key, so we
reload via load_ema_weights (the EMA copy is the eval model).
"""
from __future__ import annotations

from .base import BasePipeline, M, T


class CNNResidualPipeline(BasePipeline):
    key = "cnn-residual"
    label = "cnn-residual"
    working_size = 128
    norm = "dataset"

    def build(self):
        return M.build_cnn_residual(attention="se", se_reduction=16, p_drop=0.3)

    def load_weights(self, model, ckpt_path):
        T.load_ema_weights(ckpt_path, model, map_location=self.device)
        return model

    def target_layers(self):
        return [self.model.stages[-1]]   # last residual block (before SE/GAP)
