"""clip-probe — frozen CLIP image encoder + trained MLP head.

The MLP head is rebuilt from best_params.json (hidden / n_layers) and its
committed weights are loaded via eval_protocols.load_model; the frozen CLIP
encoder is re-downloaded and used to embed each upload exactly as in the OOD
evaluation (embed_paths).
"""
from __future__ import annotations

import torch

from .base import BasePipeline
from utils import embed   # noqa: E402


class ClipProbePipeline(BasePipeline):
    key = "clip-probe"
    label = "clip-probe"
    norm = "clip"
    downloads_backbone = True   # downloads the frozen CLIP encoder on first warm-up

    ENCODER_NAME = "ViT-B-32"
    ENCODER_PRETRAINED = "openai"

    def __init__(self, device=None):
        super().__init__(device)
        self.clip_model = None
        self.clip_pre = None

    def warmup(self):
        # Frozen encoder + its eval preprocessing (used to embed each upload).
        self.clip_model, self.clip_pre, _ = embed.load_clip(
            self.ENCODER_NAME, self.ENCODER_PRETRAINED, device=str(self.device)
        )
        # Head rebuilt from best_params.json (hidden=1024, n_layers=3) + committed weights.
        self.model = self._load_model()

    def preprocess(self, image_path):
        emb = embed.embed_paths([str(image_path)], self.clip_model, self.clip_pre, self.device)
        return torch.from_numpy(emb).to(self.device)   # (1, EMB_DIM), L2-normalized

    def explain(self, image_path):
        # The head scores a frozen global CLIP embedding — there is no spatial
        # feature map to attribute per image. Global explainability is the
        # embedding t-SNE in notebook 08.
        return {"available": False, "method": None, "overlay": None,
                "reason": "clip-probe classifies a single frozen CLIP embedding — there is no "
                          "spatial feature map to highlight. See the embedding t-SNE in notebook 08 "
                          "for how real/fake and per-generator clusters separate."}
