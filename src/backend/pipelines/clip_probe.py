"""clip-probe — frozen CLIP image encoder + trained MLP head.

Only the tiny MLP head is committed; the frozen CLIP encoder is re-downloaded.
A single upload is embedded with the same open_clip preprocessing used for the
OOD evaluation (embed_paths), then scored by the head.
"""
from __future__ import annotations

import torch

from .base import BasePipeline, M, T
from utils import embed   # noqa: E402


class ClipProbePipeline(BasePipeline):
    key = "clip-probe"
    label = "clip-probe"
    downloads_backbone = True   # downloads the frozen CLIP encoder on first warm-up

    ENCODER_NAME = "ViT-B-32"
    ENCODER_PRETRAINED = "openai"
    EMB_DIM = 512               # ViT-B-32 image embedding dim

    def __init__(self, device=None):
        super().__init__(device)
        self.clip_model = None
        self.clip_pre = None

    def build(self):
        return M.build_mlp_head(self.EMB_DIM, hidden=256, p_drop=0.3)

    def warmup(self):
        # Frozen encoder + its eval preprocessing.
        self.clip_model, self.clip_pre, _ = embed.load_clip(
            self.ENCODER_NAME, self.ENCODER_PRETRAINED, device=str(self.device)
        )
        head = self.build().to(self.device).eval()
        ckpt = self.weights_path()
        if ckpt.exists():
            T.load_checkpoint(ckpt, head, map_location=self.device)
        self.model = head.to(self.device).eval()

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
