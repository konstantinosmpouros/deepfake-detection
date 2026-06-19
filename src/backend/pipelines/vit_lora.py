"""vit-lora — ViT-Base + LoRA (224px, ImageNet norm).

Only the LoRA + head deltas are committed. build() rebuilds the frozen ViT-21k
(downloaded), then we load the trained part (strict=False) and merge LoRA into
the backbone to get a plain ViT for zero-overhead inference.
"""
from __future__ import annotations

import numpy as np

from .base import BasePipeline, D, E, M, T


class VitLoraPipeline(BasePipeline):
    key = "vit-lora"
    label = "vit-lora"
    working_size = 224
    norm = "imagenet"
    downloads_backbone = True   # rebuilds the frozen ViT-21k on first warm-up

    def build(self):
        return M.build_vit_lora(r=16, lora_alpha=16, lora_dropout=0.05)  # pretrained=True

    def load_weights(self, model, ckpt_path):
        T.load_weights(ckpt_path, model, strict=False, map_location=self.device)
        return model.merge_and_unload()   # -> plain timm ViT with LoRA merged in

    def explain(self, image_path):
        """Attention rollout (Abnar & Zuidema) instead of Grad-CAM for the ViT."""
        from pytorch_grad_cam.utils.image import show_cam_on_image

        x = self.preprocess(image_path)
        rgb = D.denormalize(x[0], self.mean, self.std).permute(1, 2, 0).cpu().numpy()
        mask = E.attention_rollout(self.model, x, out_size=self.working_size)
        overlay = show_cam_on_image(np.ascontiguousarray(rgb, dtype=np.float32), mask, use_rgb=True)
        return {"available": True, "method": "attention-rollout", "overlay": overlay, "reason": None}
