"""patch-ensemble — native-resolution patch bags + gated-attention MIL (224px).

Unlike the other image pipelines this consumes a BAG of K native-resolution
patches (no global 256 resize), reusing PatchBagDataset's deterministic eval
crops so the app reproduces the cross-generator evaluation exactly. K and the
backbone are rebuilt from best_params.json. Explainability is the MIL per-patch
attention — which crops the pooling weighted most.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .base import BasePipeline, D


class PatchEnsemblePipeline(BasePipeline):
    key = "patch-ensemble"
    label = "patch-ensemble"
    working_size = 224
    norm = "imagenet"

    def warmup(self):
        super().warmup()                       # load_model (PatchEnsemble) + imagenet stats
        from utils import eval_protocols as EP   # noqa: E402
        self.K = int(EP.best_params("patch-ensemble").get("K", 4))

    def preprocess(self, image_path):
        # Reuse the exact eval patch grid (corners / centre) from the OOD loader.
        df = pd.DataFrame({"filepath": [str(image_path)], "label": ["real"]})
        ds = D.PatchBagDataset(df, patch=self.working_size, k=self.K, train=False,
                               mean=self.mean, std=self.std)
        bag, _ = ds[0]                          # (K, 3, p, p)
        return bag.unsqueeze(0).to(self.device)   # (1, K, 3, p, p)

    @torch.no_grad()
    def explain(self, image_path):
        x = self.preprocess(image_path)
        _, attn = self.model.forward_attn(x)    # logit, (1, K) patch weights
        w = attn[0].detach().float().cpu().numpy()
        wn = w / (w.max() + 1e-8)
        patches = x[0]                          # (K, 3, p, p)
        strips = []
        for i in range(patches.shape[0]):
            rgb = D.denormalize(patches[i], self.mean, self.std).permute(1, 2, 0).cpu().numpy()
            img = (rgb * 255.0).astype(np.uint8)
            bar = np.zeros((16, img.shape[1], 3), dtype=np.uint8)
            bar[:, :, 1] = int(255 * float(wn[i]))   # green header ∝ MIL attention weight
            strips.append(np.concatenate([bar, img], axis=0))
        montage = np.concatenate(strips, axis=1)
        return {"available": True, "method": "patch-attention", "overlay": montage, "reason": None}
