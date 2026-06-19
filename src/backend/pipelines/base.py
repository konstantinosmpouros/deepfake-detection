"""BasePipeline — the shared inference contract for every pipeline.

A pipeline is a thin adapter that knows how to:
  * build()        — reconstruct the architecture (via notebooks/utils.models)
  * load_weights() — attach the committed trained part
  * preprocess()   — turn one uploaded image into a model-ready tensor
  * forward()      — produce a fused p_fake (+ per-component p_fake)

The heavy logic (architectures, transforms, weight loading) lives in
``notebooks/utils`` and is reused verbatim, so app predictions match the
notebooks' evaluation exactly. Subclasses override only what is special.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

from .. import config

# Reused research code (notebooks/utils is on sys.path via config import).
from utils import datasets as D          # noqa: E402
from utils import explain as E           # noqa: E402  (Grad-CAM / attention rollout)
from utils import models as M            # noqa: E402  (used by subclasses)
from utils import training as T          # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BasePipeline:
    """Single-output pipeline. Override class attrs + build() for most cases."""

    # --- identity --------------------------------------------------------
    key: str = ""                 # unique selector id (API + UI)
    pipeline: str = ""            # artifact / predictions folder name (CLAUDE.md)
    label: str = ""               # human label for the UI
    component_name: str = "model"  # name of the single component in the JSON

    # --- reconstruction --------------------------------------------------
    working_size: int = 224
    norm: str = "imagenet"        # 'dataset' | 'imagenet' (passed to resolve_stats)
    weights_filename: str = "best.pt"
    downloads_backbone: bool = False  # True if build() fetches a frozen backbone

    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.model = None
        self.mean = None
        self.std = None
        if not self.pipeline:
            self.pipeline = self.key
        if not self.label:
            self.label = self.key

    # --- paths / availability -------------------------------------------
    def weights_path(self) -> Path:
        return config.ARTIFACTS_DIR / self.pipeline / "models" / self.weights_filename

    def is_available(self) -> bool:
        return self.weights_path().exists()

    # --- construction (override per pipeline) ---------------------------
    def build(self) -> torch.nn.Module:
        raise NotImplementedError

    def load_weights(self, model, ckpt_path):
        """Default: full-model checkpoint (save_checkpoint format)."""
        T.load_checkpoint(ckpt_path, model, map_location=self.device)
        return model

    def warmup(self) -> None:
        """Build the architecture, attach trained weights, ready for inference."""
        model = self.build().to(self.device).eval()
        ckpt = self.weights_path()
        if ckpt.exists():
            model = self.load_weights(model, ckpt)
        self.model = model.to(self.device).eval()
        self.mean, self.std = D.resolve_stats(self.norm, config.AIR_DIR)

    # --- preprocessing (mirrors the from-files eval path / make_ood_loader) ---
    def preprocess(self, image_path) -> torch.Tensor:
        arr = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)  # HWC (writable copy)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()                  # CHW uint8
        tf = D.build_eval_tf(self.working_size, self.mean, self.std)
        return tf(t).unsqueeze(0).to(self.device)

    # --- inference -------------------------------------------------------
    @torch.no_grad()
    def predict(self, image_path) -> dict:
        x = self.preprocess(image_path)
        scores = self.forward_components(x)   # {'final': p, 'components': {name: p}}
        return {
            "final": self._decide(scores["final"]),
            "components": [
                {"name": n, **self._decide(p)} for n, p in scores["components"].items()
            ],
        }

    def forward_components(self, x) -> dict:
        """Single-logit forward → one fused probability and one component."""
        p = torch.sigmoid(self._squeeze(self.model(x))).item()
        return {"final": p, "components": {self.component_name: p}}

    # --- explainability --------------------------------------------------
    explain_method = "grad-cam"

    def target_layers(self) -> list | None:
        """CNN subclasses return the conv layer(s) to attribute (Grad-CAM).
        None → no spatial explanation for this pipeline."""
        return None

    def explain(self, image_path) -> dict:
        """Default: Grad-CAM overlay on the working-size image.

        Returns {available, method, overlay (HWC uint8 | None), reason}.
        """
        layers = self.target_layers()
        if not layers:
            return {"available": False, "method": None, "overlay": None,
                    "reason": "no spatial explanation configured for this pipeline"}
        x = self.preprocess(image_path)
        rgb = D.denormalize(x[0], self.mean, self.std).permute(1, 2, 0).cpu().numpy()
        overlay = E.gradcam_overlay(self.model, layers, x, rgb)
        return {"available": True, "method": self.explain_method, "overlay": overlay, "reason": None}

    # --- helpers ---------------------------------------------------------
    @staticmethod
    def _squeeze(out: torch.Tensor) -> torch.Tensor:
        return out.squeeze(1) if out.ndim > 1 else out

    @staticmethod
    def _decide(p_fake: float) -> dict:
        return {"label": "fake" if p_fake >= 0.5 else "real", "p_fake": round(float(p_fake), 6)}
