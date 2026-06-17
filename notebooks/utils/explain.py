"""Light Grad-CAM for the CNN pipelines (pytorch_grad_cam).

Single-purpose: produce a CAM overlay for one image + model + target layer. The
notebook picks images, chooses the target layer (must match the built model's
module names), and saves the panel. ViT attention-rollout is a later notebook.
"""
from __future__ import annotations

import numpy as np


def pick_examples(df, n_per_class: int = 3, seed: int = 0) -> list[dict]:
    """Reproducible sample of n real + n fake rows as list of dicts (filepath, label)."""
    parts = []
    for lab in ("real", "fake"):
        sub = df[df["label"] == lab]
        parts.append(sub.sample(min(n_per_class, len(sub)), random_state=seed))
    import pandas as pd
    return pd.concat(parts)[["filepath", "label"]].to_dict("records")


class _SingleLogitTarget:
    """Grad-CAM target for a 1-logit head; robust to (B,)/(B,1)/scalar per-sample output."""

    def __init__(self, index: int = 0):
        self.index = index

    def __call__(self, model_output):
        return model_output.flatten()[self.index]


def attention_rollout(model, input_tensor, out_size: int, head_fusion: str = "mean"):
    """Attention rollout (Abnar & Zuidema) for a timm ViT -> (out_size, out_size) heatmap in [0,1].

    Captures per-block attention from `attn.attn_drop`, adds the residual identity,
    row-normalizes, multiplies across layers, takes the CLS-to-patch row, reshapes to
    the patch grid, and resizes to `out_size`. Operates on a plain timm ViT (merge LoRA first).
    """
    import torch
    from PIL import Image

    attns = []
    handles = []
    for blk in model.blocks:
        if hasattr(blk.attn, "fused_attn"):
            blk.attn.fused_attn = False                    # materialize attention weights
        handles.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: attns.append(o.detach())))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(input_tensor)
    for h in handles:
        h.remove()
    if was_training:
        model.train()

    n = attns[0].shape[-1]
    result = torch.eye(n, device=attns[0].device)
    for a in attns:
        a = a[0]                                            # (heads, N, N), batch=1
        a = a.mean(0) if head_fusion == "mean" else a.max(0).values
        a = a + torch.eye(n, device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result
    mask = result[0, 1:]                                    # CLS -> patch tokens
    g = int(round(mask.numel() ** 0.5))
    mask = mask.reshape(g, g).float().cpu().numpy()
    mask = mask / (mask.max() + 1e-8)
    return np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((out_size, out_size), Image.BILINEAR),
                      dtype=np.float32) / 255.0


def gradcam_overlay(model, target_layers, input_tensor, rgb_float01, target_category: int = 0):
    """Return an H×W×3 uint8 overlay of Grad-CAM on `rgb_float01` for one image.

    input_tensor: (1,C,H,W) normalized as the model expects (on the model's device).
    rgb_float01:  (H,W,3) float in [0,1], same working size, for display.
    target_category: index of the output to explain (single-logit head -> 0 = fake score).
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale = cam(input_tensor=input_tensor, targets=[_SingleLogitTarget(target_category)])[0]
    overlay = show_cam_on_image(np.ascontiguousarray(rgb_float01, dtype=np.float32),
                                grayscale, use_rgb=True)
    return overlay
