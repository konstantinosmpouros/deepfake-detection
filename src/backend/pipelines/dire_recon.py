"""dire-recon — DIRE reconstruction-error classifier (224px, ImageNet norm).

Heavy: each upload is DDIM-inverted and reconstructed with Stable Diffusion v1.5
(via utils.dire) to build its per-pixel error map, which the committed
EfficientNet-B0 classifier then scores. Requires `diffusers` + `accelerate` (not
in the base requirements) and downloads SD v1.5 on first warm-up; per-image
inference runs the diffusion loop, so a GPU is strongly recommended.
"""
from __future__ import annotations

import torch

from .base import BasePipeline, D


class DireReconPipeline(BasePipeline):
    key = "dire-recon"
    label = "dire-recon"
    working_size = 224
    norm = "imagenet"
    downloads_backbone = True       # downloads Stable Diffusion v1.5 on first warm-up
    DDIM_STEPS = 20

    def __init__(self, device=None):
        super().__init__(device)
        self.pipe = self.fwd = self.inv = None

    def is_available(self):
        # Needs the committed classifier AND the diffusers stack to build DIRE maps.
        if not super().is_available():
            return False
        import importlib.util
        return importlib.util.find_spec("diffusers") is not None

    def warmup(self):
        super().warmup()            # the cheap classifier that scores the DIRE map
        from utils import dire      # noqa: E402  (imports diffusers)
        self.pipe, self.fwd, self.inv = dire.load_dire_pipeline(
            device=str(self.device), steps=self.DDIM_STEPS)

    @torch.no_grad()
    def preprocess(self, image_path):
        from utils import dire      # noqa: E402
        x512 = dire.read_512(image_path).unsqueeze(0)                 # (1,3,512,512) in [0,1]
        d = dire.compute_dire(x512, self.pipe, self.fwd, self.inv)    # (1,3,512,512)
        maps = dire.dire_to_uint8(d, size=256)                       # (1,256,256,3) uint8
        t = torch.from_numpy(maps[0]).permute(2, 0, 1).contiguous()  # CHW uint8
        tf = D.build_eval_tf(self.working_size, self.mean, self.std)
        return tf(t).unsqueeze(0).to(self.device)

    def explain(self, image_path):
        return {"available": False, "method": None, "overlay": None,
                "reason": "dire-recon classifies the DIRE reconstruction-error map, which IS its "
                          "input — see the DIRE map gallery in notebook 13."}
