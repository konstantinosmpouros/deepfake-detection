"""DIRE (DIffusion Reconstruction Error) helper — notebook 15 (stretch / optional).

Idea (Wang et al., ICCV 2023, arXiv:2303.09295): invert an image to noise with
DDIM and reconstruct it with a pretrained latent diffusion model (Stable Diffusion
v1.5, empty prompt). **Diffusion-generated images reconstruct with lower error than
real photos**, so the absolute error map (DIRE) is a strong real/fake signal that a
plain CNN can classify.

Compute-heavy: ~2 * steps UNet calls per image. Requires `diffusers` + `accelerate`
(NOT in the base requirements — install separately). Because of the cost this is
designed to run on a SUBSAMPLE and cache the DIRE maps to disk; the classifier then
trains on the cached maps like any image dataset.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SD_MODEL = "runwayml/stable-diffusion-v1-5"
VAE_RES = 512          # SD operates at 512; DIRE maps are downscaled to DIRE_SIZE for caching
DIRE_SIZE = 256


def load_dire_pipeline(model_id: str = SD_MODEL, device: str | None = None, steps: int = 20):
    """Load SD v1.5 + forward/inverse DDIM schedulers. Returns (pipe, fwd_sched, inv_sched)."""
    from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    fwd = DDIMScheduler.from_config(pipe.scheduler.config)
    inv = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    fwd.set_timesteps(steps, device=device)
    inv.set_timesteps(steps, device=device)
    return pipe, fwd, inv


def read_512(path) -> torch.Tensor:
    """Read an image as a (3, 512, 512) float tensor in [0,1] (square resize+center crop)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    short = min(w, h)
    nw, nh = round(w * VAE_RES / short), round(h * VAE_RES / short)
    img = img.resize((nw, nh), Image.Resampling.BILINEAR)
    left, top = (nw - VAE_RES) // 2, (nh - VAE_RES) // 2
    img = img.crop((left, top, left + VAE_RES, top + VAE_RES))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


@torch.no_grad()
def _empty_embeds(pipe, bs: int, device, dtype):
    tok = pipe.tokenizer([""] * bs, padding="max_length",
                         max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt")
    return pipe.text_encoder(tok.input_ids.to(device))[0].to(dtype)


@torch.no_grad()
def compute_dire(x01: torch.Tensor, pipe, fwd, inv) -> torch.Tensor:
    """DIRE for a batch.

    x01: (B,3,512,512) float in [0,1]. Returns (B,3,512,512) error map in [0,1] (cpu).
    """
    device = pipe.device
    dtype = pipe.unet.dtype
    sf = pipe.vae.config.scaling_factor
    x = (x01.to(device, dtype) * 2.0 - 1.0)                       # VAE expects [-1,1]
    emb = _empty_embeds(pipe, x.shape[0], device, dtype)

    lat = pipe.vae.encode(x).latent_dist.mean * sf                # deterministic latent
    z = lat.clone()
    for t in inv.timesteps:                                        # invert image -> noise
        eps = pipe.unet(z, t, encoder_hidden_states=emb).sample
        z = inv.step(eps, t, z).prev_sample
    for t in fwd.timesteps:                                        # reconstruct noise -> image
        eps = pipe.unet(z, t, encoder_hidden_states=emb).sample
        z = fwd.step(eps, t, z).prev_sample
    recon = pipe.vae.decode(z / sf).sample
    recon01 = (recon / 2.0 + 0.5).clamp(0, 1)
    dire = (x01.to(device, dtype) - recon01).abs()
    return dire.float().cpu()


def dire_to_uint8(dire: torch.Tensor, size: int = DIRE_SIZE) -> np.ndarray:
    """(B,3,512,512) [0,1] -> (B,size,size,3) uint8, contrast-stretched per image for caching."""
    import torch.nn.functional as F

    d = F.interpolate(dire, size=(size, size), mode="bilinear", align_corners=False)
    out = []
    for m in d:
        m = m - m.amin()
        m = m / (m.amax() + 1e-6)
        out.append((m.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    return np.stack(out)
