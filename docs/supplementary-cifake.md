# Supplementary study — CIFAKE (low-resolution, in-distribution)

[← docs index](README.md)

This is a **supplementary, self-contained study** sitting deliberately *outside* the ten core pipelines.
It revisits the dataset the main project [considered and dropped](02-data.md) — **CIFAKE** (32×32, real
CIFAR-10 vs. Stable-Diffusion-v1.4 fakes) — and asks a narrower question than the headline project does:
not *which detector generalizes across generators*, but *on a single low-resolution source, how far does
each family of ideas get, and which factor actually moves the needle*. It is in-distribution only (one real
source, one generator), so there is no cross-generator OOD column here; its value is as a **contrast** to
the main study and as an empirical check on the reasoning that led the project to abandon 32×32 in the first
place.

> **Scope note — why this is supplementary, not a pipeline.** Two of its choices break the project's core
> conventions on purpose. (1) It works at **32×32**, the resolution the main project rejects via the
> [resolution-shortcut neutralisation](02-data.md#236-the-resolution-shortcut--the-most-important-data-decision);
> the point here is precisely to *show* the ceiling that low resolution imposes. (2) It includes **classical
> ML baselines** (k-NN, SVM, etc. on handcrafted features), which the deep-learning-only spine forbids — kept
> here only as a representational-quality yardstick, not as a competing detector. Treat every number below as
> *in-distribution on CIFAKE*, not comparable to the `ai-real-images` / `tiny-genimage` results.

## What was run

A single end-to-end notebook covering the full ladder of approaches on a balanced 10k subset (5k/class),
stratified 70/15/15:

- **Feature extraction + classical baselines** — HOG, LBP, colour histograms, and frozen-ResNet18
  features, each fed (after PCA→150) to k-NN / Naive Bayes / Logistic Regression / SVM / Decision Tree.
- **Neural nets from scratch** — an MLP on raw pixels and a small custom CNN.
- **Transfer learning** — ResNet18 and EfficientNet-B0, full fine-tuning vs. frozen backbone.
- **Few-shot & prototypes** — 10-/5-shot splits; nearest-prototype classifier (Euclidean/Cosine).
- **Self-supervised & PEFT** — a SimCLR encoder (linear-eval + fine-tune) and LoRA on ViT-B/16.
- **Two extra architectures** — a **Dual-Branch Artifact-Aware CNN** (RGB + frequency/residual branch with
  high-pass, SRM-inspired and FFT channels) and **foundation-model probes** (frozen CLIP ViT-B/32 and
  DINOv2 ViT-S/14 + a small head).

## Results (in-distribution, CIFAKE)

Full-data test accuracy, best configuration per method:

| Method | Test acc |
|---|:--:|
| ViT-B/16 — full fine-tune | **0.962** |
| EfficientNet-B0 — full | 0.958 |
| LoRA (ViT-B/16, r=8) | 0.957 |
| **CLIP-probe** (frozen + head) | 0.947 |
| **DINOv2-probe** (frozen + head) | 0.945 |
| ResNet18 — full | 0.942 |
| **Dual-Branch Artifact-Aware CNN** | 0.935 |
| Custom CNN (from scratch) | 0.933 |
| Best classical (SVM + ResNet18 features) | 0.906 |
| MLP on CNN features | 0.900 |
| Best handcrafted (HOG + SVM) | 0.765 |
| MLP on raw pixels | 0.755 |

Few-shot (linear probe / fine-tune; foundation probes averaged over 3 seeds):

| Method | 10-shot | 5-shot |
|---|:--:|:--:|
| **CLIP-probe** | **0.743** | **0.675** |
| DINOv2-probe | 0.718 | 0.650 |
| ResNet18 / full | 0.710 | 0.619 |
| SimCLR fine-tuned | 0.703 | 0.516 |
| EfficientNet / full | 0.668 | 0.513 |
| Prototype (ResNet18) | 0.610 | 0.583 |

## What it adds to the project

- **Representation dominates.** The clean monotone ladder — handcrafted ~0.76 → ImageNet features ~0.91 →
  foundation embeddings ~0.945–0.947 (no backbone training) → full fine-tune 0.962 — is the same lesson the
  main project teaches, isolated here without the confound of a generalization gap.
- **Foundation probes are the efficiency winners.** Frozen CLIP/DINOv2 + a tiny head match the best
  no-fine-tuning accuracy *and* take the top two few-shot spots, with zero backbone training. CLIP edges
  DINOv2 and is more stable in the low-data regime.
- **LoRA confirms low-rank adaptation** — 0.957 at **0.34%** trainable parameters, within 0.5 pt of full
  fine-tuning.
- **The artifact-aware two-branch idea ties, but does not beat, a plain CNN** at 32×32 (0.935 vs. 0.933,
  with ~half the parameters). The explicit frequency/SRM branch has too little to work with at this
  resolution — **direct empirical support for the project's decision to move to higher-resolution
  photographic data**, where frequency fingerprints are richer.
- **Data augmentation consistently hurt** the from-scratch CNN here, plausibly because flip/rotation/jitter
  distort the very generative artifacts that separate real from fake — a CIFAKE-specific caution.

## Pointers

- Notebook: [`notebooks/cifake-study.ipynb`](../notebooks/cifake-study.ipynb)
- Full write-up (architectures, per-question analysis, figures): [`presentation_cifake/CIFAKE_Report.pdf`](../presentation_cifake/CIFAKE_Report.pdf)

*In-distribution CIFAKE only — no cross-generator OOD evaluation. Numbers are from a single balanced 10k
subset run; few-shot figures are 3-seed means.*
