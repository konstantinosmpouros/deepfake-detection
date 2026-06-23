# CIFAKE — Supplementary study (report)

This folder holds the deliverable for the supplementary, in-distribution **CIFAKE** study that revisits the
32×32 dataset dropped from the main project. For the narrative summary and where it fits, see
[`../docs/supplementary-cifake.md`](../docs/supplementary-cifake.md).

## Contents

- **`CIFAKE_Report.pdf`** — full write-up: dataset description, all assignment questions (Q0.1–Q5.5), nine
  architectures (handcrafted/classical baselines, MLP & CNN from scratch, transfer learning, few-shot,
  prototype classifier, SimCLR, LoRA, a dual-branch artifact-aware CNN, and frozen CLIP/DINOv2 probes),
  embedded figures, and the results analysis.

## Scope

In-distribution **CIFAKE only** (32×32; real CIFAR-10 vs. Stable-Diffusion-v1.4). These numbers are **not
comparable** to the main project's 224px `ai-real-images` / `tiny-genimage` results and there is no
cross-generator OOD evaluation here — see the [supplementary write-up](../docs/supplementary-cifake.md) and
the main [docs index](../docs/README.md) for the full project.
