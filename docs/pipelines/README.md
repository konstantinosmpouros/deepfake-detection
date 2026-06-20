# Pipelines — deep dives

[← docs index](../README.md)

This project does not chase a single best model; it treats real-vs-fake detection as a question with
**many legitimate answers** and builds one of each so they can be compared on equal footing. A *pipeline*
here is a complete, self-contained way of turning an RGB image into a probability `p_fake ∈ [0,1]` — its
own architecture, its own input representation, its own training recipe, and its own saved weights. The
ten pipelines deliberately span very different hypotheses about *where* the generative fingerprint lives:
in raw pixels (the from-scratch CNNs), in a pretrained backbone's learned features (transfer learning), in
a foundation model's semantic embedding (CLIP), in the frequency domain (two-stream, freqcross), in
high-pass noise residuals (srm-noise), in native-resolution local texture (patch-ensemble), or in how well
a diffusion model can reconstruct the image (dire-recon). Comparing them is the point: the spread between
them is what exposes the real lessons of the project — the generalization gap, the cost of over-fitting to
one generator family, and the trade-off between in-distribution accuracy and out-of-distribution survival.

Each pipeline lives in exactly one notebook (`notebooks/04…13`) and shares the same
[I/O contract and training conventions](../03-shared-methods.md), so the comparison is apples-to-apples:
the same train/val/test split, the same metrics, the same evaluation protocols. The pages below all follow
the same template, and it is worth knowing how to read them. *Purpose* states the hypothesis the pipeline
is testing. *Architecture* gives the layer-by-layer design and the intuition behind the non-obvious
choices. *Input & preprocessing* records the working resolution and normalization family (these differ by
model and matter for correctness). *Training method* makes the loop visible — loss, optimizer, schedule,
and any tricks like EMA or two-stage fine-tuning. *Optuna search* documents the hyperparameter space and
the winning configuration for the tuned pipelines. *Results* reports in-distribution metrics at both the
default 0.5 threshold and a validation-tuned threshold, plus the per-generator OOD breakdown. *Explainability*
points to the Grad-CAM / attention figures. *Saved model* gives the reload recipe.

> **How to read the table.** *In-dist AUC* is the threshold-free in-distribution discrimination on the
> held-out `ai-real-images` test set (the headline accuracy/F1/MCC live on each page). *OOD acc* is the
> overall accuracy on `tiny-genimage`'s seven unseen generators — the generalization stress test, and the
> column where the surprises are. *Tuned (trials)* is the number of Optuna trials; a dash means the
> pipeline runs on fixed, hand-set hyperparameters (the two from-scratch CNNs). Read the AUC and OOD
> columns *together*: the gap between them is the story this project is really about.

| # | Pipeline | Family | Size | Norm | In-dist AUC | OOD acc | Tuned (trials) |
|---|----------|--------|:----:|:----:|:-----------:|:-------:|:--------------:|
| 04 | [cnn-scratch](cnn-scratch.md) | from-scratch CNN | 128 | dataset | 0.9648 | 0.5488 | — |
| 05 | [cnn-residual](cnn-residual.md) | from-scratch CNN (+SE, EMA) | 128 | dataset | 0.8672 | 0.5190 | — |
| 06 | [cnn-finetune](cnn-finetune.md) | transfer · EfficientNet-B0 | 224 | imagenet | 0.9930 | 0.5636 | 12 |
| 07 | [vit-lora](vit-lora.md) | transfer + PEFT · ViT-B + LoRA | 224 | imagenet | **0.9972** | 0.6022 | 12 |
| 08 | [clip-probe](clip-probe.md) | foundation · frozen CLIP + MLP | 224 | clip | 0.9930 | 0.5837 | 80 |
| 09 | [two-stream](two-stream.md) | hybrid · RGB + FFT | 128 | dataset | 0.9609 | 0.5264 | 20 |
| 10 | [freqcross](freqcross.md) | frequency · 3-branch | 128 | dataset | 0.9651 | 0.5526 | 20 |
| 11 | [srm-noise](srm-noise.md) | forensic · SRM + Bayar | 128 | dataset | 0.9518 | 0.5231 | 24 |
| 12 | [patch-ensemble](patch-ensemble.md) | hybrid · native-patch MIL | 224 | imagenet | 0.9963 | **0.6775** | 8 |
| 13 | [dire-recon](dire-recon.md) | reconstruction · DIRE | 224 | imagenet | 0.9399 \* | 0.5415 | 20 |

\* `dire-recon` in-distribution numbers are on a 2,000-image subsample.

Architecture builders live in [`utils/models.py`](../../notebooks/utils/models.py); training pieces in
[`utils/training.py`](../../notebooks/utils/training.py); the Optuna scaffold in
[`utils/tuning.py`](../../notebooks/utils/tuning.py). Aggregated comparison: [../05-results.md](../05-results.md).
