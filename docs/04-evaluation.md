# 4 — Evaluation protocols

[← docs index](README.md) · [← 03 Shared methods](03-shared-methods.md)

A deepfake detector that scores 99% on its own test set has proven almost nothing about whether it works.
The whole scientific premise of this project ([01-overview](01-overview.md)) is that in-distribution
accuracy is the *easy* part and that the real questions are: does the detector survive a generator it has
never seen, and does it survive an image that has been compressed or blurred on its way to the model? A
single test number cannot answer those, so evaluation here is deliberately **three complementary
protocols plus explainability**, each isolating a different way a detector can be secretly broken:

1. **In-distribution** — is it any good at all, under ideal matched conditions?
2. **Cross-generator** — does that skill *transfer*, or was it memorising one generator's fingerprint?
3. **Robustness** — does that skill *survive* the everyday image corruptions of a real deployment?

All three are applied to **every** trained pipeline so the comparison is fair. Crucially, the aggregated
evaluation notebooks **do not retrain anything** — they reconstruct each model from its committed
artifacts via [`utils/eval_protocols.py`](../notebooks/utils/eval_protocols.py), then write
tables/figures to [`artifacts/evaluation/`](../notebooks/artifacts/evaluation/). Keeping evaluation
strictly read-only with respect to the trained models is what makes the numbers reproducible and the
comparison apples-to-apples: the same code path scores all ten pipelines. Numerical results:
[05-results.md](05-results.md).

---

## 4.1 The reconstruction layer — `eval_protocols.py`

Before any protocol can run, the evaluation notebooks need to load ten differently-built pipelines —
CNNs, a LoRA ViT, a CLIP probe, a patch-MIL model, a diffusion-reconstruction model — *by name*, and run
them through one shared evaluation harness. A single module provides that uniform façade so the protocol
code never has to special-case a pipeline. Key pieces:

- **`SPEC[name]`** — per-pipeline spec: input `family`, working `size`, `norm`, and checkpoint info. The
  `family` field is what tells the harness how to feed each model, because the four families consume
  fundamentally different inputs:
  - `image` — pixel input (the CNNs, ViT, two-stream, freqcross, srm-noise) at 128 or 224 px.
  - `clip` — frozen-CLIP 512-D embeddings (`clip-probe`).
  - `patch` — native-resolution patch bags for MIL (`patch-ensemble`).
  - `dire` — DDIM reconstruction-error maps (`dire-recon`).
- **`available()`** — pipelines with both a `metrics.json` and a checkpoint on disk (all 10). This guard
  means an evaluation run gracefully covers exactly the pipelines that are actually trained, rather than
  erroring on a half-finished one.
- **`load_model(name, device)`** — rebuilds the architecture **from `best_params.json`** (so the tuned
  feat/hidden/r/K/backbone/… are honoured) and loads the committed weights. This is the single most
  important line in the module: a tuned pipeline's *architecture* is itself a search result (its hidden
  width, LoRA rank `r`, number of patches `K`, chosen backbone), so rebuilding from a hard-coded default
  would silently evaluate a *different, untuned* model than the one that was trained. Reading
  `best_params.json` guarantees the architecture we score is the architecture the search actually selected
  — the same recipe the model-sharing scheme relies on ([03 §3.6](03-shared-methods.md#36-model-sharing-scheme-committing-trained-parts-to-github)).
- **`indist_probs(name, model, device)`** — probabilities on the `ai-real-images` test split.
- **`ood_frame(name, model, device)`** → DataFrame `[generator, y_true, p_fake]` over `tiny-genimage`.
  Returning a per-image frame *tagged by generator* is what lets Protocol 2 slice accuracy per generator
  without re-running inference.
- **`robustness_loader / robustness_point`** — a perturbed in-distribution loader and the accuracy/AUC at
  one (perturbation, level).
- **`best_params(name) / metrics(name) / optuna_trials(name) / split_labels(split)`** — read helpers, so
  the notebooks pull committed numbers from disk rather than recomputing them.

## 4.2 Protocol 1 — In-distribution ([`eval-comparison.ipynb`](../notebooks/eval-comparison.ipynb))

This is the matched-conditions baseline: train and test images come from the *same* dataset and the *same*
generators, so it answers "under ideal conditions, how good is this detector?" Evaluate on the held-out
`ai-real-images` **test** split (11,963 images; `dire-recon` on a 2,000 subsample because its diffusion
reconstruction is far too slow to run on the full split). It reports the full metric set (accuracy,
macro-F1, AUC-ROC, PR-AUC, precision, recall, MCC, Brier) at both the 0.5 and the val-tuned threshold,
and produces the master comparison table plus ranked bars, a ROC/PR overlay, a reliability (calibration)
overlay, and a confusion-matrix grid. Outputs: `in_distribution_comparison.csv` and figures
`indist_bars`, `roc_pr_overlay`, `reliability_overlay`, `confusion_grid`.

> **What it does and does not tell you.** A strong in-distribution score establishes that the architecture
> *can* learn the real-vs-fake boundary for these generators — a necessary sanity check, and the only
> protocol where near-perfect scores are expected. But it says nothing about *why*: a model that has
> learned a genuine, transferable artifact and a model that has memorised this dataset's particular
> fingerprints can post identical in-distribution numbers. The reliability overlay is included here for
> exactly this reason — two models with the same accuracy can have very different calibration, and that
> difference often previews which one will hold up under the next two protocols. Protocol 1 is the floor
> the comparison stands on, not the conclusion.

## 4.3 Protocol 2 — Cross-generator generalization ([`eval-generalization.ipynb`](../notebooks/eval-generalization.ipynb))

This is the **centrepiece of the whole evaluation**, because it operationalises the project's central
research question. Train on `ai-real-images`, test on `tiny-genimage`'s **7 unseen generators** (biggan,
vqdm, sdv5, wukong, adm, glide, midjourney) — the **generalization gap**. Reports per-generator accuracy,
overall OOD accuracy, and the in-dist→OOD gap. Outputs `generalization_gap.csv`, the per-generator
heatmap, and the gap plot.

Why this matters more than Protocol 1: the generators in `tiny-genimage` were *never seen during training
or tuning* ([02-data §2.1](02-data.md#21-the-two-datasets-and-why-this-pairing)), so the gap between a
pipeline's in-distribution score and its OOD score is a direct measurement of **how much of its skill was
transferable detection versus how much was generator-specific memorisation**. A pipeline that scores 99%
in-distribution and collapses toward chance OOD has learned a fingerprint, not a concept; one that holds a
meaningful margin OOD has learned something about *generated imagery* in general. The per-generator
breakdown matters because the seven generators sit at different "distances" from the training set
(GAN-era BigGAN, older diffusion ADM/GLIDE, newer diffusion SDv5/Wukong), so the heatmap shows not just
*whether* a model generalises but *to what* — which is the substance of the generalization story in
[05-results §5.2](05-results.md#52-cross-generator-generalization-ood).

**Random / dummy baseline.** Because OOD accuracy clusters near chance, the notebook adds explicit dummy
classifiers — **random** (`p ~ U(0,1)`), **always-fake**, **always-real** — computed from the OOD labels,
so each pipeline's OOD score is reported as a **lift over random** (`ood_vs_random.csv`). This step is not
decoration — when scores hover near 50%, raw accuracy is genuinely ambiguous: a 55% OOD accuracy could be
real (weak but present) signal, or it could be an artifact of class balance and a model that effectively
guesses. Pinning down the dummies makes the question *"does it actually generalize, or is it guessing?"*
explicit and answerable, rather than leaving the reader to eyeball a number near chance. It then plots
the comparison **from four angles** (`ood_random_angles.png`): ranked overall bar vs the dummies,
lift-over-random bar, per-generator lines vs the random floor, and an in-dist-vs-OOD scatter. The four
views triangulate the same conclusion — overall standing, margin above chance, per-generator detail, and
the gap relative to in-distribution — so that a near-chance result cannot be accidentally read as success.

## 4.4 Protocol 3 — Robustness ([`eval-robustness.ipynb`](../notebooks/eval-robustness.ipynb))

A detector that only works on pristine images is useless in practice, because images on the open web have
been JPEG-recompressed, resized, blurred, and noised on their way to the model. Protocol 3 measures how
gracefully each detector degrades under exactly those corruptions: apply perturbations at increasing
strength to a test subsample and plot **accuracy vs. perturbation strength**. The shape of that curve —
a gentle slope versus a cliff — is the result, not any single point. Four perturbations × five levels
(`eval_protocols.PERTURBATIONS`):

| Perturbation | Levels |
|--------------|--------|
| `jpeg_quality` | 100, 90, 80, 70, 60 |
| `gaussian_blur_sigma` | 0.0, 0.5, 1.0, 1.5, 2.0 |
| `downsample_scale` | 1.0, 0.75, 0.5, 0.35, 0.25 |
| `gaussian_noise_std` | 0.0, 0.02, 0.05, 0.1, 0.15 |

Each perturbation targets a different vulnerability, and they are revealing precisely because of what
[02-data](02-data.md#246-the-resolution-shortcut--the-most-important-data-decision) showed about *where*
the discriminative signal lives: JPEG compression and downsampling attack the **high-frequency band**
where many generative fingerprints sit, blur smooths away fine texture statistics, and additive noise
swamps subtle traces — so these curves directly probe whether a detector leans on fragile high-frequency
cues. Each starts at a no-op level (`jpeg_quality=100`, `sigma=0.0`, `scale=1.0`, `noise=0.0`) so the
curve begins at the model's clean accuracy and the *drop* is read off directly.

**Scope:** the sweep covers the **image-family** pipelines only. `clip-probe`, `patch-ensemble`, and
`dire-recon` use specialised inputs (cached embeddings / patch bags / diffusion-reconstruction maps) and
are out of scope for the pixel-perturbation sweep. Perturbing a raw pixel tensor is only well-defined for
models that consume raw pixels; the three specialised-input pipelines would each need their own
perturbation semantics (do you noise the image before or after computing the CLIP embedding? before or
after extracting patches?), which is a separate study rather than part of this shared sweep. Outputs
`robustness_summary.csv` and `robustness_curves.png`.

> This is the only eval that runs **fresh GPU inference** (the others read each pipeline's cached
> `metrics.json` numbers), so it is by far the slowest. The reason is structural: Protocols 1 and 2 score
> a fixed set of images once, but robustness must re-score the test subsample *once per (perturbation,
> level)* combination — twenty fresh inference passes per pipeline — and there are no cached numbers to
> fall back on. It is also the most data-pipeline-bound stage rather than GPU-bound: every level
> **re-applies the perturbation (e.g. JPEG re-encode) per image on the CPU**, and that encode/decode is
> often slower than the forward pass it feeds, so the GPU can sit idle waiting for the CPU to prepare the
> next batch. The speed knobs in §0 of the notebook all address this CPU bottleneck: `NUM_WORKERS`
> (parallel CPU perturbation prep — the main lever, since it lets several workers re-encode images while
> the GPU runs), `SUBSAMPLE` (images per level — trade statistical precision for wall-clock time), and a
> **tqdm** progress bar (live count + ETA, so a long sweep is observable rather than an opaque hang). See
> [05-results §Robustness](05-results.md#53-robustness) for results.

## 4.5 Explainability ([`eval-explainability.ipynb`](../notebooks/eval-explainability.ipynb))

Metrics say *how well* a model does; explainability asks *why*, which is both a grading requirement and a
genuine check against the "right answer for the wrong reason" failure that the whole evaluation is built
to catch. On a shared set of example images:

- **Grad-CAM gallery** for the CNN pipelines (`gradcam_gallery.png`) — one row per pipeline, columns =
  example images, overlay = where the model looked. This reveals whether a CNN's decision is driven by
  meaningful image regions or by a corner/border artifact, which would betray a shortcut.
- **ViT attention rollout** for `vit-lora` (`vit_attention_rollout.png`) — the transformer analogue,
  tracing which patches the attention heads actually attend to.
- **CLIP embedding t-SNE** for `clip-probe` (`clip_tsne.png`) — real-vs-fake separability in the frozen
  embedding space (the generalization story). Clean separation here visualises *why* a frozen foundation
  model generalises: the real/fake boundary already exists in CLIP's semantic space before any training.
- Per-pipeline specialised maps: `patch-ensemble` per-patch MIL attention (which patches drove the
  bag-level decision), `srm-noise` noise residuals (the high-pass signal it keys on), `freqcross`
  branch-fusion attention (how it weighs RGB vs frequency vs radial evidence), `dire-recon` DIRE error
  maps (where reconstruction error localises). Each makes the pipeline's specific hypothesis about *what
  betrays a fake* visually inspectable.

## 4.6 The Optuna analysis ([`eval-optuna.ipynb`](../notebooks/eval-optuna.ipynb))

A cross-pipeline view of the hyperparameter search, complementing the per-pipeline search artifacts from
[03 §3.3](03-shared-methods.md#33-hyperparameter-search--the-optuna-framework-utilstuningpy): best config
per pipeline (`optuna_best_params.csv`), search efficiency (completed vs pruned trials — i.e. how much
compute the MedianPruner saved), optimisation histories, and which choices mattered (e.g. focal vs BCE).
It reads each pipeline's committed `optuna_trials.json` and needs **no GPU**, because the entire search
record was persisted precisely so this analysis can reconstruct it after the fact.

## 4.7 Running the evaluations

All five notebooks are read-only w.r.t. the trained models — they reconstruct and score, never train —
which is what makes it safe to re-run them at any time. Run interactively (Restart Kernel → Run All) or
headless via the babysitter (see [07-reproducibility.md](07-reproducibility.md)):

```powershell
powershell -File notebooks/_babysit_runs.ps1 -Only eval -Hours 6        # all five
powershell -File notebooks/_babysit_runs.ps1 -Only eval-robustness      # just one
```

The `-Only eval-robustness` form is the one you reach for most, since robustness is the slow,
fresh-inference protocol (§4.4) and is often re-run on its own after the cheaper, cache-reading protocols
are already done.

Next: [05-results.md →](05-results.md)
