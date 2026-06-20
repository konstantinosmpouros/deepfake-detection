# Deepfake / AI-Generated Image Detection — Project Documentation

Deep, end-to-end documentation of a **real-vs-AI-generated image detection** system built for the
joint MSc in Artificial Intelligence (University of Piraeus + NCSR "Demokritos"), **Deep Learning**
course. The task is **binary image classification — real vs. AI-generated (fake)** on
higher-resolution photographic images.

Image generators have crossed the threshold where their output is, frame for frame, indistinguishable
from a photograph to the human eye. That capability is genuinely useful, but it also powers an arms
race: every advance in generation invites a counter-advance in detection, and every detector that
works today is, in effect, training material for the next generator. A detector is therefore never
"finished" — it is a moving target. This documentation set studies that target carefully: not just
*can we tell real from fake on the data we have*, but *what kind of detector keeps working when the
generator changes, when the image is compressed, when the artifact it relied on is no longer there*.

Detection is harder than it first appears precisely because the easy version is a trap. A model can
reach near-perfect accuracy on a fixed dataset by latching onto an incidental giveaway — an image size,
a JPEG quantisation table, a watermark, a faint colour cast — that correlates with the label in *this*
collection but says nothing about whether an image was generated. Such a model looks excellent on paper
and is worthless in the wild. Much of the engineering here is about *removing* those shortcuts so that
the numbers measure detection skill rather than dataset bookkeeping (the most important such fix is the
resolution-shortcut neutralisation in [02-data §2.3.6](02-data.md#236-the-resolution-shortcut--the-most-important-data-decision)).

> **Deep-learning only.** Every classifier is a neural network. Signal transforms (FFT/DCT) appear
> only as *inputs* to neural nets, never as features for a classical classifier. `scikit-learn` is
> used for **metrics and data splitting only**.

This constraint follows from the course, not from convenience: this is the *Deep Learning* assignment,
so the contribution must be in the networks themselves. It is also a useful discipline — frequency and
noise-residual cues, which a classical pipeline would hand to a gradient-booster, are instead fed into
learned branches and fusion heads, so the model decides *how* to weigh them rather than us hard-coding
it.

We implement **ten distinct deep-learning pipelines**, tune each with **Optuna**, and evaluate them
under three protocols (in-distribution, cross-generator, robustness) plus explainability. A local
**FastAPI + Streamlit app** serves any pipeline for interactive prediction. The ten are not redundant:
they span four families — from-scratch CNNs, transfer-learned backbones, frozen-foundation-model
probes, and frequency/forensic/reconstruction hybrids — so that the comparison answers *which family of
ideas generalizes*, not merely which single model scored highest.

---

## The scientific questions

In-distribution accuracy is largely "solvable" — the valuable contribution is studying what separates
a real detector from one that memorises a single generator. Each question below is chosen because it
probes a different failure mode of an over-fitted detector:

1. **Generalization gap** — does a detector trained on one set of generators survive *unseen*
   generators? (train on `ai-real-images`, test per-generator on `tiny-genimage`.) This is the
   headline question. A small gap means the model learned something about *generated imagery in
   general*; a large gap means it learned the fingerprint of three specific generators and nothing
   more. Because tomorrow's images come from tomorrow's generators, the gap is the closest proxy we
   have for real-world usefulness.
2. **Robustness** — how fast do detectors degrade under JPEG compression, blur, downsampling, noise?
   Real uploads are rarely pristine: they have been screenshotted, re-compressed by a messaging app,
   resized for the web. If the discriminative signal is a delicate high-frequency artifact, mild
   post-processing can erase it — so a detector's robustness curve is really a measure of how brittle
   the cue it depends on is.
3. **Frequency-domain artifacts** — generated images deviate in the high-frequency band; this motivates
   frequency-aware and two-stream models. Upsampling and transposed-convolution layers leave periodic
   spectral traces that real camera optics do not produce. These traces are often invisible yet
   discriminative, and — being a property of the *generation process* rather than the image content —
   they can transfer across generators better than semantics do.
4. **Explainability** — Grad-CAM, ViT attention rollout, embedding t-SNE, frequency spectra: *where*
   and *why* a model decides "fake". Beyond trust, explainability is a debugging tool for question 1:
   if a model's attention sits on a watermark or a border, we have caught a shortcut before it
   embarrasses us out-of-distribution.

---

## Headline results

In-distribution = `ai-real-images` test (11,963 images); OOD = `tiny-genimage` (7 unseen generators).
Sorted by in-distribution AUC. Full numbers and figures in [05-results.md](05-results.md).

| Pipeline | Family | Size | In-dist AUC | In-dist Acc | OOD Acc | Gen. gap |
|----------|--------|:----:|:-----------:|:-----------:|:-------:|:--------:|
| [vit-lora](pipelines/vit-lora.md) | Transfer · ViT-B + LoRA | 224 | **0.9972** | 0.9782 | 0.6022 | 0.376 |
| [patch-ensemble](pipelines/patch-ensemble.md) | Hybrid · native-patch MIL | 224 | 0.9963 | 0.9681 | **0.6775** | **0.291** |
| [cnn-finetune](pipelines/cnn-finetune.md) | Transfer · EfficientNet-B0 | 224 | 0.9930 | 0.9559 | 0.5636 | 0.392 |
| [clip-probe](pipelines/clip-probe.md) | Foundation · frozen CLIP | 224 | 0.9930 | 0.9592 | 0.5837 | 0.376 |
| [freqcross](pipelines/freqcross.md) | Frequency · 3-branch | 128 | 0.9651 | 0.9003 | 0.5526 | 0.348 |
| [cnn-scratch](pipelines/cnn-scratch.md) | From-scratch CNN | 128 | 0.9648 | 0.9011 | 0.5488 | 0.352 |
| [two-stream](pipelines/two-stream.md) | Hybrid · RGB + FFT | 128 | 0.9609 | 0.8977 | 0.5264 | 0.371 |
| [srm-noise](pipelines/srm-noise.md) | Forensic · SRM + Bayar | 128 | 0.9518 | 0.8824 | 0.5231 | 0.359 |
| [dire-recon](pipelines/dire-recon.md) | Reconstruction · DIRE | 224 | 0.9399 \* | 0.8730 \* | 0.5415 | 0.332 |
| [cnn-residual](pipelines/cnn-residual.md) | From-scratch CNN | 128 | 0.8672 | 0.7868 | 0.5190 | 0.268 |

\* `dire-recon` was evaluated on a **2,000-image subsample** (the diffusion reconstruction is
compute-heavy) — its in-distribution numbers are *not* directly comparable to the others.

Read the table in two directions. Read *down* the AUC column and the story is reassuring and a little
boring: almost every pipeline clears 0.95 AUC in-distribution, and the transfer-learning and
foundation-model entries flirt with perfection. Read *across* to the OOD column and the story inverts —
the same models that scored 0.99 in-distribution collapse toward the 0.5–0.6 band on unseen generators.
The gap between those two columns is the whole point of the project, and the discussion turns the
two-direction reading into concrete claims:

**Two findings worth highlighting** (see the [Discussion](05-results.md#discussion)):

- **The "CLIP generalizes best" hypothesis was *not* confirmed.** Going in, the natural expectation was
  that a frozen foundation model — having seen a vast, diverse slice of the visual world — would flag
  *anything off the real-image manifold* and therefore transfer best. It did not. The best
  cross-generator generalizer is **patch-ensemble** (OOD 0.677, smallest gap 0.291); `clip-probe` is
  only **3rd** on OOD. The lesson is that *where* you look (native-resolution local patches) can matter
  more than *how rich* your features are — a result we did not assume and the data forced on us.
- **Every pipeline still beats a random/always-fake baseline on OOD** — so all of them learned
  *something* generator-agnostic — **but most land in the 0.52–0.60 band**, i.e. far below their
  in-distribution accuracy. The generalization gap is real and large, and no single architecture in our
  set closes it. That negative-but-honest result is more informative than another 0.99 in-distribution
  number would have been.

---

## How to read these docs

The documentation is layered to be read either front-to-back as a narrative or dipped into by topic.
The numbered docs (`01`–`07`) form the spine — problem, data, methods, evaluation, results, app,
reproducibility — and read in order; the [pipelines/](pipelines/README.md) folder hangs off the spine
as one deep-dive page per model, each following the same internal structure so they are easy to compare
side by side. If you only read two things, read [01-overview.md](01-overview.md) for the framing and the
[Discussion in 05-results.md](05-results.md#discussion) for what we actually learned.

| Doc | Contents |
|-----|----------|
| [01-overview.md](01-overview.md) | Problem statement, datasets, research questions, end-to-end flow, repo map |
| [02-data.md](02-data.md) | Datasets, collection, **EDA findings** (+figures), cleaning, split, preprocessing, loaders |
| [03-shared-methods.md](03-shared-methods.md) | Training conventions, the **Optuna framework**, metric definitions, model-sharing scheme |
| [04-evaluation.md](04-evaluation.md) | The **three evaluation protocols** + explainability, `eval_protocols.py`, the random baseline |
| [05-results.md](05-results.md) | Master comparison tables, generalization gap, robustness, per-component, **Discussion** |
| [06-app.md](06-app.md) | FastAPI + Streamlit app: residency state machine, prediction schema, run commands |
| [07-reproducibility.md](07-reproducibility.md) | Environment, requirements, running notebooks/babysitter, tests, artifact layout |
| [pipelines/](pipelines/README.md) | **Deep dive per pipeline** (architecture, training, tuning, results, explainability) |

All numbers in these docs are taken from each pipeline's `notebooks/artifacts/<pipeline>/metrics/metrics.json`
and the aggregated `notebooks/artifacts/evaluation/*.csv`. Figures are linked from the same artifact tree.
Nothing here is hand-typed from memory — if a figure or number is cited, it is a pointer into that
artifact tree, so the docs and the code cannot silently drift apart.

*Last updated 2026-06-20.*
