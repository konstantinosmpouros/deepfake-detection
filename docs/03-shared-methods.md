# 3 — Shared methods: training, tuning, metrics, model-sharing

[← docs index](README.md) · [← 02 Data](02-data.md)

Twelve pipelines (notebooks 04–13) attack the same real-vs-fake problem in twelve different ways, but
they are *not* twelve independent codebases. They share one notebook skeleton, one training library, one
tuning framework, one metric definition, and one model-sharing convention. This is deliberate: when every
pipeline is trained, tuned, and scored the same way, the head-to-head comparison in
[05-results](05-results.md) measures the *architecture*, not an accidental difference in how someone wrote
their training loop. This chapter documents those shared conventions and — in keeping with the rest of
these docs — explains *why* each one exists, because most of them encode a small lesson about training
deepfake detectors that is easy to get wrong.

The one structural rule that governs everything below: the heavy, reused code lives in
[`utils/`](../notebooks/utils/), but the **training loop stays visible in each notebook**. This is partly
a grading constraint (the assignment rewards demonstrated understanding, and a loop hidden behind a single
`run_everything()` call demonstrates nothing) and partly an engineering one — when a pipeline misbehaves,
the forward/loss/backward/step is right there to read, not buried three modules deep. So the *reusable
pieces* (one epoch, one evaluation pass, checkpoint I/O) are factored into `utils/`, while the *assembly*
of those pieces into a loop is spelled out in §4 of every notebook.

---

## 3.1 The common notebook skeleton

```
0 Setup → 1 Data → 2 Model → 3 Training setup → 4 Train (visible loop)
        → 5 Curves → 6 In-dist eval → 7 OOD preview → 8 Explainability → 9 metrics.json
```

Every pipeline notebook walks the same ten stations in the same order, and that consistency is a feature,
not boilerplate. A reader who has understood one notebook can open any other and know exactly where to
find the model definition (§2), the loop (§4), or the metric dump (§9). Concretely, each notebook clears
the GPU at the start (so a notebook re-run never inherits a half-loaded model from the previous one),
builds its loaders from the shared cache, builds **and tunes** its model, trains the winning configuration
with a visible loop, then evaluates in-distribution, previews OOD behaviour, produces an explainability
figure, and finally writes a uniform `metrics.json` (§3.5) that the aggregated evaluation notebooks
([04-evaluation](04-evaluation.md)) read back without caring which pipeline produced it.

> **Why an OOD *preview* inside every pipeline notebook (§7).** The authoritative cross-generator numbers
> are produced once, centrally, in [`eval-generalization`](04-evaluation.md#43-protocol-2--cross-generator-generalization-eval-generalizationipynb).
> But each pipeline still runs a quick OOD preview on its own so that a generalization collapse is caught
> *while developing that pipeline*, not three notebooks later. It is a smoke test, flagged `"preview":
> true` in the JSON so it is never mistaken for the canonical result.

## 3.2 Training conventions ([`utils/training.py`](../notebooks/utils/training.py))

The training defaults below are not generic "good practice" copied from a tutorial — several of them are
tuned to the specific failure modes of this task (subtle, easily-destroyed generative artifacts on
balanced data). Each bullet pairs the *what* with the *why*.

- **Loss**: `BCEWithLogitsLoss` or **focal loss** (`tuning.FocalLoss`, chosen by Optuna), on a single
  logit. Optional **label smoothing** (`smooth_binary_targets`, ε≈0.05) composes with either. We operate
  on one logit (not two) because the task is binary and a single sigmoid output is the natural,
  calibration-friendly parameterisation. Focal loss is offered as a tunable alternative because it
  down-weights easy examples and concentrates gradient on the hard, ambiguous ones — which can help when a
  generator is *almost* indistinguishable from real. Label smoothing nudges the targets off the hard 0/1
  corners, which discourages the network from driving logits to ±∞ (over-confidence) and tends to improve
  the calibration that the Brier score in §3.4 measures.
- **Optimizer / schedule**: AdamW + **cosine schedule with warmup** (`build_cosine_with_warmup`,
  per-batch stepping). AdamW's decoupled weight decay is the standard, well-behaved choice for both CNNs
  and transformers. The cosine schedule starts high and decays smoothly to near-zero, which lets training
  make fast early progress and then settle into a flat minimum; the **warmup** prefix ramps the LR up over
  the first steps so that the very first, high-variance gradients (especially on a freshly-initialised
  head sitting on a pretrained backbone) do not blow the weights out before training has stabilised.
  Stepping per-batch rather than per-epoch makes the schedule smooth regardless of dataset size.
- **EMA** (`EMA`, decay 0.999) on the deeper nets (`cnn-residual`, `cnn-finetune`) — evaluate and save
  the EMA weights. An exponential moving average maintains a shadow copy of the weights that lags the
  noisy SGD trajectory. The training weights bounce around the loss surface from minibatch to minibatch;
  the EMA average sits closer to the centre of that cloud, which is typically a slightly flatter, better
  generalising point. We therefore **evaluate and ship the EMA weights**, not the last-step weights, on
  the networks deep enough to benefit. (The shallow baselines do not, so they skip it.)
- **Mixed precision**: bf16 `autocast` (no GradScaler needed), `channels_last` memory format, `cudnn.benchmark`,
  TF32 enabled, `set_float32_matmul_precision("high")`. These are pure throughput levers — they make
  training faster on the single mid-range GPU without changing the maths in any way that matters.
  **bf16** has the same exponent range as fp32, so unlike fp16 it does not need a `GradScaler` to avoid
  underflow — the loop is simpler and just as stable. **`channels_last`** lays out NHWC tensors the way
  cuDNN's convolution kernels prefer, avoiding silent layout conversions. `cudnn.benchmark` lets cuDNN
  autotune the fastest convolution algorithm for our fixed input sizes, and **TF32** /
  `matmul_precision("high")` allow Ampere-class tensor cores to run matmuls faster at a precision that is
  more than sufficient for training. The payoff is more epochs per hour, which directly enables the
  hyperparameter search in §3.3.
- **Stability**: gradient clipping (norm 1.0), `zero_grad(set_to_none=True)`. Clipping the global gradient
  norm to 1.0 caps the occasional exploding-gradient step that would otherwise undo good progress — cheap
  insurance for the deeper and transformer-based models. `set_to_none=True` is a minor speed/memory
  optimisation (it frees the gradient buffers rather than filling them with zeros).
- **Early stopping** (`EarlyStopper`, mode=max on **val AUC**, patience 5–7); best checkpoint saved by
  val AUC. We stop and checkpoint on **validation AUC, not validation loss** — and the distinction is
  deliberate. Loss is sensitive to confidence/calibration and to the exact threshold-free scaling of the
  logits; AUC measures the thing we actually care about, *ranking* fakes above reals, and is robust to the
  class-balance and confidence quirks that make loss a noisier early-stopping signal. Patience of 5–7
  epochs gives the cosine schedule room to escape a temporary plateau before we declare the run finished.

Reusable pieces: `train_one_epoch(...)` (forward/loss/backward/step, EMA update), `evaluate(...)` →
`(y_true, y_prob, loss)` with `y_prob = sigmoid(logits)`, and checkpoint helpers
(`save_checkpoint`/`load_checkpoint`, `EMA.state_dict`, plus the slim `save_weights`/`load_weights`/
`trained_state_dict` used by the parameter-efficient pipelines). These are the *single-purpose* helpers
the §3 rule allows; the loop itself — the part that demonstrates understanding — is assembled visibly in
§4 of each notebook, so that the narrative of forward → loss → backward → step → validate → checkpoint is
something a grader can read top to bottom.

## 3.3 Hyperparameter search — the Optuna framework ([`utils/tuning.py`](../notebooks/utils/tuning.py))

A model architecture is only half the story; its learning rate, weight decay, dropout, loss choice, and
architectural width can swing val AUC by several points. Hand-tuning twelve pipelines fairly is both
tedious and easy to bias (whoever tunes the hardest gets the best number). So notebooks **06–13** are
**Optuna-driven**: each defines a *visible* `objective` containing its own `trial.suggest_*` search space
(so the reader sees exactly what is being searched), and the search machinery lives in `tuning.py`:

- **`make_study(name, storage_dir, seed=42)`** — `direction="maximize"` (val AUC), **TPE sampler** +
  **MedianPruner** (`n_startup_trials=5`, `n_warmup_steps=2`), **SQLite storage** (`load_if_exists=True`
  → resumable). **TPE** (Tree-structured Parzen Estimator) is a Bayesian sampler: rather than guessing
  blindly (random search) or exhaustively (grid search), it builds a probabilistic model of which regions
  of the search space have produced good trials and preferentially samples there, so it converges on
  strong configurations in far fewer trials. The **MedianPruner** is the other half of the efficiency
  story — after `n_warmup_steps=2` epochs it compares a trial's running val AUC against the median of
  prior trials at the same epoch and *kills* trials that are tracking below it, so compute is not wasted
  finishing a configuration that is already visibly losing. `n_startup_trials=5` lets the first five trials
  run unpruned to seed that median. SQLite storage with `load_if_exists=True` means a search that is
  interrupted (or run overnight by the babysitter) **resumes** instead of restarting.
- **`quick_train_eval(model, train, val, device, *, lr, weight_decay, epochs, trial, loss_fn, …)`** —
  trains a single-logit model for a few epochs, reporting intermediate val AUC each epoch via
  `report_or_prune` so weak trials are pruned. The key word is *few* epochs: a trial is a cheap, truncated
  proxy for the full run, just long enough to tell a promising configuration from a poor one. Only
  `requires_grad` params are optimized, which is what makes this work unchanged for the LoRA / frozen-head
  pipelines (it never tries to optimise frozen weights).
- **`FocalLoss(gamma)` / `make_loss(name, gamma)`** — `"bce"` → `BCEWithLogitsLoss`, `"focal"` →
  `FocalLoss`. This is what lets the loss function itself become a searchable hyperparameter: a trial can
  propose `"focal"` with some `gamma`, and TPE learns whether the extra hard-example focus helps that
  pipeline.
- **`cleanup(*objs)`** — drop references + empty the CUDA cache between trials (avoids cross-trial OOM).
  Without this, the GPU memory from a finished trial's model can linger and accumulate until a later trial
  runs out of memory — a subtle bug that would make long searches fail unpredictably.

**The search → final-train pattern.** A study finds the best hyperparameters (reduced epochs + pruning),
then the notebook trains the **final** model on the winner for the full epoch budget. The split is
intentional: the search is allowed to be fast and approximate (truncated, pruned), because its only job is
to *rank* configurations; the single final run that we actually ship is given the full epoch budget so the
chosen configuration is trained to convergence. `TUNE=False` reuses the committed `best_params.json`
instead of re-searching, so re-running a notebook to refresh artifacts does not require repeating the
(expensive) search. `patch-ensemble` additionally exposes `EVAL_ONLY=True` (the committed default) to skip
search *and* training and just load `best.pt` for evaluation (see
[pipelines/patch-ensemble.md](pipelines/patch-ensemble.md)) — because its trials are the most expensive in
the project, re-training it on a whim is something to opt into, not the default.

### Saving strategy (per pipeline, via `save_study_artifacts`)

The search is not a black box whose output is a single magic config. We persist the entire search so it is
auditable and so the cross-pipeline Optuna analysis ([04-evaluation §4.6](04-evaluation.md#46-the-optuna-analysis-eval-optunaipynb))
can reconstruct what happened without re-running anything:

| File | Contents | Committed? |
|------|----------|:----------:|
| `metrics/best_params.json` | the winning hyperparameters | ✅ |
| `metrics/optuna_trials.json` | the **search space + every trial** (params, value, state, intermediate values, duration) + `best_value`, `n_trials`, `n_complete`, `n_pruned` | ✅ |
| `figures/optuna_{history,importances,parallel}.png` | optimisation history, param importances, parallel-coordinate | ✅ |
| `tuning/<pipeline>.db` | the live SQLite study (resumable) | ❌ gitignored |

We commit `best_params.json` because it is the recipe the final model (and the eval-time reconstruction in
[04-evaluation §4.1](04-evaluation.md#41-the-reconstruction-layer--eval_protocolspy)) is built from, and
`optuna_trials.json` because it records *every* trial — including the pruned and failed ones — which is
what lets us later say *why* a hyperparameter mattered rather than just *which* value won. The live `.db`
is gitignored because it is large, machine-local, and fully regenerable from the search; the committed
JSON + figures are the durable, shareable record.

The number of trials scales with cost-per-trial: **80** for `clip-probe` (cached embeddings → seconds
per trial) down to **8** for `patch-ensemble` (native-patch backbone passes → minutes per trial). This is
a budget allocation, not an inconsistency: a pipeline whose trials cost seconds can afford to explore its
space exhaustively, so it gets 80 trials; a pipeline whose every trial pushes many native-resolution
patches through a backbone can only afford a handful, so it gets 8 — and TPE's sample-efficiency is
exactly what makes 8 trials still informative there. Full per-pipeline search spaces and winners:
[05-results §Optuna](05-results.md#54-hyperparameter-search-optuna) and each pipeline doc.

## 3.4 Metric definitions ([`utils/metrics.py`](../notebooks/utils/metrics.py))

`classification_metrics(y_true, y_prob, threshold=0.5)` returns a flat, JSON-serialisable dict with
**accuracy, macro-F1, precision, recall, AUC-ROC, PR-AUC, MCC, Brier score**, and the confusion matrix
`[[TN, FP], [FN, TP]]`. We report this whole battery rather than accuracy alone because each number
catches a failure that accuracy hides, and in this project the *interesting* failures — over-confidence,
asymmetric errors, and OOD collapse — are exactly the ones a single accuracy figure would paper over:

- **AUC-ROC** is threshold-free: it asks whether the model *ranks* a random fake above a random real,
  independent of where we put the 0.5 cut. This is why it is also our early-stopping and tuning objective
  (§3.2/§3.3) — it measures separability, not a particular operating point.
- **PR-AUC** focuses on the positive (fake) class and is more informative than ROC-AUC when one cares
  about catching fakes specifically and the working operating regime has skewed precision/recall.
- **MCC** (Matthews correlation coefficient) is a balanced single-number summary of the whole confusion
  matrix that stays honest even when predictions are lopsided — a model that always predicts "fake" gets
  ~50% accuracy here but an MCC near 0, immediately exposing that it has learned nothing.
- **Brier score** is the mean squared error of the predicted probabilities, i.e. a **calibration** metric:
  it rewards probabilities that mean what they say. This matters because the app reports `p_fake` to a
  user as a confidence, and an over-confident detector that is "90% sure" while being wrong half the time
  is actively misleading. Calibration is also where the pipelines differ most interestingly OOD, where
  many stay accurate-ish but become badly over-confident.
- **Accuracy, precision, recall, macro-F1, confusion matrix** round out the picture at a chosen threshold
  — the precision/recall split in particular reveals *which direction* a model errs (crying fake vs.
  missing fakes), which a lone accuracy number cannot.

Threshold-free metrics (AUC, PR-AUC, Brier) use the probabilities directly; single-class slices are
guarded (AUC → NaN) so an evaluation over a generator subset that happens to contain only fakes does not
crash or report a misleading value. `best_f1_threshold(y_true, y_prob)` tunes a decision threshold on
**validation** (never test) by maximising macro-F1, and each pipeline reports metrics both **at 0.5** and
**at the tuned threshold**. Tuning the threshold matters because 0.5 is only optimal for a perfectly
calibrated, symmetric model; the true best operating point usually sits slightly off-centre, and finding
it on validation (then *applying* it to test) is the methodologically honest way to report a model's
realistic operating-point performance without peeking at the test set.

## 3.5 metrics.json schema (uniform across pipelines)

Every pipeline emits the same JSON shape, which is the contract that lets the aggregated evaluation
notebooks and the app consume any pipeline's results generically — no per-pipeline special-casing:

```jsonc
{
  "pipeline": "...", "created": "ISO-8601", "working_size": 128|224,
  "normalization": "dataset|imagenet|clip",
  "dataset": { "in_distribution": "ai-real-images", "ood": "tiny-genimage" },
  "threshold_default": 0.5, "threshold_tuned": 0.49,
  "in_distribution": { "at_0.5": { ...metrics... }, "at_tuned": { ...metrics... } },
  "ood": { "overall_accuracy": 0.55, "per_generator": { "biggan": {"accuracy":…, "n":…}, … }, "preview": true },
  "tuning": { "tuned": true, "best_val_auc": 0.99, "n_trials": 12, "best_params": { … } },
  "figures": { "training_curves": "figures/training_curves.png", … }
}
```

Both threshold variants are stored side by side (`at_0.5` and `at_tuned`) so a reader can see the effect
of threshold tuning directly; `working_size` and `normalization` are recorded because they are
reproducibility-critical preprocessing details that differ across input families (see
[02-data §2.6](02-data.md#26-preprocessing-notebook-03)); and the `tuning` block carries enough of the
search summary that the headline numbers can be read from the JSON alone.

## 3.6 Model-sharing scheme (committing trained parts to GitHub)

Trained weights and GitHub are an awkward pair — full checkpoints for a dozen pipelines would bloat the
repository past what GitHub comfortably hosts. The resolution exploits a structural fact about half of
these pipelines: their bulk is a **frozen, re-downloadable backbone** (an ImageNet ViT, a CLIP encoder, a
timm CNN) that nobody has changed. There is no point committing weights a teammate's machine can fetch
from the original source. So pipelines with a frozen backbone **save only the trained part** — the LoRA
adapters, the classifier head, the EMA weights — and a teammate rebuilds the architecture (the
`build_*()` call auto-downloads the frozen weights) and attaches the committed delta. Checkpoints live
under `artifacts/<pipeline>/models/` via `.gitignore` negations (see
[07-reproducibility.md](07-reproducibility.md)).

| Pipeline | Saved | Reload | Approx size |
|----------|-------|--------|------------:|
| cnn-scratch / two-stream / freqcross / srm-noise | full model | `load_checkpoint` / `load_weights` | ~12–20 MB |
| cnn-residual | full model + EMA | `load_ema_weights` | ~44 MB |
| cnn-finetune | EMA only (slim) | `load_weights` | EffNet-B0 ~16 MB |
| vit-lora | LoRA + head only | `load_weights(strict=False)` → `merge_and_unload()` | ~2.4 MB |
| clip-probe | MLP head only (CLIP frozen) | `load_checkpoint` | <2 MB |
| patch-ensemble / dire-recon | full model (timm backbone trained) | `load_weights` | ~16–45 MB |

The size column tells the story: the from-scratch models *are* trained end-to-end, so the full model is
the only faithful thing to save (~12–44 MB). But `vit-lora` ships in **2.4 MB** and `clip-probe` in under
**2 MB**, because for those pipelines the only weights that changed during training are the tiny adapter /
head — the 86M-parameter ViT and the CLIP encoder underneath are byte-for-byte the public pretrained
weights and are re-downloaded on load. The reload column records the exact rehydration step, including the
two-step `load_weights(strict=False)` → `merge_and_unload()` for LoRA (load the adapter non-strictly onto
the rebuilt frozen ViT, then fold `W + BA` into the weights for zero-overhead inference).

Reconstruction recipe = **committed weights file + that pipeline's notebook** (the `build_*` cell defines
the architecture; `build_*()` auto-downloads any frozen backbone). In other words the notebook *is* the
architecture spec — pairing it with the committed delta is sufficient to reproduce the exact trained model
on any machine, which is precisely the property the eval-time reconstruction layer in
[04-evaluation §4.1](04-evaluation.md#41-the-reconstruction-layer--eval_protocolspy) relies on.

Next: [04-evaluation.md →](04-evaluation.md)
