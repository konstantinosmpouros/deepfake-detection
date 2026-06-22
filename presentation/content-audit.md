# Presentation content stress test

Audit date: 2026-06-21

## Source hierarchy

1. `docs/` for experimental intent, architecture rationale, and declared caveats.
2. `notebooks/artifacts/evaluation/*.csv` for exact reported metrics.
3. `MSc_Deep_Learning_report.pdf` for the final narrative and tables.
4. Implementation code where it contradicts prose.

## Verdict

The core result is supported: excellent in-distribution discrimination does not transfer reliably to the
held-out benchmark, and `patch-ensemble` has the best measured OOD accuracy (0.677496). The original
presentation's model metrics matched the committed CSVs to four decimals. Several claims needed narrower
wording because the report and docs overstate what the experiment proves.

## Material findings

| Severity | Claim under test | Evidence | Resolution in presentation |
|---|---|---|---|
| High | “Every model loses 30–40 points.” | Gaps span 0.267870–0.392315. Competitive detectors span 0.290572–0.392315; `cnn-residual` starts from ID accuracy 0.786843. | Rewritten as 29.1–39.2 points for competitive models, with the residual exception stated explicitly. |
| High | “Patch has the smallest gap.” | `cnn-residual` is arithmetically smaller (0.267870 vs 0.290572) but reaches only 0.518973 OOD. | Patch is described as the smallest meaningful gap among competitive detectors, not the numerical minimum. |
| High | “Seven unseen generators / disjoint generator sets.” | The primary dataset names Midjourney; `tiny-genimage` also contains a Midjourney subset. The report simultaneously calls the sets disjoint and later explains that Midjourney is easiest because it exists in training. | Rewritten as seven held-out generator subsets: six new names plus an independently sourced Midjourney subset. The protocol is framed as cross-dataset and cross-generator shift. |
| High | “The 256² fix neutralises the resolution shortcut for all models.” | `PatchBagDataset` bypasses the cache and crops native files. Small files may also be bilinearly upscaled. | Stated as 9/10 cache-based pipelines. Patch now carries an unresolved resolution/upscale confound and requires matched-resolution ablation. |
| High | “Patch evaluates K=6 deterministic crops.” | Tuned training uses K=6, but `_crops()` defines four eval corners and adds a centre only when `k == 5`; for `k == 6` it returns four patches. | Architecture view now states 6 random train crops / 4 deterministic eval crops. The published score is retained because changing evaluation would produce a new experiment. |
| Medium | “All models share the same ID test population.” | `dire-recon` uses an ID subsample of 2,000; the other pipelines use 11,963. | DIRE is marked indicative and non-head-to-head in both architecture and results views. |
| Medium | “The gap compares ID AUC with OOD accuracy.” | The committed `gap` is ID accuracy minus OOD accuracy, not AUC minus accuracy. | Metric label changed to `ID acc − OOD`. |
| Medium | “Robustness covers all ten pipelines.” | The sweep covers seven image-family pipelines; CLIP, patch, and DIRE are excluded. | Scope is now explicit in the protocol copy and missing values remain `n/a`. |
| Medium | “XAI proves why the model decided.” | Grad-CAM, rollout, t-SNE, MIL weights, and residual maps have different faithfulness properties. | Reframed as architecture-native diagnostics and shortcut audits, not causal guarantees. |

## Verified headline numbers

| Pipeline | ID accuracy | ID AUC | OOD accuracy | Accuracy gap |
|---|---:|---:|---:|---:|
| patch-ensemble | 0.968068 | 0.996283 | 0.677496 | 0.290572 |
| vit-lora | 0.978183 | 0.997244 | 0.602177 | 0.376005 |
| clip-probe | 0.959208 | 0.993024 | 0.583662 | 0.375546 |
| cnn-finetune | 0.955948 | 0.993045 | 0.563632 | 0.392315 |
| freqcross | 0.900276 | 0.965120 | 0.552632 | 0.347644 |
| cnn-scratch | 0.901112 | 0.964753 | 0.548831 | 0.352280 |
| dire-recon* | 0.873000 | 0.939858 | 0.541500 | 0.331500 |
| two-stream | 0.897685 | 0.960944 | 0.526402 | 0.371283 |
| srm-noise | 0.882387 | 0.951785 | 0.523116 | 0.359272 |
| cnn-residual | 0.786843 | 0.867183 | 0.518973 | 0.267870 |

`*` DIRE ID metrics use n=2,000 rather than n=11,963.

## Robustness checks

- `vit-lora`: mean perturbed accuracy 0.922750; worst case 0.727500.
- Mean strongest-level drop: Gaussian noise 0.350, downsampling 0.221, JPEG Q60 0.003, blur approximately 0.
- The robustness claim applies only to the seven included image-family pipelines and to the tested
  perturbation strengths; it is not a general guarantee against arbitrary image transformations.

## Storytelling structure

1. The apparent success: 0.9972 ID AUC.
2. The experimental trap: shortcut leakage and an intentionally sealed OOD benchmark.
3. Ten hypotheses: each architecture is presented as a scientific bet, not a catalogue entry.
4. The common trial: ID, OOD, perturbation robustness, and architecture-native diagnostics.
5. The reversal: native-patch MIL leads measured OOD, but with explicit validity caveats.
6. The deployment conclusion: OOD coverage, calibration, and noise tolerance matter more than another ID decimal.

## Required follow-up experiments

- Re-run patch evaluation with an eval sampler that returns exactly K=6 crops.
- Run a matched-resolution ablation: native patches versus patches from the common 256² cache.
- Report bootstrap confidence intervals for OOD accuracy and pairwise model differences.
- Add a genuinely generator-disjoint OOD dataset without Midjourney overlap.
- Tune `cnn-residual` with the same search budget before interpreting its architecture.
