# 7 — Reproducibility & repo meta

[← docs index](README.md) · [← 06 App](06-app.md)

A project is only as credible as it is reproducible: the numbers in [05-results](05-results.md) mean little
if a teammate cannot re-run the pipelines and land in the same place. This chapter is the operational
manual for doing exactly that — how to set up the environment, run the notebooks (interactively or
headlessly), run the tests, and find every artifact — together with the *why* behind the few non-obvious
choices that trip people up. The biggest of those is that this machine has **two Python installs**, and
nearly every "it ran on CPU / it can't find torch" surprise traces back to that one fact, so it leads the
chapter.

How to set up, run, and reproduce everything; the artifact/data layout; and what is committed vs ignored.

---

## 7.1 Environment

- **Stack** ([`requirements.txt`](../requirements.txt)): PyTorch · torchvision · **timm** · Hugging Face
  **transformers + peft** (ViT + LoRA) · **open_clip** (CLIP embeddings) · numpy/scipy (FFT/DCT, radial
  spectrum) · Pillow/OpenCV · scikit-learn (**metrics & splitting only**) · matplotlib/seaborn ·
  **pytorch-grad-cam** · **optuna** · **kagglehub** · jupyter · tqdm. `dire-recon` additionally needs
  **`diffusers` + `accelerate`** (not in the base requirements).
- **Hardware**: a single mid-range GPU (8–12 GB) at ~224 px with sensible batch sizes. The project was
  trained on an RTX 3060 (12 GB).

The `dire-recon` dependencies are kept *out* of the base `requirements.txt` on purpose: `diffusers` and
`accelerate` pull in a heavy diffusion stack that only one optional, compute-heavy pipeline needs, so the
other nine install lighter without them. The hardware note is the practical floor — everything was sized
to fit a 12 GB consumer card, which is also why the app enforces single-model GPU residency in
[06-app §6.1](06-app.md#61-backend-srcbackend).

### ⚠️ Two Python installs (important)

| Interpreter | torch | Role |
|-------------|-------|------|
| `C:\Program Files\Python312\python.exe` | 2.x **+cu121 (CUDA)** | what `python` resolves to; what training/eval/app should use |
| Python 3.10 | **+cpu** | owns the bare `uvicorn` / `jupyter` / `streamlit` executables |

Consequences:
- **Always invoke via `python -m …`** (`python -m uvicorn`, `python -m streamlit`, `python -m pytest`) so
  you land on the CUDA interpreter. A *bare* `uvicorn`/`streamlit` runs under CPU-only torch (device shows
  `cpu`).
- Notebooks must run on the **`df312`** Jupyter kernel (registered against the Python 3.12 / CUDA
  interpreter). The babysitter passes `--ExecutePreprocessor.kernel_name=df312`.

The reason this causes trouble is that the two installs **disagree about which torch is the real one**, and
the disagreement is invisible until something runs on the wrong device. The CUDA-enabled torch lives in the
3.12 interpreter that `python` points at, but the convenience executables on `PATH` — `uvicorn`,
`streamlit`, `jupyter` — were installed by the 3.10 interpreter and carry its **CPU-only** torch with them.
Type a bare `uvicorn …` and you get a server that works perfectly except that every model is pinned to the
CPU; nothing errors, the only tell is a `cpu` device string and crawling inference (the same quiet failure
described in [06-app §6.5](06-app.md#65-running-the-app)). The fix is mechanical and absolute: prefix every
launch with `python -m`, which means "run this module under *this* (the 3.12/CUDA) interpreter" and so
guarantees the GPU build is the one in play. The notebook equivalent is the **`df312` kernel** — a Jupyter
kernel registered against the 3.12/CUDA interpreter — and the babysitter pins it explicitly with
`--ExecutePreprocessor.kernel_name=df312` so a headless run can never silently drift onto the CPU kernel.

## 7.2 Running the notebooks

Run interactively on the `df312` kernel (**Restart Kernel → Run All**), or headlessly with the babysitter.

For an interactive pass, the "Restart Kernel → Run All" discipline matters: it guarantees the notebook
executes top-to-bottom in a clean namespace, so the run reflects the saved cells and not some stale
variable left in memory from an earlier edit — which is the whole point of a *reproducible* run. For an
unattended pass — overnight, or across the full ten-pipeline sweep — the babysitter takes over.

### The babysitter ([`notebooks/_babysit_runs.ps1`](../notebooks/_babysit_runs.ps1))

Runs notebooks **one at a time, GPU-serialized**, via `nbconvert --execute --inplace`. It launches each
as a **detached** process and polls until exit, classifying success by **exit code** (`0`) **or** a fresh
`metrics.json`. At its time deadline it stops launching new work but **leaves any in-progress run alive**.

The babysitter exists because of the same constraint that shapes the app: **one GPU**. You cannot run
several training notebooks at once without them fighting over the card and OOM-ing, so the babysitter is a
headless runner that **serialises** them — exactly one notebook on the GPU at a time, the next launched only
once the previous exits. Running each notebook *detached* (rather than inline) keeps the long-lived runs
isolated from the controller, so the babysitter can poll their state without being blocked by them. Its
**dual success test** is the careful part: a notebook is considered to have succeeded if it exits with code
`0` *or* if it left behind a `metrics.json` newer than the run started. Belt-and-braces, because either
signal alone can mislead — a notebook can write its metrics and then trip on a trailing teardown cell
(non-zero exit but the real work is done), and conversely a zero exit with no fresh `metrics.json` means
nothing actually got produced. Checking both — exit code **and** fresh artifact — classifies runs
correctly in both cases. The time-deadline behaviour is deliberately gentle: at the deadline it stops
*starting* new notebooks but never kills a run already in flight, so a long from-scratch training that is
nearly done is allowed to finish rather than being thrown away.

```powershell
# Full pipeline 00 → 13 → evals (default 18h window)
powershell -File notebooks/_babysit_runs.ps1

# Longer window for a cold from-scratch run
powershell -File notebooks/_babysit_runs.ps1 -Hours 48

# Specific notebooks only (filename or pipeline name; multiple allowed; commas)
powershell -File notebooks/_babysit_runs.ps1 -Only cnn-scratch,cnn-residual,dire-recon
powershell -File notebooks/_babysit_runs.ps1 -Only eval -Hours 6
```

The two switches map directly onto the two things you actually need to control. `-Hours` sets the wall
clock window — the default 18h covers the full `00 → 13` plus evals, but a cold from-scratch CNN training
can run long, hence the 48h example. `-Only` narrows the run to specific notebooks, matched by either
filename or pipeline name and accepting a comma-separated list, so you can re-run just the two un-tuned CNNs
or just the evaluation notebooks without re-executing the whole chain. Combined (`-Only eval -Hours 6`),
they express "run only the eval notebooks, and give them a 6-hour window."

Per-run logs land in `notebooks/_run_<pipe>.{log,err.log}` and a rolling `notebooks/_babysit.log`
(all gitignored).

### Live progress ([`notebooks/_progress.py`](../notebooks/_progress.py))

Read-only reporter — the babysitter's last log line plus, per pipeline, the Optuna study's trial
count / best val AUC (the SQLite study commits after every trial) and whether `metrics.json` was written
this session. Safe to run anytime; touches nothing the kernels use.

The defining property of `_progress.py` is that it is **strictly read-only** — it inspects the babysitter's
log tail, reads each pipeline's Optuna SQLite study (which commits after every trial, so the trial count
and running-best AUC are live even mid-search), and checks for a fresh `metrics.json`. Because it only
reads files the running kernels are writing — and never opens the GPU, the live `.db` for writing, or
anything a kernel holds a lock on — it is completely safe to run at any moment without perturbing an
in-flight training. It is the right way to answer "how far along is the sweep?" without the temptation to
poke at the running process itself.

```bash
python notebooks/_progress.py
```

> **EVAL_ONLY / TUNE flags.** Set `TUNE=False` in a tuned notebook to skip the Optuna search and reuse
> the committed `best_params.json`. `patch-ensemble` additionally has `EVAL_ONLY=True` (its committed
> default) which skips search **and** training and just loads `best.pt` to evaluate — run that notebook's
> §2 (it loads params) before §3, or simply **Run All**.

These flags exist so reproduction does not force a re-search. The Optuna sweeps are the expensive part of
the project; `TUNE=False` lets a teammate skip straight to training on the **committed** winning
hyperparameters in `best_params.json`, reproducing the final model without paying for the search again.
`patch-ensemble` goes one step further with `EVAL_ONLY=True` as its committed default — it skips both
search and training and simply loads the committed `best.pt` to re-evaluate, which is why its notebook's §2
(parameter loading) must run before §3, or you just **Run All** and let the ordering take care of itself.

## 7.3 Tests ([`tests/`](../tests/))

A CPU-only pytest suite over the shared helpers and the app contract (config wiring, prediction schema,
the pipeline registry, metrics, paths, cleaning, tuning, model builders). Markers in
[`pytest.ini`](../pytest.ini): `slow` (builds real timm/peft architectures on CPU — runnable, heavier),
`download` (needs network weights; skipped unless `--run-download`), `gpu` (auto-skipped without CUDA).

The suite is **CPU-only by design** so it can run anywhere — in CI, on a laptop, or beside a training run
without demanding the GPU — and it targets the seams most likely to break silently: the config wiring that
puts `notebooks/` on the path, the shared prediction schema the UI depends on, the pipeline registry, and
the pure helpers (metrics, paths, cleaning, tuning, builders). The three **markers** exist to keep the
common case fast while still allowing the heavier checks on demand. `slow` tags the tests that actually
build real timm/peft architectures on CPU — fully runnable, just not cheap — so the everyday run can skip
them. `download` tags tests that need to pull network weights and is skipped unless you opt in with
`--run-download`, which keeps the suite **hermetic and offline** by default (no surprise network
dependency, no flakiness when a weight host is down). `gpu` auto-skips when CUDA is absent, so the same
suite passes unchanged on a CPU-only box.

```bash
python -m pytest                  # full suite (skips download-only tests)
python -m pytest -m "not slow"    # fast subset: pure helpers + schemas
python -m pytest --run-download   # also run network-dependent tests
```

> Don't run the suite while a training/eval notebook is using the GPU — the `slow` tests import torch/timm
> and compete for RAM. Run after notebooks finish.

That caution is about **RAM, not the GPU**: even though the tests run on CPU, the `slow` ones import torch
and timm and build real architectures, which is memory-hungry enough to contend with an active training
run on the same machine. Sequencing them after the notebooks finish avoids the squeeze.

## 7.4 Artifact & data layout

```text
data/  (gitignored)
├── ai-real-images/  manifest.csv · manifest_clean.csv · manifest_split.csv · norm_stats.json
│                     cache/cache_{train,val,test}_256.npy (~11 GB) · clip_emb_{train,val,test}.npy · dire maps
└── tiny-genimage/   manifest.csv · manifest_clean.csv  (+ OOD embedding/DIRE caches)

notebooks/artifacts/
├── eda/figures/
├── <pipeline>/  figures/ · models/ · metrics/{metrics.json, best_params.json, optuna_trials.json} · tuning/<pipe>.db
└── evaluation/  *.csv + figures/   ← aggregated comparison / generalization / robustness / optuna / explainability
```

The layout encodes a clean separation. Everything under `data/` is **regenerable input** — manifests, the
~11 GB 256² cache, the CLIP/DIRE embedding caches — so it is gitignored and rebuilt from the data notebooks
rather than committed. Everything under `notebooks/artifacts/` is **produced output**, and each pipeline
owns a fixed three-folder home: `figures/` for plots, `models/` for checkpoints, and `metrics/` for the
JSONs (`metrics.json`, plus `best_params.json` and `optuna_trials.json` for the tuned ones), with the live
Optuna `tuning/<pipe>.db` alongside. The `evaluation/` folder is where the per-pipeline outputs are
aggregated into the cross-pipeline CSVs and figures behind [05-results](05-results.md). This is the
contract the rest of the project relies on — the app finds a pipeline's weights under its `models/` folder,
and the comparison notebook finds every pipeline's `metrics.json` in the same place.

## 7.5 What's committed vs ignored ([`.gitignore`](../.gitignore))

- **Committed**: all source (`.py`, `.ipynb`), docs, `requirements.txt`, `pytest.ini`, the per-pipeline
  metrics/figures, and the trained model deltas.
- **Ignored**: `data/`, caches (`*.npy`), the live Optuna DBs (`tuning/`), babysitter logs
  (`notebooks/_*.log`), app inference outputs (`src/backend/predictions/`), virtualenvs/IDE files.
- **Model-sharing via negations** — `*.pt` is ignored globally, but
  `!notebooks/artifacts/**/models/*.pt` re-includes the trained checkpoints. This is what makes the
  [model-sharing scheme](03-shared-methods.md#36-model-sharing-scheme-committing-trained-parts-to-github)
  work: a teammate clones, the frozen backbones re-download on `build_*()`, and the committed delta
  attaches on top. (Note: a checkpoint must land under `notebooks/artifacts/<pipe>/models/` to match the
  negation — see the path-resolution fix in [`utils/paths.py`](../notebooks/utils/paths.py), which
  anchors the repo root on committed markers like `.git` rather than the gitignored `CLAUDE.md`.)

The governing principle is **commit results, ignore the regenerable and the bulky**: source, docs, metrics,
figures, and the small trained deltas go in version control; datasets, multi-gigabyte caches, the live
Optuna databases, runtime logs, and user-uploaded prediction records stay out. The **negation pattern** is
the clever piece and worth reading slowly. A blanket `*.pt` ignore is the safe default — it stops anyone
from accidentally committing a multi-hundred-megabyte checkpoint — but it would also exclude the trained
weights we *do* want to share. The single re-inclusion line `!notebooks/artifacts/**/models/*.pt` carves
out exactly one exception: `.pt` files that live under a pipeline's `models/` folder are committed, every
other `.pt` is ignored. That is what powers the model-sharing scheme — paired with the convention that
frozen backbones are *not* stored (they re-download on `build_*()`), only the small trained **delta** needs
committing, so cloning the repo and running `build_*()` reconstructs each model from a re-downloaded
backbone plus its committed delta. The catch the note flags is real: a checkpoint only matches the negation
if it actually sits under `notebooks/artifacts/<pipe>/models/`, which is why the repo-root resolution in
[`utils/paths.py`](../notebooks/utils/paths.py) anchors on a committed marker like `.git` rather than the
gitignored `CLAUDE.md` — get the root wrong and checkpoints would be written to a path that silently fails
to match the re-inclusion and would never be committed.

[← back to docs index](README.md) · [per-pipeline deep dives →](pipelines/README.md)
