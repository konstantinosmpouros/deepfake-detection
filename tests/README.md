# Tests

Unit tests for the shared notebook helpers (`notebooks/utils/`) and the app
backend (`src/backend/`). All tests are **CPU-only** — none require a GPU and,
by default, none download pretrained weights.

## Running

Use the CUDA-capable interpreter that owns the project stack (the same one the
notebooks use), and invoke pytest as a module so you don't hit the other
Python's `pytest.exe`:

```bash
python -m pytest                 # full suite (skips download-only tests)
python -m pytest -m "not slow"   # fast subset: pure helpers + schemas only
python -m pytest tests/test_metrics.py -v
```

> **Do not run the suite while the training babysitter is active.** The `slow`
> model tests import timm/torch and build architectures on CPU, which competes
> for RAM with the running notebooks. Run after training completes.

## Markers (see `../pytest.ini`)

| Marker | Meaning | Default |
|--------|---------|---------|
| `slow` | Builds real timm/peft architectures on CPU (random weights, no network). | runs |
| `download` | Needs to fetch pretrained weights over the network. | **skipped** unless `--run-download` |
| `gpu` | Needs a CUDA device. | skipped automatically when no GPU is present |

```bash
python -m pytest --run-download   # also run the network-dependent tests
```

## Layout

| File | Covers |
|------|--------|
| `test_metrics.py` | `utils/metrics.py` — metric values, confusion-matrix orientation, threshold invariance, JSON save round-trip. |
| `test_paths.py` | `utils/paths.py` — repo-root discovery, artifact dir creation (sandboxed in tmp). |
| `test_clean.py` | `utils/clean.py` — sha1/pHash, near-duplicate distance, corrupt-file handling. |
| `test_tuning.py` | `utils/tuning.py` — focal loss, study resumability, `save_study_artifacts` JSON, best-params merge. |
| `test_models.py` | `utils/models.py` — single-logit output contract + component outputs (timm builders are `slow`). |
| `test_schemas.py` | `src/backend/schemas.py` — the shared prediction/response JSON contract. |
| `test_config.py` | `src/backend/config.py` — repo wiring + `DF_DEVICE` device override. |
| `test_pipelines_registry.py` | `src/backend/pipelines/` — selector-key registry (`slow`). |

`conftest.py` wires `notebooks/` and the repo root onto `sys.path` so tests
import the exact modules the project ships.
