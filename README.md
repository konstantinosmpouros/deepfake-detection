# Deepfake & AI-Generated Image Detection

Detecting AI-generated (deepfake) images — an MSc Deep Learning group project for the joint MSc in Artificial Intelligence of the **University of Piraeus (UNIPI)** and **NCSR "Demokritos"**.

We frame the problem as **binary classification (real vs. AI-generated)** and tackle it with several independent **deep-learning pipelines**, then compare them. The emphasis is not only on in-distribution accuracy but on the **cross-generator generalization gap** and **robustness** that separate a real detector from one that just memorizes a single generator.

## Datasets

| Name | Size | Generators | Role |
|---|---|---|---|
| **`ai-real-images`** | 60k (30k real / 30k fake) | Stable Diffusion + Midjourney + DALL·E | Primary — training & in-distribution eval |
| **`tiny-genimage`** | 35k | 7 (biggan, vqdm, sdv5, wukong, adm, glide, midjourney) | Cross-generator out-of-distribution (OOD) test |

Photographic, higher-resolution images (the 32×32 CIFAKE dataset was considered and dropped). Datasets are downloaded via `kagglehub` and **not committed** (gitignored under `data/`); each gets a manifest at `data/<name>/manifest.csv`.

## Pipelines (all deep-learning)

`cnn-scratch` · `cnn-residual` · `cnn-finetune` (ResNet50/EfficientNet) · `vit-lora` (ViT-Base + LoRA) · `clip-probe` (frozen CLIP/DINOv2 + neural head) · `two-stream` (RGB + frequency fusion).

## Goals

1. **Baselines** — a small from-scratch CNN and a deeper residual CNN.
2. **Transfer learning & PEFT** — fine-tuned CNN/ViT backbones, including LoRA.
3. **Generalization & robustness** — CLIP/DINOv2 probes, cross-generator evaluation, robustness curves (JPEG / blur / noise).
4. **Explainability** — Grad-CAM, attention rollout, frequency-spectrum visualizations.
5. **App** — a Streamlit + FastAPI app to upload a photo, pick a pipeline, and get a prediction.

See [CLAUDE.md](CLAUDE.md) for the full plan, and [docs/research_1.md](docs/research_1.md) / [docs/research_2.md](docs/research_2.md) for the method survey.

## Project structure

```text
deepfake-detection/
├── data/          # datasets + manifests — gitignored
├── docs/          # research notes and references
├── notebooks/     # exploratory analysis and experiments
│   ├── utils/     # reusable helpers (data, models, training, eval, viz)
│   └── artifacts/ # per-pipeline figures / models / metrics
├── src/           # the app (FastAPI backend + Streamlit frontend)
├── requirements.txt
└── README.md
```

## Getting started

```bash
# 1. Clone
git clone https://github.com/konstantinosmpouros/deepfake-detection.git
cd deepfake-detection

# 2. Create an environment (Python 3.10+)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Data

Datasets are **not** committed. Run `notebooks/00_data_collection.ipynb` to download them via `kagglehub`
(needs Kaggle credentials — `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`). It builds the
per-dataset manifests used by every later notebook.

## Stack

PyTorch · timm · Hugging Face transformers + peft · open_clip · OpenCV · scikit-learn (metrics/splits) · numpy / scipy · matplotlib / seaborn · grad-cam · kagglehub. App: FastAPI + Streamlit (no Docker).

A single mid-range GPU (8–12 GB) is sufficient at ~224×224.

## License

Released under the [MIT License](LICENSE).
