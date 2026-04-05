# VIPER Forensic Engine — ArtHeist 2026

> **Detecting AI-Generated Art with Forensic Precision**
> Multi-stage computer vision pipeline: statistical analysis → deep learning → interpretability

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Setup Instructions](#setup-instructions)
5. [How to Run the Pipeline](#how-to-run-the-pipeline)
6. [How to Train the Model](#how-to-train-the-model)
7. [How to Run the Notebook](#how-to-run-the-notebook)
8. [Multi-Agent Development Guide](#multi-agent-development-guide)
9. [Results Summary](#results-summary)

---

## Project Overview

VIPER (Visual Intelligence Pipeline for Empirical Recognition) is a forensic
computer vision system that distinguishes AI-generated images from real photographs.

**Key capabilities:**
- Six forensic feature families (FFT, GLCM, noise, edge density, colour entropy, pixel stats)
- EfficientNet-B0 fine-tuned on binary AI-vs-Real classification
- Grad-CAM interpretability showing which image regions were decisive
- UMAP 2D embedding of the model's internal feature space
- JPEG robustness testing and WikiArt domain transfer analysis

**Dataset:**
- Primary: `data/Art/AiArtData/` (AI-generated) + `data/Art/RealArt/` (real images)
- Holdout: `data/WikiArt/` (unseen art domain, inference only)

---

## Architecture

```
                      ┌────────────────────────────────┐
                      │     RAW IMAGE DATA             │
                      │  AI_Art ~539 / RealArt ~436    │
                      └─────────────┬──────────────────┘
                                    │ Track Alpha
                                    ▼
                      ┌────────────────────────────────┐
                      │      src/dataloader.py         │
                      │  PyTorch Dataset + DataLoaders  │
                      │  train 70% / val 15% / test 15%│
                      └──────┬──────────────┬──────────┘
                  Track Beta │              │ Track Gamma
               ┌─────────────┘              └──────────────────┐
               ▼                                               ▼
  ┌────────────────────────┐                   ┌───────────────────────────┐
  │      src/eda.py        │                   │      src/model.py         │
  │  6 forensic features   │                   │  EfficientNet-B0 wrapper  │
  │  feature_matrix.csv    │                   │    + src/train.py         │
  └──────────┬─────────────┘                   │  checkpoints/best_model   │
             ▼                                 └────────────┬──────────────┘
  ┌────────────────────────┐            Track Gamma         │
  │    src/baseline.py     │                   ┌────────────┘
  │ Logistic Regression    │                   ▼
  │ baseline_metrics.json  │    ┌───────────────────────────┐
  └────────────────────────┘    │      src/evaluate.py      │
                                │ Metrics + Confusion Matrix │
                        Track   │      + UMAP projection     │
                        Delta   └────────────┬──────────────┘
              ┌────────────────────────────┐ │
              │       src/stretch.py       │ ▼
              │  JPEG robustness           │ ┌───────────────────────────┐
              │  WikiArt inference         │ │     src/visualize.py      │
              └────────────────────────────┘ │ Grad-CAM gallery          │
                                             │ UMAP scatter (Omni)       │
                                             └───────────────────────────┘
```

---

## Project Structure

```
ARTHEIST/
├── data/
│   ├── Art/
│   │   ├── AiArtData/          ← AI-generated images (label=1)
│   │   └── RealArt/            ← Real photographs   (label=0)
│   └── WikiArt/                ← Holdout art dataset (inference only)
│
├── src/
│   ├── config.py               ← Central config (paths, hyperparameters)
│   ├── dataloader.py           ← PyTorch Dataset + DataLoaders
│   ├── eda.py                  ← Six forensic feature extractors
│   ├── baseline.py             ← Logistic Regression baseline
│   ├── model.py                ← EfficientNet-B0 architecture
│   ├── train.py                ← Training loop (Adam + CosineAnnealingLR)
│   ├── evaluate.py             ← Metrics + UMAP embedding
│   ├── visualize.py            ← Grad-CAM gallery + UMAP scatter
│   └── stretch.py              ← JPEG robustness + WikiArt inference
│
├── notebooks/
│   └── ArtHeist_Final.ipynb    ← Master assembly notebook
│
├── results/                    ← Generated outputs (gitignored)
│   ├── feature_matrix.csv
│   ├── baseline_metrics.json
│   ├── eval_metrics.json
│   ├── confusion_matrix.png
│   └── umap_features.csv
│
├── checkpoints/                ← Model weights (gitignored)
│   └── best_model.pth
│
├── gradcam_gallery/            ← Grad-CAM images (gitignored)
├── omni_export/                ← Omni track deliverables
│
├── requirements.txt
├── .env.example
├── .gitignore
├── presentation_outline.md
└── README.md
```

---

## Setup Instructions

### 1. Clone & create virtual environment
```bash
cd ARTHEIST
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env — set DEVICE=cuda if you have a GPU
```

### 4. Verify data layout
```
data/Art/AiArtData/   ← must contain images
data/Art/RealArt/     ← must contain images
```

---

## How to Run the Pipeline

Run modules in dependency order, or all at once:

```bash
# Stage 1: Verify data loads
python src/dataloader.py

# Stage 2: Run EDA + build feature matrix
python src/eda.py

# Stage 3: Run logistic regression baseline
python src/baseline.py

# Stage 4: Train EfficientNet-B0 (requires GPU for reasonable speed)
python src/train.py

# Stage 5: Evaluate and compute UMAP
python src/evaluate.py

# Stage 6: Generate Grad-CAM gallery + UMAP scatter
python src/visualize.py

# Stage 7: Stretch goals (JPEG robustness + WikiArt)
python src/stretch.py
```

---

## How to Train the Model

```bash
# With default settings (10 epochs, lr=1e-4, batch=64)
python src/train.py

# Checkpoints saved to: checkpoints/best_model.pth
# Training history:     results/training_history.json
```

GPU recommended. Expected time on NVIDIA RTX:
- ~1-2 min/epoch with the current dataset size

---

## How to Run the Notebook

```bash
jupyter notebook notebooks/ArtHeist_Final.ipynb
# Kernel → Run All
```

The notebook assembles all four VIPER stages with narrative markdown cells.

---

## Multi-Agent Development Guide

Every module is self-contained. An AI agent can independently own one file.

| File | Agent | Can start when |
|------|-------|----------------|
| `dataloader.py` | Data Agent | Immediately |
| `eda.py` | EDA Agent | After `dataloader.py` |
| `baseline.py` | Baseline Agent | After `feature_matrix.csv` |
| `model.py` | DL Agent | After `dataloader.py` |
| `train.py` | DL Agent | After `model.py` |
| `evaluate.py` | Eval Agent | After `best_model.pth` |
| `visualize.py` | Viz Agent | After `eval_metrics.json` |
| `stretch.py` | Stretch Agent | After `best_model.pth` |

Search for `# AGENT_TASK:` comments in each file for specific extension points.

---

## Results Summary

> *(Populated after running the full pipeline)*

| Metric | Baseline (LR) | EfficientNet-B0 |
|--------|--------------|-----------------|
| Accuracy | 0.7118 | 0.8219 |
| F1 | 0.7783 | 0.8471 |
| AUC-ROC | 0.8007 | 0.9321 |

See `results/eval_metrics.json` and `results/baseline_metrics.json`.

---

## Limitations

- Dataset is relatively small (~975 images total). Results may improve significantly
  with the full CIFAKE 120K dataset from Kaggle.
- WikiArt inference is a zero-shot domain transfer; confidence scores reflect
  model uncertainty, not ground-truth labels.
- UMAP layout is non-deterministic; results may vary slightly between runs.
