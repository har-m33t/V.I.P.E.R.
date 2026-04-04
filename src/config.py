"""
src/config.py — VIPER Forensic Engine Global Configuration
Phase: ALL TRACKS (shared dependency)

Central configuration module. All hyperparameters, paths, and constants
are defined here. Agents must import from this module — never hardcode values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Project Root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ─── Data Paths ───────────────────────────────────────────────────────────────
DATA_DIR       = ROOT / "data"
CIFAKE_DIR     = DATA_DIR / "Art"          # local layout: Art/AiArtData, Art/RealArt
AI_ART_DIR     = CIFAKE_DIR / "AiArtData"
REAL_ART_DIR   = CIFAKE_DIR / "RealArt"
WIKIART_DIR    = DATA_DIR / "WikiArt"

# ─── Output Paths ─────────────────────────────────────────────────────────────
RESULTS_DIR      = ROOT / "results"
CHECKPOINTS_DIR  = ROOT / "checkpoints"
GRADCAM_DIR      = ROOT / "gradcam_gallery"
OMNI_DIR         = ROOT / "omni_export"
NOTEBOOKS_DIR    = ROOT / "notebooks"

# Create all output directories on import
for _dir in [RESULTS_DIR, CHECKPOINTS_DIR, GRADCAM_DIR, OMNI_DIR, NOTEBOOKS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Output File Paths ────────────────────────────────────────────────────────
FEATURE_MATRIX_CSV    = RESULTS_DIR / "feature_matrix.csv"
BASELINE_METRICS_JSON = RESULTS_DIR / "baseline_metrics.json"
EVAL_METRICS_JSON     = RESULTS_DIR / "eval_metrics.json"
CONFUSION_MATRIX_PNG  = RESULTS_DIR / "confusion_matrix.png"
UMAP_FEATURES_CSV     = RESULTS_DIR / "umap_features.csv"
JPEG_ROBUSTNESS_PNG   = RESULTS_DIR / "jpeg_robustness.png"
WIKIART_CONF_JSON     = RESULTS_DIR / "wikiart_confidence.json"
BEST_MODEL_PATH       = CHECKPOINTS_DIR / "best_model.pth"
UMAP_SCATTER_PNG      = OMNI_DIR / "umap_scatter.png"
OMNI_METADATA_CSV     = OMNI_DIR / "metadata.csv"

# ─── Training Hyperparameters (LOCKED by execution plan) ──────────────────────
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", 64))
NUM_EPOCHS    = int(os.getenv("NUM_EPOCHS", 10))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-4))
NUM_WORKERS   = int(os.getenv("NUM_WORKERS", 2))
SEED          = int(os.getenv("SEED", 42))

# ─── Image Configuration ──────────────────────────────────────────────────────
IMAGE_SIZE    = 224       # EfficientNet-B0 input
NUM_CLASSES   = 2         # 0 = REAL, 1 = AI_GENERATED

# ─── Class Labels ─────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "REAL", 1: "AI_GENERATED"}
LABEL_REAL  = 0
LABEL_AI    = 1

# ─── EDA Configuration ────────────────────────────────────────────────────────
EDA_KMEANS_K       = 8    # color palette clusters
EDA_SAMPLE_SIZE    = 500  # max images to sample for EDA (speed)
FFT_LOG_SCALE      = True

# ─── Model Architecture ───────────────────────────────────────────────────────
MODEL_NAME         = "efficientnet_b0"
UNFREEZE_BLOCKS    = 3    # last 3 MBConv blocks + classifier

# ─── Evaluation ───────────────────────────────────────────────────────────────
VAL_SPLIT          = 0.15
TEST_SPLIT         = 0.15

# ─── Device ───────────────────────────────────────────────────────────────────
import torch
DEVICE = torch.device(
    os.getenv("DEVICE", "cuda") if torch.cuda.is_available() else "cpu"
)

# ─── Stretch Goal Config ──────────────────────────────────────────────────────
JPEG_QUALITY_LEVELS = [95, 75, 50, 25]
STRETCH_SAMPLE_N    = 1000
