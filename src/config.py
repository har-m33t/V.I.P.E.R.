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
# Overridable so the same code runs against a local checkout and against a
# RunPod volume without editing this file.
DATA_DIR       = Path(os.getenv("DATA_DIR", ROOT / "data"))
AI_ART_DIR     = Path(os.getenv(
    "AI_ART_DIR", DATA_DIR / "ai_art_classification" / "train" / "AI_GENERATED"
))
REAL_ART_DIR   = Path(os.getenv(
    "REAL_ART_DIR", DATA_DIR / "ai_art_classification" / "train" / "NON_AI_GENERATED"
))
WIKIART_DIR    = Path(os.getenv("WIKIART_DIR", DATA_DIR / "WikiArt"))

# ─── Output Paths ─────────────────────────────────────────────────────────────
RESULTS_DIR      = ROOT / "results"
CHECKPOINTS_DIR  = ROOT / "checkpoints"
GRADCAM_DIR      = ROOT / "gradcam_gallery"
NOTEBOOKS_DIR    = ROOT / "notebooks"
TORCH_WEIGHTS_DIR = CHECKPOINTS_DIR / "torchvision_weights"

# Create all output directories on import
for _dir in [RESULTS_DIR, CHECKPOINTS_DIR, GRADCAM_DIR, NOTEBOOKS_DIR, TORCH_WEIGHTS_DIR]:
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
UMAP_SCATTER_PNG      = RESULTS_DIR / "umap_scatter.png"
TRAINING_HISTORY_JSON = RESULTS_DIR / "training_history.json"
RUN_SUMMARY_JSON      = RESULTS_DIR / "run_summary.json"

# ─── Training Hyperparameters (LOCKED by execution plan) ──────────────────────
# BATCH_SIZE is the *per-GPU* batch size. Under DDP the effective global batch
# is BATCH_SIZE * world_size, so keep it at 64 on a single GPU to reproduce the
# original CPU run exactly.
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", 64))
NUM_EPOCHS    = int(os.getenv("NUM_EPOCHS", 100))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-4))
SEED          = int(os.getenv("SEED", 42))


def _default_num_workers() -> int:
    if os.name == "nt":
        return 0
    return min(16, max(2, (os.cpu_count() or 4) - 1))


NUM_WORKERS     = int(os.getenv("NUM_WORKERS", _default_num_workers()))
PREFETCH_FACTOR = int(os.getenv("PREFETCH_FACTOR", 4))

# ─── Dataset Scale ────────────────────────────────────────────────────────────
# Cap on the number of images pulled into the train/val/test splits.
# 0 (or negative) means "use every image found" — the full 67k run.
# Set MAX_SAMPLES=10000 to reproduce the original CPU fast-track subsample.
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", 0))

# ─── Early Stopping ───────────────────────────────────────────────────────────
# Stop once the checkpointing metric has not improved for this many epochs.
EARLY_STOP_PATIENCE   = int(os.getenv("EARLY_STOP_PATIENCE", 20))
EARLY_STOP_MIN_DELTA  = float(os.getenv("EARLY_STOP_MIN_DELTA", 0.0))

# ─── GPU Acceleration ─────────────────────────────────────────────────────────
# AMP_DTYPE: auto | bf16 | fp16 | off. "auto" picks bf16 on Ampere and newer
# (no GradScaler needed, no overflow risk), fp16 elsewhere, and off on CPU.
AMP_DTYPE      = os.getenv("AMP_DTYPE", "auto").lower()
CHANNELS_LAST  = os.getenv("CHANNELS_LAST", "1") == "1"
TORCH_COMPILE  = os.getenv("TORCH_COMPILE", "0") == "1"
CUDNN_BENCHMARK = os.getenv("CUDNN_BENCHMARK", "1") == "1"
# Decode JPEGs at a reduced scale before resizing (large speedup for 512px
# sources). Off by default because it perturbs pixel values very slightly and
# would break exact comparability with the CPU baseline.
JPEG_DRAFT     = os.getenv("JPEG_DRAFT", "0") == "1"

# Reweight CrossEntropyLoss by inverse class frequency. Off by default so the
# GPU run reproduces the CPU baseline's objective exactly.
CLASS_WEIGHTED = os.getenv("CLASS_WEIGHTED", "0") == "1"

# ─── Image Configuration ──────────────────────────────────────────────────────
IMAGE_SIZE    = 224       # ConvNeXt-Tiny input
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
MODEL_NAME          = "convnext_tiny"
IMAGE_EMBED_DIM     = 768
UNFREEZE_STAGES     = 2   # last N ConvNeXt stages + classifier
CLASSIFIER_DROPOUT  = 0.3
USE_EDA_FEATURES    = False  # Disabled for 10k fast-track
FUSION_HIDDEN_DIM   = int(os.getenv("FUSION_HIDDEN_DIM", 384))
STRICT_EDA_COVERAGE = False

# ─── Evaluation ───────────────────────────────────────────────────────────────
VAL_SPLIT          = 0.15
TEST_SPLIT         = 0.15

# ─── Device / Distributed ─────────────────────────────────────────────────────
import torch

# torchrun exports these; they are absent for a plain `python src/train.py`.
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
IS_DISTRIBUTED = WORLD_SIZE > 1 and LOCAL_RANK >= 0


def _resolve_device() -> torch.device:
    requested = os.getenv("DEVICE", "cuda")
    if requested.startswith("cuda") and torch.cuda.is_available():
        # Under torchrun each process owns exactly one GPU.
        return torch.device(f"cuda:{LOCAL_RANK}" if LOCAL_RANK >= 0 else "cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _resolve_device()

# ─── Stretch Goal Config ──────────────────────────────────────────────────────
JPEG_QUALITY_LEVELS = [95, 75, 50, 25]
STRETCH_SAMPLE_N    = 1000
