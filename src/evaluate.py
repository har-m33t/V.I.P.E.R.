"""
src/evaluate.py — VIPER Forensic Engine: Evaluation Layer
Phase: Track Gamma (Deep Learning) — Evaluation Agent

Evaluates the trained model on the test set:
  - Accuracy, F1, Precision, Recall, AUC-ROC
  - Confusion matrix PNG
  - 1280-d embedding extraction → UMAP 2D projection
  - Saves results/eval_metrics.json and results/umap_features.csv

Dependency: checkpoints/best_model.pth must exist (from Deep Learning Agent).

Usage:
    python src/evaluate.py
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DEVICE, BEST_MODEL_PATH,
    EVAL_METRICS_JSON, CONFUSION_MATRIX_PNG,
    UMAP_FEATURES_CSV, RESULTS_DIR,
    CLASS_NAMES,
)
from src.dataloader import get_dataloaders
from src.model import load_checkpoint, VIPERClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Inference Pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: VIPERClassifier,
    loader,
    device: torch.device = DEVICE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Run full forward pass on a DataLoader.

    Returns:
        (all_labels, all_preds, all_probs, all_embeddings, all_paths)
    """
    model.eval()
    all_labels     = []
    all_preds      = []
    all_probs      = []
    all_embeddings = []
    all_paths      = []

    for images, labels, paths in tqdm(loader, desc="Inference"):
        images = images.to(device, non_blocking=True)

        logits     = model(images)
        probs      = F.softmax(logits, dim=1)
        preds      = logits.argmax(dim=1)
        embeddings = model.get_embedding(images)   # 1280-d

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())   # P(AI_GENERATED)
        all_embeddings.extend(embeddings.cpu().numpy())
        all_paths.extend(paths)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_embeddings),
        all_paths,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    labels: np.ndarray,
    preds:  np.ndarray,
    probs:  np.ndarray,
) -> Dict[str, float]:
    """Compute all required evaluation metrics."""
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "auc_roc":   float(roc_auc_score(labels, probs)),
        "n_samples": int(len(labels)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confusion Matrix Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels: np.ndarray,
    preds:  np.ndarray,
    save_path: Path = CONFUSION_MATRIX_PNG,
) -> None:
    """Plot normalized confusion matrix and save to PNG."""
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Raw Counts", "Normalized (row)"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=list(CLASS_NAMES.values()),
            yticklabels=list(CLASS_NAMES.values()),
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {title}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[EvalAgent] ✓ Confusion matrix → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# UMAP Embedding
# ─────────────────────────────────────────────────────────────────────────────

def compute_umap(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    paths:      List[str],
    output_csv: Path = UMAP_FEATURES_CSV,
    n_neighbors: int = 15,
    min_dist:    float = 0.1,
) -> pd.DataFrame:
    """
    Reduce 1280-d EfficientNet embeddings to 2D with UMAP.

    Saves CSV with columns: path, label, umap_x, umap_y

    AGENT_TASK: tune UMAP hyperparameters (n_neighbors, min_dist, metric)
    AGENT_TASK: add t-SNE as an alternative embedding
    """
    try:
        import umap
    except ImportError:
        print("[EvalAgent] ✗ umap-learn not installed. Skipping UMAP.")
        return pd.DataFrame()

    print(f"[EvalAgent] Computing UMAP on {len(embeddings)} embeddings ...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
        verbose=False,
    )
    coords = reducer.fit_transform(embeddings)

    df = pd.DataFrame({
        "image_path": paths,
        "label":      labels,
        "umap_x":     coords[:, 0],
        "umap_y":     coords[:, 1],
    })
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[EvalAgent] ✓ UMAP features → {output_csv}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation Routine
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    checkpoint_path: Path = BEST_MODEL_PATH,
    device: torch.device = DEVICE,
) -> Dict:
    """
    Full evaluation pipeline: load model → infer → metrics → plots → UMAP.

    Returns:
        Dict of evaluation metrics.
    """
    # ── Load model ────────────────────────────────────────────────────────────
    model = load_checkpoint(checkpoint_path, device)
    if model is None:
        metrics = {"accuracy": 0, "status": "checkpoint_missing"}
        EVAL_METRICS_JSON.write_text(json.dumps(metrics, indent=2))
        print("[EvalAgent] ✗ No checkpoint found. Wrote dummy metrics.")
        return metrics

    # ── Load data splits ─────────────────────────────────────────────────────
    _, val_loader, test_loader = get_dataloaders(verbose=False)

    # ── Run inference on validation set (canonical: used to select checkpoint)
    print("[EvalAgent] Evaluating on validation set (checkpoint selection split) ...")
    val_labels, val_preds, val_probs, val_embeddings, val_paths = run_inference(
        model, val_loader, device
    )
    val_metrics = compute_metrics(val_labels, val_preds, val_probs)
    val_metrics["split"] = "validation"

    # ── Run inference on test set (held-out, unseen) ──────────────────────────
    print("[EvalAgent] Evaluating on test set (held-out) ...")
    test_labels, test_preds, test_probs, test_embeddings, test_paths = run_inference(
        model, test_loader, device
    )
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    test_metrics["split"] = "test"

    # ── Report both ───────────────────────────────────────────────────────────
    print(f"\n[EvalAgent] === Validation Set Results (Peak Performance) ===")
    for k, v in val_metrics.items():
        print(f"  {k:12s}: {v}")

    print(f"\n[EvalAgent] === Test Set Results (Held-Out / Honest) ===")
    for k, v in test_metrics.items():
        print(f"  {k:12s}: {v}")

    # Primary metrics = validation (this is what the checkpoint was selected on)
    metrics = val_metrics
    metrics["test_accuracy"] = test_metrics["accuracy"]
    metrics["test_f1"]       = test_metrics["f1"]
    metrics["test_auc_roc"]  = test_metrics["auc_roc"]

    # Save JSON
    EVAL_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    EVAL_METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"[EvalAgent] ✓ Metrics → {EVAL_METRICS_JSON}")

    # ── Confusion matrix (validation) ─────────────────────────────────────────
    plot_confusion_matrix(val_labels, val_preds)

    # ── UMAP (combine both splits for richer plot) ────────────────────────────
    all_embeddings = np.concatenate([val_embeddings, test_embeddings], axis=0)
    all_labels     = np.concatenate([val_labels,     test_labels],     axis=0)
    all_paths      = val_paths + test_paths
    umap_df = compute_umap(all_embeddings, all_labels, all_paths)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Evaluation Agent — VIPER Forensic Engine ===")
    evaluate()
    print("[EvalAgent] ✓ Evaluation complete.")
