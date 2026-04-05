"""
src/train.py — VIPER Forensic Engine: Training Loop
Phase: Track Gamma (Deep Learning) — Deep Learning Agent

Trains VIPERClassifier (EfficientNet-B0) with:
  - Adam optimizer, lr=1e-4
  - CosineAnnealingLR scheduler
  - Best model saved by validation F1 → checkpoints/best_model.pth
  - 10 epochs (locked by execution plan)

Usage:
    python src/train.py
    # → trains and saves checkpoints/best_model.pth
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, BEST_MODEL_PATH,
    RESULTS_DIR, SEED, BATCH_SIZE,
)
from src.dataloader import get_dataloaders
from src.model import build_model, VIPERClassifier

import random
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────────────────────────────────────
# Single Epoch Train / Val
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: VIPERClassifier,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run one training epoch.

    Returns:
        Dict with "loss" and "acc".

    AGENT_TASK: add mixed-precision (AMP) training
    AGENT_TASK: implement gradient clipping
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels, _ in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "acc":  correct    / max(total, 1),
    }


@torch.no_grad()
def validate(
    model: VIPERClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on a validation/test DataLoader.

    Returns:
        Dict with "loss", "acc", "f1".
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []
    total      = 0

    for images, labels, _ in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total += images.size(0)

    f1  = f1_score(all_labels, all_preds, zero_division=0)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(total, 1)

    return {
        "loss": total_loss / max(total, 1),
        "acc":  acc,
        "f1":   f1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    n_epochs:      int   = NUM_EPOCHS,
    lr:            float = LEARNING_RATE,
    device:        torch.device = DEVICE,
    checkpoint_path: Path = BEST_MODEL_PATH,
) -> VIPERClassifier:
    """
    Full training routine: build model → load data → train → save best checkpoint.

    Args:
        n_epochs:         Number of training epochs (locked: 10).
        lr:               Adam learning rate (locked: 1e-4).
        device:           Compute device.
        checkpoint_path:  Where to save best model.

    Returns:
        Trained VIPERClassifier.

    AGENT_TASK: add early stopping based on val_f1 plateau
    AGENT_TASK: add Weights & Biases experiment tracking
    AGENT_TASK: add label smoothing to CrossEntropyLoss
    """
    set_seed()
    print(f"\n{'='*60}")
    print(f"VIPER Training — EfficientNet-B0")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {n_epochs}")
    print(f"  LR       : {lr}")
    print(f"  Batch    : {BATCH_SIZE}")
    print(f"{'='*60}\n")

    # ── Dataloaders ────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders()

    # ── Model + Loss + Optimizer + Scheduler ──────────────────────────────────
    model     = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_f1  = 0.0
    history      = []

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{n_epochs}")

        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_stats   = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"  Train loss={train_stats['loss']:.4f}  acc={train_stats['acc']:.4f} | "
            f"Val loss={val_stats['loss']:.4f}  acc={val_stats['acc']:.4f}  "
            f"F1={val_stats['f1']:.4f}  lr={lr_now:.2e}  [{elapsed:.1f}s]"
        )

        record = {"epoch": epoch, **train_stats,
                  "val_loss": val_stats["loss"],
                  "val_acc":  val_stats["acc"],
                  "val_f1":   val_stats["f1"],
                  "lr":       lr_now}
        history.append(record)

        # ── Save best checkpoint ───────────────────────────────────────────────
        if val_stats["f1"] >= best_val_f1:
            best_val_f1 = val_stats["f1"]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_f1":           best_val_f1,
                "val_acc":          val_stats["acc"],
                "history":          history,
            }, checkpoint_path)
            print(f"  ★ New best F1={best_val_f1:.4f} → saved {checkpoint_path}")

    # ── Save training history ──────────────────────────────────────────────────
    history_path = RESULTS_DIR / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"\n[DLAgent] ✓ Training complete. Best val F1={best_val_f1:.4f}")
    print(f"[DLAgent] ✓ Checkpoint → {checkpoint_path}")
    print(f"[DLAgent] ✓ History    → {history_path}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Deep Learning Agent — VIPER Forensic Engine ===")
    train()
