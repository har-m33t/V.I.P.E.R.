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
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, BEST_MODEL_PATH,
    RESULTS_DIR, SEED, BATCH_SIZE, UNFREEZE_BLOCKS,
)
from src.dataloader import get_dataloaders
from src.model import build_model, VIPERClassifier

import random
import numpy as np

HEAD_ONLY_EPOCHS = 3
BACKBONE_LR_DIVISOR = 10.0
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = None


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


def configure_training_stage(
    model: VIPERClassifier,
    unfreeze_blocks: int,
) -> None:
    """
    Freeze the EfficientNet backbone, then optionally unfreeze the last
    n MBConv blocks while always keeping the classifier head trainable.
    """
    for param in model.backbone.features.parameters():
        param.requires_grad = False

    for param in model.backbone.classifier.parameters():
        param.requires_grad = True

    if unfreeze_blocks <= 0:
        return

    feature_layers = list(model.backbone.features.children())
    for layer in feature_layers[-unfreeze_blocks:]:
        for param in layer.parameters():
            param.requires_grad = True


def build_stage_optimizer(
    model: VIPERClassifier,
    head_lr: float,
    unfreeze_blocks: int,
) -> optim.Optimizer:
    """
    Create an Adam optimizer for the active training stage.

    Stage 1: classifier only at the base learning rate.
    Stage 2: classifier at the base learning rate, last MBConv blocks at
    a 10x smaller learning rate.
    """
    classifier_params = list(model.backbone.classifier.parameters())

    if unfreeze_blocks <= 0:
        return optim.Adam(classifier_params, lr=head_lr)

    feature_layers = list(model.backbone.features.children())
    backbone_params = [
        param
        for layer in feature_layers[-unfreeze_blocks:]
        for param in layer.parameters()
        if param.requires_grad
    ]

    param_groups = [{"params": classifier_params, "lr": head_lr}]
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": head_lr / BACKBONE_LR_DIVISOR,
            }
        )

    return optim.Adam(param_groups)


def build_stage_scheduler(
    optimizer: optim.Optimizer,
    stage_epochs: int,
) -> CosineAnnealingLR:
    """Attach a cosine schedule sized to the current training stage."""
    return CosineAnnealingLR(optimizer, T_max=max(stage_epochs, 1), eta_min=1e-6)


class FocalLoss(nn.Module):
    """
    Multi-class focal loss over class-index targets.

    This keeps the existing two-logit classifier API intact while down-weighting
    easy examples and emphasizing borderline mistakes.
    """

    def __init__(self, gamma: float = FOCAL_GAMMA, alpha: float | None = FOCAL_ALPHA):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = torch.where(
                targets == 1,
                torch.full_like(ce_loss, self.alpha),
                torch.full_like(ce_loss, 1.0 - self.alpha),
            )
            focal_weight = focal_weight * alpha_t

        return (focal_weight * ce_loss).mean()


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
    print(f"  Loss     : FocalLoss(gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA})")
    print(f"  Stage 1  : classifier only for {min(HEAD_ONLY_EPOCHS, n_epochs)} epochs")
    print(f"  Stage 2  : last {UNFREEZE_BLOCKS} MBConv blocks @ {lr / BACKBONE_LR_DIVISOR:.2e}")
    print(f"{'='*60}\n")

    # ── Dataloaders ────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders()

    # ── Model + Loss + Optimizer + Scheduler ──────────────────────────────────
    model     = build_model(device)
    criterion = FocalLoss()
    head_only_epochs = min(HEAD_ONLY_EPOCHS, n_epochs)
    fine_tune_epochs = max(n_epochs - head_only_epochs, 0)

    current_unfreeze_blocks = 0
    current_stage = "classifier_head"
    configure_training_stage(model, current_unfreeze_blocks)
    optimizer = build_stage_optimizer(model, lr, current_unfreeze_blocks)
    scheduler = build_stage_scheduler(optimizer, head_only_epochs)
    trainable, total = model.count_trainable_params()
    print(
        f"[DLAgent] Stage '{current_stage}' active: "
        f"{trainable:,} / {total:,} parameters trainable"
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_f1  = 0.0
    history      = []

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        if epoch == head_only_epochs + 1 and fine_tune_epochs > 0:
            current_unfreeze_blocks = UNFREEZE_BLOCKS
            current_stage = "top_mbconv_finetune"
            configure_training_stage(model, current_unfreeze_blocks)
            optimizer = build_stage_optimizer(model, lr, current_unfreeze_blocks)
            scheduler = build_stage_scheduler(optimizer, fine_tune_epochs)
            trainable, total = model.count_trainable_params()
            print(
                f"\n[DLAgent] Switching to '{current_stage}': "
                f"unfreezing last {UNFREEZE_BLOCKS} MBConv blocks "
                f"at {lr / BACKBONE_LR_DIVISOR:.2e}"
            )
            print(
                f"[DLAgent] {trainable:,} / {total:,} parameters trainable after unfreezing"
            )

        print(f"\nEpoch {epoch}/{n_epochs}")

        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_stats   = validate(model, val_loader, criterion, device)
        scheduler.step()

        head_lr_now = optimizer.param_groups[0]["lr"]
        backbone_lr_now = (
            optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else None
        )
        elapsed = time.time() - t0

        status = (
            f"  Train loss={train_stats['loss']:.4f}  acc={train_stats['acc']:.4f} | "
            f"Val loss={val_stats['loss']:.4f}  acc={val_stats['acc']:.4f}  "
            f"F1={val_stats['f1']:.4f}  head_lr={head_lr_now:.2e}"
        )
        if backbone_lr_now is not None:
            status += f"  backbone_lr={backbone_lr_now:.2e}"
        status += f"  [{elapsed:.1f}s]"
        print(status)

        record = {"epoch": epoch, **train_stats,
                  "val_loss": val_stats["loss"],
                  "val_acc":  val_stats["acc"],
                  "val_f1":   val_stats["f1"],
                  "stage":    current_stage,
                  "lr":       head_lr_now,
                  "head_lr":  head_lr_now}
        if backbone_lr_now is not None:
            record["backbone_lr"] = backbone_lr_now
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
