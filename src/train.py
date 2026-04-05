"""
src/train.py - VIPER Forensic Engine: Training Loop

Phase 3.1 establishes a new baseline with ConvNeXt-Tiny and standard
CrossEntropyLoss. Phase 3.2 can be enabled later by setting USE_EDA_FEATURES=1
once the EDA feature matrix has full coverage.
"""

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    DEVICE,
    LEARNING_RATE,
    MODEL_NAME,
    NUM_EPOCHS,
    RESULTS_DIR,
    SEED,
    USE_EDA_FEATURES,
)
from src.dataloader import get_dataloaders
from src.model import VIPERClassifier, build_model


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _unpack_batch(
    batch,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if len(batch) == 4:
        images, eda_features, labels, _ = batch
        images = images.to(device, non_blocking=True)
        eda_features = eda_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        return images, eda_features, labels

    images, labels, _ = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return images, None, labels


def train_one_epoch(
    model: VIPERClassifier,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        images, eda_features, labels = _unpack_batch(batch, device)

        optimizer.zero_grad()
        logits = model(images, eda_features=eda_features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


@torch.no_grad()
def validate(
    model: VIPERClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        images, eda_features, labels = _unpack_batch(batch, device)

        logits = model(images, eda_features=eda_features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total += images.size(0)

    f1 = f1_score(all_labels, all_preds, zero_division=0)
    acc = sum(pred == label for pred, label in zip(all_preds, all_labels)) / max(total, 1)

    return {
        "loss": total_loss / max(total, 1),
        "acc": acc,
        "f1": f1,
    }


def train(
    n_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    device: torch.device = DEVICE,
    checkpoint_path: Path = BEST_MODEL_PATH,
    use_eda_features: bool = USE_EDA_FEATURES,
) -> VIPERClassifier:
    set_seed()
    print(f"\n{'=' * 60}")
    print(f"VIPER Training - {MODEL_NAME}")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {n_epochs}")
    print(f"  LR       : {lr}")
    print(f"  Batch    : {BATCH_SIZE}")
    print(f"  Hybrid   : {use_eda_features}")
    print(f"{'=' * 60}\n")

    train_loader, val_loader, _ = get_dataloaders(
        use_eda_features=use_eda_features,
    )
    eda_feature_dim = int(getattr(train_loader.dataset, "feature_dim", 0))

    model = build_model(
        device=device,
        eda_feature_dim=eda_feature_dim,
        pretrained=True,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    history = []

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{n_epochs}")

        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_stats = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"  Train loss={train_stats['loss']:.4f}  acc={train_stats['acc']:.4f} | "
            f"Val loss={val_stats['loss']:.4f}  acc={val_stats['acc']:.4f}  "
            f"F1={val_stats['f1']:.4f}  lr={lr_now:.2e}  [{elapsed:.1f}s]"
        )

        record = {
            "epoch": epoch,
            **train_stats,
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "val_f1": val_stats["f1"],
            "lr": lr_now,
        }
        history.append(record)

        if val_stats["f1"] >= best_val_f1:
            best_val_f1 = val_stats["f1"]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": MODEL_NAME,
                    "use_eda_features": use_eda_features,
                    "eda_feature_dim": eda_feature_dim,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_f1": best_val_f1,
                    "val_acc": val_stats["acc"],
                    "history": history,
                },
                checkpoint_path,
            )
            print(f"  * New best F1={best_val_f1:.4f} -> saved {checkpoint_path}")

    history_path = RESULTS_DIR / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"\n[DLAgent] Training complete. Best val F1={best_val_f1:.4f}")
    print(f"[DLAgent] Checkpoint -> {checkpoint_path}")
    print(f"[DLAgent] History    -> {history_path}")

    return model


if __name__ == "__main__":
    print("=== Deep Learning Agent - VIPER Forensic Engine ===")
    train()
