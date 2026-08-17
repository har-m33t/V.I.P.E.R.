"""
src/train.py - VIPER Forensic Engine: Training Loop

The image-only baseline uses ConvNeXt-Tiny with standard CrossEntropyLoss.
Forensic fusion is enabled by setting USE_EDA_FEATURES=1
once the EDA feature matrix has full coverage.

Runs on CPU, one GPU, or many GPUs from the same entrypoint:

    python src/train.py                          # CPU / single GPU
    torchrun --nproc_per_node=2 src/train.py     # 2-GPU DistributedDataParallel

Hyperparameters are unchanged from the CPU baseline. The GPU path only adds
mixed precision, channels-last memory layout, and data-parallel sharding —
none of which alter the optimisation problem being solved.
"""

import json
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    AMP_DTYPE,
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CHANNELS_LAST,
    CLASS_WEIGHTED,
    CUDNN_BENCHMARK,
    DEVICE,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    LEARNING_RATE,
    MAX_SAMPLES,
    MODEL_NAME,
    NUM_EPOCHS,
    RUN_SUMMARY_JSON,
    SEED,
    TORCH_COMPILE,
    TRAINING_HISTORY_JSON,
    USE_EDA_FEATURES,
)
from src.dataloader import get_dataloaders
from src.model import VIPERConvNeXt, build_model


# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_distributed() -> Tuple[bool, int, int, int]:
    """
    Initialise the process group when launched under torchrun.

    Returns (is_distributed, rank, local_rank, world_size). Falls back to a
    single-process (1, 0, 0, 1) configuration when the torchrun environment
    variables are absent, so the exact same script still runs on CPU.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if world_size <= 1 or local_rank < 0:
        return False, 0, 0, 1

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(minutes=30),
    )
    rank = dist.get_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def _reduce_sum(value: float, device: torch.device, is_distributed: bool) -> float:
    """Sum a scalar across every rank."""
    if not is_distributed:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _gather_predictions(
    preds: torch.Tensor,
    labels: torch.Tensor,
    is_distributed: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenate per-rank prediction tensors onto every rank.

    Ranks can hold different numbers of samples (DistributedSampler pads, but
    the val loader does not drop its tail), so sizes are exchanged first.
    """
    if not is_distributed:
        return preds, labels

    local_size = torch.tensor([preds.numel()], dtype=torch.long, device=preds.device)
    sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, local_size)
    max_size = int(max(int(s.item()) for s in sizes))

    def _pad(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == max_size:
            return tensor
        padding = torch.zeros(
            max_size - tensor.numel(), dtype=tensor.dtype, device=tensor.device
        )
        return torch.cat([tensor, padding])

    padded_preds = _pad(preds)
    padded_labels = _pad(labels)

    gathered_preds = [torch.zeros_like(padded_preds) for _ in sizes]
    gathered_labels = [torch.zeros_like(padded_labels) for _ in sizes]
    dist.all_gather(gathered_preds, padded_preds)
    dist.all_gather(gathered_labels, padded_labels)

    # Trim each rank's contribution back to its true length before merging.
    out_preds = torch.cat(
        [chunk[: int(size.item())] for chunk, size in zip(gathered_preds, sizes)]
    )
    out_labels = torch.cat(
        [chunk[: int(size.item())] for chunk, size in zip(gathered_labels, sizes)]
    )
    return out_preds, out_labels


# ─────────────────────────────────────────────────────────────────────────────
# Mixed precision
# ─────────────────────────────────────────────────────────────────────────────

def resolve_amp(device: torch.device) -> Tuple[bool, Optional[torch.dtype]]:
    """
    Decide whether to use autocast and in which dtype.

    bf16 is preferred on Ampere and newer: it has the dynamic range of fp32, so
    it needs no gradient scaling and cannot silently overflow.
    """
    if device.type != "cuda" or AMP_DTYPE == "off":
        return False, None

    if AMP_DTYPE == "bf16":
        return True, torch.bfloat16
    if AMP_DTYPE == "fp16":
        return True, torch.float16

    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def set_seed(seed: int = SEED, rank: int = 0) -> None:
    # Offsetting by rank keeps augmentation streams from being identical across
    # GPUs while remaining fully reproducible for a given world size.
    effective = seed + rank
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    torch.backends.cudnn.deterministic = not CUDNN_BENCHMARK
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK


def _unpack_batch(
    batch,
    device: torch.device,
    channels_last: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if len(batch) == 4:
        images, eda_features, labels, _ = batch
        images = images.to(device, non_blocking=True)
        eda_features = eda_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
    else:
        images, labels, _ = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        eda_features = None

    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images, eda_features, labels


# ─────────────────────────────────────────────────────────────────────────────
# Epoch loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_enabled: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    channels_last: bool = False,
    is_distributed: bool = False,
    show_progress: bool = True,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    autocast = (
        torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_enabled
        else nullcontext()
    )

    iterator = tqdm(loader, desc="  Train", leave=False, disable=not show_progress)
    for batch in iterator:
        images, eda_features, labels = _unpack_batch(batch, device, channels_last)

        optimizer.zero_grad(set_to_none=True)
        with autocast:
            logits = model(images, eda_features=eda_features)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    total_loss = _reduce_sum(total_loss, device, is_distributed)
    correct = _reduce_sum(float(correct), device, is_distributed)
    total = _reduce_sum(float(total), device, is_distributed)

    return {
        "loss": total_loss / max(total, 1.0),
        "acc": correct / max(total, 1.0),
        "n_samples": int(total),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    channels_last: bool = False,
    is_distributed: bool = False,
    show_progress: bool = True,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    preds_chunks = []
    labels_chunks = []

    autocast = (
        torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_enabled
        else nullcontext()
    )

    iterator = tqdm(loader, desc="  Val  ", leave=False, disable=not show_progress)
    for batch in iterator:
        images, eda_features, labels = _unpack_batch(batch, device, channels_last)

        with autocast:
            logits = model(images, eda_features=eda_features)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        preds_chunks.append(logits.argmax(dim=1).detach())
        labels_chunks.append(labels.detach())

    local_preds = (
        torch.cat(preds_chunks) if preds_chunks
        else torch.empty(0, dtype=torch.long, device=device)
    )
    local_labels = (
        torch.cat(labels_chunks) if labels_chunks
        else torch.empty(0, dtype=torch.long, device=device)
    )
    all_preds, all_labels = _gather_predictions(local_preds, local_labels, is_distributed)

    total_loss = _reduce_sum(total_loss, device, is_distributed)
    total = _reduce_sum(float(total), device, is_distributed)

    preds_np = all_preds.cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    f1 = f1_score(labels_np, preds_np, zero_division=0) if len(labels_np) else 0.0
    acc = float((preds_np == labels_np).mean()) if len(labels_np) else 0.0

    return {
        "loss": total_loss / max(total, 1.0),
        "acc": acc,
        "f1": float(f1),
        "n_samples": int(len(labels_np)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def train(
    n_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    checkpoint_path: Path = BEST_MODEL_PATH,
    use_eda_features: bool = USE_EDA_FEATURES,
    patience: int = EARLY_STOP_PATIENCE,
) -> Optional[VIPERConvNeXt]:
    is_distributed, rank, local_rank, world_size = setup_distributed()
    device = (
        torch.device(f"cuda:{local_rank}")
        if is_distributed and torch.cuda.is_available()
        else DEVICE
    )
    main = is_main_process(rank)

    set_seed(rank=rank)
    amp_enabled, amp_dtype = resolve_amp(device)
    channels_last = CHANNELS_LAST and device.type == "cuda"
    # GradScaler is only needed for fp16; bf16 has enough exponent range.
    scaler = (
        torch.amp.GradScaler("cuda")
        if amp_enabled and amp_dtype == torch.float16
        else None
    )

    if main:
        print(f"\n{'=' * 62}")
        print(f"VIPER Training - {MODEL_NAME}")
        print(f"  Device        : {device}")
        print(f"  World size    : {world_size} ({'DDP' if is_distributed else 'single process'})")
        print(f"  Epochs (max)  : {n_epochs}")
        print(f"  Early stop    : patience={patience} on val F1")
        print(f"  LR            : {lr}")
        print(f"  Batch         : {BATCH_SIZE}/gpu -> {BATCH_SIZE * world_size} global")
        print(f"  AMP           : {amp_dtype if amp_enabled else 'disabled'}")
        print(f"  channels_last : {channels_last}")
        print(f"  torch.compile : {TORCH_COMPILE}")
        print(f"  Max samples   : {MAX_SAMPLES or 'unlimited (full dataset)'}")
        print(f"  Hybrid EDA    : {use_eda_features}")
        print(f"{'=' * 62}\n")

    train_loader, val_loader, _ = get_dataloaders(
        use_eda_features=use_eda_features,
        verbose=main,
        distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )
    eda_feature_dim = int(getattr(train_loader.dataset, "feature_dim", 0))

    model = build_model(
        device=device,
        eda_feature_dim=eda_feature_dim,
        pretrained=True,
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    base_model = model
    if TORCH_COMPILE:
        model = torch.compile(model)
    if is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
        )

    # Default is plain CrossEntropyLoss, matching the CPU baseline exactly.
    # CLASS_WEIGHTED=1 reweights by inverse class frequency, which only matters
    # on a heavily imbalanced corpus — there it stops the model from collapsing
    # onto the majority class.
    class_weights = None
    if CLASS_WEIGHTED:
        counts = torch.bincount(
            torch.tensor(train_loader.dataset.labels), minlength=2
        ).float()
        class_weights = (counts.sum() / (2.0 * counts.clamp(min=1))).to(device)
        if main:
            print(f"[DLAgent] Class-weighted loss: counts={counts.tolist()}, "
                  f"weights={[round(w, 4) for w in class_weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    stopped_early = False
    wall_start = time.time()

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        if main:
            print(f"\nEpoch {epoch}/{n_epochs}")

        # Reshuffles each rank's shard differently every epoch.
        if is_distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            is_distributed=is_distributed,
            show_progress=main,
        )
        val_stats = validate(
            model, val_loader, criterion, device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            is_distributed=is_distributed,
            show_progress=main,
        )
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        throughput = train_stats["n_samples"] / max(elapsed, 1e-6)

        if main:
            print(
                f"  Train loss={train_stats['loss']:.4f}  acc={train_stats['acc']:.4f} | "
                f"Val loss={val_stats['loss']:.4f}  acc={val_stats['acc']:.4f}  "
                f"F1={val_stats['f1']:.4f}  lr={lr_now:.2e}  "
                f"[{elapsed:.1f}s, {throughput:.0f} img/s]"
            )

        history.append({
            "epoch": epoch,
            "loss": train_stats["loss"],
            "acc": train_stats["acc"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "val_f1": val_stats["f1"],
            "lr": lr_now,
            "epoch_seconds": elapsed,
            "images_per_second": throughput,
        })

        improved = val_stats["f1"] > best_val_f1 + EARLY_STOP_MIN_DELTA
        if improved:
            best_val_f1 = val_stats["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            if main:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_name": MODEL_NAME,
                        "use_eda_features": use_eda_features,
                        "eda_feature_dim": eda_feature_dim,
                        "model_state_dict": base_model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_f1": best_val_f1,
                        "val_acc": val_stats["acc"],
                        "history": history,
                    },
                    checkpoint_path,
                )
                print(f"  * New best F1={best_val_f1:.4f} -> saved {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if main:
                print(
                    f"  no improvement ({epochs_without_improvement}/{patience}) "
                    f"— best F1={best_val_f1:.4f} @ epoch {best_epoch}"
                )

        # Every rank must reach the same stop decision or DDP will deadlock.
        should_stop = epochs_without_improvement >= patience
        if is_distributed:
            flag = torch.tensor([1.0 if should_stop else 0.0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            should_stop = bool(flag.item() > 0)

        if should_stop:
            stopped_early = True
            if main:
                print(
                    f"\n[DLAgent] Early stop: val F1 flat for {patience} epochs. "
                    f"Best F1={best_val_f1:.4f} @ epoch {best_epoch}."
                )
            break

    total_wall = time.time() - wall_start

    if main:
        TRAINING_HISTORY_JSON.write_text(json.dumps(history, indent=2))
        summary = {
            "model": MODEL_NAME,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "world_size": world_size,
            "distributed": is_distributed,
            "amp_dtype": str(amp_dtype) if amp_enabled else "disabled",
            "channels_last": channels_last,
            "torch_compile": TORCH_COMPILE,
            "batch_size_per_gpu": BATCH_SIZE,
            "global_batch_size": BATCH_SIZE * world_size,
            "learning_rate": lr,
            "seed": SEED,
            "max_samples": MAX_SAMPLES or None,
            "class_weighted": CLASS_WEIGHTED,
            "epochs_configured": n_epochs,
            "epochs_run": len(history),
            "stopped_early": stopped_early,
            "early_stop_patience": patience,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "train_samples": train_stats["n_samples"],
            "val_samples": val_stats["n_samples"],
            "total_wall_seconds": total_wall,
            "mean_epoch_seconds": (
                sum(h["epoch_seconds"] for h in history) / len(history) if history else None
            ),
            "mean_images_per_second": (
                sum(h["images_per_second"] for h in history) / len(history) if history else None
            ),
        }
        RUN_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

        print(f"\n[DLAgent] Training complete in {total_wall / 60:.1f} min "
              f"over {len(history)} epoch(s).")
        print(f"[DLAgent] Best val F1={best_val_f1:.4f} (epoch {best_epoch})")
        print(f"[DLAgent] Checkpoint -> {checkpoint_path}")
        print(f"[DLAgent] History    -> {TRAINING_HISTORY_JSON}")
        print(f"[DLAgent] Summary    -> {RUN_SUMMARY_JSON}")

    cleanup_distributed(is_distributed)
    return base_model if main else None


if __name__ == "__main__":
    print("=== Deep Learning Agent - VIPER Forensic Engine ===")
    train()
