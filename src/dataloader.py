"""
src/dataloader.py — VIPER Forensic Engine Data Layer
Phase: Track Alpha (Foundation) — Data Agent

Builds PyTorch DataLoaders for the AiArt vs RealArt CIFAKE-equivalent dataset.
Expected directory layout:
    data/Art/AiArtData/   ← AI-generated images  (label=1)
    data/Art/RealArt/     ← Real photographs      (label=0)

Usage:
    from src.dataloader import get_dataloaders, ArtDataset
    train_loader, val_loader, test_loader = get_dataloaders()

Exit contract: raises RuntimeError if directories are missing so that
downstream agents fail fast rather than silently.
"""

import sys
import random
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── Import project config ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    AI_ART_DIR, REAL_ART_DIR,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, SEED,
    LABEL_REAL, LABEL_AI,
    VAL_SPLIT, TEST_SPLIT,
)

# ── Supported image extensions ─────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    """Augmented transform for the training set."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


def get_val_transform() -> transforms.Compose:
    """Deterministic transform for val/test sets."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────────────────────────────────────

class ArtDataset(Dataset):
    """
    Binary classification dataset for AI-generated vs Real art images.

    Args:
        image_paths: List of Path objects to image files.
        labels:      Corresponding integer labels (0=REAL, 1=AI_GENERATED).
        transform:   torchvision transform to apply.
    """

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ):
        assert len(image_paths) == len(labels), "Paths/labels length mismatch"
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform or get_val_transform()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path  = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            # Return a black image on corrupt file rather than crashing
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=0)
        image = self.transform(image)
        return image, label, str(path)

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced datasets."""
        counts  = torch.zeros(2)
        for lbl in self.labels:
            counts[lbl] += 1
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return weights


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Discovery
# ─────────────────────────────────────────────────────────────────────────────

def _collect_images(directory: Path, label: int) -> Tuple[List[Path], List[int]]:
    """
    Recursively collect all valid images in a directory.

    Returns:
        Tuple of (image_paths, labels)

    Raises:
        RuntimeError: if the directory does not exist.
    """
    if not directory.exists():
        raise RuntimeError(
            f"[DataAgent] Required directory missing: {directory}\n"
            f"  → Cannot continue. Add images to {directory} and retry."
        )
    paths = [
        p for p in sorted(directory.rglob("*"))
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    ]
    if not paths:
        raise RuntimeError(
            f"[DataAgent] Directory exists but contains no images: {directory}"
        )
    return paths, [label] * len(paths)


def _validate_image(path: Path) -> bool:
    """Quick image validation without loading full pixel data."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    ai_dir:   Path = AI_ART_DIR,
    real_dir: Path = REAL_ART_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
    seed: int = SEED,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for binary art classification.

    Args:
        ai_dir:      Path to AI-generated images.
        real_dir:    Path to real images.
        batch_size:  Mini-batch size (default from config).
        num_workers: Parallel loading workers.
        val_split:   Fraction of data for validation.
        test_split:  Fraction of data for test.
        seed:        Random seed for reproducibility.
        verbose:     Print dataset statistics.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # ── Collect all images ────────────────────────────────────────────────────
    ai_paths,   ai_labels   = _collect_images(ai_dir,   LABEL_AI)
    real_paths, real_labels = _collect_images(real_dir, LABEL_REAL)

    all_paths  = ai_paths   + real_paths
    all_labels = ai_labels  + real_labels

    if verbose:
        print(f"[DataAgent] Found {len(ai_paths)} AI images, "
              f"{len(real_paths)} real images → {len(all_paths)} total")

    # ── Shuffle and split ──────────────────────────────────────────────────────
    rng = random.Random(seed)
    indices = list(range(len(all_paths)))
    rng.shuffle(indices)

    n_total = len(indices)
    n_test  = int(n_total * test_split)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_test - n_val

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    def _make_loader(idx_list, transform, shuffle: bool) -> DataLoader:
        paths  = [all_paths[i]  for i in idx_list]
        labels = [all_labels[i] for i in idx_list]
        ds = ArtDataset(paths, labels, transform=transform)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(shuffle),   # drop_last only for training
        )

    train_loader = _make_loader(train_idx, get_train_transform(), shuffle=True)
    val_loader   = _make_loader(val_idx,   get_val_transform(),   shuffle=False)
    test_loader  = _make_loader(test_idx,  get_val_transform(),   shuffle=False)

    if verbose:
        print(f"[DataAgent] Split → train={len(train_idx)}, "
              f"val={len(val_idx)}, test={len(test_idx)}")
        print(f"[DataAgent] DataLoaders ready. batch_size={batch_size}")

    return train_loader, val_loader, test_loader


def get_full_dataset(
    ai_dir:   Path = AI_ART_DIR,
    real_dir: Path = REAL_ART_DIR,
    transform=None,
) -> ArtDataset:
    """
    Return unsplit ArtDataset (used by EDA and visualization agents).

    AGENT_TASK: add support for loading CIFAKE-format folders (train/test subdirs)
    """
    ai_paths,   ai_labels   = _collect_images(ai_dir,   LABEL_AI)
    real_paths, real_labels = _collect_images(real_dir, LABEL_REAL)
    return ArtDataset(
        ai_paths + real_paths,
        ai_labels + real_labels,
        transform=transform or get_val_transform(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== DataAgent Verification ===")
    train, val, test = get_dataloaders(verbose=True)

    # Inspect one batch
    images, labels, paths = next(iter(train))
    print(f"Batch shape : {images.shape}")
    print(f"Labels      : {labels[:8].tolist()}")
    print(f"Sample path : {paths[0]}")
    print("[DataAgent] ✓ dataloader.py verification passed")
