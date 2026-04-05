"""
src/dataloader.py - VIPER Forensic Engine Data Layer

Phase 3.1 keeps the pipeline image-only by default while Phase 3.2 can opt into
late-fusion by attaching standardized EDA feature vectors from
results/feature_matrix.csv.
"""

import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    AI_ART_DIR,
    BATCH_SIZE,
    FEATURE_MATRIX_CSV,
    IMAGE_SIZE,
    LABEL_AI,
    LABEL_REAL,
    NUM_WORKERS,
    REAL_ART_DIR,
    SEED,
    STRICT_EDA_COVERAGE,
    TEST_SPLIT,
    USE_EDA_FEATURES,
    VAL_SPLIT,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _normalize_image_path(path: Path | str) -> str:
    return str(Path(path).resolve())


def get_train_transform() -> transforms.Compose:
    """Standard Phase 3.1 augmentation: flip + color jitter only."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transform() -> transforms.Compose:
    """Deterministic transform for validation and test sets."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class ArtDataset(Dataset):
    """
    Binary classification dataset for AI-generated vs real art images.

    When `feature_dim > 0`, each sample returns:
        (image_tensor, eda_features_tensor, label, path)

    Otherwise it returns the original Phase 3.1 tuple:
        (image_tensor, label, path)
    """

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        feature_lookup: Optional[Dict[str, torch.Tensor]] = None,
        feature_dim: int = 0,
    ):
        if len(image_paths) != len(labels):
            raise ValueError("Paths/labels length mismatch")
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or get_val_transform()
        self.feature_lookup = feature_lookup or {}
        self.feature_dim = int(feature_dim)
        self._zero_features = (
            torch.zeros(self.feature_dim, dtype=torch.float32)
            if self.feature_dim > 0
            else None
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=0)
        image = self.transform(image)

        if self.feature_dim > 0:
            feature_key = _normalize_image_path(path)
            eda_features = self.feature_lookup.get(feature_key)
            if eda_features is None:
                eda_features = self._zero_features
            return image, eda_features.clone(), label, str(path)

        return image, label, str(path)

    def get_class_weights(self) -> torch.Tensor:
        counts = torch.zeros(2)
        for label in self.labels:
            counts[label] += 1
        weights = 1.0 / counts
        return weights / weights.sum()


def _collect_images(directory: Path, label: int) -> Tuple[List[Path], List[int]]:
    if not directory.exists():
        raise RuntimeError(
            f"[DataAgent] Required directory missing: {directory}\n"
            f"  -> Cannot continue. Add images to {directory} and retry."
        )

    paths = [
        path for path in sorted(directory.rglob("*"))
        if path.suffix.lower() in IMAGE_EXTS and path.is_file()
    ]
    if not paths:
        raise RuntimeError(
            f"[DataAgent] Directory exists but contains no images: {directory}"
        )
    return paths, [label] * len(paths)


def _validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _load_feature_lookup(
    feature_csv: Path,
    train_paths: List[Path],
    required_paths: List[Path],
    strict: bool,
) -> Tuple[Dict[str, torch.Tensor], int]:
    if not feature_csv.exists():
        raise RuntimeError(
            f"[DataAgent] EDA feature matrix not found at {feature_csv}. "
            "Run src/eda.py before enabling hybrid fusion."
        )

    raw_lookup: Dict[str, np.ndarray] = {}
    with feature_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError("[DataAgent] EDA feature matrix is empty.")

        feature_columns = [
            column for column in reader.fieldnames
            if column not in {"image_path", "label"}
        ]
        if not feature_columns:
            raise RuntimeError(
                "[DataAgent] EDA feature matrix does not contain numeric features."
            )

        for row in reader:
            raw_path = row.get("image_path")
            if not raw_path:
                continue
            try:
                vector = np.asarray(
                    [float(row[column]) for column in feature_columns],
                    dtype=np.float32,
                )
            except (TypeError, ValueError):
                continue
            raw_lookup[_normalize_image_path(raw_path)] = vector

    if not raw_lookup:
        raise RuntimeError("[DataAgent] No usable EDA feature rows were loaded.")

    required_keys = [_normalize_image_path(path) for path in required_paths]
    missing_keys = [key for key in required_keys if key not in raw_lookup]
    if missing_keys and strict:
        missing_preview = "\n".join(f"    - {key}" for key in missing_keys[:5])
        raise RuntimeError(
            "[DataAgent] EDA coverage is incomplete for hybrid fusion.\n"
            f"  Missing rows: {len(missing_keys)} / {len(required_keys)}\n"
            "  Rebuild results/feature_matrix.csv for the full dataset before "
            "enabling USE_EDA_FEATURES.\n"
            f"{missing_preview}"
        )

    train_keys = [
        _normalize_image_path(path)
        for path in train_paths
        if _normalize_image_path(path) in raw_lookup
    ]
    if not train_keys:
        raise RuntimeError(
            "[DataAgent] No training images matched the EDA feature matrix."
        )

    train_matrix = np.stack([raw_lookup[key] for key in train_keys], axis=0)
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0)
    std[std < 1e-6] = 1.0

    feature_lookup = {
        key: torch.from_numpy((value - mean) / std).to(dtype=torch.float32)
        for key, value in raw_lookup.items()
    }

    if missing_keys and not strict:
        print(
            f"[DataAgent] Warning: {len(missing_keys)} images are missing EDA rows. "
            "Zero vectors will be used for those samples."
        )

    return feature_lookup, train_matrix.shape[1]


def get_dataloaders(
    ai_dir: Path = AI_ART_DIR,
    real_dir: Path = REAL_ART_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
    seed: int = SEED,
    verbose: bool = True,
    use_eda_features: bool = USE_EDA_FEATURES,
    feature_csv: Path = FEATURE_MATRIX_CSV,
    strict_eda_coverage: bool = STRICT_EDA_COVERAGE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for binary art classification.
    """
    effective_num_workers = 0 if sys.platform == "win32" else num_workers

    ai_paths, ai_labels = _collect_images(ai_dir, LABEL_AI)
    real_paths, real_labels = _collect_images(real_dir, LABEL_REAL)

    all_paths = ai_paths + real_paths
    all_labels = ai_labels + real_labels

    if verbose:
        print(
            f"[DataAgent] Found {len(ai_paths)} AI images, "
            f"{len(real_paths)} real images -> {len(all_paths)} total"
        )

    rng = random.Random(seed)
    indices = list(range(len(all_paths)))
    rng.shuffle(indices)

    n_total = len(indices)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    n_train = n_total - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    feature_lookup: Optional[Dict[str, torch.Tensor]] = None
    feature_dim = 0
    if use_eda_features:
        feature_lookup, feature_dim = _load_feature_lookup(
            feature_csv=feature_csv,
            train_paths=[all_paths[i] for i in train_idx],
            required_paths=all_paths,
            strict=strict_eda_coverage,
        )

    def _make_loader(
        idx_list: List[int],
        transform: transforms.Compose,
        shuffle: bool,
    ) -> DataLoader:
        paths = [all_paths[i] for i in idx_list]
        labels = [all_labels[i] for i in idx_list]
        dataset = ArtDataset(
            image_paths=paths,
            labels=labels,
            transform=transform,
            feature_lookup=feature_lookup,
            feature_dim=feature_dim,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=effective_num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
        )

    train_loader = _make_loader(train_idx, get_train_transform(), shuffle=True)
    val_loader = _make_loader(val_idx, get_val_transform(), shuffle=False)
    test_loader = _make_loader(test_idx, get_val_transform(), shuffle=False)

    if verbose:
        print(
            f"[DataAgent] Split -> train={len(train_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}"
        )
        if use_eda_features:
            print(
                f"[DataAgent] Hybrid fusion enabled with {feature_dim} EDA features."
            )
        print(
            f"[DataAgent] DataLoaders ready. batch_size={batch_size}, "
            f"num_workers={effective_num_workers}"
        )

    return train_loader, val_loader, test_loader


def get_full_dataset(
    ai_dir: Path = AI_ART_DIR,
    real_dir: Path = REAL_ART_DIR,
    transform=None,
) -> ArtDataset:
    ai_paths, ai_labels = _collect_images(ai_dir, LABEL_AI)
    real_paths, real_labels = _collect_images(real_dir, LABEL_REAL)
    return ArtDataset(
        ai_paths + real_paths,
        ai_labels + real_labels,
        transform=transform or get_val_transform(),
    )


if __name__ == "__main__":
    print("=== DataAgent Verification ===")
    train_loader, val_loader, test_loader = get_dataloaders(verbose=True)

    batch = next(iter(train_loader))
    if len(batch) == 4:
        images, eda_features, labels, paths = batch
        print(f"Batch shape   : {images.shape}")
        print(f"EDA shape     : {eda_features.shape}")
        print(f"Labels        : {labels[:8].tolist()}")
        print(f"Sample path   : {paths[0]}")
    else:
        images, labels, paths = batch
        print(f"Batch shape   : {images.shape}")
        print(f"Labels        : {labels[:8].tolist()}")
        print(f"Sample path   : {paths[0]}")

    print("[DataAgent] dataloader.py verification passed")
