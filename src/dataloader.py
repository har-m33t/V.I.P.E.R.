"""
src/dataloader.py - VIPER Forensic Engine Data Layer

Phase 3.1 keeps the pipeline image-only by default while Phase 3.2 can opt into
late-fusion by attaching standardized EDA feature vectors from
results/feature_matrix.csv.
"""

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    AI_ART_DIR,
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    FEATURE_MATRIX_CSV,
    IMAGE_EMBED_DIM,
    IMAGE_SIZE,
    LABEL_AI,
    LABEL_REAL,
    NUM_WORKERS,
    REAL_ART_DIR,
    RESULTS_DIR,
    SEED,
    STRICT_EDA_COVERAGE,
    TEST_SPLIT,
    USE_EDA_FEATURES,
    VAL_SPLIT,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
MAX_SAMPLES_FAST_TRACK = 10000
HIGH_CONFIDENCE_THRESHOLD = 0.80


def _normalize_image_path(path: Path | str) -> str:
    return str(Path(path).resolve())


def _prediction_name(label: int) -> str:
    return str(CLASS_NAMES.get(int(label), int(label)))


def _error_type(label: int, prediction: int) -> str:
    if int(label) == int(prediction):
        return "correct"
    if int(label) == LABEL_REAL and int(prediction) == LABEL_AI:
        return "false_positive"
    return "false_negative"


def _confidence_bucket(confidence: float) -> str:
    if float(confidence) >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    return "low"


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

    if len(indices) > MAX_SAMPLES_FAST_TRACK:
        indices = indices[:MAX_SAMPLES_FAST_TRACK]
        if verbose:
            print(f"[DataAgent] ⚡ FAST-TRACK ENGAGED: Sub-sampled to {MAX_SAMPLES_FAST_TRACK} images.")

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


def _compute_projection(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.10,
) -> tuple[np.ndarray, str]:
    if len(embeddings) == 0:
        return np.empty((0, 2), dtype=np.float32), "none"
    if len(embeddings) == 1:
        return np.zeros((1, 2), dtype=np.float32), "degenerate"

    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=max(2, min(int(n_neighbors), len(embeddings) - 1)),
            min_dist=float(min_dist),
            n_components=2,
            metric="cosine",
            random_state=SEED,
            verbose=False,
        )
        coords = reducer.fit_transform(embeddings)
        return coords.astype(np.float32), "umap"
    except Exception:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=SEED).fit_transform(embeddings)
        return coords.astype(np.float32), "pca_fallback"


def _compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    ai_probabilities: np.ndarray,
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "n_samples": int(len(labels)),
    }
    try:
        metrics["auc_roc"] = float(roc_auc_score(labels, ai_probabilities))
    except ValueError:
        metrics["auc_roc"] = 0.0
    return metrics


def _build_error_breakdown(predictions_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if predictions_df.empty:
        return []

    errors_df = predictions_df[predictions_df["error_type"] != "correct"].copy()
    if errors_df.empty:
        return []

    grouped = (
        errors_df.groupby(["error_type", "confidence_bucket"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["error_type", "confidence_bucket"])
    )
    return grouped.to_dict(orient="records")


@torch.no_grad()
def precompute_validation_artifacts(
    checkpoint_path: Path = BEST_MODEL_PATH,
    predictions_csv: Path = RESULTS_DIR / "validation_predictions.csv",
    predictions_json: Path = RESULTS_DIR / "validation_predictions.json",
    embeddings_csv: Path = RESULTS_DIR / "validation_embeddings.csv",
    summary_json: Path = RESULTS_DIR / "validation_summary.json",
) -> Dict[str, Any]:
    """
    Precompute validation-set predictions, confidences, and 768D embeddings.

    The dashboard can read these artifacts directly instead of running model
    inference on every load.
    """
    from src.model import load_checkpoint

    model = load_checkpoint(checkpoint_path=checkpoint_path, device=DEVICE)
    if model is None:
        raise FileNotFoundError(
            f"[DataAgent] Unable to load checkpoint from {checkpoint_path}"
        )

    _, val_loader, _ = get_dataloaders(verbose=False)
    model.eval()

    prediction_rows: List[Dict[str, Any]] = []
    embedding_rows: List[Dict[str, Any]] = []
    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_ai_probabilities: List[float] = []
    all_embeddings: List[np.ndarray] = []

    print("[DataAgent] Precomputing validation artifacts ...")
    for batch in val_loader:
        if len(batch) == 4:
            images, eda_features, labels, paths = batch
            eda_features = eda_features.to(DEVICE, non_blocking=True)
        else:
            images, labels, paths = batch
            eda_features = None

        images = images.to(DEVICE, non_blocking=True)
        logits = model(images, eda_features=eda_features)
        probabilities = F.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        predicted_confidences = probabilities.max(dim=1).values
        ai_probabilities = probabilities[:, LABEL_AI]
        embeddings = model.get_embedding(images)

        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        predictions_np = predictions.detach().cpu().numpy().astype(np.int64)
        confidence_np = predicted_confidences.detach().cpu().numpy().astype(np.float32)
        ai_prob_np = ai_probabilities.detach().cpu().numpy().astype(np.float32)
        embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)

        for idx, raw_path in enumerate(paths):
            image_path = _normalize_image_path(raw_path)
            filename = Path(raw_path).name
            label = int(labels_np[idx])
            prediction = int(predictions_np[idx])
            confidence = float(confidence_np[idx])
            ai_probability = float(ai_prob_np[idx])
            error_type = _error_type(label, prediction)

            prediction_rows.append({
                "split": "validation",
                "image_path": image_path,
                "filename": filename,
                "label": label,
                "label_name": _prediction_name(label),
                "predicted_label": prediction,
                "predicted_name": _prediction_name(prediction),
                "ai_probability": ai_probability,
                "predicted_confidence": confidence,
                "confidence_bucket": _confidence_bucket(confidence),
                "correct": bool(label == prediction),
                "error_type": error_type,
            })

            embedding_row: Dict[str, Any] = {
                "image_path": image_path,
                "filename": filename,
            }
            embedding_row.update({
                f"emb_{embedding_index:03d}": float(embeddings_np[idx, embedding_index])
                for embedding_index in range(IMAGE_EMBED_DIM)
            })
            embedding_rows.append(embedding_row)

        all_labels.extend(labels_np.tolist())
        all_predictions.extend(predictions_np.tolist())
        all_ai_probabilities.extend(ai_prob_np.tolist())
        all_embeddings.extend(embeddings_np)

    predictions_df = pd.DataFrame(prediction_rows)
    embeddings_df = pd.DataFrame(embedding_rows)
    embedding_matrix = np.asarray(all_embeddings, dtype=np.float32)
    coords, projection_method = _compute_projection(embedding_matrix)
    if len(coords) == len(predictions_df):
        predictions_df["umap_x"] = coords[:, 0]
        predictions_df["umap_y"] = coords[:, 1]
    else:
        predictions_df["umap_x"] = np.nan
        predictions_df["umap_y"] = np.nan

    predictions_df = predictions_df.sort_values(
        ["error_type", "predicted_confidence"],
        ascending=[True, False],
    ).reset_index(drop=True)
    embeddings_df = embeddings_df.sort_values("filename").reset_index(drop=True)

    metrics = _compute_metrics(
        labels=np.asarray(all_labels, dtype=np.int64),
        predictions=np.asarray(all_predictions, dtype=np.int64),
        ai_probabilities=np.asarray(all_ai_probabilities, dtype=np.float32),
    )
    error_breakdown = _build_error_breakdown(predictions_df)

    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_csv, index=False)
    embeddings_df.to_csv(embeddings_csv, index=False)

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "n_samples": int(len(predictions_df)),
        "embedding_dim": IMAGE_EMBED_DIM,
        "projection_method": projection_method,
        "confidence_threshold_high": HIGH_CONFIDENCE_THRESHOLD,
        "metrics": metrics,
        "error_breakdown": error_breakdown,
        "artifacts": {
            "predictions_csv": str(predictions_csv),
            "embeddings_csv": str(embeddings_csv),
        },
    }
    payload = {
        **summary_payload,
        "records": predictions_df.to_dict(orient="records"),
    }
    predictions_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"[DataAgent] Validation predictions -> {predictions_csv}")
    print(f"[DataAgent] Validation embeddings  -> {embeddings_csv}")
    print(f"[DataAgent] Validation manifest    -> {predictions_json}")
    print(f"[DataAgent] Validation summary     -> {summary_json}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIPER dataloader utilities")
    parser.add_argument(
        "--precompute-validation",
        action="store_true",
        help="Write validation predictions/confidences/embeddings for the Streamlit dashboard.",
    )
    args = parser.parse_args()

    if args.precompute_validation:
        precompute_validation_artifacts()
    else:
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
