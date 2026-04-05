"""
src/visualize_comparison.py - VIPER vs Logistic Regression comparison visuals.

This module builds presentation-ready comparison charts between the saved
VIPERClassifier checkpoint and a reproducibly reconstructed Logistic Regression
baseline on the same train/test split.

Outputs:
    results/roc_overlay.png
    results/calibration_curve.png
    results/radar_comparison.png
    results/error_venn.png
    results/blur_robustness_sweep.png
"""

import csv
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib_venn import venn2
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.baseline import LogisticRegression as _unused  # noqa: F401
from src.config import (
    BATCH_SIZE,
    DEVICE,
    FEATURE_MATRIX_CSV,
    IMAGE_SIZE,
    RESULTS_DIR,
    SEED,
)
from src.dataloader import get_dataloaders, get_val_transform
from src.eda import (
    compute_color_entropy,
    compute_pixel_stats,
    edge_density,
    fft_analysis,
    glcm_features,
    noise_residuals,
)
from src.model import load_checkpoint

warnings.filterwarnings("ignore", category=UserWarning)


ROC_OVERLAY_PNG = RESULTS_DIR / "roc_overlay.png"
CALIBRATION_CURVE_PNG = RESULTS_DIR / "calibration_curve.png"
RADAR_COMPARISON_PNG = RESULTS_DIR / "radar_comparison.png"
ERROR_VENN_PNG = RESULTS_DIR / "error_venn.png"
BLUR_SWEEP_PNG = RESULTS_DIR / "blur_robustness_sweep.png"

VIPER_COLOR = "#0F766E"
BASELINE_COLOR = "#C2410C"
GRID_COLOR = "#D7E3EA"
BACKGROUND_COLOR = "#F6F3ED"
PANEL_COLOR = "#FFFEFA"


def set_plot_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": 220,
        "savefig.dpi": 220,
        "figure.facecolor": BACKGROUND_COLOR,
        "axes.facecolor": PANEL_COLOR,
        "axes.edgecolor": "#94A3B8",
        "axes.labelcolor": "#0F172A",
        "axes.titleweight": "bold",
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "xtick.color": "#1E293B",
        "ytick.color": "#1E293B",
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.9,
        "legend.frameon": True,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#CBD5E1",
    })


def load_rgb_image(path: Path) -> np.ndarray:
    try:
        image = Image.open(path).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        return np.asarray(image, dtype=np.uint8)
    except Exception:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


def blur_rgb_image(image_rgb: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return image_rgb
    return cv2.GaussianBlur(image_rgb, (kernel_size, kernel_size), sigmaX=0)


def feature_dict_from_rgb(image_rgb: np.ndarray) -> Dict[str, float]:
    features: Dict[str, float] = {}
    features.update(compute_pixel_stats(image_rgb))
    features.update(fft_analysis(image_rgb))
    features.update(compute_color_entropy(image_rgb))
    features.update(noise_residuals(image_rgb))
    features.update(glcm_features(image_rgb))
    features.update(edge_density(image_rgb))
    return features


def vector_from_feature_dict(
    feature_dict: Dict[str, float],
    feature_columns: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [float(feature_dict[column]) for column in feature_columns],
        dtype=np.float32,
    )


def read_feature_matrix(
    feature_csv: Path,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    if not feature_csv.exists():
        raise FileNotFoundError(
            f"Missing {feature_csv}. Run src/eda.py before comparison."
        )

    with feature_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError("feature_matrix.csv is empty.")
        feature_columns = [
            column for column in reader.fieldnames
            if column not in {"image_path", "label"}
        ]
        lookup: Dict[str, np.ndarray] = {}
        for row in reader:
            image_path = row.get("image_path")
            if not image_path:
                continue
            resolved_path = str(Path(image_path).resolve())
            lookup[resolved_path] = np.asarray(
                [float(row[column]) for column in feature_columns],
                dtype=np.float32,
            )

    return feature_columns, lookup


def ensure_feature_lookup(
    feature_lookup: Dict[str, np.ndarray],
    required_paths: Sequence[Path],
    feature_columns: Sequence[str],
) -> Dict[str, np.ndarray]:
    missing_paths = [
        path for path in required_paths
        if str(path.resolve()) not in feature_lookup
    ]
    if missing_paths:
        print(
            f"[CompareViz] Backfilling {len(missing_paths)} missing EDA rows "
            "for aligned baseline evaluation."
        )

    for path in tqdm(missing_paths, desc="Backfill EDA", leave=False):
        image_rgb = load_rgb_image(path)
        feature_lookup[str(path.resolve())] = vector_from_feature_dict(
            feature_dict_from_rgb(image_rgb),
            feature_columns,
        )

    return feature_lookup


def collect_split_data() -> Tuple[List[Path], List[int], List[Path], List[int]]:
    train_loader, _, test_loader = get_dataloaders(verbose=False)
    train_paths = list(train_loader.dataset.image_paths)
    train_labels = list(train_loader.dataset.labels)
    test_paths = list(test_loader.dataset.image_paths)
    test_labels = list(test_loader.dataset.labels)
    return train_paths, train_labels, test_paths, test_labels


def reconstruct_baseline_model(
    train_paths: Sequence[Path],
    train_labels: Sequence[int],
    feature_lookup: Dict[str, np.ndarray],
) -> Pipeline:
    train_matrix = np.stack(
        [feature_lookup[str(path.resolve())] for path in train_paths],
        axis=0,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=SEED,
        )),
    ])
    pipeline.fit(train_matrix, np.asarray(train_labels))
    return pipeline


@torch.no_grad()
def infer_viper(
    model,
    path_to_image: Dict[str, np.ndarray],
    paths: Sequence[Path],
    labels: Sequence[int],
    blur_kernel: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = get_val_transform()
    probabilities: List[float] = []
    predictions: List[int] = []
    labels_array = np.asarray(labels, dtype=np.int64)

    model.eval()
    for start in tqdm(range(0, len(paths), BATCH_SIZE), desc=f"VIPER blur={blur_kernel}", leave=False):
        batch_paths = paths[start:start + BATCH_SIZE]
        batch_images = []
        for path in batch_paths:
            image_rgb = path_to_image[str(path.resolve())]
            image_rgb = blur_rgb_image(image_rgb, blur_kernel)
            image = Image.fromarray(image_rgb)
            batch_images.append(transform(image))
        image_tensor = torch.stack(batch_images).to(DEVICE)
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        probabilities.extend(probs.tolist())
        predictions.extend(preds.tolist())

    return labels_array, np.asarray(predictions, dtype=np.int64), np.asarray(probabilities, dtype=np.float32)


def infer_baseline(
    pipeline: Pipeline,
    path_to_image: Dict[str, np.ndarray],
    paths: Sequence[Path],
    labels: Sequence[int],
    feature_lookup: Dict[str, np.ndarray],
    feature_columns: Sequence[str],
    blur_kernel: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_rows = []
    for path in tqdm(paths, desc=f"Baseline blur={blur_kernel}", leave=False):
        resolved_path = str(path.resolve())
        if blur_kernel <= 1:
            feature_rows.append(feature_lookup[resolved_path])
            continue
        image_rgb = blur_rgb_image(path_to_image[resolved_path], blur_kernel)
        feature_rows.append(
            vector_from_feature_dict(
                feature_dict_from_rgb(image_rgb),
                feature_columns,
            )
        )

    x_matrix = np.stack(feature_rows, axis=0)
    probabilities = pipeline.predict_proba(x_matrix)[:, 1]
    predictions = pipeline.predict(x_matrix)
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float32),
    )


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(labels, predictions)),
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Recall": float(recall_score(labels, predictions, zero_division=0)),
        "F1-Score": float(f1_score(labels, predictions, zero_division=0)),
        "AUC-ROC": float(roc_auc_score(labels, probabilities)),
    }


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.35)


def plot_roc_overlay(
    labels: np.ndarray,
    viper_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    save_path: Path = ROC_OVERLAY_PNG,
) -> None:
    viper_fpr, viper_tpr, _ = roc_curve(labels, viper_probabilities)
    base_fpr, base_tpr, _ = roc_curve(labels, baseline_probabilities)
    viper_auc = auc(viper_fpr, viper_tpr)
    base_auc = auc(base_fpr, base_tpr)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(
        viper_fpr,
        viper_tpr,
        color=VIPER_COLOR,
        linewidth=3,
        label=f"VIPER (AUC = {viper_auc:.3f})",
    )
    ax.plot(
        base_fpr,
        base_tpr,
        color=BASELINE_COLOR,
        linewidth=3,
        linestyle="--",
        label=f"Baseline LR (AUC = {base_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], color="#64748B", linewidth=1.5, linestyle=":")
    ax.set_title("ROC Overlay: VIPER vs Logistic Regression")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_curve(
    labels: np.ndarray,
    viper_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    save_path: Path = CALIBRATION_CURVE_PNG,
) -> None:
    vip_frac_pos, vip_mean_pred = calibration_curve(
        labels,
        viper_probabilities,
        n_bins=10,
        strategy="uniform",
    )
    base_frac_pos, base_mean_pred = calibration_curve(
        labels,
        baseline_probabilities,
        n_bins=10,
        strategy="uniform",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], color="#64748B", linestyle=":", linewidth=1.5, label="Perfect calibration")
    ax.plot(
        vip_mean_pred,
        vip_frac_pos,
        marker="o",
        markersize=8,
        linewidth=2.8,
        color=VIPER_COLOR,
        label="VIPER",
    )
    ax.plot(
        base_mean_pred,
        base_frac_pos,
        marker="s",
        markersize=8,
        linewidth=2.8,
        color=BASELINE_COLOR,
        label="Baseline LR",
    )
    ax.set_title("Probability Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_radar_comparison(
    viper_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    save_path: Path = RADAR_COMPARISON_PNG,
) -> None:
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    viper_values = [viper_metrics[name] for name in metric_names]
    baseline_values = [baseline_metrics[name] for name in metric_names]
    viper_values += viper_values[:1]
    baseline_values += baseline_values[:1]

    fig = plt.figure(figsize=(10, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor(PANEL_COLOR)
    ax.plot(angles, viper_values, color=VIPER_COLOR, linewidth=3, label="VIPER")
    ax.fill(angles, viper_values, color=VIPER_COLOR, alpha=0.18)
    ax.plot(
        angles,
        baseline_values,
        color=BASELINE_COLOR,
        linewidth=3,
        linestyle="--",
        label="Baseline LR",
    )
    ax.fill(angles, baseline_values, color=BASELINE_COLOR, alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_title("Comparative Performance Radar", pad=28)
    ax.grid(alpha=0.35)
    ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.12))
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_error_venn(
    test_paths: Sequence[Path],
    labels: np.ndarray,
    viper_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    save_path: Path = ERROR_VENN_PNG,
) -> None:
    viper_errors = {
        path.name
        for path, label, pred in zip(test_paths, labels, viper_predictions)
        if int(label) != int(pred)
    }
    baseline_errors = {
        path.name
        for path, label, pred in zip(test_paths, labels, baseline_predictions)
        if int(label) != int(pred)
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    venn = venn2(
        [viper_errors, baseline_errors],
        set_labels=("VIPER Errors", "Baseline Errors"),
        set_colors=(VIPER_COLOR, BASELINE_COLOR),
        alpha=0.60,
        ax=ax,
    )
    if venn.get_patch_by_id("10") is not None:
        venn.get_patch_by_id("10").set_edgecolor("#0F172A")
    if venn.get_patch_by_id("01") is not None:
        venn.get_patch_by_id("01").set_edgecolor("#0F172A")
    if venn.get_patch_by_id("11") is not None:
        venn.get_patch_by_id("11").set_edgecolor("#0F172A")

    ax.set_title("Error Overlap on the Shared Test Set")
    ax.text(
        0.5,
        -0.10,
        "Intersection shows images both models misclassified.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_blur_sweep(
    kernel_sizes: Sequence[int],
    viper_f1_scores: Sequence[float],
    baseline_f1_scores: Sequence[float],
    save_path: Path = BLUR_SWEEP_PNG,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(
        kernel_sizes,
        viper_f1_scores,
        marker="o",
        markersize=9,
        linewidth=3,
        color=VIPER_COLOR,
        label="VIPER",
    )
    ax.plot(
        kernel_sizes,
        baseline_f1_scores,
        marker="s",
        markersize=9,
        linewidth=3,
        linestyle="--",
        color=BASELINE_COLOR,
        label="Baseline LR",
    )
    ax.set_title("Adversarial Blur Sweep")
    ax.set_xlabel("Gaussian Blur Kernel Size")
    ax.set_ylabel("F1 Score")
    ax.set_xticks(list(kernel_sizes))
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def run_blur_sweep(
    viper_model,
    baseline_pipeline: Pipeline,
    path_to_image: Dict[str, np.ndarray],
    test_paths: Sequence[Path],
    test_labels: Sequence[int],
    feature_lookup: Dict[str, np.ndarray],
    feature_columns: Sequence[str],
) -> Tuple[List[int], List[float], List[float]]:
    kernel_sizes = [0, 3, 5, 7, 11]
    viper_scores: List[float] = []
    baseline_scores: List[float] = []

    for kernel_size in kernel_sizes:
        labels, viper_preds, _ = infer_viper(
            model=viper_model,
            path_to_image=path_to_image,
            paths=test_paths,
            labels=test_labels,
            blur_kernel=kernel_size,
        )
        _, baseline_preds, _ = infer_baseline(
            pipeline=baseline_pipeline,
            path_to_image=path_to_image,
            paths=test_paths,
            labels=test_labels,
            feature_lookup=feature_lookup,
            feature_columns=feature_columns,
            blur_kernel=kernel_size,
        )
        viper_scores.append(float(f1_score(labels, viper_preds, zero_division=0)))
        baseline_scores.append(float(f1_score(labels, baseline_preds, zero_division=0)))

    return kernel_sizes, viper_scores, baseline_scores


def print_metric_summary(
    viper_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
) -> None:
    print("\n[CompareViz] Shared test-set metrics")
    for name in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]:
        print(
            f"  {name:10s} | VIPER={viper_metrics[name]:.4f} "
            f"| Baseline={baseline_metrics[name]:.4f}"
        )


def visualize_comparison(
    feature_csv: Path = FEATURE_MATRIX_CSV,
) -> None:
    set_plot_theme()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_paths, train_labels, test_paths, test_labels = collect_split_data()
    comparison_paths = list(train_paths) + list(test_paths)

    feature_columns, feature_lookup = read_feature_matrix(feature_csv)
    feature_lookup = ensure_feature_lookup(
        feature_lookup=feature_lookup,
        required_paths=comparison_paths,
        feature_columns=feature_columns,
    )

    path_to_image = {
        str(path.resolve()): load_rgb_image(path)
        for path in tqdm(comparison_paths, desc="Load images", leave=False)
    }

    baseline_pipeline = reconstruct_baseline_model(
        train_paths=train_paths,
        train_labels=train_labels,
        feature_lookup=feature_lookup,
    )
    viper_model = load_checkpoint(device=DEVICE)
    if viper_model is None:
        raise FileNotFoundError("VIPER checkpoint missing or incompatible.")

    labels, viper_preds, viper_probs = infer_viper(
        model=viper_model,
        path_to_image=path_to_image,
        paths=test_paths,
        labels=test_labels,
        blur_kernel=0,
    )
    _, baseline_preds, baseline_probs = infer_baseline(
        pipeline=baseline_pipeline,
        path_to_image=path_to_image,
        paths=test_paths,
        labels=test_labels,
        feature_lookup=feature_lookup,
        feature_columns=feature_columns,
        blur_kernel=0,
    )

    viper_metrics = compute_metrics(labels, viper_preds, viper_probs)
    baseline_metrics = compute_metrics(labels, baseline_preds, baseline_probs)
    print_metric_summary(viper_metrics, baseline_metrics)

    plot_roc_overlay(labels, viper_probs, baseline_probs)
    plot_calibration_curve(labels, viper_probs, baseline_probs)
    plot_radar_comparison(viper_metrics, baseline_metrics)
    plot_error_venn(test_paths, labels, viper_preds, baseline_preds)

    kernel_sizes, viper_f1_scores, baseline_f1_scores = run_blur_sweep(
        viper_model=viper_model,
        baseline_pipeline=baseline_pipeline,
        path_to_image=path_to_image,
        test_paths=test_paths,
        test_labels=test_labels,
        feature_lookup=feature_lookup,
        feature_columns=feature_columns,
    )
    plot_blur_sweep(
        kernel_sizes=kernel_sizes,
        viper_f1_scores=viper_f1_scores,
        baseline_f1_scores=baseline_f1_scores,
    )

    print(f"[CompareViz] Saved -> {ROC_OVERLAY_PNG}")
    print(f"[CompareViz] Saved -> {CALIBRATION_CURVE_PNG}")
    print(f"[CompareViz] Saved -> {RADAR_COMPARISON_PNG}")
    print(f"[CompareViz] Saved -> {ERROR_VENN_PNG}")
    print(f"[CompareViz] Saved -> {BLUR_SWEEP_PNG}")


if __name__ == "__main__":
    print("=== Comparison Visualization Agent - VIPER Forensic Engine ===")
    visualize_comparison()
