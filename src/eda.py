"""
src/eda.py — VIPER Forensic Engine: Exploratory Data Analysis Layer
Phase: Track Beta (Analytical Pipeline) — EDA Agent

Computes six forensic feature families for each image:
  1. compute_pixel_stats()   — mean/std per RGB channel
  2. fft_analysis()          — high-frequency energy from FFT
  3. compute_color_entropy() — color palette entropy via KMeans
  4. noise_residuals()       — Gaussian residual + PRNU-style noise fingerprints
  5. glcm_features()         — GLCM texture (contrast, energy, homogeneity, correlation)
  6. edge_density()          — Canny edge pixel fraction

Outputs:
  results/feature_matrix.csv  — one row per image with all features + label

Usage:
    python src/eda.py
    # → writes results/feature_matrix.csv and prints summary statistics
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from skimage.restoration import denoise_wavelet
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    AI_ART_DIR, REAL_ART_DIR, WIKIART_DIR,
    FEATURE_MATRIX_CSV, RESULTS_DIR,
    EDA_KMEANS_K, EDA_SAMPLE_SIZE, IMAGE_SIZE,
    LABEL_REAL, LABEL_AI,
)

warnings.filterwarnings("ignore")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# Individual Feature Extractors
# ─────────────────────────────────────────────────────────────────────────────

def compute_pixel_stats(img_rgb: np.ndarray) -> Dict[str, float]:
    """
    Compute per-channel mean and standard deviation.

    Args:
        img_rgb: H×W×3 uint8 numpy array (RGB).

    Returns:
        Dict with keys: mean_r, mean_g, mean_b, std_r, std_g, std_b, global_brightness
    """
    img_f = img_rgb.astype(np.float32) / 255.0
    means = img_f.mean(axis=(0, 1))
    stds  = img_f.std(axis=(0, 1))
    return {
        "mean_r":          float(means[0]),
        "mean_g":          float(means[1]),
        "mean_b":          float(means[2]),
        "std_r":           float(stds[0]),
        "std_g":           float(stds[1]),
        "std_b":           float(stds[2]),
        "global_brightness": float(means.mean()),
    }


def fft_analysis(img_rgb: np.ndarray) -> Dict[str, float]:
    """
    Measure high-frequency energy in the image via 2D FFT on grayscale.

    AI-generated images often show characteristic frequency spectra with
    either over-smoothing or periodic artifacts.

    Args:
        img_rgb: H×W×3 uint8 numpy array.

    Returns:
        Dict with keys: fft_high_freq_energy, fft_low_freq_energy, fft_ratio
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fft  = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    h, w   = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 8      # inner 1/8 = low-frequency region

    # Build masks
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    low_mask  = dist <= radius
    high_mask = ~low_mask

    low_energy  = float(magnitude[low_mask].sum())
    high_energy = float(magnitude[high_mask].sum())
    total       = low_energy + high_energy + 1e-8

    return {
        "fft_high_freq_energy": high_energy / total,
        "fft_low_freq_energy":  low_energy  / total,
        "fft_ratio":            high_energy / (low_energy + 1e-8),
    }


def compute_color_entropy(img_rgb: np.ndarray, k: int = EDA_KMEANS_K) -> Dict[str, float]:
    """
    Estimate color palette entropy using KMeans clustering in CIELAB space.

    Converts RGB pixels to Lab, clusters into k color centers, and computes
    Shannon entropy of the cluster distribution.

    Args:
        img_rgb: H×W×3 uint8 numpy array.
        k:       Number of color clusters (default EDA_KMEANS_K=8).

    Returns:
        Dict with keys: color_entropy, dominant_color_r/g/b, hue_std

    AGENT_TASK: extend with cross-image Lab consistency metrics
    """
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    pixels_lab = img_lab.reshape(-1, 3).astype(np.float32)

    # Downsample for speed
    if len(pixels_lab) > 5000:
        idx = np.random.choice(len(pixels_lab), 5000, replace=False)
        pixels_lab = pixels_lab[idx]

    k_eff = max(1, min(k, len(pixels_lab)))

    km = KMeans(n_clusters=k_eff, n_init=5, random_state=42)
    km.fit(pixels_lab)
    counts = np.bincount(km.labels_, minlength=k_eff).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))

    dominant_lab = np.clip(km.cluster_centers_[np.argmax(counts)], 0, 255).astype(np.uint8)
    dominant_rgb = cv2.cvtColor(
        dominant_lab.reshape(1, 1, 3), cv2.COLOR_LAB2RGB
    )[0, 0].astype(np.float32)

    # Hue std via HSV
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hue_std = float(img_hsv[:, :, 0].std())

    return {
        "color_entropy":    entropy,
        "dominant_color_r": float(dominant_rgb[0]) / 255.0,
        "dominant_color_g": float(dominant_rgb[1]) / 255.0,
        "dominant_color_b": float(dominant_rgb[2]) / 255.0,
        "hue_std":          hue_std,
    }


def noise_residuals(img_rgb: np.ndarray) -> Dict[str, float]:
    """
    Extract noise residuals plus a basic PRNU-style estimator.

    Real photographs carry camera sensor noise; AI images often produce
    cleaner or differently-structured residuals.

    Args:
        img_rgb: H×W×3 uint8 numpy array.

    Returns:
        Dict with keys:
          noise_mean, noise_std, noise_energy,
          prnu_mean, prnu_std, prnu_energy, prnu_fft_ratio, prnu_autocorr_peak
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Legacy residual: Gaussian subtraction
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.5)
    residual = gray - blurred

    # PRNU proxy:
    # 1) Wavelet-denoise to estimate content
    # 2) Noise residual = image - denoised image
    # 3) Normalize by local intensity and apply light high-pass filtering
    gray_norm = gray / 255.0
    try:
        denoised = denoise_wavelet(
            gray_norm,
            channel_axis=None,
            method="BayesShrink",
            mode="soft",
            rescale_sigma=True,
        ).astype(np.float32)
    except ImportError:
        denoised = cv2.GaussianBlur(gray_norm, (0, 0), sigmaX=1.0)
    noise = gray_norm - denoised
    prnu = noise / (gray_norm + 1e-3)
    prnu = prnu - cv2.GaussianBlur(prnu, (3, 3), sigmaX=0.8)

    prnu_mean = float(prnu.mean())
    prnu_std = float(prnu.std())
    prnu_energy = float((prnu ** 2).mean())

    prnu_fft = np.fft.fftshift(np.fft.fft2(prnu))
    prnu_mag = np.abs(prnu_fft)
    h, w = prnu_mag.shape
    cy, cx = h // 2, w // 2
    radius = max(1, min(h, w) // 8)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    low_mask = dist <= radius
    high_mask = ~low_mask
    low_power = float(prnu_mag[low_mask].sum())
    high_power = float(prnu_mag[high_mask].sum())
    prnu_fft_ratio = high_power / (low_power + 1e-8)

    prnu_zero_mean = prnu - prnu_mean
    ac = np.fft.ifft2(np.abs(np.fft.fft2(prnu_zero_mean)) ** 2).real
    ac = np.fft.fftshift(ac)
    center = float(ac[cy, cx]) + 1e-8
    ac[cy, cx] = 0.0
    prnu_autocorr_peak = float(np.max(np.abs(ac)) / center)

    return {
        "noise_mean":   float(residual.mean()),
        "noise_std":    float(residual.std()),
        "noise_energy": float((residual ** 2).mean()),
        "prnu_mean":    prnu_mean,
        "prnu_std":     prnu_std,
        "prnu_energy":  prnu_energy,
        "prnu_fft_ratio": prnu_fft_ratio,
        "prnu_autocorr_peak": prnu_autocorr_peak,
    }


def glcm_features(img_rgb: np.ndarray) -> Dict[str, float]:
    """
    Compute Gray-Level Co-occurrence Matrix (GLCM) texture features.

    Captures structural texture properties. AI images tend to have more
    homogeneous textures than real photographs.

    Args:
        img_rgb: H×W×3 uint8 numpy array.

    Returns:
        Dict with keys: glcm_contrast, glcm_energy, glcm_homogeneity, glcm_correlation

    AGENT_TASK: add multi-angle GLCM averaging and entropy feature
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Downsample for speed (GLCM is O(N²))
    gray_small = cv2.resize(gray, (64, 64))
    # Reduce to 64 levels to make GLCM tractable
    gray_scaled = (gray_small // 4).astype(np.uint8)

    glcm = graycomatrix(
        gray_scaled,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64,
        symmetric=True,
        normed=True,
    )

    contrast     = float(graycoprops(glcm, "contrast").mean())
    energy       = float(graycoprops(glcm, "energy").mean())
    homogeneity  = float(graycoprops(glcm, "homogeneity").mean())
    correlation  = float(graycoprops(glcm, "correlation").mean())

    return {
        "glcm_contrast":    contrast,
        "glcm_energy":      energy,
        "glcm_homogeneity": homogeneity,
        "glcm_correlation": correlation,
    }


def edge_density(img_rgb: np.ndarray) -> Dict[str, float]:
    """
    Compute Canny edge pixel fraction and gradient statistics.

    AI images frequently have overly sharp or overly smooth edges.

    Args:
        img_rgb: H×W×3 uint8 numpy array.

    Returns:
        Dict with keys: canny_edge_density, sobel_mean, sobel_std

    AGENT_TASK: implement oriented edge histograms for direction-aware analysis
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Canny edge density
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    density = float(edges.sum()) / (255.0 * edges.size)

    # Sobel gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    return {
        "canny_edge_density": density,
        "sobel_mean":         float(magnitude.mean()),
        "sobel_std":          float(magnitude.std()),
    }


def load_image_rgb(
    image_path: Path,
    target_size: int = IMAGE_SIZE,
) -> Optional[np.ndarray]:
    """Load an image from disk, resize, and return an RGB array."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    img_bgr = cv2.resize(img_bgr, (target_size, target_size))
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def extract_features_from_rgb(
    img_rgb: np.ndarray,
    label: Optional[int] = None,
    image_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute the full forensic feature dictionary from an in-memory RGB image.

    When `label` or `image_path` are provided, they are included so the result
    can be written directly into the feature matrix schema.
    """
    features: Dict[str, float] = {}
    if image_path is not None:
        features["image_path"] = str(image_path)
    if label is not None:
        features["label"] = int(label)

    features.update(compute_pixel_stats(img_rgb))
    features.update(fft_analysis(img_rgb))
    features.update(compute_color_entropy(img_rgb))
    features.update(noise_residuals(img_rgb))
    features.update(glcm_features(img_rgb))
    features.update(edge_density(img_rgb))
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Full Feature Extractor (all six families)
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(
    image_path: Path,
    label: int,
    target_size: int = IMAGE_SIZE,
) -> Optional[Dict]:
    """
    Load image and run all six feature extractors.

    Args:
        image_path:  Path to image file.
        label:       Ground truth label (0=REAL, 1=AI_GENERATED).
        target_size: Resize to this square resolution before extraction.

    Returns:
        Dict of all features + path + label, or None on failure.
    """
    try:
        img_rgb = load_image_rgb(image_path, target_size=target_size)
        if img_rgb is None:
            return None
        return extract_features_from_rgb(
            img_rgb,
            label=label,
            image_path=str(image_path),
        )
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature Matrix Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    ai_dir:   Path = AI_ART_DIR,
    real_dir: Path = REAL_ART_DIR,
    output_csv: Path = FEATURE_MATRIX_CSV,
    sample_size: int = EDA_SAMPLE_SIZE,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Collect images from both classes, extract all features, and write CSV.

    Args:
        ai_dir:      Path to AI-generated images.
        real_dir:    Path to real images.
        output_csv:  Destination CSV path.
        sample_size: Max images per class to process (for speed).
        seed:        Reproducibility seed for sampling.

    Returns:
        pandas DataFrame of the feature matrix.
    """
    def _collect(directory: Path, label: int) -> List[Path]:
        paths = [
            p for p in sorted(directory.rglob("*"))
            if p.suffix.lower() in IMAGE_EXTS and p.is_file()
        ]
        if not paths:
            raise RuntimeError(f"[EDAAgent] No images found in {directory}")
        rng = np.random.default_rng(seed)
        if len(paths) > sample_size:
            idx   = rng.choice(len(paths), sample_size, replace=False)
            paths = [paths[i] for i in idx]
        return paths

    ai_paths   = _collect(ai_dir,   LABEL_AI)
    real_paths = _collect(real_dir, LABEL_REAL)
    all_paths  = [(p, LABEL_AI) for p in ai_paths] + [(p, LABEL_REAL) for p in real_paths]

    print(f"[EDAAgent] Processing {len(all_paths)} images "
          f"({len(ai_paths)} AI, {len(real_paths)} real) ...")

    rows = []
    for path, label in tqdm(all_paths, desc="Extracting features"):
        row = extract_all_features(path, label)
        if row:
            rows.append(row)

    if not rows:
        # Fail-forward: write empty CSV so downstream agents don't hang
        pd.DataFrame().to_csv(output_csv, index=False)
        raise RuntimeError("[EDAAgent] No features extracted. Check image paths.")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[EDAAgent] ✓ Feature matrix saved → {output_csv}")
    print(f"           Rows={len(df)}, Columns={len(df.columns)}")
    print(df.describe().to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EDA Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_eda_summary(df: pd.DataFrame, save_dir: Path = RESULTS_DIR) -> None:
    """
    Generate and save EDA summary plots:
      - Feature distributions by class
      - Correlation heatmap
      - FFT energy comparison

    AGENT_TASK: add interactive Plotly versions of each figure
    AGENT_TASK: add t-SNE 2D projection of raw features
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    feature_cols = [c for c in df.columns if c not in ("image_path", "label")]
    labels_map   = {0: "Real", 1: "AI-Generated"}
    df["class"]  = df["label"].map(labels_map)

    # ── 1. Feature distributions ──────────────────────────────────────────────
    fig, axes = plt.subplots(4, 5, figsize=(22, 16))
    axes = axes.flatten()
    for i, feat in enumerate(feature_cols[:20]):
        ax = axes[i]
        for cls, grp in df.groupby("class"):
            ax.hist(grp[feat].dropna(), bins=30, alpha=0.6, label=cls)
        ax.set_title(feat, fontsize=8)
        ax.legend(fontsize=6)
    plt.suptitle("Feature Distributions: Real vs AI-Generated", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_dir / "eda_distributions.png", dpi=120)
    plt.close(fig)
    print(f"[EDAAgent] ✓ Saved eda_distributions.png")

    # ── 2. Correlation heatmap ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 12))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(save_dir / "eda_correlation.png", dpi=120)
    plt.close(fig)
    print(f"[EDAAgent] ✓ Saved eda_correlation.png")

    # ── 3. FFT ratio by class ─────────────────────────────────────────────────
    if "fft_ratio" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for cls, grp in df.groupby("class"):
            ax.hist(grp["fft_ratio"].dropna(), bins=40, alpha=0.6, label=cls)
        ax.set_xlabel("FFT High/Low Frequency Ratio")
        ax.set_ylabel("Count")
        ax.set_title("FFT Analysis: AI vs Real")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / "eda_fft_ratio.png", dpi=120)
        plt.close(fig)
        print(f"[EDAAgent] ✓ Saved eda_fft_ratio.png")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== EDA Agent — VIPER Forensic Engine ===")
    df = build_feature_matrix()
    plot_eda_summary(df)
    print("[EDAAgent] ✓ EDA complete.")
