"""
src/stretch.py - VIPER Forensic Engine: Stretch Goal Agent
Phase: Track Delta - Stretch Agent

Runs robustness and zero-shot experiments using the trained classifier:
  1. JPEG robustness: accuracy vs compression quality (Q=95,75,50,25)
     -> results/jpeg_robustness.png
  2. WikiArt zero-shot inference on a deterministic 100-image subset
     -> results/wikiart_confidence.json

Usage:
    python src/stretch.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    IMAGE_SIZE,
    JPEG_QUALITY_LEVELS,
    JPEG_ROBUSTNESS_PNG,
    SEED,
    STRETCH_SAMPLE_N,
    WIKIART_CONF_JSON,
    WIKIART_DIR,
)
from src.dataloader import get_dataloaders, get_val_transform
from src.model import load_checkpoint

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
WIKIART_DATASET_HANDLE = "steubk/wikiart"
WIKIART_DOWNLOAD_DIRNAME = "_kaggle_download"
WIKIART_ZERO_SHOT_SAMPLE_N = min(100, STRETCH_SAMPLE_N)
WIKIART_REMBRANDT_QUERY = "rembrandt"
SALT_PEPPER_NOISE_LEVELS = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
SALT_PEPPER_ROBUSTNESS_PNG = JPEG_ROBUSTNESS_PNG.with_name("salt_pepper_robustness.png")
ROBUSTNESS_METRICS_JSON = JPEG_ROBUSTNESS_PNG.with_name("robustness_metrics.json")


def compress_jpeg(image_array: np.ndarray, quality: int) -> np.ndarray:
    """JPEG-compress and decompress an image at the requested quality."""
    import cv2

    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode(".jpg", img_bgr, encode_param)
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def inject_salt_and_pepper_noise(image_array: np.ndarray, intensity: float) -> np.ndarray:
    """
    Inject salt-and-pepper noise into an RGB image.

    Args:
        image_array: RGB uint8 image array.
        intensity: Fraction of pixels to corrupt in [0, 1].
    """
    if intensity <= 0:
        return image_array.copy()

    noisy = image_array.copy()
    h, w, _ = noisy.shape
    n_corrupt = int(round(h * w * intensity))
    if n_corrupt == 0:
        return noisy

    rng = np.random.default_rng(SEED)
    coords = rng.choice(h * w, size=n_corrupt, replace=False)
    half = n_corrupt // 2

    salt_coords = coords[:half]
    pepper_coords = coords[half:]

    noisy.reshape(-1, 3)[salt_coords] = 255
    noisy.reshape(-1, 3)[pepper_coords] = 0
    return noisy


@torch.no_grad()
def evaluate_at_quality(
    model,
    paths: List[Path],
    labels: List[int],
    quality: int,
    device: torch.device,
    transform,
) -> float:
    """Evaluate model accuracy on a set of images compressed at a given quality."""
    model.eval()
    correct = 0
    total = 0

    for path, label in zip(paths, labels):
        try:
            img = np.array(Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
            compressed = compress_jpeg(img, quality)
            pil_img = Image.fromarray(compressed)
            tensor = transform(pil_img).unsqueeze(0).to(device)

            logits = model(tensor)
            pred = logits.argmax(dim=1).item()
            if pred == label:
                correct += 1
            total += 1
        except Exception:
            continue

    return correct / max(total, 1)


@torch.no_grad()
def evaluate_at_noise(
    model,
    paths: List[Path],
    labels: List[int],
    intensity: float,
    device: torch.device,
    transform,
) -> float:
    """Evaluate model accuracy on images corrupted with salt-and-pepper noise."""
    model.eval()
    correct = 0
    total = 0

    for path, label in zip(paths, labels):
        try:
            img = np.array(Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
            corrupted = inject_salt_and_pepper_noise(img, intensity)
            pil_img = Image.fromarray(corrupted)
            tensor = transform(pil_img).unsqueeze(0).to(device)

            logits = model(tensor)
            pred = logits.argmax(dim=1).item()
            if pred == label:
                correct += 1
            total += 1
        except Exception:
            continue

    return correct / max(total, 1)


def _collect_eval_sample(
    loader,
    sample_n: int,
    seed: int = SEED,
) -> Tuple[List[Path], List[int]]:
    """Collect a deterministic evaluation subset from the test loader."""
    all_paths, all_labels = [], []
    for _, labels, paths in loader:
        all_paths.extend(paths)
        all_labels.extend(labels.tolist())
        if len(all_paths) >= sample_n:
            break

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_paths), min(sample_n, len(all_paths)), replace=False)
    sampled_paths = [Path(all_paths[i]) for i in idx]
    sampled_labels = [all_labels[i] for i in idx]
    return sampled_paths, sampled_labels


def _save_robustness_metrics(experiment_name: str, results: Dict[str, float]) -> None:
    """Persist robustness curves to a shared results JSON file."""
    payload: Dict[str, object] = {}
    if ROBUSTNESS_METRICS_JSON.exists():
        try:
            payload = json.loads(ROBUSTNESS_METRICS_JSON.read_text())
        except json.JSONDecodeError:
            payload = {}

    payload[experiment_name] = results
    ROBUSTNESS_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    ROBUSTNESS_METRICS_JSON.write_text(json.dumps(payload, indent=2))


def _plot_robustness_curve(
    x_values: Sequence[float],
    y_values: Sequence[float],
    save_path: Path,
    xlabel: str,
    title: str,
    color: str,
) -> None:
    """Plot and save a robustness curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_values, y_values, marker="o", linewidth=2, color=color, markersize=8)
    ax.fill_between(x_values, y_values, alpha=0.15, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.set_xticks(x_values)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_jpeg_robustness(
    model,
    loader,
    device: torch.device = DEVICE,
    sample_n: int = STRETCH_SAMPLE_N,
    quality_levels: List[int] = JPEG_QUALITY_LEVELS,
    save_path: Path = JPEG_ROBUSTNESS_PNG,
) -> Dict[int, float]:
    """Sample test-set images and evaluate accuracy at each JPEG quality."""
    transform = get_val_transform()
    sampled_paths, sampled_labels = _collect_eval_sample(loader, sample_n, seed=SEED)

    print(f"[StretchAgent] JPEG robustness test on {len(sampled_paths)} images ...")
    results: Dict[int, float] = {}
    for quality in quality_levels:
        acc = evaluate_at_quality(model, sampled_paths, sampled_labels, quality, device, transform)
        results[quality] = acc
        print(f"  Q={quality:3d} -> accuracy={acc:.4f}")

    qualities = sorted(results.keys(), reverse=True)
    accuracies = [results[q] for q in qualities]
    _save_robustness_metrics("jpeg_accuracy", {str(k): v for k, v in results.items()})
    plot_rendered = True
    try:
        _plot_robustness_curve(
            x_values=qualities,
            y_values=accuracies,
            save_path=save_path,
            xlabel="JPEG Quality Level",
            title="VIPER Robustness: Accuracy vs JPEG Compression Quality",
            color="#1565C0",
        )
    except ModuleNotFoundError as exc:
        plot_rendered = False
        print(f"[StretchAgent] Could not render JPEG robustness plot: {exc}")

    diffs = [abs(accuracies[i] - accuracies[i + 1]) for i in range(len(accuracies) - 1)]
    if diffs:
        cliff_idx = int(np.argmax(diffs))
        cliff_q = qualities[cliff_idx + 1]
        print(f"[StretchAgent] JPEG degradation cliff detected near Q={cliff_q}")

    if plot_rendered:
        print(f"[StretchAgent] Saved JPEG robustness plot -> {save_path}")
    print(f"[StretchAgent] Logged JPEG robustness metrics -> {ROBUSTNESS_METRICS_JSON}")
    return results


def run_salt_pepper_robustness(
    model,
    loader,
    device: torch.device = DEVICE,
    sample_n: int = STRETCH_SAMPLE_N,
    noise_levels: Sequence[float] = SALT_PEPPER_NOISE_LEVELS,
    save_path: Path = SALT_PEPPER_ROBUSTNESS_PNG,
) -> Dict[float, float]:
    """Evaluate accuracy degradation under increasing salt-and-pepper noise."""
    transform = get_val_transform()
    sampled_paths, sampled_labels = _collect_eval_sample(loader, sample_n, seed=SEED)

    print(f"[StretchAgent] Salt-and-pepper robustness test on {len(sampled_paths)} images ...")
    results: Dict[float, float] = {}
    for intensity in noise_levels:
        acc = evaluate_at_noise(model, sampled_paths, sampled_labels, intensity, device, transform)
        results[float(intensity)] = acc
        print(f"  noise={intensity:0.2f} -> accuracy={acc:.4f}")

    ordered_levels = sorted(results.keys())
    accuracies = [results[level] for level in ordered_levels]
    _save_robustness_metrics("salt_pepper_accuracy", {f"{k:.2f}": v for k, v in results.items()})
    plot_rendered = True
    try:
        _plot_robustness_curve(
            x_values=ordered_levels,
            y_values=accuracies,
            save_path=save_path,
            xlabel="Salt-and-Pepper Noise Intensity",
            title="VIPER Robustness: Accuracy vs Salt-and-Pepper Noise",
            color="#C62828",
        )
    except ModuleNotFoundError as exc:
        plot_rendered = False
        print(f"[StretchAgent] Could not render salt-and-pepper plot: {exc}")
    if plot_rendered:
        print(f"[StretchAgent] Saved salt-and-pepper robustness plot -> {save_path}")
    print(f"[StretchAgent] Logged salt-and-pepper robustness metrics -> {ROBUSTNESS_METRICS_JSON}")
    return results


def _list_image_paths(root: Path, exclude_dirs: Optional[Iterable[str]] = None) -> List[Path]:
    """Recursively collect image files under a directory."""
    if not root.exists():
        return []

    excluded = {name.lower() for name in (exclude_dirs or [])}
    paths: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if excluded and any(part.lower() in excluded for part in path.parts):
            continue
        paths.append(path)
    return paths


def _sample_paths(paths: Sequence[Path], sample_n: int, seed: int = SEED) -> List[Path]:
    """Select a deterministic subset without replacement."""
    if len(paths) <= sample_n:
        return list(paths)

    rng = np.random.default_rng(seed)
    sampled_idx = sorted(rng.choice(len(paths), size=sample_n, replace=False).tolist())
    return [paths[i] for i in sampled_idx]


def _relative_path_str(path: Path, root: Path) -> str:
    """Return a path relative to root when possible."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _extract_wikiart_metadata(path: Path, source_root: Path) -> Dict[str, str]:
    """
    Infer lightweight artist/style hints from the dataset directory structure.

    The Kaggle WikiArt layout can vary, so these are best-effort hints only.
    """
    rel = Path(_relative_path_str(path, source_root))
    parent_parts = rel.parts[:-1]

    artist_hint = parent_parts[0] if len(parent_parts) >= 1 else ""
    style_hint = parent_parts[1] if len(parent_parts) >= 2 else ""
    genre_hint = parent_parts[2] if len(parent_parts) >= 3 else ""

    return {
        "relative_path": str(rel),
        "artist_hint": artist_hint,
        "style_hint": style_hint,
        "genre_hint": genre_hint,
    }


def _summarize_probabilities(probabilities: Sequence[float]) -> Dict[str, float]:
    """Compute a compact confidence summary."""
    if not probabilities:
        return {
            "count": 0,
            "mean_conf": 0.0,
            "std_conf": 0.0,
            "median_conf": 0.0,
            "min_conf": 0.0,
            "max_conf": 0.0,
            "pct_flagged": 0.0,
        }

    arr = np.asarray(probabilities, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean_conf": float(arr.mean()),
        "std_conf": float(arr.std()),
        "median_conf": float(np.median(arr)),
        "min_conf": float(arr.min()),
        "max_conf": float(arr.max()),
        "pct_flagged": float((arr > 0.5).mean()),
    }


def _kaggle_credentials_available() -> bool:
    """Check whether Kaggle credentials are exposed through the environment."""
    return bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))


def _download_wikiart_dataset(wikiart_dir: Path) -> Tuple[Optional[Path], Dict[str, str]]:
    """
    Download the Kaggle WikiArt dataset into a local cache directory.

    This uses the repo's existing Kaggle credentials convention:
    KAGGLE_USERNAME and KAGGLE_KEY in the environment.
    """
    if not _kaggle_credentials_available():
        return None, {
            "status": "missing_credentials",
            "dataset_handle": WIKIART_DATASET_HANDLE,
            "message": "Set KAGGLE_USERNAME and KAGGLE_KEY in .env before downloading WikiArt.",
        }

    try:
        import kagglehub  # type: ignore
    except ImportError:
        return None, {
            "status": "missing_dependency",
            "dataset_handle": WIKIART_DATASET_HANDLE,
            "message": "kagglehub is not installed in the active environment.",
        }

    download_fn = getattr(kagglehub, "dataset_download", None)
    if download_fn is None:
        return None, {
            "status": "unsupported_kagglehub",
            "dataset_handle": WIKIART_DATASET_HANDLE,
            "message": "The installed kagglehub build does not expose dataset_download().",
        }

    download_dir = wikiart_dir / WIKIART_DOWNLOAD_DIRNAME
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset_root = Path(
            download_fn(
                WIKIART_DATASET_HANDLE,
                output_dir=str(download_dir),
            )
        )
    except Exception as exc:
        return None, {
            "status": "download_failed",
            "dataset_handle": WIKIART_DATASET_HANDLE,
            "message": str(exc),
        }

    return dataset_root, {
        "status": "downloaded",
        "dataset_handle": WIKIART_DATASET_HANDLE,
        "path": str(dataset_root),
    }


def _resolve_wikiart_source(wikiart_dir: Path) -> Tuple[Optional[Path], Dict[str, str]]:
    """
    Resolve a WikiArt image source in priority order:
      1. Existing local images in data/WikiArt
      2. Previously downloaded Kaggle cache
      3. Fresh Kaggle download
    """
    local_images = _list_image_paths(wikiart_dir, exclude_dirs={WIKIART_DOWNLOAD_DIRNAME})
    if local_images:
        return wikiart_dir, {
            "status": "local_existing",
            "path": str(wikiart_dir),
            "message": f"Using {len(local_images)} existing WikiArt images from disk.",
        }

    cached_dir = wikiart_dir / WIKIART_DOWNLOAD_DIRNAME
    cached_images = _list_image_paths(cached_dir)
    if cached_images:
        return cached_dir, {
            "status": "cached_download",
            "path": str(cached_dir),
            "message": f"Using {len(cached_images)} cached WikiArt images.",
        }

    return _download_wikiart_dataset(wikiart_dir)


@torch.no_grad()
def _score_paths(
    model,
    image_paths: Sequence[Path],
    source_root: Path,
    transform,
    device: torch.device,
    desc: str,
) -> List[Dict[str, object]]:
    """Run inference on a list of image paths and return per-image scores."""
    model.eval()
    rows: List[Dict[str, object]] = []

    for path in tqdm(image_paths, desc=desc):
        try:
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            p_ai = float(probs[1].item())
        except Exception:
            continue

        metadata = _extract_wikiart_metadata(path, source_root)
        rows.append(
            {
                "path": str(path),
                "relative_path": metadata["relative_path"],
                "artist_hint": metadata["artist_hint"],
                "style_hint": metadata["style_hint"],
                "genre_hint": metadata["genre_hint"],
                "predicted_label": pred_idx,
                "predicted_class": CLASS_NAMES[pred_idx],
                "p_ai_generated": p_ai,
            }
        )

    return rows


def _artist_matches(paths: Sequence[Path], source_root: Path, query: str) -> List[Path]:
    """Find artist-labelled files using a case-insensitive substring match."""
    query_lower = query.lower()
    matches = []
    for path in paths:
        rel_text = _relative_path_str(path, source_root).lower()
        if query_lower in rel_text:
            matches.append(path)
    return matches


def _summarize_artist(scores: Sequence[Dict[str, object]], artist_query: str) -> Dict[str, object]:
    """Summarize whether the model tends to flag an artist as AI-generated."""
    probabilities = [float(row["p_ai_generated"]) for row in scores]
    summary = _summarize_probabilities(probabilities)

    if not scores:
        return {
            "artist_query": artist_query,
            "answer": (
                "Unknown: no files with this artist name were found in the available WikiArt tree."
            ),
            "summary": summary,
            "top_flagged_work": None,
        }

    top_flagged = max(scores, key=lambda row: float(row["p_ai_generated"]))
    pct_flagged = float(summary["pct_flagged"])
    mean_conf = float(summary["mean_conf"])

    if pct_flagged >= 0.5:
        verdict = "Yes"
    elif pct_flagged == 0.0 and mean_conf < 0.5:
        verdict = "No"
    else:
        verdict = "Mixed"

    answer = (
        f"{verdict}: across {int(summary['count'])} {artist_query.title()}-labelled works, "
        f"mean P(AI)={mean_conf:.3f} and {pct_flagged:.1%} exceeded the 0.5 decision threshold."
    )

    return {
        "artist_query": artist_query,
        "answer": answer,
        "summary": summary,
        "top_flagged_work": top_flagged,
    }


@torch.no_grad()
def run_wikiart_inference(
    model,
    wikiart_dir: Path = WIKIART_DIR,
    output_json: Path = WIKIART_CONF_JSON,
    device: torch.device = DEVICE,
    sample_n: int = WIKIART_ZERO_SHOT_SAMPLE_N,
) -> Dict[str, object]:
    """
    Run a zero-shot inference pass on a deterministic 100-image WikiArt subset.

    The function first tries the local WikiArt folder. If no images are present,
    it attempts a Kaggle download using the credentials referenced in src/config.py.
    """
    output_json.parent.mkdir(parents=True, exist_ok=True)
    transform = get_val_transform()

    source_root, source_info = _resolve_wikiart_source(wikiart_dir)
    if source_root is None:
        result = {
            "dataset_handle": WIKIART_DATASET_HANDLE,
            "sample_size_requested": sample_n,
            "download": source_info,
            "reason": source_info.get("message", "WikiArt data is unavailable."),
            "confidence_summary": _summarize_probabilities([]),
            "rembrandt_analysis": _summarize_artist([], WIKIART_REMBRANDT_QUERY),
            "per_image": [],
        }
        output_json.write_text(json.dumps(result, indent=2))
        print(f"[StretchAgent] WikiArt unavailable. Details -> {output_json}")
        return result

    all_paths = _list_image_paths(source_root)
    if not all_paths:
        result = {
            "dataset_handle": WIKIART_DATASET_HANDLE,
            "sample_size_requested": sample_n,
            "download": source_info,
            "reason": f"No image files found under {source_root}.",
            "confidence_summary": _summarize_probabilities([]),
            "rembrandt_analysis": _summarize_artist([], WIKIART_REMBRANDT_QUERY),
            "per_image": [],
        }
        output_json.write_text(json.dumps(result, indent=2))
        print(f"[StretchAgent] No WikiArt images found. Details -> {output_json}")
        return result

    sampled_paths = _sample_paths(all_paths, sample_n, seed=SEED)
    print(
        f"[StretchAgent] WikiArt zero-shot on {len(sampled_paths)} sampled images "
        f"(source pool={len(all_paths)}) ..."
    )
    per_image_scores = _score_paths(
        model=model,
        image_paths=sampled_paths,
        source_root=source_root,
        transform=transform,
        device=device,
        desc="WikiArt sample",
    )

    rembrandt_paths = _artist_matches(all_paths, source_root, WIKIART_REMBRANDT_QUERY)
    rembrandt_scores = _score_paths(
        model=model,
        image_paths=rembrandt_paths,
        source_root=source_root,
        transform=transform,
        device=device,
        desc="Rembrandt",
    ) if rembrandt_paths else []

    confidence_summary = _summarize_probabilities(
        [float(row["p_ai_generated"]) for row in per_image_scores]
    )
    rembrandt_analysis = _summarize_artist(rembrandt_scores, WIKIART_REMBRANDT_QUERY)

    result = {
        "dataset_handle": WIKIART_DATASET_HANDLE,
        "sample_size_requested": sample_n,
        "source_image_count": len(all_paths),
        "source_root": str(source_root),
        "download": source_info,
        "confidence_summary": confidence_summary,
        "rembrandt_analysis": rembrandt_analysis,
        "per_image": per_image_scores,
    }

    output_json.write_text(json.dumps(result, indent=2))
    print(f"[StretchAgent] Saved WikiArt zero-shot report -> {output_json}")
    print(
        "[StretchAgent] WikiArt summary: "
        f"mean P(AI)={confidence_summary['mean_conf']:.4f}, "
        f"flagged={confidence_summary['pct_flagged']:.2%}"
    )
    print(f"[StretchAgent] Rembrandt verdict: {rembrandt_analysis['answer']}")
    return result


def run_stretch(checkpoint_path: Path = BEST_MODEL_PATH) -> None:
    """Run the stretch-goal experiments."""
    model = load_checkpoint(checkpoint_path, DEVICE)
    if model is None:
        print("[StretchAgent] No checkpoint found. Exiting.")
        return

    _, _, test_loader = get_dataloaders(verbose=False)
    try:
        run_jpeg_robustness(model, test_loader, DEVICE)
    except ModuleNotFoundError as exc:
        print(f"[StretchAgent] Skipping JPEG robustness because a dependency is missing: {exc}")
    try:
        run_salt_pepper_robustness(model, test_loader, DEVICE)
    except ModuleNotFoundError as exc:
        print(
            "[StretchAgent] Skipping salt-and-pepper robustness because a dependency is missing: "
            f"{exc}"
        )
    run_wikiart_inference(model, device=DEVICE)
    print("[StretchAgent] All stretch goals complete.")


if __name__ == "__main__":
    print("=== Stretch Agent - VIPER Forensic Engine ===")
    run_stretch()
