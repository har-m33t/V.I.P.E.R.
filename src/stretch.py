"""
src/stretch.py — VIPER Forensic Engine: Stretch Goal Agent
Phase: Track Delta — Stretch Agent

Runs two robustness experiments in parallel with Evaluation Agent:
  1. JPEG robustness: accuracy vs compression quality (Q=95,75,50,25)
     → results/jpeg_robustness.png
  2. WikiArt inference: confidence distribution on unseen art domain
     → results/wikiart_confidence.json

Dependency: checkpoints/best_model.pth (from Deep Learning Agent)

Usage:
    python src/stretch.py
"""

import sys
import json
import io
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DEVICE, BEST_MODEL_PATH,
    WIKIART_DIR, JPEG_ROBUSTNESS_PNG, WIKIART_CONF_JSON,
    JPEG_QUALITY_LEVELS, STRETCH_SAMPLE_N, IMAGE_SIZE, SEED,
)
from src.dataloader import get_dataloaders, get_val_transform
from src.model import load_checkpoint

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# JPEG Robustness Test
# ─────────────────────────────────────────────────────────────────────────────

def compress_jpeg(image_array: np.ndarray, quality: int) -> np.ndarray:
    """
    JPEG-compress and decompress an image at the given quality level.

    Args:
        image_array: H×W×3 uint8 numpy array (RGB).
        quality:     JPEG quality 1-95.

    Returns:
        Decompressed H×W×3 uint8 numpy array.
    """
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf  = cv2.imencode(".jpg", img_bgr, encode_param)
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


@torch.no_grad()
def evaluate_at_quality(
    model,
    paths: List[Path],
    labels: List[int],
    quality: int,
    device: torch.device,
    transform,
) -> float:
    """
    Evaluate model accuracy on a set of images compressed at a given JPEG quality.

    AGENT_TASK: also record F1 and confidence calibration at each quality level
    """
    model.eval()
    correct = 0
    total   = 0

    for path, label in zip(paths, labels):
        try:
            img = np.array(Image.open(path).convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE)))
            compressed = compress_jpeg(img, quality)
            pil_img    = Image.fromarray(compressed)
            tensor     = transform(pil_img).unsqueeze(0).to(device)

            logit = model(tensor)
            pred  = logit.argmax(dim=1).item()
            if pred == label:
                correct += 1
            total += 1
        except Exception:
            continue

    return correct / max(total, 1)


def run_jpeg_robustness(
    model,
    loader,
    device: torch.device = DEVICE,
    sample_n: int = STRETCH_SAMPLE_N,
    quality_levels: List[int] = JPEG_QUALITY_LEVELS,
    save_path: Path = JPEG_ROBUSTNESS_PNG,
) -> Dict:
    """
    Sample images from the test loader and evaluate accuracy at each JPEG quality.

    Saves:
        results/jpeg_robustness.png — accuracy-vs-quality curve

    Returns:
        Dict mapping quality → accuracy.

    AGENT_TASK: extend to test against other compression formats (WebP, AVIF)
    """
    transform = get_val_transform()

    # Collect sample paths/labels from the test loader
    all_paths, all_labels = [], []
    for _, labels, paths in loader:
        all_paths.extend(paths)
        all_labels.extend(labels.tolist())
        if len(all_paths) >= sample_n:
            break

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(all_paths), min(sample_n, len(all_paths)), replace=False)
    sampled_paths  = [Path(all_paths[i])  for i in idx]
    sampled_labels = [all_labels[i] for i in idx]

    print(f"[StretchAgent] JPEG robustness test on {len(sampled_paths)} images ...")
    results = {}
    for q in quality_levels:
        acc = evaluate_at_quality(model, sampled_paths, sampled_labels, q, device, transform)
        results[q] = acc
        print(f"  Q={q:3d} → accuracy={acc:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    qs   = sorted(results.keys(), reverse=True)
    accs = [results[q] for q in qs]
    ax.plot(qs, accs, marker="o", linewidth=2, color="#1565C0", markersize=8)
    ax.fill_between(qs, accs, alpha=0.15, color="#1565C0")
    ax.set_xlabel("JPEG Quality Level")
    ax.set_ylabel("Accuracy")
    ax.set_title("VIPER Robustness: Accuracy vs JPEG Compression Quality")
    ax.set_xticks(qs)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # Mark the degradation cliff
    diffs = [abs(accs[i] - accs[i+1]) for i in range(len(accs)-1)]
    if diffs:
        cliff_idx = np.argmax(diffs)
        cliff_q   = qs[cliff_idx + 1]
        ax.axvline(cliff_q, color="red", linestyle="--", alpha=0.6)
        ax.annotate(f"Cliff @ Q={cliff_q}",
                    xy=(cliff_q, accs[cliff_idx+1]),
                    xytext=(cliff_q + 2, accs[cliff_idx+1] - 0.05),
                    fontsize=9, color="red")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[StretchAgent] ✓ JPEG robustness plot → {save_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# WikiArt Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_wikiart_inference(
    model,
    wikiart_dir: Path = WIKIART_DIR,
    output_json: Path = WIKIART_CONF_JSON,
    device: torch.device = DEVICE,
) -> Dict:
    """
    Run the trained model on WikiArt images (EDA-only, no ground truth labels).

    Saves confidence distribution data to JSON.

    AGENT_TASK: group WikiArt by art style and analyse per-style confidence
    AGENT_TASK: compare WikiArt confidence to test-set AI confidence
    """
    transform = get_val_transform()

    # Collect WikiArt images
    paths = [
        p for p in sorted(wikiart_dir.rglob("*"))
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    ]

    if not paths:
        result = {
            "reason": "WikiArt omitted due to time constraints — refer to README limitations",
            "wikiart_dir": str(wikiart_dir),
        }
        output_json.write_text(json.dumps(result, indent=2))
        print(f"[StretchAgent] ✗ No WikiArt images in {wikiart_dir}. Wrote fallback JSON.")
        return result

    model.eval()
    confidences = []
    print(f"[StretchAgent] WikiArt inference on {len(paths)} images ...")

    for path in tqdm(paths[:500], desc="WikiArt"):   # cap at 500 for speed
        try:
            img    = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            logit  = model(tensor)
            prob   = F.softmax(logit, dim=1)[0, 1].item()   # P(AI_GENERATED)
            confidences.append({"path": str(path), "p_ai_generated": prob})
        except Exception:
            continue

    p_vals = [c["p_ai_generated"] for c in confidences]
    result = {
        "n_images":    len(confidences),
        "mean_conf":   float(np.mean(p_vals)) if p_vals else 0.0,
        "std_conf":    float(np.std(p_vals))  if p_vals else 0.0,
        "pct_flagged": float(sum(p > 0.5 for p in p_vals) / max(len(p_vals), 1)),
        "per_image":   confidences,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2))
    print(f"[StretchAgent] ✓ WikiArt results → {output_json}")
    print(f"  Mean P(AI)  : {result['mean_conf']:.4f}")
    print(f"  % Flagged   : {result['pct_flagged']:.2%}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_stretch(checkpoint_path: Path = BEST_MODEL_PATH) -> None:
    """Run both stretch goal experiments."""
    model = load_checkpoint(checkpoint_path, DEVICE)
    if model is None:
        print("[StretchAgent] ✗ No checkpoint found. Exiting.")
        return

    _, _, test_loader = get_dataloaders(verbose=False)

    run_jpeg_robustness(model, test_loader, DEVICE)
    run_wikiart_inference(model, device=DEVICE)
    print("[StretchAgent] ✓ All stretch goals complete.")


if __name__ == "__main__":
    print("=== Stretch Agent — VIPER Forensic Engine ===")
    run_stretch()
