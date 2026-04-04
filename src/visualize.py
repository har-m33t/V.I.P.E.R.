"""
src/visualize.py — VIPER Forensic Engine: Interpretability Layer
Phase: Track Gamma (Deep Learning) — Visualization Agent

Generates:
  1. Grad-CAM heatmaps for 20 correct + 5 misclassified images
     → saved to gradcam_gallery/
  2. UMAP scatter plot
     → saved to omni_export/umap_scatter.png
  3. Omni metadata CSV linking images to UMAP coordinates
     → saved to omni_export/metadata.csv

Dependency: checkpoints/best_model.pth + results/umap_features.csv

Usage:
    python src/visualize.py
"""

import sys
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DEVICE, BEST_MODEL_PATH,
    GRADCAM_DIR, OMNI_DIR, OMNI_METADATA_CSV, UMAP_SCATTER_PNG,
    UMAP_FEATURES_CSV, RESULTS_DIR, IMAGE_SIZE, CLASS_NAMES,
    EVAL_METRICS_JSON,
)
from src.dataloader import get_dataloaders, get_val_transform
from src.model import load_checkpoint, VIPERClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM Implementation
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Hooks into the target layer to capture activations and gradients
    during a forward/backward pass.

    AGENT_TASK: implement GradCAM++ for sharper localisation
    AGENT_TASK: add EigenCAM as a gradient-free alternative
    """

    def __init__(self, model: VIPERClassifier, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients:   Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def _forward_hook(module, input, output):
            self.activations = output.detach()

        def _backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(_forward_hook)
        self.target_layer.register_full_backward_hook(_backward_hook)

    def generate(
        self,
        image_tensor: torch.Tensor,   # (1, C, H, W)
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a normalized Grad-CAM heatmap.

        Args:
            image_tensor: Single image tensor on the model device.
            target_class: Class index to compute CAM for (defaults to predicted class).

        Returns:
            H×W numpy array of heatmap values in [0, 1].
        """
        self.model.zero_grad()
        logits = self.model(image_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        score = logits[0, target_class]
        score.backward()

        # Grad-CAM: global-average pooled gradients × activations
        grads = self.gradients[0]          # (C, H, W)
        acts  = self.activations[0]        # (C, H, W)
        weights = grads.mean(dim=(1, 2))   # (C,)

        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam)

        # Normalize
        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(
    original_img: np.ndarray,   # H×W×3 uint8 RGB
    heatmap: np.ndarray,        # H×W float [0,1]
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_img: RGB uint8 image.
        heatmap:      Normalized CAM array.
        alpha:        Heatmap transparency.

    Returns:
        Blended RGB uint8 image.
    """
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = (cm.jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * heatmap_colored + (1 - alpha) * original_img).astype(np.uint8)
    return blended


# ─────────────────────────────────────────────────────────────────────────────
# Gallery Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_gradcam_gallery(
    model: VIPERClassifier,
    loader,
    device: torch.device = DEVICE,
    n_correct: int = 20,
    n_misclassified: int = 5,
    save_dir: Path = GRADCAM_DIR,
) -> List[str]:
    """
    Generate Grad-CAM heatmaps for correct and misclassified examples.

    Saves:
        gradcam_gallery/correct_XXXX.png
        gradcam_gallery/misclassified_XXXX.png

    Returns:
        List of saved file paths.

    AGENT_TASK: add side-by-side comparison grid visualization
    AGENT_TASK: rank misclassifications by confidence for more informative selection
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    gcam = GradCAM(model, model.gradcam_target_layer)

    correct_saved       = []
    misclassified_saved = []
    val_transform       = get_val_transform()

    print(f"[VizAgent] Generating Grad-CAM gallery ...")

    model.eval()
    for images, labels, paths in tqdm(loader, desc="Grad-CAM"):
        if (len(correct_saved) >= n_correct and
                len(misclassified_saved) >= n_misclassified):
            break

        for i in range(len(images)):
            if (len(correct_saved) >= n_correct and
                    len(misclassified_saved) >= n_misclassified):
                break

            img_t  = images[i:i+1].to(device)
            label  = labels[i].item()
            path   = paths[i]

            with torch.enable_grad():
                logits = model(img_t)
                pred   = logits.argmax(dim=1).item()

            is_correct = (pred == label)

            # Decide whether to save this example
            if is_correct and len(correct_saved) < n_correct:
                prefix = "correct"
            elif not is_correct and len(misclassified_saved) < n_misclassified:
                prefix = "misclassified"
            else:
                continue

            # Generate CAM
            with torch.enable_grad():
                heatmap = gcam.generate(img_t, target_class=pred)

            # Load original image for overlay
            try:
                orig = cv2.imread(str(path))
                orig = cv2.cvtColor(cv2.resize(orig, (IMAGE_SIZE, IMAGE_SIZE)),
                                    cv2.COLOR_BGR2RGB)
            except Exception:
                continue

            blended = overlay_heatmap(orig, heatmap)

            # Compose figure
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(orig)
            axes[0].set_title(f"Original\nTrue: {CLASS_NAMES[label]}")
            axes[0].axis("off")

            axes[1].imshow(blended)
            axes[1].set_title(f"Grad-CAM\nPred: {CLASS_NAMES[pred]}")
            axes[1].axis("off")

            idx = len(correct_saved) if is_correct else len(misclassified_saved)
            fname = f"{prefix}_{idx:04d}.png"
            save_path = save_dir / fname
            fig.tight_layout()
            fig.savefig(save_path, dpi=120)
            plt.close(fig)

            if is_correct:
                correct_saved.append(str(save_path))
            else:
                misclassified_saved.append(str(save_path))

    all_saved = correct_saved + misclassified_saved
    print(f"[VizAgent] ✓ Saved {len(correct_saved)} correct + "
          f"{len(misclassified_saved)} misclassified Grad-CAMs → {save_dir}")
    return all_saved


# ─────────────────────────────────────────────────────────────────────────────
# UMAP Scatter Plot (Omni Track)
# ─────────────────────────────────────────────────────────────────────────────

def plot_umap_scatter(
    umap_csv: Path = UMAP_FEATURES_CSV,
    save_png: Path = UMAP_SCATTER_PNG,
) -> None:
    """
    Plot 2D UMAP scatter coloured by class label.

    AGENT_TASK: add hover labels with Plotly for interactive exploration
    AGENT_TASK: overlay centroids and decision boundaries
    """
    if not umap_csv.exists():
        print(f"[VizAgent] ✗ {umap_csv} not found. Run evaluate.py first.")
        return

    df = pd.read_csv(umap_csv)
    label_map = {0: "Real", 1: "AI-Generated"}
    colors    = {0: "#4CAF50", 1: "#E53935"}

    fig, ax = plt.subplots(figsize=(10, 8))

    for label_id, group in df.groupby("label"):
        ax.scatter(
            group["umap_x"], group["umap_y"],
            c=colors[label_id],
            label=label_map[label_id],
            alpha=0.5,
            s=12,
            edgecolors="none",
        )

    ax.set_title("UMAP Projection of EfficientNet-B0 Feature Space\n"
                 "VIPER Forensic Engine — ArtHeist 2026", fontsize=13)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(markerscale=3, fontsize=10)
    ax.grid(alpha=0.2)

    save_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_png, dpi=150)
    plt.close(fig)
    print(f"[VizAgent] ✓ UMAP scatter → {save_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Omni Metadata CSV
# ─────────────────────────────────────────────────────────────────────────────

def build_omni_metadata(
    umap_csv:    Path = UMAP_FEATURES_CSV,
    gradcam_dir: Path = GRADCAM_DIR,
    output_csv:  Path = OMNI_METADATA_CSV,
) -> None:
    """
    Build the Omni metadata CSV linking UMAP points to Grad-CAM files.

    AGENT_TASK: add confidence score column from eval predictions
    """
    if not umap_csv.exists():
        return

    df = pd.read_csv(umap_csv)
    gradcam_files = sorted(gradcam_dir.glob("*.png"))
    gradcam_map   = {f.stem: str(f) for f in gradcam_files}

    # Best-effort: link by filename stem
    df["gradcam_path"] = df["image_path"].apply(
        lambda p: gradcam_map.get(Path(p).stem, "")
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[VizAgent] ✓ Omni metadata → {output_csv}  ({len(df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Main Visualization Routine
# ─────────────────────────────────────────────────────────────────────────────

def visualize(checkpoint_path: Path = BEST_MODEL_PATH) -> None:
    """Full visualization pipeline."""
    model = load_checkpoint(checkpoint_path, DEVICE)
    if model is None:
        # Fail-forward: create empty placeholder files
        GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
        OMNI_DIR.mkdir(parents=True, exist_ok=True)
        (GRADCAM_DIR / ".placeholder").touch()
        (OMNI_DIR / ".placeholder").touch()
        print("[VizAgent] ✗ No checkpoint. Created placeholder files.")
        return

    _, val_loader, test_loader = get_dataloaders(verbose=False)

    # Grad-CAM gallery
    generate_gradcam_gallery(model, test_loader, DEVICE)

    # UMAP scatter
    plot_umap_scatter()

    # Omni metadata
    build_omni_metadata()

    print("[VizAgent] ✓ Visualization complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Visualization Agent — VIPER Forensic Engine ===")
    visualize()
