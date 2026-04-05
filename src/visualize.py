"""
src/visualize.py - VIPER Forensic Engine: Interpretability Layer

Generates:
  1. Grad-CAM++ heatmaps for correct and misclassified images
     -> saved to gradcam_gallery/
  2. UMAP scatter plot
     -> saved to omni_export/umap_scatter.png
  3. Omni metadata CSV linking images to UMAP coordinates
     -> saved to omni_export/metadata.csv

Usage:
    python src/visualize.py
"""

import sys
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

matplotlib.use("Agg")

try:
    from pytorch_grad_cam import GradCAMPlusPlus as TorchGradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError as exc:
    TorchGradCAMPlusPlus = None
    ClassifierOutputTarget = None
    _GRADCAM_IMPORT_ERROR = exc
else:
    _GRADCAM_IMPORT_ERROR = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    GRADCAM_DIR,
    IMAGE_SIZE,
    OMNI_DIR,
    OMNI_METADATA_CSV,
    UMAP_FEATURES_CSV,
    UMAP_SCATTER_PNG,
)
from src.dataloader import get_dataloaders  # noqa: E402
from src.model import VIPERClassifier, load_checkpoint  # noqa: E402


class GradCAMPlusPlusEngine:
    """
    Wrapper around pytorch-grad-cam's Grad-CAM++ implementation.

    AGENT_TASK: add EigenCAM as a gradient-free alternative
    """

    def __init__(self, model: VIPERClassifier, target_layer: torch.nn.Module):
        if TorchGradCAMPlusPlus is None:
            raise ImportError(
                "pytorch-grad-cam is required for Grad-CAM++ visualizations."
            ) from _GRADCAM_IMPORT_ERROR

        self.cam = TorchGradCAMPlusPlus(
            model=model,
            target_layers=[target_layer],
        )

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a normalized Grad-CAM++ heatmap."""
        targets = None
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]

        cam = self.cam(input_tensor=image_tensor, targets=targets)[0]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)

    def close(self) -> None:
        """Release hooks registered by pytorch-grad-cam."""
        activations_and_grads = getattr(self.cam, "activations_and_grads", None)
        release = getattr(activations_and_grads, "release", None)
        if callable(release):
            release()


def overlay_heatmap(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a Grad-CAM++ heatmap on the original image."""
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = (cm.jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * heatmap_colored + (1 - alpha) * original_img).astype(np.uint8)
    return blended


def generate_gradcam_gallery(
    model: VIPERClassifier,
    loader,
    device: torch.device = DEVICE,
    n_correct: int = 20,
    n_misclassified: int = 5,
    save_dir: Path = GRADCAM_DIR,
) -> List[str]:
    """
    Generate Grad-CAM++ heatmaps for correct and misclassified examples.

    Saves:
        gradcam_gallery/correct_XXXX.png
        gradcam_gallery/misclassified_XXXX.png

    Returns:
        List of saved file paths.

    AGENT_TASK: add side-by-side comparison grid visualization
    AGENT_TASK: rank misclassifications by confidence for more informative selection
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    gcam = GradCAMPlusPlusEngine(model, model.gradcam_target_layer)

    correct_saved: List[str] = []
    misclassified_saved: List[str] = []

    print("[VizAgent] Generating Grad-CAM++ gallery ...")

    model.eval()
    try:
        for images, labels, paths in tqdm(loader, desc="Grad-CAM++"):
            if len(correct_saved) >= n_correct and len(misclassified_saved) >= n_misclassified:
                break

            for i in range(len(images)):
                if len(correct_saved) >= n_correct and len(misclassified_saved) >= n_misclassified:
                    break

                img_t = images[i : i + 1].to(device)
                label = labels[i].item()
                path = paths[i]

                with torch.enable_grad():
                    logits = model(img_t)
                    pred = logits.argmax(dim=1).item()

                is_correct = pred == label
                if is_correct and len(correct_saved) < n_correct:
                    prefix = "correct"
                elif (not is_correct) and len(misclassified_saved) < n_misclassified:
                    prefix = "misclassified"
                else:
                    continue

                with torch.enable_grad():
                    heatmap = gcam.generate(img_t, target_class=pred)

                try:
                    orig = cv2.imread(str(path))
                    orig = cv2.cvtColor(
                        cv2.resize(orig, (IMAGE_SIZE, IMAGE_SIZE)),
                        cv2.COLOR_BGR2RGB,
                    )
                except Exception:
                    continue

                blended = overlay_heatmap(orig, heatmap)

                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(orig)
                axes[0].set_title(f"Original\nTrue: {CLASS_NAMES[label]}")
                axes[0].axis("off")

                axes[1].imshow(blended)
                axes[1].set_title(f"Grad-CAM++\nPred: {CLASS_NAMES[pred]}")
                axes[1].axis("off")

                idx = len(correct_saved) if is_correct else len(misclassified_saved)
                save_path = save_dir / f"{prefix}_{idx:04d}.png"
                fig.tight_layout()
                fig.savefig(save_path, dpi=120)
                plt.close(fig)

                if is_correct:
                    correct_saved.append(str(save_path))
                else:
                    misclassified_saved.append(str(save_path))
    finally:
        gcam.close()

    all_saved = correct_saved + misclassified_saved
    print(
        f"[VizAgent] Saved {len(correct_saved)} correct + "
        f"{len(misclassified_saved)} misclassified Grad-CAM++ views -> {save_dir}"
    )
    return all_saved


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
        print(f"[VizAgent] {umap_csv} not found. Run evaluate.py first.")
        return

    df = pd.read_csv(umap_csv)
    label_map = {0: "Real", 1: "AI-Generated"}
    colors = {0: "#4CAF50", 1: "#E53935"}

    fig, ax = plt.subplots(figsize=(10, 8))

    for label_id, group in df.groupby("label"):
        ax.scatter(
            group["umap_x"],
            group["umap_y"],
            c=colors[label_id],
            label=label_map[label_id],
            alpha=0.5,
            s=12,
            edgecolors="none",
        )

    ax.set_title(
        "UMAP Projection of EfficientNet-B0 Feature Space\n"
        "VIPER Forensic Engine - ArtHeist 2026",
        fontsize=13,
    )
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(markerscale=3, fontsize=10)
    ax.grid(alpha=0.2)

    save_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_png, dpi=150)
    plt.close(fig)
    print(f"[VizAgent] UMAP scatter -> {save_png}")


def build_omni_metadata(
    umap_csv: Path = UMAP_FEATURES_CSV,
    gradcam_dir: Path = GRADCAM_DIR,
    output_csv: Path = OMNI_METADATA_CSV,
) -> None:
    """
    Build the Omni metadata CSV linking UMAP points to Grad-CAM files.

    AGENT_TASK: add confidence score column from eval predictions
    """
    if not umap_csv.exists():
        return

    df = pd.read_csv(umap_csv)
    gradcam_files = sorted(gradcam_dir.glob("*.png"))
    gradcam_map = {f.stem: str(f) for f in gradcam_files}

    df["gradcam_path"] = df["image_path"].apply(lambda p: gradcam_map.get(Path(p).stem, ""))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[VizAgent] Omni metadata -> {output_csv} ({len(df)} rows)")


def visualize(checkpoint_path: Path = BEST_MODEL_PATH) -> None:
    """Run the full visualization pipeline."""
    model = load_checkpoint(checkpoint_path, DEVICE)
    if model is None:
        GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
        OMNI_DIR.mkdir(parents=True, exist_ok=True)
        (GRADCAM_DIR / ".placeholder").touch()
        (OMNI_DIR / ".placeholder").touch()
        print("[VizAgent] No checkpoint found. Created placeholder files.")
        return

    _, _, test_loader = get_dataloaders(verbose=False)

    generate_gradcam_gallery(model, test_loader, DEVICE)
    plot_umap_scatter()
    build_omni_metadata()

    print("[VizAgent] Visualization complete.")


if __name__ == "__main__":
    print("=== Visualization Agent - VIPER Forensic Engine ===")
    visualize()
