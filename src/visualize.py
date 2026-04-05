"""
src/visualize.py - VIPER Forensic Engine: Interpretability Layer

Generates:
  1. Grad-CAM++ heatmaps for correct and misclassified images
     -> saved to gradcam_gallery/
  2. Interactive UMAP scatter
     -> saved to omni_export/umap_scatter.html
  3. Omni metadata CSV linking images to UMAP coordinates
     -> saved to omni_export/metadata.csv

Usage:
    python src/visualize.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as exc:
    px = None
    go = None
    _PLOTLY_IMPORT_ERROR = exc
else:
    _PLOTLY_IMPORT_ERROR = None

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
    BASELINE_METRICS_JSON,
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    EVAL_METRICS_JSON,
    GRADCAM_DIR,
    IMAGE_SIZE,
    OMNI_DIR,
    OMNI_METADATA_CSV,
    UMAP_FEATURES_CSV,
    UMAP_SCATTER_PNG,
)
from src.dataloader import get_dataloaders, get_val_transform  # noqa: E402
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


def _require_plotly() -> None:
    if px is None or go is None:
        raise ImportError(
            "plotly is required for interactive VIPER visualizations."
        ) from _PLOTLY_IMPORT_ERROR


def _load_metric_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _predict_umap_confidence(
    model: VIPERClassifier,
    image_paths: Sequence[str],
    device: torch.device = DEVICE,
    batch_size: int = 32,
) -> Dict[str, List[float]]:
    transform = get_val_transform()
    ai_confidence: List[float] = []
    predicted_label: List[int] = []
    prediction_confidence: List[float] = []

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), batch_size), desc="UMAP confidence"):
            batch_paths = image_paths[start:start + batch_size]
            batch_tensors = []
            for image_path in batch_paths:
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception:
                    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=0)
                batch_tensors.append(transform(image))

            image_tensor = torch.stack(batch_tensors).to(device)
            probabilities = torch.softmax(model(image_tensor), dim=1)

            ai_confidence.extend(probabilities[:, 1].cpu().numpy().astype(float).tolist())
            predicted_label.extend(probabilities.argmax(dim=1).cpu().numpy().astype(int).tolist())
            prediction_confidence.extend(
                probabilities.max(dim=1).values.cpu().numpy().astype(float).tolist()
            )

    return {
        "ai_confidence": ai_confidence,
        "predicted_label": predicted_label,
        "prediction_confidence": prediction_confidence,
    }


def prepare_umap_dataframe(
    model: Optional[VIPERClassifier] = None,
    umap_csv: Path = UMAP_FEATURES_CSV,
    device: torch.device = DEVICE,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Load and enrich the UMAP export with confidence and error metadata.
    """
    if not umap_csv.exists():
        raise FileNotFoundError(f"{umap_csv} not found. Run evaluate.py first.")

    df = pd.read_csv(umap_csv).dropna(subset=["umap_x", "umap_y"]).copy()
    if df.empty:
        return df

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    if "ai_confidence" not in df.columns and "confidence_score" in df.columns:
        df["ai_confidence"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    if "predicted_label" not in df.columns and "ai_confidence" in df.columns:
        df["predicted_label"] = (df["ai_confidence"] >= 0.5).astype(int)
    if "prediction_confidence" not in df.columns and "ai_confidence" in df.columns:
        df["prediction_confidence"] = np.maximum(df["ai_confidence"], 1.0 - df["ai_confidence"])

    required_columns = {"ai_confidence", "predicted_label", "prediction_confidence"}
    if not required_columns.issubset(df.columns):
        if model is None:
            raise ValueError(
                "Model is required to enrich UMAP data with confidence metadata."
            )
        predictions = _predict_umap_confidence(
            model=model,
            image_paths=df["image_path"].astype(str).tolist(),
            device=device,
            batch_size=batch_size,
        )
        for column, values in predictions.items():
            df[column] = values

    label_map = {0: "Real", 1: "AI-Generated"}
    df["filename"] = df["image_path"].astype(str).map(lambda value: Path(value).name)
    df["label_name"] = df["label"].map(label_map).fillna("Unknown")
    df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce").fillna(0).astype(int)
    df["predicted_name"] = df["predicted_label"].map(label_map).fillna("Unknown")
    df["ai_confidence"] = pd.to_numeric(df["ai_confidence"], errors="coerce").fillna(0.5)
    prediction_confidence = pd.to_numeric(
        df["prediction_confidence"], errors="coerce"
    )
    fallback_confidence = pd.Series(
        np.maximum(df["ai_confidence"], 1.0 - df["ai_confidence"]),
        index=df.index,
    )
    df["prediction_confidence"] = prediction_confidence.where(
        prediction_confidence.notna(),
        fallback_confidence,
    )
    df["confidence_pct"] = df["prediction_confidence"].map(lambda score: f"{score:.1%}")
    df["misclassified"] = df["predicted_label"] != df["label"]
    df["classification_status"] = np.where(df["misclassified"], "Misclassified", "Correct")
    df["error_type"] = np.where(
        df["misclassified"],
        np.where(df["predicted_label"] == 1, "False Positive", "False Negative"),
        "Correct",
    )
    df["decision_margin"] = np.abs(df["ai_confidence"] - 0.5) * 2.0
    return df


def build_umap_scatter_figure(
    df: pd.DataFrame,
    edge_cases_only: bool = False,
) -> Any:
    """
    Build an interactive 2D UMAP scatter colored by prediction confidence.
    """
    _require_plotly()

    plot_df = df[df["misclassified"]].copy() if edge_cases_only else df.copy()
    title = (
        "VIPER Edge Cases<br><sup>Only misclassified UMAP points, colored by model confidence</sup>"
        if edge_cases_only
        else "VIPER 2D UMAP Confidence Map<br><sup>Prediction confidence drives color intensity; symbol encodes correctness</sup>"
    )

    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title={"text": title, "x": 0.5},
            template="plotly_dark",
            paper_bgcolor="#040B18",
            plot_bgcolor="#040B18",
            font={"family": "Inter, Segoe UI, sans-serif", "color": "#E5EDF8"},
            width=1180,
            height=760,
        )
        fig.add_annotation(
            text="No misclassified edge cases are available in the current export.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 15, "color": "#94A3B8"},
        )
        return fig

    symbol_column = "error_type" if edge_cases_only else "classification_status"
    symbol_map = {
        "Correct": "circle",
        "Misclassified": "x",
        "False Positive": "x",
        "False Negative": "diamond",
    }

    fig = px.scatter(
        plot_df,
        x="umap_x",
        y="umap_y",
        color="prediction_confidence",
        color_continuous_scale=[
            (0.00, "#1D4ED8"),
            (0.45, "#38BDF8"),
            (1.00, "#F97316"),
        ],
        symbol=symbol_column,
        symbol_map=symbol_map,
        hover_name="filename",
        hover_data={
            "label_name": True,
            "predicted_name": True,
            "error_type": True,
            "confidence_pct": True,
            "ai_confidence": ":.3f",
            "decision_margin": ":.3f",
            "image_path": True,
            "umap_x": ":.3f",
            "umap_y": ":.3f",
            "prediction_confidence": False,
            "misclassified": False,
            "classification_status": False,
        },
        labels={
            "umap_x": "UMAP Dimension 1",
            "umap_y": "UMAP Dimension 2",
            "prediction_confidence": "Prediction confidence",
            "label_name": "True class",
            "predicted_name": "Predicted class",
            "ai_confidence": "AI probability",
            "decision_margin": "Decision margin",
            "image_path": "Image path",
            "error_type": "Failure mode",
            "confidence_pct": "Confidence",
        },
        title=title,
        template="plotly_dark",
        opacity=0.86,
        render_mode="webgl",
    )

    for trace in fig.data:
        trace.marker.size = 13 if trace.name in {"Misclassified", "False Positive", "False Negative"} else 10
        trace.marker.line = {"width": 0.8, "color": "rgba(2, 8, 23, 0.85)"}

    fig.update_layout(
        width=1180,
        height=760,
        paper_bgcolor="#040B18",
        plot_bgcolor="#040B18",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#E5EDF8"},
        title={"x": 0.5},
        margin={"l": 60, "r": 40, "t": 90, "b": 60},
        legend_title_text="Point type",
        coloraxis_colorbar={"title": "Confidence"},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.16)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.16)", zeroline=False)
    fig.add_annotation(
        text=f"{len(plot_df)} points | {int(plot_df['misclassified'].sum())} errors",
        x=0.995,
        y=1.04,
        xref="paper",
        yref="paper",
        xanchor="right",
        showarrow=False,
        font={"size": 12, "color": "#93C5FD"},
    )
    return fig


def plot_umap_scatter(
    model: Optional[VIPERClassifier] = None,
    umap_csv: Path = UMAP_FEATURES_CSV,
    save_png: Path = UMAP_SCATTER_PNG,
    save_html: Optional[Path] = None,
    device: torch.device = DEVICE,
    edge_cases_only: bool = False,
    prepared_df: Optional[pd.DataFrame] = None,
    batch_size: int = 32,
) -> Any:
    """
    Export the full UMAP scatter or the misclassified edge-case view.
    """
    df = prepared_df if prepared_df is not None else prepare_umap_dataframe(
        model=model,
        umap_csv=umap_csv,
        device=device,
        batch_size=batch_size,
    )
    fig = build_umap_scatter_figure(df=df, edge_cases_only=edge_cases_only)

    default_html = (
        save_png.with_name(f"{save_png.stem}_edge_cases.html")
        if edge_cases_only
        else save_png.with_suffix(".html")
    )
    save_html = save_html or default_html
    save_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_html, include_plotlyjs="cdn", full_html=True)
    print(f"[VizAgent] Interactive UMAP -> {save_html}")
    return fig


def plot_fast_track_comparison(
    eval_metrics_path: Path = EVAL_METRICS_JSON,
    baseline_metrics_path: Path = BASELINE_METRICS_JSON,
    save_html: Optional[Path] = None,
) -> Any:
    """
    Compare the baseline against the fast-track fine-tuned model with Plotly.
    """
    _require_plotly()

    eval_metrics = _load_metric_json(eval_metrics_path)
    baseline_metrics = _load_metric_json(baseline_metrics_path)
    comparison_rows = [
        {
            "metric": "Accuracy",
            "Baseline": float(baseline_metrics["accuracy"]),
            "Fast-Track": float(eval_metrics.get("test_accuracy", eval_metrics["accuracy"])),
        },
        {
            "metric": "F1",
            "Baseline": float(baseline_metrics["f1"]),
            "Fast-Track": float(eval_metrics.get("test_f1", eval_metrics["f1"])),
        },
        {
            "metric": "AUC-ROC",
            "Baseline": float(baseline_metrics["auc_roc"]),
            "Fast-Track": float(eval_metrics.get("test_auc_roc", eval_metrics["auc_roc"])),
        },
    ]

    plot_df = pd.DataFrame(comparison_rows).melt(
        id_vars="metric",
        value_vars=["Baseline", "Fast-Track"],
        var_name="model",
        value_name="score",
    )

    fig = px.bar(
        plot_df,
        x="metric",
        y="score",
        color="model",
        barmode="group",
        text="score",
        color_discrete_map={"Baseline": "#64748B", "Fast-Track": "#F97316"},
        title="Baseline vs Fast-Track Scale-Up<br><sup>Held-out metrics show the lift from the fine-tuned ConvNeXt run</sup>",
        template="plotly_dark",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        width=940,
        height=620,
        paper_bgcolor="#040B18",
        plot_bgcolor="#040B18",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#E5EDF8"},
        title={"x": 0.5},
        yaxis_title="Score",
        xaxis_title="Metric",
        legend_title_text="Model",
        yaxis={"range": [0.0, 1.08]},
        margin={"l": 60, "r": 30, "t": 90, "b": 60},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.16)", zeroline=False)

    for row in comparison_rows:
        delta = row["Fast-Track"] - row["Baseline"]
        fig.add_annotation(
            x=row["metric"],
            y=max(row["Baseline"], row["Fast-Track"]) + 0.06,
            text=f"{delta:+.1%}",
            showarrow=False,
            font={"size": 13, "color": "#FDBA74"},
        )

    if save_html is not None:
        save_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_html, include_plotlyjs="cdn", full_html=True)
        print(f"[VizAgent] Fast-track comparison -> {save_html}")

    return fig


def build_omni_metadata(
    umap_csv: Path = UMAP_FEATURES_CSV,
    gradcam_dir: Path = GRADCAM_DIR,
    output_csv: Path = OMNI_METADATA_CSV,
    prepared_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the Omni metadata CSV linking UMAP points to Grad-CAM files and scores.
    """
    if prepared_df is not None:
        df = prepared_df.copy()
    elif umap_csv.exists():
        df = pd.read_csv(umap_csv)
    else:
        return pd.DataFrame()

    gradcam_files = sorted(gradcam_dir.glob("*.png"))
    gradcam_map = {f.stem: str(f) for f in gradcam_files}
    df["gradcam_path"] = df["image_path"].apply(lambda p: gradcam_map.get(Path(p).stem, ""))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[VizAgent] Omni metadata -> {output_csv} ({len(df)} rows)")
    return df


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
    umap_df = prepare_umap_dataframe(model=model, device=DEVICE)
    plot_umap_scatter(model=model, device=DEVICE, prepared_df=umap_df)
    plot_umap_scatter(
        model=model,
        device=DEVICE,
        prepared_df=umap_df,
        edge_cases_only=True,
    )
    build_omni_metadata(prepared_df=umap_df)
    plot_fast_track_comparison(save_html=OMNI_DIR / "fast_track_comparison.html")

    print("[VizAgent] Visualization complete.")


if __name__ == "__main__":
    print("=== Visualization Agent - VIPER Forensic Engine ===")
    visualize()
