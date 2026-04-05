from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
GRADCAM_DIR = ROOT / "gradcam_gallery"

VALIDATION_PREDICTIONS_CSV = RESULTS_DIR / "validation_predictions.csv"
VALIDATION_PREDICTIONS_JSON = RESULTS_DIR / "validation_predictions.json"
VALIDATION_EMBEDDINGS_CSV = RESULTS_DIR / "validation_embeddings.csv"
VALIDATION_SUMMARY_JSON = RESULTS_DIR / "validation_summary.json"
BASELINE_METRICS_JSON = RESULTS_DIR / "baseline_metrics.json"
EVAL_METRICS_JSON = RESULTS_DIR / "eval_metrics.json"

DARK_CSS = """
<style>
  :root {
    --bg: #07111f;
    --panel: #0c182a;
    --panel-soft: #111f35;
    --line: rgba(132, 154, 196, 0.22);
    --text: #edf4ff;
    --muted: #9fb4d3;
    --accent: #4fd1c5;
    --warn: #f59e0b;
    --danger: #fb7185;
  }
  .stApp {
    background:
      radial-gradient(circle at top left, rgba(79, 209, 197, 0.12), transparent 28%),
      radial-gradient(circle at top right, rgba(59, 130, 246, 0.14), transparent 24%),
      linear-gradient(180deg, #06101c 0%, #091423 100%);
    color: var(--text);
  }
  .block-container {
    max-width: 1260px;
    padding-top: 2rem;
    padding-bottom: 2rem;
  }
  div[data-testid="stMetric"] {
    background: rgba(12, 24, 42, 0.9);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 0.9rem 1rem;
  }
  div[data-testid="stFileUploader"] {
    background: rgba(12, 24, 42, 0.75);
    border: 1px dashed rgba(132, 154, 196, 0.35);
    border-radius: 18px;
    padding: 0.5rem;
  }
  .viper-banner {
    border: 1px solid var(--line);
    background: linear-gradient(135deg, rgba(12, 24, 42, 0.96), rgba(17, 31, 53, 0.92));
    border-radius: 22px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 24px 80px rgba(0, 0, 0, 0.26);
  }
  .viper-kicker {
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.78rem;
    margin-bottom: 0.45rem;
  }
  .viper-title {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 0;
  }
  .viper-subtitle {
    color: var(--muted);
    margin-top: 0.55rem;
    max-width: 58rem;
  }
</style>
"""


def _mtime(path: Path) -> int:
    return path.stat().st_mtime_ns if path.exists() else 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_json_artifact(path_str: str, version: int) -> dict[str, Any]:
    del version
    return _read_json(Path(path_str))


@st.cache_data(show_spinner=False)
def load_csv_artifact(path_str: str, version: int) -> pd.DataFrame:
    del version
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def compute_projection(embeddings: np.ndarray) -> tuple[np.ndarray, str]:
    if len(embeddings) == 0:
        return np.empty((0, 2), dtype=np.float32), "none"
    if len(embeddings) == 1:
        return np.zeros((1, 2), dtype=np.float32), "degenerate"

    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=max(2, min(15, len(embeddings) - 1)),
            min_dist=0.10,
            n_components=2,
            metric="cosine",
            random_state=42,
            verbose=False,
        )
        return reducer.fit_transform(embeddings).astype(np.float32), "umap"
    except Exception:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        return coords.astype(np.float32), "pca_fallback"


@st.cache_resource(show_spinner=False)
def load_validation_projection(
    predictions_path: str,
    embeddings_path: str,
    predictions_version: int,
    embeddings_version: int,
) -> pd.DataFrame:
    del predictions_version, embeddings_version
    predictions_df = load_csv_artifact(predictions_path, _mtime(Path(predictions_path)))
    if predictions_df.empty:
        return predictions_df

    if {
        "umap_x",
        "umap_y",
    }.issubset(predictions_df.columns) and predictions_df[["umap_x", "umap_y"]].notna().all().all():
        return predictions_df

    embeddings_df = load_csv_artifact(embeddings_path, _mtime(Path(embeddings_path)))
    embedding_columns = [column for column in embeddings_df.columns if column.startswith("emb_")]
    if not embedding_columns:
        return predictions_df

    aligned_embeddings = (
        predictions_df[["image_path"]]
        .merge(embeddings_df, on="image_path", how="left")
    )
    coords, _ = compute_projection(aligned_embeddings[embedding_columns].to_numpy(dtype=np.float32))
    projected_df = predictions_df.copy()
    projected_df["umap_x"] = coords[:, 0]
    projected_df["umap_y"] = coords[:, 1]
    return projected_df


@st.cache_data(show_spinner=False)
def build_error_breakdown(predictions_path: str, version: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions_df = load_csv_artifact(predictions_path, version)
    if predictions_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    errors_df = predictions_df[predictions_df["error_type"] != "correct"].copy()
    if errors_df.empty:
        return pd.DataFrame(), errors_df

    breakdown_df = (
        errors_df.groupby(["error_type", "confidence_bucket"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    breakdown_df["failure_mode"] = (
        breakdown_df["error_type"].str.replace("_", " ").str.title()
        + " / "
        + breakdown_df["confidence_bucket"].str.title()
    )
    return breakdown_df, errors_df


@st.cache_data(show_spinner=False)
def list_gradcam_samples(directory: str, version: int) -> list[str]:
    del version
    return [path.name for path in sorted(Path(directory).glob("*.png"))]


@st.cache_resource(show_spinner=False)
def load_runtime() -> tuple[Optional[Any], Optional[str]]:
    try:
        from backend.app import ModelRuntime

        runtime = ModelRuntime()
        if not runtime.ready:
            return None, runtime.detail
        return runtime, None
    except Exception as exc:
        return None, str(exc)


def generate_gradcam_overlay(runtime: Any, image_bytes: bytes, predicted_index: int) -> tuple[Optional[np.ndarray], Optional[str]]:
    try:
        from backend.app import (
            ClassifierOutputTarget,
            GradCAMPlusPlus,
            _FusionGradCAMWrapper,
        )
        from src.config import IMAGE_SIZE
        from src.eda import extract_features_from_rgb
        from src.visualize import overlay_heatmap
    except Exception as exc:
        return None, f"Grad-CAM dependencies unavailable: {exc}"

    if runtime.model is None or GradCAMPlusPlus is None or ClassifierOutputTarget is None:
        return None, "Grad-CAM++ is unavailable in the current environment."

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    resized_rgb = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.uint8)
    features = extract_features_from_rgb(resized_rgb)
    image_tensor = runtime.transform(image).unsqueeze(0).to(runtime.device)
    eda_tensor = runtime._standardize_eda_features(features)

    wrapper = _FusionGradCAMWrapper(runtime.model, eda_features=eda_tensor)
    cam = GradCAMPlusPlus(model=wrapper, target_layers=[runtime.model.gradcam_target_layer])
    try:
        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=[ClassifierOutputTarget(int(predicted_index))],
        )[0].astype(np.float32)
    finally:
        activations_and_grads = getattr(cam, "activations_and_grads", None)
        release = getattr(activations_and_grads, "release", None)
        if callable(release):
            release()

    return overlay_heatmap(resized_rgb, grayscale_cam), None


@st.cache_data(show_spinner=False, max_entries=16)
def analyze_uploaded_image(image_bytes: bytes, filename: str) -> tuple[dict[str, Any], Optional[np.ndarray], Optional[str]]:
    runtime, runtime_error = load_runtime()
    if runtime is None:
        return {}, None, runtime_error

    response = runtime.predict(image_bytes=image_bytes, filename=filename)
    response_dict = response.model_dump() if hasattr(response, "model_dump") else response.dict()
    overlay, overlay_error = generate_gradcam_overlay(
        runtime=runtime,
        image_bytes=image_bytes,
        predicted_index=int(response_dict["predicted_index"]),
    )
    return response_dict, overlay, overlay_error


def render_banner() -> None:
    st.markdown(
        """
        <section class="viper-banner">
          <div class="viper-kicker">VIPER Forensic Engine</div>
          <h1 class="viper-title">Validation-ready dashboard with cached model intelligence.</h1>
          <p class="viper-subtitle">
            The intelligence view reads precomputed validation artifacts only, while uploaded
            image analysis keeps the checkpoint hot in one cached runtime for live demo speed.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    st.sidebar.title("VIPER Control")
    st.sidebar.caption("Presentation-safe: validation artifacts only.")
    view = st.sidebar.radio(
        "View",
        options=("Image Forensics", "Model Intelligence"),
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.write("Artifacts")
    st.sidebar.code(
        "\n".join(
            [
                VALIDATION_PREDICTIONS_CSV.name,
                VALIDATION_EMBEDDINGS_CSV.name,
                VALIDATION_SUMMARY_JSON.name,
            ]
        )
    )
    return view


def render_metrics_strip(eval_metrics: dict[str, Any], baseline_metrics: dict[str, Any], summary: dict[str, Any]) -> None:
    summary_metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    accuracy = float(summary_metrics.get("accuracy", eval_metrics.get("accuracy", 0.0)))
    f1_score = float(summary_metrics.get("f1", eval_metrics.get("f1", 0.0)))
    auc_roc = float(summary_metrics.get("auc_roc", eval_metrics.get("auc_roc", 0.0)))

    baseline_accuracy = float(baseline_metrics.get("accuracy", 0.0))
    baseline_f1 = float(baseline_metrics.get("f1", 0.0))
    baseline_auc = float(baseline_metrics.get("auc_roc", 0.0))

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Validation Accuracy",
        f"{accuracy:.1%}",
        f"{accuracy - baseline_accuracy:+.1%} vs baseline",
    )
    metric_cols[1].metric(
        "Validation F1",
        f"{f1_score:.3f}",
        f"{f1_score - baseline_f1:+.3f} vs baseline",
    )
    metric_cols[2].metric(
        "Validation AUC",
        f"{auc_roc:.3f}",
        f"{auc_roc - baseline_auc:+.3f} vs baseline",
    )
    metric_cols[3].metric(
        "Held-out Test Accuracy",
        f"{float(eval_metrics.get('test_accuracy', 0.0)):.1%}",
        summary.get("projection_method", "precomputed"),
    )


def render_image_forensics() -> None:
    st.subheader("Image Forensics")
    st.caption("Live inference stays cached per uploaded file so tab switches do not rerun the same analysis.")

    uploader_col, gallery_col = st.columns([1.25, 0.75])
    with uploader_col:
        uploaded_file = st.file_uploader(
            "Upload an image for live inference",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=False,
        )
    with gallery_col:
        sample_names = list_gradcam_samples(str(GRADCAM_DIR), _mtime(GRADCAM_DIR))
        sample_choice = st.selectbox(
            "Cached Grad-CAM gallery",
            options=["None"] + sample_names,
            index=0,
        )

    runtime, runtime_error = load_runtime()
    if runtime_error:
        st.info(f"Live model runtime unavailable: {runtime_error}")

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        original_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        with st.spinner("Running cached VIPER inference..."):
            result, overlay, overlay_error = analyze_uploaded_image(image_bytes, uploaded_file.name)

        if not result:
            st.error(overlay_error or "Inference failed.")
            return

        image_cols = st.columns(2)
        with image_cols[0]:
            st.image(original_image, caption="Uploaded image", use_container_width=True)
        with image_cols[1]:
            if overlay is not None:
                st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
            else:
                st.warning(overlay_error or "Grad-CAM overlay unavailable.")

        verdict_cols = st.columns(4)
        verdict_cols[0].metric("Prediction", result["prediction"])
        verdict_cols[1].metric("Confidence", result["confidence_pct"])
        verdict_cols[2].metric("AI Probability", f"{float(result['ai_probability']):.1%}")
        verdict_cols[3].metric("Fusion Mode", "EDA On" if result["uses_eda_fusion"] else "Image Only")

        st.write(result["verdict"])

        evidence_df = pd.DataFrame(result["evidence_breakdown"])
        if not evidence_df.empty:
            st.dataframe(
                evidence_df[["label", "status", "score", "value", "detail"]],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Technical notes", expanded=False):
            st.write(
                "Evidence signals are reused from the backend runtime so toggling views does not rerun "
                "model loading or image-level feature extraction for the same upload."
            )
            st.json(
                {
                    "filename": result["filename"],
                    "model_name": result["model_name"],
                    "gradcam_available": result["gradcam_available"],
                }
            )
    elif sample_choice != "None":
        st.image(
            str(GRADCAM_DIR / sample_choice),
            caption=f"Cached gallery sample: {sample_choice}",
            use_container_width=True,
        )
    else:
        st.info("Upload an image or preview a cached Grad-CAM sample.")


def build_umap_figure(projection_df: pd.DataFrame, only_errors: bool) -> Any:
    filtered_df = projection_df.copy()
    if only_errors:
        filtered_df = filtered_df[filtered_df["error_type"] != "correct"]

    if filtered_df.empty:
        return None

    filtered_df["marker_group"] = filtered_df["error_type"].str.replace("_", " ").str.title()
    return px.scatter(
        filtered_df,
        x="umap_x",
        y="umap_y",
        color="predicted_confidence",
        color_continuous_scale=["#1b3358", "#2563eb", "#4fd1c5", "#f59e0b", "#fb7185"],
        symbol="marker_group",
        hover_name="filename",
        hover_data={
            "label_name": True,
            "predicted_name": True,
            "predicted_confidence": ":.1%",
            "ai_probability": ":.1%",
            "error_type": True,
            "umap_x": ":.3f",
            "umap_y": ":.3f",
        },
        title="Validation latent space colored by prediction confidence",
        labels={
            "umap_x": "Projection X",
            "umap_y": "Projection Y",
            "predicted_confidence": "Confidence",
            "marker_group": "Outcome",
        },
        template="plotly_dark",
        height=620,
    ).update_traces(marker={"size": 9, "line": {"width": 0.6, "color": "rgba(255,255,255,0.15)"}})


def render_model_intelligence() -> None:
    summary_path = VALIDATION_SUMMARY_JSON if VALIDATION_SUMMARY_JSON.exists() else VALIDATION_PREDICTIONS_JSON
    summary = load_json_artifact(str(summary_path), _mtime(summary_path))
    eval_metrics = load_json_artifact(str(EVAL_METRICS_JSON), _mtime(EVAL_METRICS_JSON))
    baseline_metrics = load_json_artifact(str(BASELINE_METRICS_JSON), _mtime(BASELINE_METRICS_JSON))
    projection_df = load_validation_projection(
        str(VALIDATION_PREDICTIONS_CSV),
        str(VALIDATION_EMBEDDINGS_CSV),
        _mtime(VALIDATION_PREDICTIONS_CSV),
        _mtime(VALIDATION_EMBEDDINGS_CSV),
    )
    breakdown_df, errors_df = build_error_breakdown(
        str(VALIDATION_PREDICTIONS_CSV),
        _mtime(VALIDATION_PREDICTIONS_CSV),
    )

    st.subheader("Model Intelligence")
    st.caption("Every chart below is sourced from validation-only artifacts to avoid train/test presentation leakage.")

    if projection_df.empty:
        st.warning(
            "Validation artifacts are missing. Run `python src/dataloader.py --precompute-validation` first."
        )
        return

    render_metrics_strip(eval_metrics, baseline_metrics, summary)

    controls_col, table_col = st.columns([1.1, 0.9])
    with controls_col:
        only_errors = st.toggle("Show only misclassified edge cases", value=False)
        umap_figure = build_umap_figure(projection_df, only_errors=only_errors)
        if umap_figure is not None:
            st.plotly_chart(umap_figure, use_container_width=True)
    with table_col:
        if not errors_df.empty:
            st.markdown("**Highest-confidence misses**")
            st.dataframe(
                errors_df.sort_values("predicted_confidence", ascending=False).head(12)[
                    [
                        "filename",
                        "label_name",
                        "predicted_name",
                        "predicted_confidence",
                        "confidence_bucket",
                        "error_type",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No validation errors found in the current artifact set.")

    chart_cols = st.columns(2)
    if not breakdown_df.empty:
        with chart_cols[0]:
            bar_figure = px.bar(
                breakdown_df,
                x="error_type",
                y="count",
                color="confidence_bucket",
                barmode="group",
                text="count",
                title="Failure modes by confidence bucket",
                labels={
                    "error_type": "Failure mode",
                    "count": "Images",
                    "confidence_bucket": "Confidence",
                },
                template="plotly_dark",
                color_discrete_map={"high": "#fb7185", "low": "#4fd1c5"},
            )
            st.plotly_chart(bar_figure, use_container_width=True)
        with chart_cols[1]:
            pie_figure = px.pie(
                breakdown_df,
                names="failure_mode",
                values="count",
                title="Share of validation errors",
                template="plotly_dark",
                hole=0.5,
                color="confidence_bucket",
                color_discrete_map={"high": "#fb7185", "low": "#4fd1c5"},
            )
            st.plotly_chart(pie_figure, use_container_width=True)

    with st.expander("Artifact summary", expanded=False):
        st.json(
            {
                "split": summary.get("split", "validation"),
                "projection_method": summary.get("projection_method", "unknown"),
                "confidence_threshold_high": summary.get("confidence_threshold_high", 0.80),
                "error_breakdown": summary.get("error_breakdown", []),
            }
        )


def main() -> None:
    st.set_page_config(
        page_title="VIPER Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    render_banner()
    view = render_sidebar()

    if view == "Image Forensics":
        render_image_forensics()
    else:
        render_model_intelligence()


if __name__ == "__main__":
    main()
