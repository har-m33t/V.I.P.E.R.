from __future__ import annotations

import base64
import html
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
    --bg: #0b0c10;
    --panel: #12171f;
    --panel-soft: #1f2933;
    --line: rgba(148, 163, 184, 0.14);
    --text: #f3f4f6;
    --muted: #9aa5b1;
    --warn: #f59e0b;
    --danger: #e11d48;
    --safe: #10b981;
    --shadow: 0 22px 60px rgba(0, 0, 0, 0.35);
  }
  html, body, [class*="css"] {
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  }
  .stApp {
    background:
      radial-gradient(circle at 0% 0%, rgba(225, 29, 72, 0.16), transparent 30%),
      radial-gradient(circle at 100% 0%, rgba(16, 185, 129, 0.14), transparent 28%),
      linear-gradient(135deg, #050608 0%, #0b0c10 50%, #11151b 100%);
    color: var(--text);
  }
  header[data-testid="stHeader"] {
    background: transparent;
  }
  #MainMenu, footer {
    visibility: hidden;
  }
  .block-container {
    max-width: 1440px;
    padding: 0.9rem 1.1rem 2rem;
  }
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(11, 12, 16, 0.98), rgba(17, 21, 27, 0.98));
    border-right: 1px solid rgba(225, 29, 72, 0.16);
  }
  section[data-testid="stSidebar"] > div {
    padding-top: 0.8rem;
  }
  div[data-testid="stSidebar"] .stMarkdown,
  div[data-testid="stSidebar"] label,
  div[data-testid="stSidebar"] p,
  div[data-testid="stSidebar"] span {
    color: var(--text);
  }
  .heist-header,
  .mission-shell,
  .artifact-shell,
  .section-shell,
  .scan-terminal,
  .scan-sidecard {
    border: 1px solid var(--line);
    border-radius: 24px;
    background: linear-gradient(180deg, rgba(18, 23, 31, 0.92), rgba(11, 12, 16, 0.96));
    box-shadow: var(--shadow);
  }
  .heist-header {
    padding: 1.2rem 1.25rem 1.15rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
  }
  .heist-header::before {
    content: "";
    position: absolute;
    inset: auto -4rem -4rem auto;
    width: 16rem;
    height: 16rem;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(225, 29, 72, 0.22), transparent 60%);
    pointer-events: none;
  }
  .heist-kicker {
    color: var(--danger);
    text-transform: uppercase;
    letter-spacing: 0.22em;
    font-size: 0.72rem;
    margin-bottom: 0.45rem;
  }
  .heist-title-row {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    justify-content: space-between;
  }
  .heist-title {
    margin: 0;
    font-size: clamp(2rem, 4vw, 3.5rem);
    line-height: 0.95;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .heist-title span {
    color: var(--danger);
    display: block;
    margin-top: 0.35rem;
  }
  .heist-subtitle {
    margin: 0.9rem 0 0;
    max-width: 50rem;
    color: var(--muted);
    font-size: 0.98rem;
    line-height: 1.6;
  }
  .status-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.55rem;
    min-width: 15.5rem;
  }
  .status-pill {
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.6rem 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem;
    text-align: center;
    background: rgba(31, 41, 51, 0.55);
  }
  .status-pill.danger {
    border-color: rgba(225, 29, 72, 0.5);
    color: #fecdd3;
    background: linear-gradient(90deg, rgba(225, 29, 72, 0.22), rgba(225, 29, 72, 0.08));
  }
  .status-pill.safe {
    border-color: rgba(16, 185, 129, 0.5);
    color: #d1fae5;
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.22), rgba(16, 185, 129, 0.08));
  }
  .status-pill.neutral {
    border-color: rgba(148, 163, 184, 0.28);
    color: #e5e7eb;
  }
  .mission-shell,
  .artifact-shell,
  .section-shell {
    padding: 0.95rem 1rem;
    margin-bottom: 1rem;
  }
  .mission-title,
  .section-title {
    margin: 0.15rem 0 0;
    font-size: 1.1rem;
    font-weight: 700;
  }
  .mission-kicker,
  .section-kicker,
  .artifact-kicker {
    color: var(--danger);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.7rem;
  }
  .mission-copy,
  .section-copy,
  .artifact-copy {
    margin: 0.55rem 0 0;
    color: var(--muted);
    line-height: 1.55;
    font-size: 0.92rem;
  }
  .artifact-list {
    margin: 0.75rem 0 0;
    padding: 0;
    list-style: none;
  }
  .artifact-list li {
    padding: 0.5rem 0.65rem;
    border: 1px solid var(--line);
    border-radius: 14px;
    background: rgba(31, 41, 51, 0.5);
    font-family: Consolas, monospace;
    font-size: 0.82rem;
  }
  div[data-testid="stSidebar"] div[role="radiogroup"] {
    display: grid;
    gap: 0.55rem;
  }
  div[data-testid="stSidebar"] div[role="radiogroup"] > label,
  div[data-testid="stSidebar"] label[data-baseweb="radio"] {
    margin: 0;
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 16px;
    background: rgba(31, 41, 51, 0.45);
    padding: 0.65rem 0.8rem;
  }
  div[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked),
  div[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
    border-color: rgba(225, 29, 72, 0.55);
    background: linear-gradient(135deg, rgba(225, 29, 72, 0.18), rgba(31, 41, 51, 0.88));
    box-shadow: 0 0 0 1px rgba(225, 29, 72, 0.18);
  }
  div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover,
  div[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
    border-color: rgba(225, 29, 72, 0.34);
  }
  .scan-frame {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    border: 1px solid rgba(16, 185, 129, 0.26);
    background:
      linear-gradient(180deg, rgba(11, 12, 16, 0.22), rgba(11, 12, 16, 0.54)),
      linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(31, 41, 51, 0.22));
    box-shadow:
      0 24px 80px rgba(0, 0, 0, 0.34),
      inset 0 0 0 1px rgba(255, 255, 255, 0.03);
  }
  .scan-frame::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(rgba(16, 185, 129, 0.06) 1px, transparent 1px),
      linear-gradient(90deg, rgba(16, 185, 129, 0.06) 1px, transparent 1px);
    background-size: 26px 26px;
    pointer-events: none;
  }
  .scan-frame img {
    display: block;
    width: 100%;
    height: auto;
    aspect-ratio: 1 / 1;
    object-fit: cover;
    filter: contrast(1.03) saturate(0.95);
  }
  .scan-frame-label {
    position: absolute;
    top: 1rem;
    left: 1rem;
    z-index: 2;
    padding: 0.35rem 0.65rem;
    border-radius: 999px;
    border: 1px solid rgba(16, 185, 129, 0.28);
    background: rgba(11, 12, 16, 0.78);
    color: #d1fae5;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
  }
  .scan-frame-reticle {
    position: absolute;
    inset: 0;
    z-index: 2;
    pointer-events: none;
  }
  .scan-frame-corner {
    position: absolute;
    width: 40px;
    height: 40px;
    border-color: rgba(16, 185, 129, 0.7);
    border-style: solid;
    filter: drop-shadow(0 0 10px rgba(16, 185, 129, 0.22));
  }
  .scan-frame-corner.tl {
    top: 14px;
    left: 14px;
    border-width: 2px 0 0 2px;
    border-top-left-radius: 12px;
  }
  .scan-frame-corner.tr {
    top: 14px;
    right: 14px;
    border-width: 2px 2px 0 0;
    border-top-right-radius: 12px;
  }
  .scan-frame-corner.bl {
    bottom: 14px;
    left: 14px;
    border-width: 0 0 2px 2px;
    border-bottom-left-radius: 12px;
  }
  .scan-frame-corner.br {
    bottom: 14px;
    right: 14px;
    border-width: 0 2px 2px 0;
    border-bottom-right-radius: 12px;
  }
  .scan-frame-sweep {
    position: absolute;
    left: 0;
    right: 0;
    top: 14%;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.85), transparent);
    box-shadow: 0 0 18px rgba(16, 185, 129, 0.32);
    animation: scanSweep 3.8s ease-in-out infinite;
  }
  @keyframes scanSweep {
    0%, 100% { transform: translateY(-20%); opacity: 0.15; }
    50% { transform: translateY(320px); opacity: 0.92; }
  }
  .scan-banner {
    margin: 1rem 0 0.75rem;
    padding: 1rem 1.2rem;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    text-align: center;
    font-family: "Courier New", monospace;
    font-size: clamp(1.2rem, 2.6vw, 2rem);
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.3);
    animation: verdictPulse 1.8s ease-in-out infinite;
  }
  .scan-banner.authentic {
    color: #d8ffe7;
    background: linear-gradient(135deg, rgba(5, 74, 49, 0.94), rgba(13, 148, 88, 0.82));
    border-color: rgba(74, 222, 128, 0.38);
    text-shadow: 0 0 18px rgba(74, 222, 128, 0.26);
  }
  .scan-banner.forgery {
    color: #ffe5eb;
    background: linear-gradient(135deg, rgba(120, 18, 42, 0.94), rgba(225, 29, 72, 0.82));
    border-color: rgba(251, 113, 133, 0.4);
    text-shadow: 0 0 18px rgba(251, 113, 133, 0.24);
  }
  @keyframes verdictPulse {
    0%, 100% { transform: scale(1); box-shadow: 0 18px 48px rgba(0, 0, 0, 0.3); }
    50% { transform: scale(1.015); box-shadow: 0 22px 62px rgba(0, 0, 0, 0.4); }
  }
  .scan-terminal {
    margin-bottom: 1rem;
    overflow: hidden;
    border-radius: 22px;
    background: linear-gradient(180deg, rgba(11, 12, 16, 0.98), rgba(18, 23, 31, 0.96));
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
  }
  .scan-terminal-header {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.65rem 0.9rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.14);
    color: #fda4af;
    font-size: 0.76rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-family: Consolas, monospace;
    background: rgba(255, 255, 255, 0.02);
  }
  .scan-terminal-body {
    margin: 0;
    padding: 1rem 1rem 1.05rem;
    color: #d1fae5;
    line-height: 1.65;
    font-size: 0.98rem;
    font-family: Consolas, monospace;
    white-space: normal;
  }
  .scan-sidecard {
    padding: 1rem;
    margin-bottom: 1rem;
  }
  .scan-sidecard-label {
    color: var(--danger);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.76rem;
    margin-bottom: 0.4rem;
  }
  .scan-sidecard-title {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 700;
  }
  div[data-testid="stFileUploader"],
  div[data-testid="stPlotlyChart"],
  div[data-testid="stDataFrame"],
  div[data-testid="stImage"],
  div[data-testid="stExpander"] details {
    border-radius: 22px;
    overflow: hidden;
  }
  div[data-testid="stFileUploader"] {
    background: rgba(18, 23, 31, 0.76);
    border: 1px dashed rgba(225, 29, 72, 0.36);
    padding: 0.55rem;
  }
  div[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(180deg, rgba(18, 23, 31, 0.84), rgba(11, 12, 16, 0.88));
  }
  div[data-testid="stFileUploaderDropzone"] button,
  .stButton > button,
  button[kind="secondary"],
  button[kind="primary"] {
    border-radius: 999px;
    border: 1px solid rgba(225, 29, 72, 0.45);
    background: linear-gradient(90deg, rgba(225, 29, 72, 0.18), rgba(31, 41, 51, 0.88));
    color: var(--text);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
    min-height: 2.7rem;
    box-shadow: 0 10px 32px rgba(0, 0, 0, 0.25);
  }
  div[data-testid="stFileUploaderDropzone"] button:hover,
  .stButton > button:hover,
  button[kind="secondary"]:hover,
  button[kind="primary"]:hover {
    border-color: rgba(225, 29, 72, 0.68);
    color: #ffffff;
  }
  div[data-baseweb="select"] > div,
  div[data-testid="stTextInputRootElement"] > div,
  div[data-baseweb="base-input"] > div {
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(18, 23, 31, 0.82);
    color: var(--text);
  }
  div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(31, 41, 51, 0.82), rgba(18, 23, 31, 0.92));
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 20px;
    padding: 1rem 1.05rem;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
  }
  div[data-testid="stMetric"] label {
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }
  div[data-testid="stMetricValue"] {
    color: var(--text);
  }
  div[data-testid="stMetricDelta"] {
    color: var(--muted);
  }
  .signal-label {
    margin: 0.35rem 0 0.55rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem;
    color: var(--muted);
  }
  div[data-testid="stExpander"] details {
    background: rgba(18, 23, 31, 0.84);
    border: 1px solid rgba(148, 163, 184, 0.12);
  }
  div[data-testid="stExpander"] summary {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text);
  }
  .stAlert {
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.14);
  }
  @media (max-width: 960px) {
    .block-container {
      padding-left: 0.75rem;
      padding-right: 0.75rem;
    }
    .heist-title-row {
      flex-direction: column;
    }
    .status-grid {
      width: 100%;
      min-width: 0;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  @media (max-width: 640px) {
    .heist-header,
    .mission-shell,
    .artifact-shell,
    .section-shell,
    .scan-terminal,
    .scan-sidecard,
    .scan-frame {
      border-radius: 18px;
    }
    .status-grid {
      grid-template-columns: 1fr;
    }
    .heist-title {
      font-size: 1.8rem;
    }
  }
</style>
"""

SIGNAL_CONFIG = {
    "fft": {"label": "FFT Anomaly", "fallback": "FFT irregularity"},
    "prnu": {"label": "PRNU Inconsistency", "fallback": "PRNU variation"},
    "lab": {"label": "LAB Saturation", "fallback": "LAB saturation"},
}


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


@st.cache_data(show_spinner=False, max_entries=16)
def build_png_data_uri(image_bytes: bytes) -> str:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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


def build_omni_explanation(report_card: dict[str, Any]) -> str:
    try:
        from src.omni import explain_forensic_report_card

        explanation = explain_forensic_report_card(report_card)
        if explanation.strip():
            return explanation
    except Exception:
        pass
    return str(report_card.get("verdict") or "VIPER completed the scan, but no narrative explanation is available.")


def render_security_camera_feed(image_data_uri: str, filename: str) -> None:
    safe_filename = html.escape(filename)
    st.markdown(
        f"""
        <div class="scan-frame">
          <div class="scan-frame-label">Security Camera Feed</div>
          <img src="{image_data_uri}" alt="{safe_filename}" />
          <div class="scan-frame-reticle" aria-hidden="true">
            <span class="scan-frame-corner tl"></span>
            <span class="scan-frame-corner tr"></span>
            <span class="scan-frame-corner bl"></span>
            <span class="scan-frame-corner br"></span>
            <span class="scan-frame-sweep"></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scan_banner(prediction: str) -> None:
    normalized = prediction.strip().lower()
    is_authentic = "real" in normalized or "authentic" in normalized
    banner_text = (
        "[ AUTHENTIC: INITIATE HEIST ]"
        if is_authentic
        else "[ FORGERY DETECTED: ABORT MISSION ]"
    )
    banner_class = "authentic" if is_authentic else "forgery"
    st.markdown(
        f'<div class="scan-banner {banner_class}">{html.escape(banner_text)}</div>',
        unsafe_allow_html=True,
    )


def render_terminal_explainer(explanation: str) -> None:
    st.markdown(
        f"""
        <div class="scan-terminal">
          <div class="scan-terminal-header">
            <span>Omni Lite Explainer</span>
            <span>Retinal Scan Summary</span>
          </div>
          <p class="scan-terminal-body">{html.escape(explanation)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <section class="section-shell">
          <div class="section-kicker">{html.escape(kicker)}</div>
          <div class="section-title">{html.escape(title)}</div>
          <p class="section-copy">{html.escape(copy)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def apply_plot_theme(figure: Any, height: Optional[int] = None) -> Any:
    layout_updates: dict[str, Any] = {
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "plot_bgcolor": "rgba(31, 41, 51, 0.55)",
        "font": {"color": "#f3f4f6", "family": "IBM Plex Sans, Segoe UI, sans-serif"},
        "margin": {"l": 18, "r": 18, "t": 64, "b": 18},
        "legend": {
            "bgcolor": "rgba(11, 12, 16, 0.72)",
            "bordercolor": "rgba(148, 163, 184, 0.12)",
            "borderwidth": 1,
        },
    }
    if height is not None:
        layout_updates["height"] = height
    figure.update_layout(**layout_updates)
    return figure


def render_banner() -> None:
    st.markdown(
        """
        <section class="heist-header">
          <div class="heist-kicker">Mission Control</div>
          <div class="heist-title-row">
            <div>
              <h1 class="heist-title">VIPER TERMINAL <span>// OPERATION: ARTHEIST</span></h1>
              <p class="heist-subtitle">
                Live artwork scans, cached validation optics, and judge-safe evidence review in one
                dark heist deck. The backend stays intact while the terminal look shifts fully into
                the mission theme.
              </p>
            </div>
            <div class="status-grid">
              <div class="status-pill danger">Forgery Alert Grid</div>
              <div class="status-pill safe">Authenticity Gate Armed</div>
              <div class="status-pill neutral">Cached Runtime Locked</div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    st.sidebar.markdown(
        """
        <section class="mission-shell">
          <div class="mission-kicker">Mission Briefing</div>
          <div class="mission-title">Choose the active surface.</div>
          <p class="mission-copy">
            The Target runs the live artwork scan. Forensic Optics stays locked to validation artifacts
            so the evidence deck remains clean and demo-safe.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    view = st.sidebar.radio(
        "Mission Surface",
        options=("The Target (Scan)", "Forensic Optics (VIPER Data)"),
        index=0,
        label_visibility="collapsed",
    )
    st.sidebar.markdown(
        f"""
        <section class="artifact-shell">
          <div class="artifact-kicker">Cold Storage</div>
          <p class="artifact-copy">Validation files currently wired into the optics deck.</p>
          <ul class="artifact-list">
            <li>{html.escape(VALIDATION_PREDICTIONS_CSV.name)}</li>
            <li>{html.escape(VALIDATION_EMBEDDINGS_CSV.name)}</li>
            <li>{html.escape(VALIDATION_SUMMARY_JSON.name)}</li>
          </ul>
        </section>
        """,
        unsafe_allow_html=True,
    )
    return view


def _normalize_signal_status(status: Any) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == "moderate":
        normalized = "medium"
    if normalized in {"low", "medium", "high"}:
        return normalized.title()
    return normalized.title() or "Unknown"


def _signal_delta_text(status: Any) -> str:
    risk_label = _normalize_signal_status(status)
    direction = "-" if risk_label == "Low" else "+"
    return f"{direction} {risk_label} risk"


def _signal_items(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items_by_id: dict[str, dict[str, Any]] = {}
    for raw_item in result.get("evidence_breakdown", []):
        if not isinstance(raw_item, dict):
            continue
        signal_id = str(raw_item.get("id", "")).lower()
        items_by_id[signal_id] = raw_item
    return items_by_id


def render_signal_buckets(result: dict[str, Any]) -> None:
    signal_items = _signal_items(result)
    metric_cols = st.columns(3)

    for column, signal_id in zip(metric_cols, SIGNAL_CONFIG):
        config = SIGNAL_CONFIG[signal_id]
        signal = signal_items.get(signal_id)
        with column:
            if signal is None:
                st.metric(config["label"], "N/A", "Unavailable", delta_color="off")
                st.caption("Signal export missing from this scan.")
                continue

            score = float(signal.get("score", 0.0))
            st.metric(
                config["label"],
                f"{score:.0%}",
                _signal_delta_text(signal.get("status")),
                delta_color="inverse",
            )
            detail = str(signal.get("value") or signal.get("detail") or config["fallback"])
            st.caption(detail)


def build_fast_track_comparison_figure(
    eval_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> Any:
    metric_labels = {
        "accuracy": "Accuracy",
        "f1": "F1",
        "precision": "Precision",
        "recall": "Recall",
        "auc_roc": "AUC-ROC",
    }
    comparison_rows: list[dict[str, Any]] = []

    for metric_key, metric_label in metric_labels.items():
        baseline_value = baseline_metrics.get(metric_key)
        eval_value = eval_metrics.get(metric_key)
        if baseline_value is None or eval_value is None:
            continue

        comparison_rows.append(
            {"metric": metric_label, "run": "Baseline", "score": float(baseline_value)}
        )
        comparison_rows.append(
            {"metric": metric_label, "run": "Fast-Track 10k", "score": float(eval_value)}
        )

    if not comparison_rows:
        return None

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df["score_label"] = comparison_df["score"].map(lambda value: f"{value:.1%}")
    figure = px.bar(
        comparison_df,
        x="metric",
        y="score",
        color="run",
        barmode="group",
        text="score_label",
        title="Fast-Track validation lift over the baseline",
        labels={"metric": "Metric", "score": "Score", "run": "Run"},
        template="plotly_dark",
        color_discrete_map={
            "Baseline": "#64748b",
            "Fast-Track 10k": "#10b981",
        },
    )
    apply_plot_theme(figure, height=420)
    figure.update_traces(textposition="outside", cliponaxis=False)
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(range=[0, 1.05], tickformat=".0%", showgrid=True, gridcolor="rgba(148, 163, 184, 0.10)", zeroline=False)
    return figure


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
    render_section_intro(
        "The Target",
        "Live scan a suspect artwork through the VIPER stack.",
        "Upload a painting or inspect a cached breach overlay. The inference runtime stays hot, so navigation and repeat scans do not trigger a cold model reload.",
    )

    uploader_col, gallery_col = st.columns([1.25, 0.75])
    with uploader_col:
        uploaded_file = st.file_uploader(
            "Upload the target artwork",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=False,
        )
    with gallery_col:
        sample_names = list_gradcam_samples(str(GRADCAM_DIR), _mtime(GRADCAM_DIR))
        sample_choice = st.selectbox(
            "Vaulted overlay archive",
            options=["None"] + sample_names,
            index=0,
        )

    runtime, runtime_error = load_runtime()
    if runtime_error:
        st.info(f"Live scan runtime unavailable: {runtime_error}")

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image_data_uri = build_png_data_uri(image_bytes)
        with st.spinner("Running cached VIPER inference..."):
            result, overlay, overlay_error = analyze_uploaded_image(image_bytes, uploaded_file.name)

        if not result:
            st.error(overlay_error or "Inference failed.")
            return

        explanation = build_omni_explanation(result)

        scan_cols = st.columns(2)
        with scan_cols[0]:
            st.markdown("**Security Feed**")
            render_security_camera_feed(image_data_uri=image_data_uri, filename=uploaded_file.name)
        with scan_cols[1]:
            st.markdown("**Thermal Trace**")
            if overlay is not None:
                st.image(overlay, use_container_width=True)
            else:
                st.warning(overlay_error or "Grad-CAM overlay unavailable.")

        render_scan_banner(result["prediction"])

        verdict_cols = st.columns(4)
        verdict_cols[0].metric("Classification", result["prediction"])
        verdict_cols[1].metric("Confidence Lock", result["confidence_pct"])
        verdict_cols[2].metric("Forgery Probability", f"{float(result['ai_probability']):.1%}")
        verdict_cols[3].metric("Fusion Stack", "EDA On" if result["uses_eda_fusion"] else "Image Only")

        render_terminal_explainer(explanation)

        st.markdown('<div class="signal-label">Signal Buckets</div>', unsafe_allow_html=True)
        render_signal_buckets(result)

        gradcam_item = _signal_items(result).get("gradcam")
        if gradcam_item is not None:
            st.caption(
                f"Grad-CAM focus: {gradcam_item.get('detail', 'Attention trace unavailable.')} "
                f"({gradcam_item.get('value', 'No coverage summary')})"
            )

        with st.expander("Signal Trace", expanded=False):
            evidence_df = pd.DataFrame(result["evidence_breakdown"])
            if not evidence_df.empty:
                st.dataframe(
                    evidence_df[["label", "status", "score", "value", "detail"]],
                    use_container_width=True,
                    hide_index=True,
                )
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
            caption=f"Vault archive sample: {sample_choice}",
            use_container_width=True,
        )
    else:
        st.info("Upload a target or inspect a cached overlay from the archive.")


def build_umap_figure(projection_df: pd.DataFrame, only_errors: bool) -> Any:
    filtered_df = projection_df.copy()
    if only_errors:
        filtered_df = filtered_df[filtered_df["error_type"] != "correct"]

    if filtered_df.empty:
        return None

    filtered_df["marker_group"] = filtered_df["error_type"].str.replace("_", " ").str.title()
    figure = px.scatter(
        filtered_df,
        x="umap_x",
        y="umap_y",
        color="predicted_confidence",
        color_continuous_scale=["#1f2933", "#334155", "#10b981", "#f59e0b", "#e11d48"],
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
        title="Forensic Optics vault map colored by prediction confidence",
        labels={
            "umap_x": "Projection X",
            "umap_y": "Projection Y",
            "predicted_confidence": "Confidence",
            "marker_group": "Outcome",
        },
        template="plotly_dark",
    )
    figure.update_traces(marker={"size": 9, "line": {"width": 0.6, "color": "rgba(255,255,255,0.15)"}})
    figure.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.10)", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.10)", zeroline=False)
    figure.update_layout(coloraxis_colorbar_title="Confidence", legend_title_text="Outcome")
    return apply_plot_theme(figure, height=620)


def render_forensic_optics() -> None:
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

    render_section_intro(
        "Forensic Optics",
        "Inspect the cached validation vault behind the live scan.",
        "This surface stays locked to validation-only artifacts, giving the judges the model's confidence landscape and edge-case behavior without touching the training split.",
    )

    if projection_df.empty:
        st.warning(
            "Validation artifacts are missing. Run `python src/dataloader.py --precompute-validation` first."
        )
        return

    render_metrics_strip(eval_metrics, baseline_metrics, summary)

    toggle_col, count_col = st.columns([0.7, 0.3])
    with toggle_col:
        only_errors = st.toggle("Isolate misclassified edge cases", value=False)
    with count_col:
        st.metric("Edge Cases", f"{len(errors_df)}", "Validation misses")

    umap_figure = build_umap_figure(projection_df, only_errors=only_errors)
    if umap_figure is not None:
        st.plotly_chart(umap_figure, use_container_width=True)

    with st.expander("Open Mission Telemetry", expanded=False):
        comparison_figure = build_fast_track_comparison_figure(eval_metrics, baseline_metrics)
        if comparison_figure is not None:
            st.plotly_chart(comparison_figure, use_container_width=True)

        if not breakdown_df.empty:
            chart_cols = st.columns(2)
            with chart_cols[0]:
                bar_figure = px.bar(
                    breakdown_df,
                    x="error_type",
                    y="count",
                    color="confidence_bucket",
                    barmode="group",
                    text="count",
                    title="Error breakdown by confidence bucket",
                    labels={
                        "error_type": "Failure mode",
                        "count": "Images",
                        "confidence_bucket": "Confidence",
                    },
                    template="plotly_dark",
                    color_discrete_map={"high": "#e11d48", "medium": "#f59e0b", "low": "#10b981"},
                )
                apply_plot_theme(bar_figure, height=420)
                bar_figure.update_xaxes(showgrid=False)
                bar_figure.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.10)", zeroline=False)
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
                    color_discrete_map={"high": "#e11d48", "medium": "#f59e0b", "low": "#10b981"},
                )
                apply_plot_theme(pie_figure, height=420)
                st.plotly_chart(pie_figure, use_container_width=True)
        else:
            st.success("No validation misses found in the current artifact set.")

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
        page_title="VIPER TERMINAL",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    render_banner()
    view = render_sidebar()

    if view == "The Target (Scan)":
        render_image_forensics()
    else:
        render_forensic_optics()


if __name__ == "__main__":
    main()
