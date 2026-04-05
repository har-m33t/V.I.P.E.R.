from __future__ import annotations

import base64
import html
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.config import IMAGE_SIZE


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
    --bg: #06080d;
    --panel: #0f1419;
    --panel-elevated: #151c26;
    --panel-soft: #1a2330;
    --line: rgba(148, 163, 184, 0.12);
    --text: #e8eaef;
    --muted: #8b95a8;
    --gold: #c9a227;
    --gold-dim: rgba(201, 162, 39, 0.35);
    --electric: #2ec4ff;
    --electric-dim: rgba(46, 196, 255, 0.22);
    --risk: #e85d5d;
    --risk-dim: rgba(232, 93, 93, 0.2);
    --safe: #3ecf8e;
    --shadow: 0 24px 64px rgba(0, 0, 0, 0.45);
  }
  html, body, [class*="css"] {
    font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  }
  .stApp {
    background:
      radial-gradient(ellipse 80% 50% at 50% -20%, rgba(46, 196, 255, 0.08), transparent 55%),
      radial-gradient(ellipse 60% 40% at 100% 50%, rgba(201, 162, 39, 0.06), transparent 50%),
      radial-gradient(ellipse 50% 35% at 0% 80%, rgba(232, 93, 93, 0.05), transparent 45%),
      linear-gradient(165deg, #030406 0%, #0a0e14 40%, #0d1219 100%);
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
    background: linear-gradient(180deg, rgba(6, 8, 13, 0.99), rgba(15, 20, 28, 0.98));
    border-right: 1px solid rgba(46, 196, 255, 0.12);
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
  .ah-hero,
  .mission-shell,
  .artifact-shell,
  .section-shell,
  .scan-terminal,
  .scan-sidecard,
  .ah-panel {
    border: 1px solid var(--line);
    border-radius: 20px;
    background: linear-gradient(145deg, rgba(21, 28, 38, 0.94), rgba(10, 13, 18, 0.97));
    box-shadow: var(--shadow);
  }
  .ah-hero {
    padding: 1.35rem 1.4rem 1.3rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
  }
  .ah-hero::before {
    content: "";
    position: absolute;
    inset: -40% -10% auto auto;
    width: 55%;
    height: 120%;
    background: radial-gradient(ellipse, rgba(46, 196, 255, 0.09), transparent 65%);
    pointer-events: none;
  }
  .ah-hero::after {
    content: "";
    position: absolute;
    inset: auto auto -50% -15%;
    width: 45%;
    height: 100%;
    background: radial-gradient(ellipse, rgba(201, 162, 39, 0.07), transparent 60%);
    pointer-events: none;
  }
  .ah-kicker {
    color: var(--electric);
    text-transform: uppercase;
    letter-spacing: 0.2em;
    font-size: 0.68rem;
    margin-bottom: 0.5rem;
    font-weight: 600;
  }
  .ah-title-row {
    display: flex;
    gap: 1.25rem;
    align-items: flex-start;
    justify-content: space-between;
    position: relative;
    z-index: 1;
  }
  .ah-title {
    margin: 0;
    font-size: clamp(1.65rem, 3.2vw, 2.65rem);
    line-height: 1.08;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: var(--text);
  }
  .ah-title-gold {
    color: var(--gold);
    font-weight: 600;
  }
  .ah-subtitle {
    margin: 0.75rem 0 0;
    max-width: 46rem;
    color: var(--muted);
    font-size: 0.95rem;
    line-height: 1.65;
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
    border-color: var(--risk-dim);
    color: #fecaca;
    background: linear-gradient(90deg, var(--risk-dim), rgba(26, 31, 40, 0.6));
  }
  .status-pill.safe {
    border-color: var(--gold-dim);
    color: #f5e6b8;
    background: linear-gradient(90deg, rgba(201, 162, 39, 0.18), rgba(26, 31, 40, 0.55));
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
    color: var(--electric);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.68rem;
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
    border-color: rgba(46, 196, 255, 0.45);
    background: linear-gradient(135deg, var(--electric-dim), rgba(21, 28, 38, 0.9));
    box-shadow: 0 0 0 1px rgba(46, 196, 255, 0.12);
  }
  div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover,
  div[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
    border-color: rgba(46, 196, 255, 0.28);
  }
  .scan-frame {
    position: relative;
    overflow: hidden;
    border-radius: 18px;
    border: 1px solid rgba(201, 162, 39, 0.22);
    background:
      linear-gradient(180deg, rgba(6, 8, 13, 0.35), rgba(10, 14, 20, 0.75)),
      linear-gradient(135deg, var(--electric-dim), rgba(21, 28, 38, 0.15));
    box-shadow:
      0 20px 70px rgba(0, 0, 0, 0.42),
      inset 0 0 0 1px rgba(255, 255, 255, 0.025);
  }
  .scan-frame::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(rgba(46, 196, 255, 0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(46, 196, 255, 0.04) 1px, transparent 1px);
    background-size: 28px 28px;
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
    top: 0.85rem;
    left: 0.85rem;
    z-index: 2;
    padding: 0.32rem 0.7rem;
    border-radius: 6px;
    border: 1px solid var(--gold-dim);
    background: rgba(6, 8, 13, 0.88);
    color: #f0e6c8;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
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
    width: 36px;
    height: 36px;
    border-color: rgba(46, 196, 255, 0.45);
    border-style: solid;
    filter: drop-shadow(0 0 8px var(--electric-dim));
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
    top: 12%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(46, 196, 255, 0.55), transparent);
    box-shadow: 0 0 12px rgba(46, 196, 255, 0.2);
    animation: scanSweep 5s ease-in-out infinite;
    opacity: 0.85;
  }
  @keyframes scanSweep {
    0%, 100% { transform: translateY(-15%); opacity: 0.12; }
    50% { transform: translateY(260px); opacity: 0.55; }
  }
  .ah-panel {
    padding: 1rem 1.05rem;
    margin-bottom: 0.85rem;
  }
  .ah-panel-kicker {
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.65rem;
    margin-bottom: 0.35rem;
    font-weight: 600;
  }
  .ah-panel-title {
    margin: 0;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
  }
  .ah-feedback {
    margin-top: 0.75rem;
    padding: 0.65rem 0.75rem;
    border-radius: 10px;
    border: 1px solid var(--line);
    background: rgba(15, 20, 28, 0.65);
    font-size: 0.86rem;
    line-height: 1.5;
    color: var(--muted);
  }
  .ah-score-line {
    margin-top: 0.55rem;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.04em;
  }
  .ah-verdict-chip {
    display: inline-block;
    padding: 0.4rem 0.75rem;
    border-radius: 8px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.65rem;
  }
  .ah-verdict-chip.real {
    border: 1px solid var(--gold-dim);
    color: #f5e6b8;
    background: rgba(201, 162, 39, 0.12);
  }
  .ah-verdict-chip.ai {
    border: 1px solid var(--risk-dim);
    color: #fecaca;
    background: var(--risk-dim);
  }
  .ah-confidence-bar {
    height: 6px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.06);
    overflow: hidden;
    margin: 0.5rem 0 0.75rem;
  }
  .ah-confidence-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--electric), #5dd8ff);
    transition: width 0.35s ease;
  }
  .ah-confidence-fill.ai-risk {
    background: linear-gradient(90deg, #c94a4a, var(--risk));
  }
  div[class*="st-key-mark_authentic"] button {
    border-color: var(--gold-dim) !important;
    background: linear-gradient(95deg, rgba(201, 162, 39, 0.2), rgba(21, 28, 38, 0.95)) !important;
    color: #f5f0e6 !important;
  }
  div[class*="st-key-mark_authentic"] button:hover {
    border-color: rgba(201, 162, 39, 0.55) !important;
    box-shadow: 0 0 20px rgba(201, 162, 39, 0.15) !important;
  }
  div[class*="st-key-mark_ai"] button {
    border-color: var(--risk-dim) !important;
    background: linear-gradient(95deg, var(--risk-dim), rgba(21, 28, 38, 0.95)) !important;
    color: #fde8e8 !important;
  }
  div[class*="st-key-mark_ai"] button:hover {
    border-color: rgba(232, 93, 93, 0.5) !important;
    box-shadow: 0 0 18px rgba(232, 93, 93, 0.12) !important;
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
    padding: 0.6rem 0.85rem;
    border-bottom: 1px solid var(--line);
    color: var(--electric);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: "IBM Plex Mono", Consolas, monospace;
    background: rgba(46, 196, 255, 0.04);
  }
  .scan-terminal-body {
    margin: 0;
    padding: 0.95rem 0.95rem 1rem;
    color: #c8d0dc;
    line-height: 1.6;
    font-size: 0.9rem;
    font-family: "IBM Plex Mono", Consolas, monospace;
    white-space: normal;
  }
  .scan-sidecard {
    padding: 1rem;
    margin-bottom: 1rem;
  }
  .scan-sidecard-label {
    color: var(--electric);
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
    background: rgba(15, 20, 28, 0.85);
    border: 1px dashed rgba(46, 196, 255, 0.22);
    padding: 0.55rem;
  }
  div[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(180deg, rgba(18, 23, 31, 0.84), rgba(11, 12, 16, 0.88));
  }
  div[data-testid="stFileUploaderDropzone"] button,
  .stButton > button,
  button[kind="secondary"],
  button[kind="primary"] {
    border-radius: 10px;
    border: 1px solid rgba(46, 196, 255, 0.25);
    background: linear-gradient(95deg, rgba(46, 196, 255, 0.12), rgba(21, 28, 38, 0.92));
    color: var(--text);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    min-height: 2.55rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.28);
  }
  div[data-testid="stFileUploaderDropzone"] button:hover,
  .stButton > button:hover,
  button[kind="secondary"]:hover,
  button[kind="primary"]:hover {
    border-color: rgba(46, 196, 255, 0.45);
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
    .ah-title-row {
      flex-direction: column;
    }
    .status-grid {
      width: 100%;
      min-width: 0;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  @media (max-width: 640px) {
    .ah-hero,
    .mission-shell,
    .artifact-shell,
    .section-shell,
    .scan-terminal,
    .scan-sidecard,
    .scan-frame,
    .ah-panel {
      border-radius: 16px;
    }
    .status-grid {
      grid-template-columns: 1fr;
    }
    .ah-title {
      font-size: 1.45rem;
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


def generate_gradcam_views(
    runtime: Any, image_bytes: bytes, predicted_index: int
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    try:
        from backend.app import (
            ClassifierOutputTarget,
            GradCAMPlusPlus,
            _FusionGradCAMWrapper,
        )
        from matplotlib import cm as mpl_cm
        from src.config import IMAGE_SIZE
        from src.eda import extract_features_from_rgb
    except Exception as exc:
        return None, None, None, f"Grad-CAM dependencies unavailable: {exc}"

    if runtime.model is None or GradCAMPlusPlus is None or ClassifierOutputTarget is None:
        return None, None, None, "Grad-CAM++ is unavailable in the current environment."

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

    h, w = resized_rgb.shape[:2]
    heatmap_resized = cv2.resize(grayscale_cam, (w, h))
    heatmap_rgb = (mpl_cm.jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    alpha = 0.42
    blended = (alpha * heatmap_rgb + (1.0 - alpha) * resized_rgb.astype(np.float32)).astype(np.uint8)
    return resized_rgb, blended, heatmap_rgb, None


@st.cache_data(show_spinner=False, max_entries=16)
def analyze_uploaded_image(
    image_bytes: bytes, filename: str
) -> tuple[dict[str, Any], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    runtime, runtime_error = load_runtime()
    if runtime is None:
        return {}, None, None, None, runtime_error

    response = runtime.predict(image_bytes=image_bytes, filename=filename)
    response_dict = response.model_dump() if hasattr(response, "model_dump") else response.dict()
    original_rgb, blended, heatmap_rgb, overlay_error = generate_gradcam_views(
        runtime=runtime,
        image_bytes=image_bytes,
        predicted_index=int(response_dict["predicted_index"]),
    )
    return response_dict, original_rgb, blended, heatmap_rgb, overlay_error


def build_omni_explanation(report_card: dict[str, Any]) -> str:
    try:
        from src.omni import explain_forensic_report_card

        explanation = explain_forensic_report_card(report_card)
        if explanation.strip():
            return explanation
    except Exception:
        pass
    return str(report_card.get("verdict") or "VIPER completed the scan, but no narrative explanation is available.")


def prediction_reads_real(prediction: str) -> bool:
    p = (prediction or "").strip().lower()
    return "real" in p or "authentic" in p


def build_forensic_blurb_from_evidence(result: dict[str, Any]) -> str:
    elevated: list[str] = []
    for raw in result.get("evidence_breakdown", []):
        if not isinstance(raw, dict):
            continue
        sid = str(raw.get("id", "")).lower()
        if sid not in {"fft", "prnu", "lab"}:
            continue
        status = str(raw.get("status", "")).lower()
        if status in {"high", "medium", "moderate"}:
            elevated.append(str(raw.get("label", sid)))

    if len(elevated) >= 2:
        return (
            "Elevated " + ", ".join(elevated[:2]).lower()
            + " suggest atypical frequency or color structure versus the trained corpus."
        )
    if len(elevated) == 1:
        return f"Notable {elevated[0].lower()} relative to baseline reference statistics."
    return (
        "Frequency balance, residual noise, and chroma spread remain close to the corpus baseline "
        "for this resolution."
    )


def compute_prnu_map_display(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_norm = gray / 255.0
    try:
        from skimage.restoration import denoise_wavelet

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
    z = (prnu - float(prnu.mean())) / max(float(prnu.std()), 1e-6)
    z = np.clip(z / 3.0, -1.0, 1.0)
    return ((z + 1.0) * 127.5).astype(np.uint8)


def build_fft_spectrum_figure(img_rgb: np.ndarray) -> Any:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log(np.abs(fft) + 1e-8)
    figure = px.imshow(
        mag,
        aspect="equal",
        color_continuous_scale=[[0, "#0a1628"], [0.5, "#1e4976"], [1, "#2ec4ff"]],
        title="",
    )
    apply_plot_theme(figure, height=200)
    figure.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(len=0.75, thickness=12, tickfont=dict(size=9)),
    )
    figure.update_xaxes(showticklabels=False)
    figure.update_yaxes(showticklabels=False)
    return figure


def build_prnu_map_figure(img_rgb: np.ndarray) -> Any:
    prnu_u8 = compute_prnu_map_display(img_rgb)
    figure = px.imshow(prnu_u8, aspect="equal", color_continuous_scale="gray")
    apply_plot_theme(figure, height=200)
    figure.update_layout(margin=dict(l=0, r=0, t=8, b=0), coloraxis_showscale=False)
    figure.update_xaxes(showticklabels=False)
    figure.update_yaxes(showticklabels=False)
    return figure


def build_lab_distribution_figure(img_rgb: np.ndarray) -> Any:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a_flat = lab[:, :, 1].flatten()
    b_flat = lab[:, :, 2].flatten()
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=a_flat,
            name="a*",
            nbinsx=40,
            marker_color="rgba(46, 196, 255, 0.55)",
        )
    )
    figure.add_trace(
        go.Histogram(
            x=b_flat,
            name="b*",
            nbinsx=40,
            marker_color="rgba(201, 162, 39, 0.45)",
        )
    )
    apply_plot_theme(figure, height=200)
    figure.update_layout(
        barmode="overlay",
        bargap=0.02,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=8, r=8, t=28, b=8),
    )
    return figure


def render_verdict_card(result: dict[str, Any], confidence_ratio: float, blurb: str) -> None:
    model_real = prediction_reads_real(str(result.get("prediction", "")))
    chip_class = "real" if model_real else "ai"
    chip_text = "REAL" if model_real else "AI-GENERATED"
    fill_class = "" if model_real else " ai-risk"
    width_pct = max(3.0, min(100.0, confidence_ratio * 100.0))
    st.markdown(
        f"""
        <div class="ah-panel">
          <div class="ah-panel-kicker">Verdict</div>
          <div class="ah-verdict-chip {chip_class}">Model prediction · {html.escape(chip_text)}</div>
          <div class="ah-panel-title">Confidence {html.escape(str(result.get("confidence_pct", "")))}</div>
          <div class="ah-confidence-bar"><div class="ah-confidence-fill{fill_class}" style="width:{width_pct:.1f}%"></div></div>
          <p style="margin:0;color:var(--muted);font-size:0.88rem;line-height:1.55;">{html.escape(blurb)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_current_piece_frame(image_data_uri: str, filename: str) -> None:
    safe_filename = html.escape(filename)
    st.markdown(
        f"""
        <div class="scan-frame">
          <div class="scan-frame-label">Current Piece</div>
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


def render_terminal_explainer(explanation: str) -> None:
    st.markdown(
        f"""
        <div class="scan-terminal">
          <div class="scan-terminal-header">
            <span>System Insight</span>
            <span>Analysis log</span>
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
        <section class="ah-hero">
          <div class="ah-kicker">Museum vault · forensic channel</div>
          <div class="ah-title-row">
            <div>
              <h1 class="ah-title">ArtHeist: <span class="ah-title-gold">Forensic Intelligence System</span></h1>
              <p class="ah-subtitle">
                Classify artwork as real or AI-generated with fusion-model verdicts, Grad-CAM evidence,
                and validation intelligence—presented as a secure lab console with light analyst decisions.
              </p>
            </div>
            <div class="status-grid">
              <div class="status-pill safe">Authentic · gold track</div>
              <div class="status-pill danger">AI / risk channel</div>
              <div class="status-pill neutral">Analysis · electric trace</div>
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
          <div class="mission-kicker">Navigation</div>
          <div class="mission-title">Workspace</div>
          <p class="mission-copy">
            Inspect a piece under the vault lamp, then open the intelligence deck for latent space,
            confidence, and error analytics from cached validation runs.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    view = st.sidebar.radio(
        "Workspace",
        options=("Artwork Inspection", "Model Intelligence Dashboard"),
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


def build_model_performance_triple_figure(
    eval_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> Optional[Any]:
    rows: list[dict[str, Any]] = []
    b_f1 = baseline_metrics.get("f1")
    v_f1 = eval_metrics.get("f1")
    t_f1 = eval_metrics.get("test_f1")
    if b_f1 is not None:
        rows.append({"stage": "Baseline", "F1": float(b_f1)})
    if v_f1 is not None:
        rows.append({"stage": "Improved (validation)", "F1": float(v_f1)})
    if t_f1 is not None:
        rows.append({"stage": "Final (held-out test)", "F1": float(t_f1)})
    if len(rows) < 2:
        return None
    frame = pd.DataFrame(rows)
    frame["label"] = frame["F1"].map(lambda v: f"{v:.1%}")
    figure = px.bar(
        frame,
        x="stage",
        y="F1",
        text="label",
        title="Model performance · F1 progression",
        template="plotly_dark",
        color="stage",
        color_discrete_map={
            "Baseline": "#64748b",
            "Improved (validation)": "#2ec4ff",
            "Final (held-out test)": "#c9a227",
        },
    )
    apply_plot_theme(figure, height=380)
    figure.update_traces(textposition="outside", cliponaxis=False)
    figure.update_layout(showlegend=False)
    figure.update_yaxes(range=[0, 1.08], tickformat=".0%", showgrid=True, gridcolor="rgba(148, 163, 184, 0.10)", zeroline=False)
    figure.update_xaxes(showgrid=False)
    return figure


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
        "Inspection bay",
        "Artwork under the vault lamp.",
        "Upload a piece to run the fusion classifier, compare your judgment with the model, and review Grad-CAM plus lightweight forensic plots. Cached gallery samples remain available for overlay reference.",
    )

    uploader_col, gallery_col = st.columns([1.25, 0.75])
    with uploader_col:
        uploaded_file = st.file_uploader(
            "Secure intake",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=False,
            help="JPEG / PNG / WebP — routed through the VIPER runtime.",
        )
    with gallery_col:
        sample_names = list_gradcam_samples(str(GRADCAM_DIR), _mtime(GRADCAM_DIR))
        sample_choice = st.selectbox(
            "Grad-CAM gallery (reference)",
            options=["None"] + sample_names,
            index=0,
        )

    if "ah_matches" not in st.session_state:
        st.session_state.ah_matches = 0
    if "ah_decisions" not in st.session_state:
        st.session_state.ah_decisions = 0
    if "ah_feedback" not in st.session_state:
        st.session_state.ah_feedback = ""

    runtime, runtime_error = load_runtime()
    if runtime_error:
        st.info(f"Live inference unavailable: {runtime_error}")

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image_data_uri = build_png_data_uri(image_bytes)
        with st.spinner("Running forensic inference…"):
            result, orig_rgb, blended, heatmap_rgb, gradcam_error = analyze_uploaded_image(
                image_bytes, uploaded_file.name
            )

        if not result:
            st.error(gradcam_error or "Inference failed.")
            return

        forensic_rgb = orig_rgb
        if forensic_rgb is None:
            try:
                forensic_rgb = np.array(
                    Image.open(BytesIO(image_bytes))
                    .convert("RGB")
                    .resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS),
                    dtype=np.uint8,
                )
            except Exception:
                forensic_rgb = np.zeros((64, 64, 3), dtype=np.uint8)

        explanation = build_omni_explanation(result)
        blurb = build_forensic_blurb_from_evidence(result)
        conf_ratio = float(result.get("confidence_score", 0.0))

        left, center, right = st.columns([0.22, 0.38, 0.40])

        with left:
            st.markdown(
                """
                <div class="ah-panel">
                  <div class="ah-panel-kicker">Analyst decision</div>
                  <div class="ah-panel-title">Your call</div>
                  <p style="margin:0.5rem 0 0;font-size:0.82rem;color:var(--muted);line-height:1.5;">
                    Record judgment against the model. Score tracks agreement only (not ground truth).
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")
            model_real = prediction_reads_real(str(result["prediction"]))
            if st.button("Mark as Authentic", key="mark_authentic", use_container_width=True):
                st.session_state.ah_decisions += 1
                if model_real:
                    st.session_state.ah_matches += 1
                    st.session_state.ah_feedback = (
                        "Aligned — your assessment matches the model: authentic artwork."
                    )
                else:
                    st.session_state.ah_feedback = (
                        "Disagreement — the model classifies this as AI-generated."
                    )
            if st.button("Flag as AI-Generated", key="mark_ai", use_container_width=True):
                st.session_state.ah_decisions += 1
                if not model_real:
                    st.session_state.ah_matches += 1
                    st.session_state.ah_feedback = (
                        "Aligned — your assessment matches the model: AI-generated."
                    )
                else:
                    st.session_state.ah_feedback = (
                        "Disagreement — the model reads this as authentic artwork."
                    )

            if st.session_state.ah_feedback:
                st.markdown(
                    f'<div class="ah-feedback">{html.escape(st.session_state.ah_feedback)}</div>',
                    unsafe_allow_html=True,
                )
            total_d = int(st.session_state.ah_decisions)
            matches = int(st.session_state.ah_matches)
            st.markdown(
                f'<p class="ah-score-line">Correct identifications (vs model): {matches} / {total_d}</p>',
                unsafe_allow_html=True,
            )

        with center:
            st.markdown(
                '<div class="ah-panel-kicker" style="margin-bottom:0.35rem;">Exhibit display</div>',
                unsafe_allow_html=True,
            )
            render_current_piece_frame(image_data_uri=image_data_uri, filename=uploaded_file.name)

        with right:
            render_verdict_card(result, conf_ratio, blurb)
            st.markdown(
                '<div class="ah-panel-kicker" style="margin:0.85rem 0 0.35rem;">Visual evidence · Grad-CAM</div>',
                unsafe_allow_html=True,
            )
            view_mode = st.radio(
                "View",
                ("Original", "Heatmap", "Blended"),
                horizontal=True,
                label_visibility="collapsed",
            )
            display_img: Optional[np.ndarray] = None
            if view_mode == "Original" and orig_rgb is not None:
                display_img = orig_rgb
            elif view_mode == "Heatmap" and heatmap_rgb is not None:
                display_img = heatmap_rgb
            elif view_mode == "Blended" and blended is not None:
                display_img = blended
            if display_img is not None:
                st.image(display_img, use_container_width=True)
            else:
                st.warning(gradcam_error or "Grad-CAM views unavailable.")

            gradcam_item = _signal_items(result).get("gradcam")
            if gradcam_item is not None:
                st.caption(
                    f"{gradcam_item.get('detail', '')} · {gradcam_item.get('value', '')}"
                )

            mini = st.columns(4)
            mini[0].metric("P(real)", f"{1.0 - float(result['ai_probability']):.1%}")
            mini[1].metric("P(AI)", f"{float(result['ai_probability']):.1%}")
            mini[2].metric("Fusion", "On" if result["uses_eda_fusion"] else "Img")
            mini[3].metric("CAM", "OK" if result["gradcam_available"] else "—")

            st.markdown(
                '<div class="ah-panel-kicker" style="margin:0.85rem 0 0.5rem;">Forensic signals</div>',
                unsafe_allow_html=True,
            )
            with st.expander("FFT spectrum · frequency domain", expanded=False):
                st.caption("Log-magnitude spectrum; irregular high-frequency energy can indicate synthesis artifacts.")
                st.plotly_chart(build_fft_spectrum_figure(forensic_rgb), use_container_width=True)
            with st.expander("PRNU noise map · residual field", expanded=False):
                st.caption("Wavelet residual emphasis; camera-like vs synthetic noise structure.")
                st.plotly_chart(build_prnu_map_figure(forensic_rgb), use_container_width=True)
            with st.expander("LAB color distribution", expanded=False):
                st.caption("a* / b* histograms in LAB space for chroma spread.")
                st.plotly_chart(build_lab_distribution_figure(forensic_rgb), use_container_width=True)

            st.markdown('<div class="signal-label">Signal scores</div>', unsafe_allow_html=True)
            render_signal_buckets(result)

            render_terminal_explainer(explanation)

            with st.expander("Raw evidence table", expanded=False):
                evidence_df = pd.DataFrame(result["evidence_breakdown"])
                if not evidence_df.empty:
                    st.dataframe(
                        evidence_df[["label", "status", "score", "value", "detail"]],
                        use_container_width=True,
                        hide_index=True,
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
            caption=f"Gallery reference: {sample_choice}",
            use_container_width=True,
        )
    else:
        st.info("Upload artwork for a full inspection, or pick a gallery overlay for a quick visual reference.")


def build_umap_figure(
    projection_df: pd.DataFrame,
    only_errors: bool,
    color_mode: str,
) -> Any:
    filtered_df = projection_df.copy()
    if only_errors:
        filtered_df = filtered_df[filtered_df["error_type"] != "correct"]

    if filtered_df.empty:
        return None

    filtered_df["marker_group"] = filtered_df["error_type"].str.replace("_", " ").str.title()
    hover_data = {
        "label_name": True,
        "predicted_name": True,
        "predicted_confidence": ":.1%",
        "ai_probability": ":.1%",
        "error_type": True,
        "umap_x": ":.3f",
        "umap_y": ":.3f",
    }

    if color_mode == "truth" and "label_name" in filtered_df.columns:
        figure = px.scatter(
            filtered_df,
            x="umap_x",
            y="umap_y",
            color="label_name",
            symbol="marker_group",
            hover_name="filename",
            hover_data=hover_data,
            title="Latent space map · true label (validation)",
            labels={
                "umap_x": "UMAP-1",
                "umap_y": "UMAP-2",
                "label_name": "True class",
                "marker_group": "Outcome",
            },
            template="plotly_dark",
            color_discrete_map={
                "REAL": "#c9a227",
                "AI_GENERATED": "#e85d5d",
            },
        )
        figure.update_layout(legend_title_text="True label")
    else:
        figure = px.scatter(
            filtered_df,
            x="umap_x",
            y="umap_y",
            color="predicted_confidence",
            color_continuous_scale=["#0a1628", "#1e4976", "#2ec4ff", "#c9a227", "#e85d5d"],
            symbol="marker_group",
            hover_name="filename",
            hover_data=hover_data,
            title="Latent space map · model confidence",
            labels={
                "umap_x": "UMAP-1",
                "umap_y": "UMAP-2",
                "predicted_confidence": "Confidence",
                "marker_group": "Outcome",
            },
            template="plotly_dark",
        )
        figure.update_layout(coloraxis_colorbar_title="Confidence", legend_title_text="Outcome")

    figure.update_traces(marker={"size": 9, "line": {"width": 0.5, "color": "rgba(255,255,255,0.12)"}})
    figure.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.08)", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.08)", zeroline=False)
    return apply_plot_theme(figure, height=560)


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
        "Model intelligence",
        "Validation-only intelligence deck.",
        "UMAP projections, confidence coloring, F1 progression, and error taxonomies are computed from cached validation predictions—no training leakage.",
    )

    if projection_df.empty:
        st.warning(
            "Validation artifacts are missing. Run `python src/dataloader.py --precompute-validation` first."
        )
        return

    render_metrics_strip(eval_metrics, baseline_metrics, summary)

    st.markdown(
        """
        <div class="ah-panel-kicker" style="margin:1rem 0 0.35rem;">Latent space map</div>
        <p style="margin:0 0 0.75rem;color:var(--muted);font-size:0.9rem;max-width:52rem;">
          Two-dimensional embedding of validation embeddings. Toggle coloring to separate true Real vs AI clusters
          or inspect certainty via continuous confidence.
        </p>
        """,
        unsafe_allow_html=True,
    )
    ctrl1, ctrl2, ctrl3 = st.columns([0.42, 0.28, 0.3])
    with ctrl1:
        color_mode = st.radio(
            "Color by",
            options=("truth", "confidence"),
            format_func=lambda m: "True label (REAL / AI)" if m == "truth" else "Model confidence",
            horizontal=True,
            label_visibility="visible",
        )
    with ctrl2:
        only_errors = st.toggle("Misclassified only", value=False)
    with ctrl3:
        st.metric("Edge cases in validation", f"{len(errors_df)}")

    umap_figure = build_umap_figure(
        projection_df,
        only_errors=only_errors,
        color_mode=str(color_mode),
    )
    if umap_figure is not None:
        st.plotly_chart(umap_figure, use_container_width=True)

    st.markdown(
        '<div class="ah-panel-kicker" style="margin:1rem 0 0.35rem;">Model performance</div>',
        unsafe_allow_html=True,
    )
    perf_row = st.columns(2)
    with perf_row[0]:
        triple_fig = build_model_performance_triple_figure(eval_metrics, baseline_metrics)
        if triple_fig is not None:
            st.plotly_chart(triple_fig, use_container_width=True)
    with perf_row[1]:
        comparison_figure = build_fast_track_comparison_figure(eval_metrics, baseline_metrics)
        if comparison_figure is not None:
            st.plotly_chart(comparison_figure, use_container_width=True)

    st.markdown(
        '<div class="ah-panel-kicker" style="margin:1rem 0 0.35rem;">Error analysis</div>',
        unsafe_allow_html=True,
    )
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
                title="Misclassifications by confidence bucket",
                labels={
                    "error_type": "Failure mode",
                    "count": "Images",
                    "confidence_bucket": "Confidence",
                },
                template="plotly_dark",
                color_discrete_map={"high": "#e85d5d", "medium": "#c9a227", "low": "#2ec4ff"},
            )
            apply_plot_theme(bar_figure, height=400)
            bar_figure.update_xaxes(showgrid=False)
            bar_figure.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.10)", zeroline=False)
            st.plotly_chart(bar_figure, use_container_width=True)
        with chart_cols[1]:
            pie_figure = px.pie(
                breakdown_df,
                names="failure_mode",
                values="count",
                title="Share of failure modes",
                template="plotly_dark",
                hole=0.52,
                color="confidence_bucket",
                color_discrete_map={"high": "#e85d5d", "medium": "#c9a227", "low": "#2ec4ff"},
            )
            apply_plot_theme(pie_figure, height=400)
            st.plotly_chart(pie_figure, use_container_width=True)
    else:
        st.success("No validation misses in the current artifact set.")

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

    with st.expander("Telemetry JSON", expanded=False):
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
        page_title="ArtHeist · Forensic Intelligence",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    render_banner()
    view = render_sidebar()

    if view == "Artwork Inspection":
        render_image_forensics()
    else:
        render_forensic_optics()


if __name__ == "__main__":
    main()
