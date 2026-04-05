"""
Regenerate a clean, minimal 2D UMAP scatter from umap_features.csv.
Saves to omni_export/umap_scatter.html
"""
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.config import UMAP_FEATURES_CSV, OMNI_DIR

def build_2d_umap():
    if not UMAP_FEATURES_CSV.exists():
        print(f"[ERROR] {UMAP_FEATURES_CSV} not found. Run: python src/evaluate.py")
        return

    df = pd.read_csv(UMAP_FEATURES_CSV)

    # Defensive: drop rows with missing coordinates
    df = df.dropna(subset=["umap_x", "umap_y"])

    # Human-readable label
    df["Class"] = df["label"].map({0: "Real Art", 1: "AI Generated"})
    df["filename"] = df["image_path"].apply(lambda p: Path(p).name)

    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="Class",
        color_discrete_map={"Real Art": "#3b82f6", "AI Generated": "#ef4444"},
        hover_data={"filename": True, "umap_x": ":.3f", "umap_y": ":.3f", "Class": True},
        title="VIPER — 2D UMAP Latent Space<br><sup>ConvNeXt-Tiny feature embeddings compressed to 2D (cosine metric)</sup>",
        labels={"umap_x": "UMAP Dimension 1", "umap_y": "UMAP Dimension 2"},
        template="plotly_dark",
        opacity=0.85,
        width=900,
        height=650,
    )

    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="rgba(255,255,255,0.2)")))
    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(family="Inter, sans-serif", color="#f3f4f6"),
        legend=dict(
            title="Class",
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
        title_font=dict(size=16),
        margin=dict(l=40, r=40, t=70, b=40),
    )

    OMNI_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OMNI_DIR / "umap_scatter.html"
    fig.write_html(
        out_path,
        include_plotlyjs="cdn",   # lightweight: loads Plotly from CDN, not inline
        full_html=True,
    )
    print(f"[OK] 2D UMAP saved -> {out_path}")
    print(f"     Points: {len(df)} | AI: {(df['label']==1).sum()} | Real: {(df['label']==0).sum()}")

if __name__ == "__main__":
    build_2d_umap()
