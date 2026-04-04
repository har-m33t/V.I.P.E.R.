"""
src/baseline.py — VIPER Forensic Engine: Classical ML Baseline
Phase: Track Beta (Analytical Pipeline) — Baseline Agent

Trains a Logistic Regression classifier on the EDA feature matrix.
Outputs top-3 features by coefficient magnitude and saves metrics to JSON.

Dependency: results/feature_matrix.csv must exist (produced by EDA Agent).

Usage:
    python src/baseline.py
    # → writes results/baseline_metrics.json
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import FEATURE_MATRIX_CSV, BASELINE_METRICS_JSON, SEED


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Trainer
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(
    feature_csv: Path = FEATURE_MATRIX_CSV,
    output_json: Path = BASELINE_METRICS_JSON,
    seed: int = SEED,
) -> dict:
    """
    Train Logistic Regression on EDA features and evaluate on held-out split.

    Args:
        feature_csv: Path to feature_matrix.csv (from EDA Agent).
        output_json: Path to write baseline metrics JSON.
        seed:        Random seed for reproducibility.

    Returns:
        Dict of metrics.

    AGENT_TASK: try additional classical models — RandomForest, XGBoost, SVM
    AGENT_TASK: implement cross-validation instead of a single split
    """
    # ── Load feature matrix ───────────────────────────────────────────────────
    if not feature_csv.exists():
        out = {"status": "failed", "reason": f"Missing {feature_csv}"}
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(out, indent=2))
        raise FileNotFoundError(f"[BaselineAgent] {feature_csv} not found. Run EDA first.")

    df = pd.read_csv(feature_csv)
    print(f"[BaselineAgent] Loaded feature matrix: {df.shape}")

    feature_cols = [c for c in df.columns if c not in ("image_path", "label")]
    X = df[feature_cols].fillna(0.0).values
    y = df["label"].values

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )

    # ── Pipeline: scale → logistic regression ─────────────────────────────────
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=seed,
        )),
    ])
    pipe.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_test, y_proba)),
        "n_train":   int(len(X_train)),
        "n_test":    int(len(X_test)),
    }

    # ── Top-3 features by |coefficient| ───────────────────────────────────────
    coef = pipe.named_steps["clf"].coef_[0]
    ranked = sorted(
        zip(feature_cols, coef),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    top3 = [{"feature": f, "coefficient": float(c)} for f, c in ranked[:3]]
    metrics["top_3_features"] = top3

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2))

    print(f"\n[BaselineAgent] === Results ===")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"  Top features:")
    for item in top3:
        print(f"    {item['feature']}: {item['coefficient']:+.4f}")
    print(f"\n[BaselineAgent] ✓ Metrics saved → {output_json}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "AI-Generated"]))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Baseline Agent — VIPER Forensic Engine ===")
    try:
        run_baseline()
    except FileNotFoundError as e:
        print(f"[BaselineAgent] ✗ {e}")
        sys.exit(1)
