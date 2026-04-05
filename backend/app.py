from __future__ import annotations

import random
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    AI_ART_DIR,
    BEST_MODEL_PATH,
    DEVICE,
    FEATURE_MATRIX_CSV,
    IMAGE_SIZE,
    LABEL_AI,
    LABEL_REAL,
    MODEL_NAME,
    REAL_ART_DIR,
    SEED,
    TEST_SPLIT,
    USE_EDA_FEATURES,
    VAL_SPLIT,
)
from src.dataloader import _collect_images, _normalize_image_path, get_val_transform  # noqa: E402
from src.eda import FEATURE_META_COLUMNS, extract_features_from_rgb  # noqa: E402
from src.model import VIPERClassifier, load_checkpoint  # noqa: E402

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    GradCAMPlusPlus = None
    ClassifierOutputTarget = None


class EvidenceItem(BaseModel):
    id: str
    label: str
    status: str
    score: float
    value: str
    detail: str


class PredictResponse(BaseModel):
    filename: str
    prediction: str
    predicted_index: int
    confidence_score: float
    confidence_pct: str
    ai_probability: float
    verdict: str
    evidence_breakdown: list[EvidenceItem]
    model_name: str
    uses_eda_fusion: bool
    gradcam_available: bool


class HealthResponse(BaseModel):
    ready: bool
    model_name: str
    checkpoint_loaded: bool
    uses_eda_fusion: bool
    gradcam_available: bool
    detail: str


@dataclass
class FeatureReference:
    columns: list[str]
    train_mean: np.ndarray
    train_std: np.ndarray
    full_mean: dict[str, float]
    full_std: dict[str, float]


class _FusionGradCAMWrapper(nn.Module):
    """Adapter so Grad-CAM++ can call a fusion model with fixed EDA features."""

    def __init__(
        self,
        model: VIPERClassifier,
        eda_features: Optional[torch.Tensor],
    ):
        super().__init__()
        self.model = model
        self._eda_features = eda_features

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        eda_features = self._eda_features
        if eda_features is not None and eda_features.shape[0] != image_tensor.shape[0]:
            eda_features = eda_features.expand(image_tensor.shape[0], -1)
        return self.model(image_tensor, eda_features=eda_features)


class ModelRuntime:
    def __init__(self) -> None:
        self.device = DEVICE
        self.transform = get_val_transform()
        self.model = load_checkpoint(BEST_MODEL_PATH, device=self.device)
        self.feature_reference = self._load_feature_reference()
        self.ready, self.detail = self._validate_runtime()

        if self.model is not None:
            self.model.eval()

    def _validate_runtime(self) -> tuple[bool, str]:
        if self.model is None:
            return False, f"Checkpoint unavailable at {BEST_MODEL_PATH}"

        if self.model.eda_feature_dim > 0 and self.feature_reference is None:
            return False, f"EDA feature matrix unavailable at {FEATURE_MATRIX_CSV}"

        if self.model.eda_feature_dim > 0:
            actual_dim = len(self.feature_reference.columns)
            if actual_dim != self.model.eda_feature_dim:
                return (
                    False,
                    "EDA feature dimension mismatch between checkpoint and "
                    f"feature matrix ({self.model.eda_feature_dim} != {actual_dim})",
                )

        return True, "VIPER runtime ready"

    def _load_feature_reference(self) -> Optional[FeatureReference]:
        if not FEATURE_MATRIX_CSV.exists():
            return None

        df = pd.read_csv(FEATURE_MATRIX_CSV)
        if df.empty:
            return None

        columns = [
            column for column in df.columns
            if column not in FEATURE_META_COLUMNS
        ]
        if not columns:
            return None

        train_df = df
        try:
            ai_paths, _ = _collect_images(AI_ART_DIR, LABEL_AI)
            real_paths, _ = _collect_images(REAL_ART_DIR, LABEL_REAL)
            all_paths = ai_paths + real_paths

            rng = random.Random(SEED)
            indices = list(range(len(all_paths)))
            rng.shuffle(indices)

            n_total = len(indices)
            n_test = int(n_total * TEST_SPLIT)
            n_val = int(n_total * VAL_SPLIT)
            n_train = n_total - n_test - n_val
            train_keys = {
                _normalize_image_path(all_paths[i])
                for i in indices[:n_train]
            }

            normalized_paths = df["image_path"].map(_normalize_image_path)
            matched_df = df[normalized_paths.isin(train_keys)]
            if not matched_df.empty:
                train_df = matched_df
        except Exception:
            train_df = df

        mean = train_df[columns].astype(np.float32).mean(axis=0)
        std = train_df[columns].astype(np.float32).std(axis=0, ddof=0)
        std = std.mask(std < 1e-6, 1.0)

        full_mean = df[columns].astype(np.float32).mean(axis=0).to_dict()
        full_std = (
            df[columns]
            .astype(np.float32)
            .std(axis=0, ddof=0)
            .mask(lambda series: series < 1e-6, 1.0)
            .to_dict()
        )

        return FeatureReference(
            columns=columns,
            train_mean=mean.to_numpy(dtype=np.float32),
            train_std=std.to_numpy(dtype=np.float32),
            full_mean={key: float(value) for key, value in full_mean.items()},
            full_std={key: float(value) for key, value in full_std.items()},
        )

    def _standardize_eda_features(self, features: dict[str, float]) -> Optional[torch.Tensor]:
        if self.model is None or self.model.eda_feature_dim <= 0:
            return None
        if self.feature_reference is None:
            raise RuntimeError("EDA feature reference is not loaded.")

        vector = np.asarray(
            [float(features[column]) for column in self.feature_reference.columns],
            dtype=np.float32,
        )
        standardized = (vector - self.feature_reference.train_mean) / self.feature_reference.train_std
        tensor = torch.from_numpy(standardized).unsqueeze(0)
        return tensor.to(device=self.device, dtype=torch.float32)

    def _feature_zscore(self, name: str, value: float) -> float:
        if self.feature_reference is None:
            return 0.0
        mean = float(self.feature_reference.full_mean.get(name, 0.0))
        std = float(self.feature_reference.full_std.get(name, 1.0))
        if std < 1e-6:
            return 0.0
        return (float(value) - mean) / std

    @staticmethod
    def _clamp_score(value: float) -> float:
        return float(max(0.0, min(value, 1.0)))

    def _severity(self, score: float) -> str:
        if score >= 0.78:
            return "high"
        if score >= 0.52:
            return "moderate"
        return "low"

    @staticmethod
    def _sigmoid(value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-value)))

    def _lab_saturation(self, image_rgb: np.ndarray) -> dict[str, float]:
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        a_channel = lab[:, :, 1] - 128.0
        b_channel = lab[:, :, 2] - 128.0
        chroma = np.sqrt(a_channel ** 2 + b_channel ** 2)
        return {
            "lab_chroma_mean": float(chroma.mean() / 181.0),
            "lab_chroma_std": float(chroma.std() / 181.0),
        }

    def _summarize_gradcam(
        self,
        image_tensor: torch.Tensor,
        predicted_index: int,
        eda_features: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        if self.model is None or GradCAMPlusPlus is None or ClassifierOutputTarget is None:
            return {
                "available": False,
                "score": 0.0,
                "hotspot_fraction": 0.0,
                "description": "Grad-CAM++ is unavailable in the current environment.",
            }

        wrapper = _FusionGradCAMWrapper(self.model, eda_features=eda_features)
        cam = GradCAMPlusPlus(
            model=wrapper,
            target_layers=[self.model.gradcam_target_layer],
        )
        try:
            targets = [ClassifierOutputTarget(predicted_index)]
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0].astype(np.float32)
        finally:
            activations_and_grads = getattr(cam, "activations_and_grads", None)
            release = getattr(activations_and_grads, "release", None)
            if callable(release):
                release()

        heatmap = np.clip(grayscale_cam, 0.0, 1.0)
        hotspot_threshold = float(np.quantile(heatmap, 0.85))
        hotspot_fraction = float((heatmap >= hotspot_threshold).mean())
        h, w = heatmap.shape
        center = heatmap[h // 4:(3 * h) // 4, w // 4:(3 * w) // 4]
        center_bias = float(center.mean() / (heatmap.mean() + 1e-6))
        peak_intensity = float(np.percentile(heatmap, 95))

        if hotspot_fraction < 0.18 and center_bias > 1.2:
            description = "Localized activation around a concentrated anomaly cluster."
        elif hotspot_fraction < 0.28:
            description = "Selective attention over a few narrow forensic hot spots."
        else:
            description = "Diffuse frame-wide activation across texture-heavy regions."

        score = self._clamp_score((peak_intensity * 0.7) + min(hotspot_fraction / 0.4, 1.0) * 0.3)
        return {
            "available": True,
            "score": score,
            "hotspot_fraction": hotspot_fraction,
            "description": description,
        }

    def _build_evidence(
        self,
        features: dict[str, float],
        image_rgb: np.ndarray,
        gradcam_summary: dict[str, Any],
    ) -> list[EvidenceItem]:
        lab_metrics = self._lab_saturation(image_rgb)

        fft_ratio = float(features["fft_ratio"])
        fft_z = self._feature_zscore("fft_ratio", fft_ratio)
        fft_score = self._clamp_score(self._sigmoid(max(fft_z, 0.0)))

        prnu_ratio = float(features["prnu_fft_ratio"])
        prnu_peak = float(features["prnu_autocorr_peak"])
        prnu_z = max(
            self._feature_zscore("prnu_fft_ratio", prnu_ratio),
            self._feature_zscore("prnu_autocorr_peak", prnu_peak),
        )
        prnu_score = self._clamp_score(self._sigmoid(max(prnu_z, 0.0)))

        lab_score = self._clamp_score(
            min(lab_metrics["lab_chroma_mean"] * 1.4, 1.0) * 0.7
            + self._sigmoid(max(self._feature_zscore("hue_std", float(features["hue_std"])), 0.0)) * 0.3
        )

        gradcam_score = float(gradcam_summary["score"])

        return [
            EvidenceItem(
                id="fft",
                label="FFT anomaly",
                status=self._severity(fft_score),
                score=fft_score,
                value=f"{fft_ratio:.2f}x high/low ratio",
                detail=(
                    "High-frequency energy is elevated against the VIPER corpus baseline."
                    if fft_score >= 0.52
                    else "Frequency balance stays close to the learned corpus baseline."
                ),
            ),
            EvidenceItem(
                id="prnu",
                label="PRNU inconsistency",
                status=self._severity(prnu_score),
                score=prnu_score,
                value=f"{prnu_ratio:.1f}x residual skew",
                detail=(
                    "Noise residuals look less camera-like and more spectrally irregular."
                    if prnu_score >= 0.52
                    else "Residual noise remains comparatively stable for a natural capture."
                ),
            ),
            EvidenceItem(
                id="lab",
                label="LAB saturation",
                status=self._severity(lab_score),
                score=lab_score,
                value=f"{lab_metrics['lab_chroma_mean']:.2f} normalized chroma",
                detail=(
                    "Color separation and chroma spread are unusually vivid."
                    if lab_score >= 0.52
                    else "Chroma spread sits within a restrained natural range."
                ),
            ),
            EvidenceItem(
                id="gradcam",
                label="Grad-CAM focus",
                status=self._severity(gradcam_score),
                score=gradcam_score,
                value=(
                    f"{gradcam_summary['hotspot_fraction']:.0%} hotspot coverage"
                    if gradcam_summary["available"]
                    else "Unavailable"
                ),
                detail=str(gradcam_summary["description"]),
            ),
        ]

    def predict(self, image_bytes: bytes, filename: str) -> PredictResponse:
        if not self.ready or self.model is None:
            raise RuntimeError(self.detail)

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("Uploaded file is not a valid image.") from exc

        image_rgb = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.uint8)
        features = extract_features_from_rgb(image_rgb)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        eda_tensor = self._standardize_eda_features(features)

        with torch.no_grad():
            logits = self.model(image_tensor, eda_features=eda_tensor)
            probabilities = F.softmax(logits, dim=1)[0]
            predicted_index = int(probabilities.argmax().item())
            predicted_confidence = float(probabilities[predicted_index].item())
            ai_probability = float(probabilities[LABEL_AI].item())

        gradcam_summary = self._summarize_gradcam(
            image_tensor=image_tensor,
            predicted_index=predicted_index,
            eda_features=eda_tensor,
        )
        evidence = self._build_evidence(features, image_rgb=image_rgb, gradcam_summary=gradcam_summary)

        prediction = "AI-Generated" if predicted_index == LABEL_AI else "Real"
        verdict = (
            "Synthetic artifact pattern detected."
            if predicted_index == LABEL_AI
            else "Natural capture signature favored."
        )

        return PredictResponse(
            filename=filename,
            prediction=prediction,
            predicted_index=predicted_index,
            confidence_score=round(predicted_confidence, 4),
            confidence_pct=f"{predicted_confidence:.1%}",
            ai_probability=round(ai_probability, 4),
            verdict=verdict,
            evidence_breakdown=evidence,
            model_name=MODEL_NAME,
            uses_eda_fusion=bool(self.model.eda_feature_dim > 0),
            gradcam_available=bool(gradcam_summary["available"]),
        )


runtime: Optional[ModelRuntime] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime
    runtime = ModelRuntime()
    app.state.runtime = runtime
    yield


app = FastAPI(
    title="VIPER Forensic Engine API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "VIPER Forensic Engine API",
        "model": MODEL_NAME,
        "predict_endpoint": "/predict",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    active_runtime = runtime or ModelRuntime()
    return HealthResponse(
        ready=active_runtime.ready,
        model_name=MODEL_NAME,
        checkpoint_loaded=bool(active_runtime.model is not None),
        uses_eda_fusion=bool(active_runtime.model and active_runtime.model.eda_feature_dim > 0),
        gradcam_available=GradCAMPlusPlus is not None,
        detail=active_runtime.detail,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    active_runtime = runtime
    if active_runtime is None:
        raise HTTPException(status_code=503, detail="Runtime is still initializing.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename for uploaded image.")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be 10 MB or smaller.")

    try:
        return active_runtime.predict(image_bytes=image_bytes, filename=file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
