"""
src/model.py - VIPER Forensic Engine: ConvNeXt-Tiny Architecture

Phase 3.1 upgrades the image backbone from EfficientNet-B0 to ConvNeXt-Tiny.
The model also exposes an optional late-fusion path for Phase 3.2, but that
path is only active when a non-zero EDA feature dimension is provided.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    BEST_MODEL_PATH,
    CLASSIFIER_DROPOUT,
    DEVICE,
    FUSION_HIDDEN_DIM,
    IMAGE_EMBED_DIM,
    MODEL_NAME,
    NUM_CLASSES,
    TORCH_WEIGHTS_DIR,
    UNFREEZE_STAGES,
)


class VIPERClassifier(nn.Module):
    """
    ConvNeXt-Tiny binary classifier for AI-generated image detection.

    The base model exposes a 768D image embedding. When `eda_feature_dim > 0`,
    the forward pass concatenates the image embedding with the forensic feature
    vector and routes the result through a 2-layer fusion head.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = CLASSIFIER_DROPOUT,
        eda_feature_dim: int = 0,
        fusion_hidden_dim: int = FUSION_HIDDEN_DIM,
        pretrained: bool = True,
    ):
        super().__init__()
        self.eda_feature_dim = int(eda_feature_dim)
        self.embedding_dim = IMAGE_EMBED_DIM
        self._warned_missing_eda = False

        torch.hub.set_dir(str(TORCH_WEIGHTS_DIR))
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            self.backbone = models.convnext_tiny(weights=weights)
            self.using_pretrained_backbone = weights is not None
        except Exception as exc:
            print(
                f"[DLAgent] Warning: failed to load ConvNeXt-Tiny pretrained weights "
                f"({exc}). Falling back to random initialization."
            )
            self.backbone = models.convnext_tiny(weights=None)
            self.using_pretrained_backbone = False

        self.embedding_norm = self.backbone.classifier[0]
        self.backbone.classifier = nn.Identity()

        if self.eda_feature_dim > 0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout, inplace=False),
                nn.Linear(self.embedding_dim + self.eda_feature_dim, fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(fusion_hidden_dim, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout, inplace=False),
                nn.Linear(self.embedding_dim, num_classes),
            )

        self._stage_groups = [
            (self.backbone.features[0], self.backbone.features[1]),
            (self.backbone.features[2], self.backbone.features[3]),
            (self.backbone.features[4], self.backbone.features[5]),
            (self.backbone.features[6], self.backbone.features[7]),
        ]

        self._freeze_backbone()
        self.set_trainable_stages(UNFREEZE_STAGES)

        # Final CNBlock in the last ConvNeXt stage.
        self.gradcam_target_layer = self.backbone.features[-1][-1]

    def _freeze_backbone(self) -> None:
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.embedding_norm.parameters():
            param.requires_grad = True

    def set_trainable_stages(self, n_stages: int) -> None:
        """
        Unfreeze the final `n_stages` ConvNeXt stages plus the head.

        Stage groups include each stage's downsampling transition so the
        trainable boundary remains coherent.
        """
        n_stages = max(0, min(int(n_stages), len(self._stage_groups)))
        for stage_group in self._stage_groups[-n_stages:]:
            for module in stage_group:
                for param in module.parameters():
                    param.requires_grad = True

        for param in self.head.parameters():
            param.requires_grad = True

    def _extract_image_embedding(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(x)
        pooled = self.backbone.avgpool(features)
        normalized = self.embedding_norm(pooled)
        return torch.flatten(normalized, 1)

    def _prepare_eda_features(
        self,
        eda_features: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.eda_feature_dim <= 0:
            return None

        if eda_features is None or eda_features.numel() == 0:
            if not self._warned_missing_eda:
                print(
                    "[DLAgent] Warning: model expects EDA features but none were "
                    "provided. Using zero vectors for this forward pass."
                )
                self._warned_missing_eda = True
            return torch.zeros(
                (batch_size, self.eda_feature_dim),
                device=device,
                dtype=dtype,
            )

        if eda_features.dim() == 1:
            eda_features = eda_features.unsqueeze(0)

        if eda_features.shape[1] != self.eda_feature_dim:
            raise ValueError(
                f"Expected {self.eda_feature_dim} EDA features, "
                f"received {eda_features.shape[1]}."
            )

        return eda_features.to(device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        eda_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedding = self._extract_image_embedding(x)
        eda_features = self._prepare_eda_features(
            eda_features=eda_features,
            batch_size=embedding.shape[0],
            device=embedding.device,
            dtype=embedding.dtype,
        )
        if eda_features is not None:
            embedding = torch.cat([embedding, eda_features], dim=1)
        return self.head(embedding)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the 768D image embedding before any classifier or fusion head."""
        return self._extract_image_embedding(x)

    def count_trainable_params(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def build_model(
    device: torch.device = DEVICE,
    eda_feature_dim: int = 0,
    pretrained: bool = True,
) -> VIPERClassifier:
    """
    Build and return a VIPERClassifier ready for training or inference.
    """
    model = VIPERClassifier(
        eda_feature_dim=eda_feature_dim,
        pretrained=pretrained,
    ).to(device)
    trainable, total = model.count_trainable_params()
    print(f"[DLAgent] VIPERClassifier built:")
    print(f"          Backbone         : {MODEL_NAME}")
    print(f"          Pretrained       : {model.using_pretrained_backbone}")
    print(f"          EDA feature dim  : {eda_feature_dim}")
    print(f"          Trainable params : {trainable:,} / {total:,}")
    print(f"          Grad-CAM target  : {type(model.gradcam_target_layer).__name__}")
    return model


def load_checkpoint(
    checkpoint_path: Path = BEST_MODEL_PATH,
    device: torch.device = DEVICE,
) -> Optional[VIPERClassifier]:
    """
    Load a saved model checkpoint.
    """
    if not checkpoint_path.exists():
        print(f"[DLAgent] No checkpoint found at {checkpoint_path}")
        return None

    state = torch.load(checkpoint_path, map_location=device)
    eda_feature_dim = int(state.get("eda_feature_dim", 0))
    model_name = state.get("model_name")
    if model_name != MODEL_NAME:
        legacy_name = model_name or "legacy_checkpoint"
        print(
            f"[DLAgent] Checkpoint model '{legacy_name}' does not match the current "
            f"architecture '{MODEL_NAME}'. Retrain the model before evaluation."
        )
        return None

    model = build_model(
        device=device,
        eda_feature_dim=eda_feature_dim,
        pretrained=False,
    )
    try:
        model.load_state_dict(state["model_state_dict"])
    except RuntimeError as exc:
        print(f"[DLAgent] Failed to load checkpoint: {exc}")
        return None

    epoch = state.get("epoch", "?")
    f1 = state.get("val_f1")
    f1_text = f"{f1:.4f}" if isinstance(f1, (float, int)) else "?"
    print(f"[DLAgent] Loaded checkpoint (epoch={epoch}, val_f1={f1_text})")
    return model


if __name__ == "__main__":
    print("=== Model Agent - VIPER Forensic Engine ===")
    model = build_model()
    print("\nTrainable layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  * {name}")
