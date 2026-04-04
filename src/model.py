"""
src/model.py — VIPER Forensic Engine: EfficientNet-B0 Architecture
Phase: Track Gamma (Deep Learning) — Deep Learning Agent

Wraps torchvision's EfficientNet-B0 with:
  - Selective layer freezing (base frozen, last 3 MBConv + classifier unfrozen)
  - Feature extraction hook for UMAP embeddings
  - Grad-CAM compatible forward pass

Usage:
    from src.model import build_model, get_feature_extractor
    model = build_model()           # full classifier
    extractor = get_feature_extractor(model)  # 512-d embeddings
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import NUM_CLASSES, DEVICE, UNFREEZE_BLOCKS, BEST_MODEL_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────────────────────────────────────

class VIPERClassifier(nn.Module):
    """
    EfficientNet-B0 binary classifier for AI-generated image detection.

    Architecture:
        - Feature extractor: EfficientNet-B0 backbone (pretrained on ImageNet)
        - Frozen layers: all except last UNFREEZE_BLOCKS MBConv + classifier
        - Head: AdaptiveAvgPool2d → Dropout(0.3) → Linear(1280, 2)

    The final MBConv block (`features[-1]`) is the Grad-CAM target layer.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # ── Replace classifier head ────────────────────────────────────────────
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        # ── Apply selective freezing ───────────────────────────────────────────
        self._freeze_backbone()
        self._unfreeze_last_blocks(UNFREEZE_BLOCKS)

        # ── Expose the Grad-CAM target layer (last conv block) ─────────────────
        # EfficientNet-B0 has 9 feature blocks (features[0]…features[8])
        self.gradcam_target_layer = self.backbone.features[-1]

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters (base training stage)."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def _unfreeze_last_blocks(self, n_blocks: int = 3) -> None:
        """
        Unfreeze the last n_blocks MBConv blocks and the classifier.

        EfficientNet-B0 features layout:
            [0] ConvBNSiLU (stem)
            [1..8] MBConv blocks

        AGENT_TASK: implement progressive unfreezing schedule
        """
        feature_layers = list(self.backbone.features.children())
        n_total = len(feature_layers)

        for layer in feature_layers[n_total - n_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 1280-d feature vector before the classifier head.
        Used by the Evaluation Agent for UMAP embedding.
        """
        features = self.backbone.features(x)
        pooled   = self.backbone.avgpool(features)
        return torch.flatten(pooled, 1)   # (B, 1280)

    def count_trainable_params(self) -> Tuple[int, int]:
        """Return (trainable, total) parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return trainable, total


# ─────────────────────────────────────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────────────────────────────────────

def build_model(device: torch.device = DEVICE) -> VIPERClassifier:
    """
    Build and return a VIPERClassifier ready for training.

    Args:
        device: Target device (cuda / cpu).

    Returns:
        Initialized VIPERClassifier on the specified device.
    """
    model = VIPERClassifier()
    model = model.to(device)
    trainable, total = model.count_trainable_params()
    print(f"[DLAgent] VIPERClassifier built:")
    print(f"          Trainable params : {trainable:,} / {total:,}")
    print(f"          Grad-CAM target  : {type(model.gradcam_target_layer).__name__}")
    return model


def load_checkpoint(
    checkpoint_path: Path = BEST_MODEL_PATH,
    device: torch.device = DEVICE,
) -> Optional[VIPERClassifier]:
    """
    Load a saved model checkpoint.

    Args:
        checkpoint_path: Path to .pth file.
        device:          Target device.

    Returns:
        Loaded VIPERClassifier or None if checkpoint not found.
    """
    if not checkpoint_path.exists():
        print(f"[DLAgent] No checkpoint found at {checkpoint_path}")
        return None

    model = build_model(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    epoch = state.get("epoch", "?")
    f1    = state.get("val_f1", "?")
    print(f"[DLAgent] ✓ Loaded checkpoint (epoch={epoch}, val_f1={f1:.4f})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model inspection helper
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Model Agent — VIPER Forensic Engine ===")
    m = build_model()
    print("\nTrainable layers:")
    for name, param in m.named_parameters():
        if param.requires_grad:
            print(f"  ✓ {name}")
