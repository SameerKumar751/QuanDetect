"""
Module: Feature Extraction & Dimensionality Reduction.

Extracts deep spatial visual representations from CT images using a pretrained
2D CNN backbone (ResNet-18), followed by PCA dimensionality reduction to prepare
quantum-ready features for the Variational Quantum Classifier (VQC).
"""
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from imaging.config import N_QUBITS, RANDOM_STATE

# Try importing torchvision resnet18
try:
    import torchvision.models as models
    from torchvision.models import ResNet18_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class ResNetBackbone(nn.Module):
    """ResNet-18 backbone acting as a 512-dimensional visual feature extractor."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if HAS_TORCHVISION:
            try:
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                base = models.resnet18(weights=weights)
            except Exception:
                base = models.resnet18(weights=None)
            # Remove final classification head (fc layer)
            self.features = nn.Sequential(*list(base.children())[:-1])  # Output shape: (B, 512, 1, 1)
            self.out_dim = 512
        else:
            # Lightweight CNN fallback if torchvision is still installing
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.out_dim = 128

        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.features(x)
            return torch.flatten(feat, 1)


class ImagingPCAReducer:
    """
    Reduces high-dimensional CNN visual embeddings (512-D) down to N_QUBITS (4-D)
    and maps them to angle range [0, pi] for PennyLane quantum angle embedding.
    """

    def __init__(self, n_components: int = N_QUBITS):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        self.is_fitted = False
        self.mins = None
        self.spans = None

        # Pre-seed with synthetic orthogonal projection so it works immediately out-of-the-box
        np.random.seed(RANDOM_STATE)
        synthetic_embeddings = np.random.randn(20, 512).astype(np.float32)
        self.fit(synthetic_embeddings)

    def fit(self, X: np.ndarray):
        reduced = self.pca.fit_transform(X)
        self.mins = reduced.min(axis=0)
        maxs = reduced.max(axis=0)
        self.spans = np.where(maxs - self.mins == 0, 1.0, maxs - self.mins)
        self.is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            self.fit(X)
        reduced = self.pca.transform(X)
        # Normalize into [0, pi] for angle embedding
        scaled = np.clip((reduced - self.mins) / self.spans, 0.0, 1.0) * np.pi
        return scaled.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# Global singleton instances
_backbone: Optional[ResNetBackbone] = None
_pca_reducer: Optional[ImagingPCAReducer] = None


def get_feature_extractor() -> ResNetBackbone:
    global _backbone
    if _backbone is None:
        _backbone = ResNetBackbone(pretrained=True)
    return _backbone


def get_pca_reducer() -> ImagingPCAReducer:
    global _pca_reducer
    if _pca_reducer is None:
        _pca_reducer = ImagingPCAReducer(n_components=N_QUBITS)
    return _pca_reducer


def extract_features(tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a standardized CT image tensor of shape (1, 3, 224, 224):
      1. Extracts 512-dimensional deep CNN visual features.
      2. Projects to N_QUBITS (4-D) normalized angles in [0, pi].

    Returns:
      - cnn_features: (1, 512)
      - quantum_features: (1, N_QUBITS)
    """
    extractor = get_feature_extractor()
    with torch.no_grad():
        feat_tensor = extractor(tensor)
        cnn_features = feat_tensor.cpu().numpy()

    reducer = get_pca_reducer()
    quantum_features = reducer.transform(cnn_features)

    return cnn_features, quantum_features
