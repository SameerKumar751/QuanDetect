"""
Module: Hybrid Quantum-Classical Imaging Model (HQ-CNN).

Fuses the deep representation power of the ResNet-18 convolutional backbone
with the high-dimensional Hilbert space mapping of the Variational Quantum Classifier (VQC)
to produce a robust ensemble decision for lung lesion assessment.
"""
from typing import Dict, Optional
import numpy as np

from imaging.config import (
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    RISK_LOW_MAX,
    RISK_MODERATE_MAX,
    STAGE_STATUS_MESSAGE,
)


def predict_hybrid(cnn_pred: Dict, quantum_pred: Dict) -> Dict:
    """
    Fuses predictions from the 2D ResNet-18 CNN and the PennyLane VQC.
    Weighted fusion: 55% CNN + 45% Quantum VQC.
    """
    cnn_probs = np.array([
        cnn_pred["class_probabilities"]["Normal"],
        cnn_pred["class_probabilities"]["Benign"],
        cnn_pred["class_probabilities"]["Malignant"],
    ])

    q_probs = np.array([
        quantum_pred["class_probabilities"]["Normal"],
        quantum_pred["class_probabilities"]["Benign"],
        quantum_pred["class_probabilities"]["Malignant"],
    ])

    # Ensemble weighting
    fused_probs = 0.55 * cnn_probs + 0.45 * q_probs
    fused_probs = fused_probs / fused_probs.sum()

    pred_idx = int(np.argmax(fused_probs))
    pred_label = IDX_TO_CLASS[pred_idx]

    malignancy_risk = float(round(fused_probs[2] + 0.15 * fused_probs[1], 4))
    if malignancy_risk < RISK_LOW_MAX:
        risk_tier = "Low"
    elif malignancy_risk < RISK_MODERATE_MAX:
        risk_tier = "Moderate"
    else:
        risk_tier = "High"

    sorted_probs = np.sort(fused_probs)
    confidence = float(round(sorted_probs[-1] - sorted_probs[-2], 4))

    return {
        "model_name": "Hybrid Quantum-CNN (HQ-CNN)",
        "model_type": "hybrid",
        "predicted_class": pred_idx,
        "predicted_label": pred_label,
        "class_probabilities": {
            "Normal": round(float(fused_probs[0]), 4),
            "Benign": round(float(fused_probs[1]), 4),
            "Malignant": round(float(fused_probs[2]), 4),
        },
        "malignancy_risk_score": malignancy_risk,
        "risk_tier": risk_tier,
        "confidence": confidence,
        "predicted_stage": STAGE_STATUS_MESSAGE,
    }
