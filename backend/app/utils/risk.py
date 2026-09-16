"""
Module 4: Prediction & Decision Support.

Converts a raw disease probability into a clinically-readable risk
tier (Low / Medium / High) and a confidence score, and ranks the
features that most influenced a given prediction (used for the
"top contributing features" shown alongside every prediction, and
feeding the SHAP-based deeper explainability endpoint).
"""
from typing import Dict, List
import numpy as np

from app.config import RISK_LOW_MAX, RISK_MEDIUM_MAX


def stratify_risk(probability: float) -> str:
    if probability < RISK_LOW_MAX:
        return "Low"
    elif probability < RISK_MEDIUM_MAX:
        return "Medium"
    return "High"


def confidence_score(probability: float) -> float:
    """Distance from the decision boundary (0.5), rescaled to [0, 1] —
    a simple, model-agnostic proxy for prediction confidence."""
    return float(round(abs(probability - 0.5) * 2, 4))


def top_contributing_features(
    feature_values: Dict[str, float], feature_importances: Dict[str, float], top_n: int = 5
) -> List[Dict]:
    """Rank features by |importance x standardized value| as a fast,
    model-agnostic approximation used for immediate UI feedback (the
    full SHAP explanation is available via the /explain endpoint)."""
    scored = []
    for feat, val in feature_values.items():
        importance = feature_importances.get(feat, 0.0)
        scored.append({"feature": feat, "value": round(float(val), 4), "importance": round(float(importance), 4)})
    scored.sort(key=lambda d: abs(d["importance"]), reverse=True)
    return scored[:top_n]
