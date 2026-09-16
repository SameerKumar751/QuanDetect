"""
Module: Classical baseline models (Random Forest, XGBoost, SVM).

These provide the benchmark against which the hybrid Variational
Quantum Classifier is compared, per SIH26139's requirement for
classical-vs-quantum performance benchmarking (accuracy, sensitivity,
specificity, F1).
"""
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

from app.config import RANDOM_STATE

MODEL_REGISTRY = {
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=RANDOM_STATE
    ),
    "xgboost": lambda: XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False,
    ),
    "svm": lambda: SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}


def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """Compute accuracy / sensitivity (recall) / specificity / precision / F1 / ROC-AUC."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() \
        if len(set(y_true)) == 2 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, average="binary" if len(set(y_true)) == 2 else "macro", zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, average="binary" if len(set(y_true)) == 2 else "macro", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary" if len(set(y_true)) == 2 else "macro", zero_division=0)),
    }
    if y_prob is not None and len(set(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def train_classical_model(
    model_key: str, X_train, y_train, X_test, y_test
) -> Tuple[object, Dict[str, float], float]:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown classical model '{model_key}'")

    model = MODEL_REGISTRY[model_key]()
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["training_time_seconds"] = round(elapsed, 4)
    metrics["model_name"] = model_key
    return model, metrics, elapsed


def train_all_classical(
    model_keys: List[str], X_train, y_train, X_test, y_test
) -> Tuple[Dict[str, object], List[Dict[str, float]]]:
    trained_models = {}
    results = []
    for key in model_keys:
        model, metrics, _ = train_classical_model(key, X_train, y_train, X_test, y_test)
        trained_models[key] = model
        results.append(metrics)
    return trained_models, results
