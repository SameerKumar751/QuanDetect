"""
REST API endpoints for QuanDetect.

Flow: /upload -> /preprocess -> /train/classical & /train/quantum ->
/predict -> /compare -> /explain
"""
import io
from typing import Optional

import numpy as np
import pandas as pd
import shap
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import SAMPLE_DATASET_PATH
from app.preprocessing import (
    PreprocessingPipeline,
    detect_target_column,
    missing_value_summary,
    clean_dataframe,
    split_data,
)
from app.models.classical import train_all_classical, MODEL_REGISTRY
from app.models.quantum import train_vqc, predict_vqc, circuit_text_diagram
from app.utils.data_utils import new_session, save_session, require_session
from app.utils.risk import stratify_risk, confidence_score, top_contributing_features
from app.schemas import (
    UploadResponse,
    PreprocessRequest,
    PreprocessResponse,
    TrainClassicalRequest,
    TrainClassicalResponse,
    TrainQuantumRequest,
    TrainQuantumResponse,
    PredictRequest,
    PredictResponse,
    CompareResponse,
    ExplainRequest,
    ExplainResponse,
    ModelMetrics,
)

router = APIRouter()


# --------------------------------------------------------------------------
# Dataset upload
# --------------------------------------------------------------------------
@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: Optional[UploadFile] = File(None)):
    """Upload any tabular biomedical CSV. If no file is supplied, the
    bundled Breast Cancer Wisconsin sample dataset is loaded instead,
    so the platform is always demo-able out of the box."""
    if file is not None:
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    else:
        df = pd.read_csv(SAMPLE_DATASET_PATH)

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")

    df = clean_dataframe(df)
    target_col = detect_target_column(df)

    session_id = new_session()
    save_session(session_id, {"raw_df": df, "target_column": target_col})

    preview = df.head(5).replace({np.nan: None}).to_dict(orient="records")

    return UploadResponse(
        session_id=session_id,
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        columns=df.columns.tolist(),
        detected_target_column=target_col,
        preview=preview,
        missing_value_summary=missing_value_summary(df),
    )


# --------------------------------------------------------------------------
# Preprocessing & feature engineering
# --------------------------------------------------------------------------
@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_dataset(req: PreprocessRequest):
    session = require_session(req.session_id)
    df = session.get("raw_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found for this session.")
    if req.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{req.target_column}' not in dataset.")

    pipeline = PreprocessingPipeline(
        target_column=req.target_column,
        drop_columns=req.drop_columns,
        missing_strategy=req.missing_strategy or "median",
        scale=req.scale if req.scale is not None else True,
        feature_selection_k=req.feature_selection_k,
    )
    X, X_quantum, y = pipeline.fit_transform(df)
    X_train, X_test, Xq_train, Xq_test, y_train, y_test, idx_train, idx_test = split_data(X, y, X_quantum)

    classes, counts = np.unique(y, return_counts=True)
    class_balance = {str(int(c)): int(cnt) for c, cnt in zip(classes, counts)}

    save_session(
        req.session_id,
        {
            "pipeline": pipeline,
            "X_train": X_train,
            "X_test": X_test,
            "Xq_train": Xq_train,
            "Xq_test": Xq_test,
            "y_train": y_train,
            "y_test": y_test,
            "idx_test": idx_test,
            "target_column": req.target_column,
        },
    )

    return PreprocessResponse(
        session_id=req.session_id,
        n_train=len(y_train),
        n_test=len(y_test),
        n_features_original=len(pipeline.original_feature_names),
        n_features_selected=len(pipeline.selected_feature_names),
        selected_features=pipeline.selected_feature_names,
        class_balance=class_balance,
    )


# --------------------------------------------------------------------------
# Classical training
# --------------------------------------------------------------------------
@router.post("/train/classical", response_model=TrainClassicalResponse)
async def train_classical(req: TrainClassicalRequest):
    session = require_session(req.session_id)
    for key in ["X_train", "y_train", "X_test", "y_test"]:
        if key not in session:
            raise HTTPException(status_code=400, detail="Run /preprocess before training.")

    model_keys = [m for m in req.models if m in MODEL_REGISTRY]
    if not model_keys:
        raise HTTPException(status_code=400, detail=f"No valid models requested. Choose from {list(MODEL_REGISTRY.keys())}")

    trained_models, results = train_all_classical(
        model_keys, session["X_train"], session["y_train"], session["X_test"], session["y_test"]
    )
    save_session(req.session_id, {"trained_classical_models": trained_models, "classical_results": results})

    return TrainClassicalResponse(session_id=req.session_id, results=[ModelMetrics(**r) for r in results])


# --------------------------------------------------------------------------
# Quantum (VQC) training
# --------------------------------------------------------------------------
@router.post("/train/quantum", response_model=TrainQuantumResponse)
async def train_quantum(req: TrainQuantumRequest):
    session = require_session(req.session_id)
    for key in ["Xq_train", "y_train", "Xq_test", "y_test"]:
        if key not in session:
            raise HTTPException(status_code=400, detail="Run /preprocess before training.")

    model, metrics, loss_history, scaling_params = train_vqc(
        session["Xq_train"],
        session["y_train"],
        session["Xq_test"],
        session["y_test"],
        n_qubits=req.n_qubits,
        n_layers=req.n_layers,
        epochs=req.epochs,
        lr=req.learning_rate,
    )
    n_qubits_used = session["Xq_train"].shape[1] if req.n_qubits is None else req.n_qubits
    n_layers_used = req.n_layers or 3

    save_session(
        req.session_id,
        {
            "vqc_model": model,
            "vqc_metrics": metrics,
            "vqc_scaling_params": scaling_params,
        },
    )

    return TrainQuantumResponse(
        session_id=req.session_id,
        metrics=ModelMetrics(**metrics),
        loss_history=[round(float(l), 5) for l in loss_history],
        n_qubits_used=n_qubits_used,
        n_layers_used=n_layers_used,
        circuit_diagram=circuit_text_diagram(n_qubits_used, n_layers_used),
    )


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    session = require_session(req.session_id)
    pipeline: PreprocessingPipeline = session.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Run /preprocess before predicting.")

    X_selected, X_quantum = pipeline.transform_single(req.features)

    if req.model_name == "vqc":
        model = session.get("vqc_model")
        if model is None:
            raise HTTPException(status_code=400, detail="Train the quantum model first via /train/quantum.")
        prob = float(predict_vqc(model, X_quantum, session["vqc_scaling_params"])[0])
        importances = {f: 1.0 for f in pipeline.selected_feature_names}  # circuit-level importances are opaque; approximate uniformly
    else:
        trained = session.get("trained_classical_models", {})
        model = trained.get(req.model_name)
        if model is None:
            raise HTTPException(status_code=400, detail=f"Train '{req.model_name}' first via /train/classical.")
        prob = float(model.predict_proba(X_selected)[0][1])
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(pipeline.selected_feature_names, model.feature_importances_))
        elif hasattr(model, "coef_"):
            importances = dict(zip(pipeline.selected_feature_names, np.abs(model.coef_[0])))
        else:
            importances = {f: 1.0 for f in pipeline.selected_feature_names}

    predicted_class = int(prob >= 0.5)
    risk_level = stratify_risk(prob)
    conf = confidence_score(prob)

    feature_subset = {f: req.features.get(f, 0.0) for f in pipeline.selected_feature_names}
    top_features = top_contributing_features(feature_subset, importances)

    return PredictResponse(
        model_name=req.model_name,
        disease_probability=round(prob, 4),
        predicted_class=predicted_class,
        risk_level=risk_level,
        confidence=conf,
        top_contributing_features=top_features,
    )


# --------------------------------------------------------------------------
# Comparison / benchmarking
# --------------------------------------------------------------------------
@router.get("/compare/{session_id}", response_model=CompareResponse)
async def compare(session_id: str):
    session = require_session(session_id)
    classical_results = session.get("classical_results", [])
    vqc_metrics = session.get("vqc_metrics")

    all_results = list(classical_results)
    if vqc_metrics:
        all_results.append(vqc_metrics)

    best_model = None
    if all_results:
        best_model = max(all_results, key=lambda r: r["f1_score"])["model_name"]

    return CompareResponse(
        session_id=session_id,
        classical_results=[ModelMetrics(**r) for r in classical_results],
        quantum_result=ModelMetrics(**vqc_metrics) if vqc_metrics else None,
        best_model=best_model,
    )


# --------------------------------------------------------------------------
# Explainability (SHAP) - classical models only (tree/kernel explainers)
# --------------------------------------------------------------------------
@router.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    session = require_session(req.session_id)
    pipeline: PreprocessingPipeline = session.get("pipeline")
    trained = session.get("trained_classical_models", {})
    model = trained.get(req.model_name)
    if model is None or pipeline is None:
        raise HTTPException(status_code=400, detail="Train the classical model first via /train/classical.")

    X_test = session["X_test"]
    idx = min(req.sample_index or 0, len(X_test) - 1)
    sample = X_test[idx : idx + 1]

    try:
        if hasattr(model, "predict_proba") and model.__class__.__name__ in ("RandomForestClassifier", "XGBClassifier"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
            if isinstance(shap_values, list):
                sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3:  # (n_samples, n_features, n_classes)
                    sv = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
                elif shap_values.ndim == 2 and shap_values.shape[1] == 2:
                    sv = shap_values[:, 1]
                else:
                    sv = shap_values[0]
            else:
                sv = shap_values

            if isinstance(explainer.expected_value, (list, np.ndarray)):
                base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
            else:
                base_value = explainer.expected_value
        else:
            background = shap.kmeans(session["X_train"], min(20, session["X_train"].shape[0]))
            explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], background)
            sv = explainer.shap_values(sample, nsamples=100)[0]
            base_value = explainer.expected_value
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {e}")

    prediction = float(model.predict_proba(sample)[0][1])

    return ExplainResponse(
        model_name=req.model_name,
        feature_names=pipeline.selected_feature_names,
        shap_values=[round(float(v), 5) for v in np.ravel(sv)[:len(pipeline.selected_feature_names)]],
        base_value=round(float(base_value), 5),
        prediction=round(prediction, 5),
    )


# --------------------------------------------------------------------------
# Sample dataset metadata (used by the frontend upload page to offer a
# "use demo dataset" shortcut, and to show which features exist so a
# real hospital CSV can be mapped onto the same schema later).
# --------------------------------------------------------------------------
@router.get("/sample-dataset-info")
async def sample_dataset_info():
    df = pd.read_csv(SAMPLE_DATASET_PATH)
    df = clean_dataframe(df)
    target_col = detect_target_column(df) or "LUNG_CANCER"
    return {
        "name": "Survey Lung Cancer Dataset",
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "columns": df.columns.tolist(),
        "target_column": target_col,
        "description": (
            "Clinical survey and patient risk factor dataset (age, gender, smoking, "
            "respiratory symptoms, and chronic diseases) for early lung cancer detection. "
            "Bundled for development/demo; replace with any tabular biomedical CSV via the Upload page."
        ),
    }
