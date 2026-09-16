"""
Pydantic models describing the shape of every API request/response.
Kept separate from routes so the contract is easy to scan and reuse.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class UploadResponse(BaseModel):
    session_id: str
    n_rows: int
    n_columns: int
    columns: List[str]
    detected_target_column: Optional[str] = None
    preview: List[Dict[str, Any]]
    missing_value_summary: Dict[str, int]


class PreprocessRequest(BaseModel):
    session_id: str
    target_column: str
    drop_columns: Optional[List[str]] = []
    missing_strategy: Optional[str] = "median"  # median | mean | most_frequent
    scale: Optional[bool] = True
    feature_selection_k: Optional[int] = None  # top-k features via ANOVA F-test


class PreprocessResponse(BaseModel):
    session_id: str
    n_train: int
    n_test: int
    n_features_original: int
    n_features_selected: int
    selected_features: List[str]
    class_balance: Dict[str, int]


class TrainClassicalRequest(BaseModel):
    session_id: str
    models: Optional[List[str]] = ["random_forest", "xgboost", "svm"]


class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    sensitivity: float  # recall / TPR
    specificity: float
    precision: float
    f1_score: float
    roc_auc: Optional[float] = None
    training_time_seconds: float


class TrainClassicalResponse(BaseModel):
    session_id: str
    results: List[ModelMetrics]


class TrainQuantumRequest(BaseModel):
    session_id: str
    n_qubits: Optional[int] = None
    n_layers: Optional[int] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None


class TrainQuantumResponse(BaseModel):
    session_id: str
    metrics: ModelMetrics
    loss_history: List[float]
    n_qubits_used: int
    n_layers_used: int
    circuit_diagram: str


class PredictRequest(BaseModel):
    session_id: str
    model_name: str  # random_forest | xgboost | svm | vqc
    features: Dict[str, float]


class PredictResponse(BaseModel):
    model_name: str
    disease_probability: float
    predicted_class: int
    risk_level: str  # Low | Medium | High
    confidence: float
    top_contributing_features: List[Dict[str, Any]]


class CompareResponse(BaseModel):
    session_id: str
    classical_results: List[ModelMetrics]
    quantum_result: Optional[ModelMetrics] = None
    best_model: Optional[str] = None


class ExplainRequest(BaseModel):
    session_id: str
    model_name: str
    sample_index: Optional[int] = 0


class ExplainResponse(BaseModel):
    model_name: str
    feature_names: List[str]
    shap_values: List[float]
    base_value: float
    prediction: float
