"""
Pydantic schemas for the QuanDetect Medical Imaging API.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SampleScanItem(BaseModel):
    id: str
    name: str
    category: str  # Normal | Benign | Malignant
    description: str
    filename: str


class ImagingUploadResponse(BaseModel):
    session_id: str
    filename: str
    category_hint: Optional[str] = None
    original_base64: str
    segmented_roi_base64: str
    lung_mask_base64: str
    image_shape: List[int]
    min_intensity: float
    max_intensity: float
    mean_intensity: float


class SingleModelPrediction(BaseModel):
    model_name: str
    model_type: str  # cnn | quantum | hybrid
    predicted_class: int
    predicted_label: str
    class_probabilities: Dict[str, float]
    malignancy_risk_score: float
    risk_tier: str  # Low | Moderate | High
    confidence: float
    predicted_stage: str


class ImagingPredictRequest(BaseModel):
    session_id: str
    model_type: str = Field(default="all", description="cnn | quantum | hybrid | all")


class ImagingPredictResponse(BaseModel):
    session_id: str
    predictions: List[SingleModelPrediction]
    gradcam_base64: Optional[str] = None


class GradCamRequest(BaseModel):
    session_id: str
    target_class: Optional[int] = None


class GradCamResponse(BaseModel):
    session_id: str
    model_name: str
    target_class_label: str
    gradcam_overlay_base64: str
    attention_summary: str


class ModelEvaluationMetric(BaseModel):
    model_name: str
    model_type: str
    accuracy: float
    precision: float
    sensitivity: float  # Recall
    specificity: float
    f1_score: float
    roc_auc: Optional[float] = None
    confusion_matrix: List[List[int]]
    inference_time_ms: float


class ImagingBenchmarkResponse(BaseModel):
    models: List[ModelEvaluationMetric]
    best_model: str
    sample_count: int
    dataset_name: str
