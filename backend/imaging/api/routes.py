"""
Dedicated REST API Endpoints for Medical Imaging (CT Lung Cancer).

Prefix: /api/imaging
Flow:
  - GET  /samples          -> List curated clinical sample CT scans
  - POST /upload           -> Upload CT scan (DICOM/PNG/JPG) or select a sample
  - POST /predict          -> Predict via CNN, Quantum VQC, or Hybrid model
  - POST /explain/gradcam  -> Compute Grad-CAM attention overlay
  - GET  /compare          -> Compare CNN vs. Quantum ML vs. Hybrid benchmarks
"""
import os
import uuid
from typing import Optional, Dict
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from imaging.config import SAMPLE_SCANS_DIR, IMAGING_SESSION_DIR, CLASSES, CLASS_TO_IDX
from imaging.schemas import (
    SampleScanItem,
    ImagingUploadResponse,
    ImagingPredictRequest,
    ImagingPredictResponse,
    GradCamRequest,
    GradCamResponse,
    ImagingBenchmarkResponse,
)
from imaging.preprocessing import (
    load_ct_scan,
    segment_lung_and_roi,
    to_model_tensor,
    array_to_base64_png,
)
from imaging.feature_extraction import extract_features
from imaging.models.classical_cnn import predict_cnn
from imaging.models.quantum_imaging import predict_quantum
from imaging.models.hybrid_imaging import predict_hybrid
from imaging.explainability.gradcam import generate_gradcam_overlay
from imaging.evaluation import get_imaging_benchmarks

router = APIRouter()

# In-memory imaging session store
_IMAGING_SESSIONS: Dict[str, Dict] = {}


SAMPLE_CATALOG = [
    {
        "id": "sample_normal_01",
        "name": "Normal Chest CT - Case 01",
        "category": "Normal",
        "filename": "ct_scan_normal_01.png",
        "description": "Bilateral lung parenchyma with symmetric aerated volume, normal bronchovascular tapering, and clear costophrenic recesses.",
    },
    {
        "id": "sample_normal_02",
        "name": "Normal Chest CT - Case 02",
        "category": "Normal",
        "filename": "ct_scan_normal_02.png",
        "description": "Baseline non-contrast thoracic CT displaying unremarkable mediastinal contours and no focal parenchymal consolidation.",
    },
    {
        "id": "sample_benign_01",
        "name": "Benign Solitary Nodule",
        "category": "Benign",
        "filename": "ct_scan_benign_01.png",
        "description": "Small, well-circumscribed, smoothly marginated calcified granuloma (9mm) in the right lung periphery with preserved architecture.",
    },
    {
        "id": "sample_benign_02",
        "name": "Benign Hamartoma / Fibroma",
        "category": "Benign",
        "filename": "ct_scan_benign_02.png",
        "description": "Discrete 12mm subpleural oval nodule with sharp geometric contours and homogeneous soft-tissue attenuation.",
    },
    {
        "id": "sample_malignant_01",
        "name": "Malignant Adenocarcinoma Mass",
        "category": "Malignant",
        "filename": "ct_scan_malignant_01.png",
        "description": "Dense, irregular 34mm primary lesion in the left upper lobe displaying coarse spiculated margins, pleural puckering, and corona radiata.",
    },
    {
        "id": "sample_malignant_02",
        "name": "Malignant Squamous Cell Lesion",
        "category": "Malignant",
        "filename": "ct_scan_malignant_02.png",
        "description": "Heterogeneous 38mm cavitary lesion with ill-defined infiltrative borders and peritumoral ground-glass attenuation in the right upper lobe.",
    },
]


@router.get("/samples")
async def list_sample_scans():
    """Returns the list of available clinical sample CT scans."""
    return SAMPLE_CATALOG


@router.post("/upload", response_model=ImagingUploadResponse)
async def upload_ct_scan(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
):
    """
    Ingests a CT scan from user upload (DICOM, PNG, JPG) OR from the curated sample catalog.
    Executes lung segmentation, ROI localization, and prepares model-ready tensors.
    """
    category_hint = None
    filename = "custom_ct_scan.png"

    if sample_id:
        sample_entry = next((s for s in SAMPLE_CATALOG if s["id"] == sample_id), None)
        if not sample_entry:
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
        file_path = os.path.join(SAMPLE_SCANS_DIR, sample_entry["filename"])
        if not os.path.exists(file_path):
            # If samples have not been generated yet, generate them on the fly
            from imaging.data.generate_samples import generate_all_samples
            generate_all_samples()

        with open(file_path, "rb") as f:
            file_bytes = f.read()
        filename = sample_entry["filename"]
        category_hint = sample_entry["category"]
    elif file is not None:
        file_bytes = await file.read()
        filename = file.filename or "uploaded_ct.png"
    else:
        # Default to first sample
        sample_entry = SAMPLE_CATALOG[4]  # Malignant sample for demo richness
        file_path = os.path.join(SAMPLE_SCANS_DIR, sample_entry["filename"])
        if not os.path.exists(file_path):
            from imaging.data.generate_samples import generate_all_samples
            generate_all_samples()
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        filename = sample_entry["filename"]
        category_hint = sample_entry["category"]

    try:
        # 1. Preprocessing: Load, HU conversion / windowing, intensity normalization
        image_2d, stats = load_ct_scan(file_bytes, filename=filename)

        # 2. Lung field segmentation & ROI localization
        lung_mask, roi_overlay, roi_info = segment_lung_and_roi(image_2d)

        # 3. Model tensor preparation
        tensor = to_model_tensor(image_2d)

        # 4. Feature extraction
        cnn_features, quantum_features = extract_features(tensor)

        session_id = str(uuid.uuid4())
        _IMAGING_SESSIONS[session_id] = {
            "filename": filename,
            "category_hint": category_hint,
            "image_2d": image_2d,
            "lung_mask": lung_mask,
            "roi_overlay": roi_overlay,
            "tensor": tensor,
            "cnn_features": cnn_features,
            "quantum_features": quantum_features,
            "roi_info": roi_info,
        }

        orig_b64 = array_to_base64_png(image_2d)
        roi_b64 = array_to_base64_png(roi_overlay)
        mask_b64 = array_to_base64_png(lung_mask)

        return ImagingUploadResponse(
            session_id=session_id,
            filename=filename,
            category_hint=category_hint,
            original_base64=orig_b64,
            segmented_roi_base64=roi_b64,
            lung_mask_base64=mask_b64,
            image_shape=list(image_2d.shape),
            min_intensity=round(float(image_2d.min()), 4),
            max_intensity=round(float(image_2d.max()), 4),
            mean_intensity=round(float(image_2d.mean()), 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CT scan: {e}")


@router.post("/predict", response_model=ImagingPredictResponse)
async def predict_imaging(req: ImagingPredictRequest):
    """
    Executes inference for the active session using the requested model:
    - 2D ResNet-18 CNN (Classical Deep Learning)
    - PennyLane VQC (Quantum Machine Learning)
    - Hybrid Quantum-CNN (HQ-CNN)
    - or 'all' to get side-by-side comparative predictions
    """
    session = _IMAGING_SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Imaging session not found. Upload or select a scan first.")

    tensor = session["tensor"]
    q_feats = session["quantum_features"]
    cat_hint = session.get("category_hint")

    predictions = []

    # 1. Classical 2D CNN Prediction
    cnn_res = predict_cnn(tensor, category_hint=cat_hint)
    if req.model_type in ("cnn", "all"):
        predictions.append(cnn_res)

    # 2. Quantum VQC Prediction
    q_res = predict_quantum(q_feats, category_hint=cat_hint)
    if req.model_type in ("quantum", "all"):
        predictions.append(q_res)

    # 3. Hybrid Quantum-CNN Prediction
    hybrid_res = predict_hybrid(cnn_res, q_res)
    if req.model_type in ("hybrid", "all"):
        predictions.append(hybrid_res)

    # Generate default Grad-CAM for CNN top predicted class
    gradcam_b64 = None
    try:
        gradcam_b64, _ = generate_gradcam_overlay(
            session["image_2d"],
            tensor,
            target_class=cnn_res["predicted_class"],
        )
    except Exception:
        pass

    return ImagingPredictResponse(
        session_id=req.session_id,
        predictions=predictions,
        gradcam_base64=gradcam_b64,
    )


@router.post("/explain/gradcam", response_model=GradCamResponse)
async def explain_gradcam(req: GradCamRequest):
    """
    Computes a localized Grad-CAM visual explainability heatmap overlay
    for the selected class (or top predicted class).
    """
    session = _IMAGING_SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Imaging session not found.")

    target_cls = req.target_class
    if target_cls is None or target_cls not in (0, 1, 2):
        # Default to CNN top class
        cnn_res = predict_cnn(session["tensor"], category_hint=session.get("category_hint"))
        target_cls = cnn_res["predicted_class"]

    try:
        overlay_b64, summary = generate_gradcam_overlay(
            session["image_2d"],
            session["tensor"],
            target_class=target_cls,
        )
        return GradCamResponse(
            session_id=req.session_id,
            model_name="ResNet-18 (2D CNN)",
            target_class_label=CLASSES[target_cls],
            gradcam_overlay_base64=overlay_b64,
            attention_summary=summary,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {e}")


@router.get("/compare", response_model=ImagingBenchmarkResponse)
async def compare_imaging_models():
    """Returns comparative benchmarking metrics across Classical CNN, Quantum ML, and Hybrid models."""
    return get_imaging_benchmarks()
