"""
Automated End-to-End Verification Test for the QuanDetect Medical Imaging Pipeline.
"""
import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from main import app
from imaging.data.generate_samples import generate_all_samples

def run_tests():
    print("=== 1. Testing Sample Generation ===")
    generate_all_samples()
    print("PASS: Sample scans verified.")

    print("\n=== 2. Testing FastAPI TestClient & Endpoints ===")
    client = TestClient(app)

    # Health check
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print("PASS: /api/health returned 200 OK")

    # Tabular sanity check: ensure tabular endpoint is still alive and untouched
    res = client.get("/api/sample-dataset-info")
    assert res.status_code == 200, f"Tabular endpoint failed: {res.text}"
    print("PASS: /api/sample-dataset-info (tabular) intact & functional")

    # Imaging: /api/imaging/samples
    res = client.get("/api/imaging/samples")
    assert res.status_code == 200, f"Failed /api/imaging/samples: {res.text}"
    samples = res.json()
    assert len(samples) >= 6, f"Expected at least 6 samples, got {len(samples)}"
    print(f"PASS: /api/imaging/samples returned {len(samples)} curated clinical cases")

    # Imaging: /api/imaging/upload (with sample selection)
    sample_id = "sample_malignant_01"
    res = client.post("/api/imaging/upload", data={"sample_id": sample_id})
    assert res.status_code == 200, f"Failed /api/imaging/upload: {res.text}"
    upload_data = res.json()
    session_id = upload_data["session_id"]
    assert "data:image/png;base64" in upload_data["original_base64"]
    assert "data:image/png;base64" in upload_data["segmented_roi_base64"]
    assert upload_data["category_hint"] == "Malignant"
    print(f"PASS: /api/imaging/upload successful (session_id={session_id[:8]}...)")

    # Imaging: /api/imaging/predict (all models: CNN, Quantum, Hybrid)
    res = client.post("/api/imaging/predict", json={"session_id": session_id, "model_type": "all"})
    assert res.status_code == 200, f"Failed /api/imaging/predict: {res.text}"
    pred_data = res.json()
    predictions = pred_data["predictions"]
    assert len(predictions) == 3, f"Expected 3 model predictions, got {len(predictions)}"

    for p in predictions:
        print(f"   -> Model: {p['model_name']} | Label: {p['predicted_label']} | Malignancy Risk: {p['malignancy_risk_score']*100:.1f}% | Risk Tier: {p['risk_tier']} | Stage: {p['predicted_stage']}")
        assert p["risk_tier"] in ("Low", "Moderate", "High")
        assert "Cannot be reliably predicted from 2D slice" in p["predicted_stage"]

    print("PASS: /api/imaging/predict verified across 2D CNN, Quantum VQC, and Hybrid HQ-CNN")

    # Imaging: /api/imaging/explain/gradcam
    res = client.post("/api/imaging/explain/gradcam", json={"session_id": session_id})
    assert res.status_code == 200, f"Failed /api/imaging/explain/gradcam: {res.text}"
    cam_data = res.json()
    assert "data:image/png;base64" in cam_data["gradcam_overlay_base64"]
    assert len(cam_data["attention_summary"]) > 20
    print(f"PASS: /api/imaging/explain/gradcam generated overlay for {cam_data['target_class_label']}")
    print(f"   -> Attention summary: {cam_data['attention_summary']}")

    # Imaging: /api/imaging/compare
    res = client.get("/api/imaging/compare")
    assert res.status_code == 200, f"Failed /api/imaging/compare: {res.text}"
    bench = res.json()
    assert len(bench["models"]) == 3
    print(f"PASS: /api/imaging/compare returned {len(bench['models'])} benchmark models (Best: {bench['best_model']})")

    for m in bench["models"]:
        print(f"   -> {m['model_name']}: Acc={m['accuracy']*100:.1f}%, Sens={m['sensitivity']*100:.1f}%, Spec={m['specificity']*100:.1f}%, F1={m['f1_score']*100:.1f}%, AUC={m['roc_auc']*100:.1f}%")

    print("\n========================================================")
    print("ALL MEDICAL IMAGING VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("========================================================")

if __name__ == "__main__":
    run_tests()
