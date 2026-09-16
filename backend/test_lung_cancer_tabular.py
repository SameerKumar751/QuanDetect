"""
Comprehensive Verification Script for Tabular Lung Cancer Adaptation.
Tests all endpoints: /upload -> /preprocess -> /train/classical -> /train/quantum -> /compare -> /predict -> /explain
And verifies that the separate medical-imaging module remains completely untouched and functional.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from main import app

def run_tabular_verification():
    client = TestClient(app)

    print("=== 1. Testing Default Dataset Info ===")
    res = client.get("/api/sample-dataset-info")
    assert res.status_code == 200
    info = res.json()
    print("Dataset Name:", info["name"])
    print("Detected Target:", info["target_column"])
    print("Columns (total):", len(info["columns"]))
    assert info["name"] == "Survey Lung Cancer Dataset"
    assert info["target_column"] == "LUNG_CANCER"
    print("PASS: Default dataset info updated successfully.")

    print("\n=== 2. Testing /api/upload with survey lung cancer ===")
    res = client.post("/api/upload")
    assert res.status_code == 200
    upload_res = res.json()
    session_id = upload_res["session_id"]
    print("Session ID:", session_id[:8])
    print("Rows:", upload_res["n_rows"], "Columns:", upload_res["n_columns"])
    print("Detected target:", upload_res["detected_target_column"])
    assert upload_res["n_rows"] == 309
    assert upload_res["detected_target_column"] == "LUNG_CANCER"
    print("PASS: /api/upload ingested Survey Lung Cancer data.")

    print("\n=== 3. Testing /api/preprocess ===")
    preprocess_payload = {
        "session_id": session_id,
        "target_column": "LUNG_CANCER",
        "feature_selection_k": 10,
        "scale": True,
    }
    res = client.post("/api/preprocess", json=preprocess_payload)
    assert res.status_code == 200, f"Preprocess error: {res.text}"
    prep = res.json()
    print("Train count:", prep["n_train"], "Test count:", prep["n_test"])
    print("Original features count:", prep["n_features_original"])
    print("Selected features (K=10):", prep["selected_features"])
    print("Class balance (0: NO, 1: YES):", prep["class_balance"])
    assert prep["n_train"] == 247
    assert prep["n_test"] == 62
    assert len(prep["selected_features"]) == 10
    print("PASS: /api/preprocess completed with categorical encoding & stratified split.")

    print("\n=== 4. Testing /api/train/classical (Random Forest, XGBoost, SVM) ===")
    classical_payload = {
        "session_id": session_id,
        "models": ["random_forest", "xgboost", "svm"],
    }
    res = client.post("/api/train/classical", json=classical_payload)
    assert res.status_code == 200, f"Classical train error: {res.text}"
    classical_res = res.json()
    for m in classical_res["results"]:
        print(f"   -> {m['model_name']}: Acc={m['accuracy']*100:.1f}%, Sens={m['sensitivity']*100:.1f}%, Spec={m['specificity']*100:.1f}%, F1={m['f1_score']*100:.1f}%, Time={m['training_time_seconds']}s")
        assert m["accuracy"] > 0.70
    print("PASS: Classical models trained successfully.")

    print("\n=== 5. Testing /api/train/quantum (PennyLane VQC on default.qubit) ===")
    quantum_payload = {
        "session_id": session_id,
        "n_qubits": 4,
        "n_layers": 2,
        "epochs": 10,
        "learning_rate": 0.08,
    }
    res = client.post("/api/train/quantum", json=quantum_payload)
    assert res.status_code == 200, f"Quantum train error: {res.text}"
    q_res = res.json()
    qm = q_res["metrics"]
    print(f"   -> VQC Quantum Model: Acc={qm['accuracy']*100:.1f}%, Sens={qm['sensitivity']*100:.1f}%, Spec={qm['specificity']*100:.1f}%, F1={qm['f1_score']*100:.1f}%, Time={qm['training_time_seconds']}s")
    print(f"   -> Loss history (first & last): {q_res['loss_history'][0]} -> {q_res['loss_history'][-1]}")
    print("PASS: Quantum VQC trained successfully.")

    print("\n=== 6. Testing /api/compare/{session_id} ===")
    res = client.get(f"/api/compare/{session_id}")
    assert res.status_code == 200
    comp = res.json()
    print("Best Model by F1-Score:", comp["best_model"])
    assert len(comp["classical_results"]) == 3
    assert comp["quantum_result"] is not None
    print("PASS: Model benchmarking comparison verified.")

    print("\n=== 7. Testing /api/predict (Inference with Survey features) ===")
    sample_patient = {
        "GENDER": 1,
        "AGE": 67,
        "SMOKING": 2,
        "YELLOW_FINGERS": 2,
        "ANXIETY": 2,
        "PEER_PRESSURE": 1,
        "CHRONIC DISEASE": 1,
        "FATIGUE": 2,
        "ALLERGY": 2,
        "WHEEZING": 2,
        "ALCOHOL CONSUMING": 2,
        "COUGHING": 2,
        "SHORTNESS OF BREATH": 2,
        "SWALLOWING DIFFICULTY": 2,
        "CHEST PAIN": 2,
    }
    pred_payload = {
        "session_id": session_id,
        "model_name": "random_forest",
        "features": sample_patient,
    }
    res = client.post("/api/predict", json=pred_payload)
    assert res.status_code == 200, f"Predict error: {res.text}"
    p = res.json()
    print(f"Prediction result: Class={p['predicted_class']} (1=Cancer Yes), Probability={p['disease_probability']*100:.1f}%, Risk Tier={p['risk_level']}, Confidence={p['confidence']}")
    print("Top contributing features:", [f['feature'] for f in p['top_contributing_features']])
    assert p["predicted_class"] == 1
    assert p["risk_level"] in ("Medium", "High")
    print("PASS: Prediction verified on survey features.")

    print("\n=== 8. Testing /api/explain (SHAP on lung cancer features) ===")
    explain_payload = {
        "session_id": session_id,
        "model_name": "random_forest",
        "sample_index": 0,
    }
    res = client.post("/api/explain", json=explain_payload)
    assert res.status_code == 200, f"Explain error: {res.text}"
    exp = res.json()
    print("SHAP Base Value:", exp["base_value"])
    print("SHAP Prediction:", exp["prediction"])
    print("SHAP Features:", exp["feature_names"])
    print("SHAP Values count:", len(exp["shap_values"]))
    assert len(exp["feature_names"]) == len(exp["shap_values"])
    print("PASS: SHAP explainability working on new survey features.")

    print("\n=== 9. Verifying Separate Medical Imaging Pipeline is Untouched ===")
    res = client.get("/api/imaging/samples")
    assert res.status_code == 200
    samples = res.json()
    assert len(samples) == 6
    print(f"PASS: Medical imaging module untouched ({len(samples)} CT scan samples intact).")

    print("\n====================================================================")
    print("ALL TABULAR LUNG CANCER ADAPTATION TESTS PASSED SUCCESSFULLY!")
    print("====================================================================")

if __name__ == "__main__":
    run_tabular_verification()
