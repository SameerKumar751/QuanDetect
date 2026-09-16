# QuanDetect
### Hybrid Quantum-Classical Machine Learning Platform for Early Disease Detection
*Built for Smart India Hackathon — Problem Statement SIH26139*

QuanDetect fuses classical machine learning (Random Forest, XGBoost, SVM) with a
genuine **Variational Quantum Classifier (VQC)**, built with **PennyLane + PyTorch**
and simulated on PennyLane's `default.qubit` device, to detect disease risk early
and transparently. Classical and quantum models are trained on identical data
splits and benchmarked side-by-side (accuracy, sensitivity, specificity,
precision, F1-score), with SHAP-based explainability and Low/Medium/High risk
stratification for clinical decision support.

A **Breast Cancer Wisconsin (Diagnostic)** sample dataset ships out of the box —
but the entire pipeline (upload → preprocessing → feature selection → training →
prediction) is dataset-agnostic. Drop in any tabular biomedical CSV (real
hospital records, genomics panels, etc.) with a binary/categorical outcome
column and it works the same way.

---

## 1. Project Structure

```
QuanDetect/
├── backend/                     # FastAPI application
│   ├── main.py                  # App entrypoint (mounts /api and /api/imaging)
│   ├── requirements.txt
│   ├── data/
│   │   └── sample_breast_cancer.csv   # Bundled demo dataset
│   ├── sessions/                # Per-upload session cache (auto-created)
│   ├── app/                     # Tabular Pipeline (isolated)
│   │   ├── config.py            # Central configuration
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── preprocessing.py     # Module 1: cleaning, imputation, scaling,
│   │   │                        #   feature selection, PCA for quantum encoding
│   │   ├── models/
│   │   │   ├── classical.py     # Random Forest / XGBoost / SVM baselines
│   │   │   └── quantum.py       # Module 2 & 3: Hybrid VQC (PennyLane + PyTorch)
│   │   ├── utils/
│   │   │   ├── data_utils.py    # Session store
│   │   │   └── risk.py          # Module 4: risk stratification & decision support
│   │   └── api/
│   │       └── routes.py        # All REST endpoints (/api/*)
│   └── imaging/                 # Dedicated Medical Imaging Pipeline (CT Scans)
│       ├── config.py            # CT windowing, resolutions, VQC hyperparams
│       ├── schemas.py           # Pydantic models for imaging API
│       ├── preprocessing.py     # DICOM/PNG ingestion, HU windowing, lung segmentation & ROI
│       ├── feature_extraction.py# Pretrained ResNet-18 visual embeddings & PCA reduction
│       ├── models/
│       │   ├── classical_cnn.py # 2D ResNet-18 lung lesion classifier
│       │   ├── quantum_imaging.py # PennyLane VQC on extracted CT features
│       │   └── hybrid_imaging.py# Hybrid Quantum-CNN (HQ-CNN) fusion
│       ├── explainability/
│       │   └── gradcam.py       # Grad-CAM attention heatmaps over CT lung parenchyma
│       ├── evaluation.py        # Multi-model benchmarking (Acc, Sens, Spec, F1, ROC-AUC)
│       ├── data/
│       │   └── sample_ct_scans/ # Curated Normal, Benign, and Malignant CT slices
│       └── api/
│           └── routes.py        # Dedicated imaging endpoints (/api/imaging/*)
│
├── frontend/                    # React + Vite application
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js       # "Soft Clinical + Quantum" theme
│   ├── index.html
│   └── src/
│       ├── App.jsx              # Routes + layout
│       ├── main.jsx
│       ├── index.css
│       ├── lib/
│       │   ├── api.js           # Axios client / API endpoint map
│       │   ├── utils.js         # cn(), formatters, risk color helpers
│       │   └── AppStateContext.jsx  # Shared session state across pages
│       ├── components/
│       │   ├── Sidebar.jsx      # Modern nav with GSAP animation
│       │   ├── StatCard.jsx
│       │   └── ui/              # shadcn-style Button, Card, Badge, Input, etc.
│       └── pages/
│           ├── Landing.jsx      # 1. Overview
│           ├── Upload.jsx       # 2. Dataset Upload
│           ├── Training.jsx     # 3. Model Training (classical + quantum)
│           ├── Prediction.jsx   # 4. Prediction
│           ├── Comparison.jsx   # 5. Comparison Dashboard
│           └── Results.jsx      # 6. Risk Stratification + SHAP Explainability
│
└── README.md                    # You are here
```

---

## 2. Tech Stack

**Backend:** FastAPI · PennyLane · PyTorch · scikit-learn · XGBoost · Pandas · NumPy · SHAP
**Frontend:** React + Vite · Tailwind CSS · shadcn/ui-style components · GSAP · Recharts · Framer Motion · Lucide React

---

## 3. Prerequisites

- Python 3.10–3.11 (PennyLane/PyTorch compatibility)
- Node.js 18+ and npm
- ~2 GB free disk space for Python ML dependencies (PyTorch, XGBoost)

---

## 4. Backend Setup

```bash
cd backend

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API (auto-reload for development)
uvicorn main:app --reload --port 8000
```

The API will be live at **http://localhost:8000**, with interactive Swagger docs
at **http://localhost:8000/docs**.

### Key endpoints

| Method | Endpoint                     | Purpose                                            |
|--------|-------------------------------|-----------------------------------------------------|
| POST   | `/api/upload`                 | Upload a CSV (or omit file to load the sample set) |
| POST   | `/api/preprocess`              | Clean, impute, scale, select features               |
| POST   | `/api/train/classical`         | Train Random Forest / XGBoost / SVM                 |
| POST   | `/api/train/quantum`           | Train the Hybrid VQC on `default.qubit`             |
| POST   | `/api/predict`                 | Predict disease probability + risk tier for a sample |
| GET    | `/api/compare/{session_id}`    | Benchmark all trained models                         |
| POST   | `/api/explain`                 | SHAP explanation for a classical model's prediction  |
| GET    | `/api/sample-dataset-info`     | Metadata about the bundled demo dataset               |

### Medical Imaging (CT) Endpoints

| Method | Endpoint                     | Purpose                                            |
|--------|-------------------------------|-----------------------------------------------------|
| GET    | `/api/imaging/samples`        | List curated clinical CT scan cases (Normal/Benign/Malignant) |
| POST   | `/api/imaging/upload`         | Ingest CT scan (DICOM/PNG/JPG) or select sample ID  |
| POST   | `/api/imaging/predict`        | Multi-model inference (ResNet-18 CNN, VQC, Hybrid)  |
| POST   | `/api/imaging/explain/gradcam`| Compute localized Grad-CAM attention heatmap overlay|
| GET    | `/api/imaging/compare`        | Benchmark imaging models across clinical metrics    |

---

## 5. Frontend Setup

```bash
cd frontend

npm install

# Optional: copy .env.example -> .env and set VITE_API_BASE_URL if the
# backend is not running on http://localhost:8000
cp .env.example .env

npm run dev
```

The app will be live at **http://localhost:5173**. In development, Vite proxies
all `/api/*` requests to `http://localhost:8000` (see `vite.config.js`), so no
extra configuration is needed if both servers run locally on their default ports.

To build for production:

```bash
npm run build
npm run preview
```

---

## 6. Using the Platform

1. **Overview** — Landing page explaining the 4-stage pipeline.
2. **Dataset Upload** — Drag & drop a CSV, or click "Use Demo Dataset" to load
   the bundled Breast Cancer Wisconsin data instantly. Pick the target/label
   column and how many top features to keep (ANOVA F-test), then run
   preprocessing.
3. **Model Training** — Train the classical baselines (Random Forest, XGBoost,
   SVM) and the Hybrid VQC (configurable epochs & entangling layers) on the
   PennyLane `default.qubit` simulator. Live loss curve and circuit diagram
   are shown for the quantum model.
4. **Prediction** — Enter feature values (or prefill from a sample row),
   choose a trained model (including the quantum VQC), and get a disease
   probability, risk tier, and the top contributing features.
5. **Comparison Dashboard** — Bar chart and radar chart benchmarking every
   trained model across Accuracy, Sensitivity, Specificity, Precision, and
   F1-score, plus a "best model" badge.
6. **Risk Stratification & Explainability** — Legend for the Low / Medium /
   High risk tiers, pipeline completion tracker, and a SHAP-based
   bar chart explaining any test-set prediction from a classical model.

---

## 7. Swapping in a Different Dataset

The pipeline never hard-codes column names. To use your own biomedical CSV:

1. Go to **Dataset Upload** and upload the file instead of using the demo set.
2. The backend auto-detects a likely target column (`target`, `label`,
   `diagnosis`, `class`, `outcome`, or the last column if it looks
   categorical) — or you can pick any column manually from the dropdown.
3. Non-numeric / ID-like columns are automatically dropped during
   preprocessing; missing values are imputed; features are scaled and reduced
   via ANOVA F-test + PCA (for the quantum circuit) automatically.
4. Proceed to **Model Training** as usual — no code changes required.

---

## 8. Notes on the Quantum Model

- Uses PennyLane's `default.qubit` simulator, as required.
- Classical PCA-reduced features (angle-encoded into `[0, π]`) are embedded via
  `qml.AngleEmbedding`, followed by `qml.StronglyEntanglingLayers`
  (the variational ansatz), with Pauli-Z expectation values read out per qubit.
- A small classical linear + sigmoid head converts the quantum expectation
  values into a calibrated disease probability — this classical-quantum
  coupling, combined with the classical PCA pre-processing stage, is what
  makes the architecture "hybrid."
- Qubit count defaults to 4 (configurable in `backend/app/config.py` via
  `N_QUBITS`) to keep the simulator responsive; increase for more expressive
  circuits at the cost of training speed.

---

## 9. Error Handling & Robustness

- Upload endpoint validates CSV parseability and non-empty datasets.
- Preprocessing validates the requested target column exists.
- Training/prediction endpoints check that prerequisite steps
  (preprocessing → training) have been completed for the active session and
  return clear `4xx` errors with descriptive messages if not.
- The frontend surfaces all backend errors inline with contextual guidance
  (e.g. "Train the classical model first via Model Training").

---

## 10. Disclaimer

QuanDetect is a research/education platform built for a hackathon
demonstration. It is **not** a certified medical device and should not be
used for real clinical diagnosis without proper regulatory validation.
