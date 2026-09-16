# QuanDetect

### Hybrid Quantum-Classical Machine Learning Platform for Explainable Disease Detection

QuanDetect is a research-oriented platform developed for **Smart India Hackathon — Problem Statement SIH26139**. It combines classical machine learning, quantum machine learning, deep learning, medical imaging, and explainable AI into a unified platform for early disease-risk analysis.

The platform currently provides two complementary pipelines:

- **Tabular Disease Detection** for structured biomedical datasets
- **CT Lung Diagnostics** for medical imaging analysis

---

## Overview

QuanDetect is designed to provide a common environment for developing, evaluating, comparing, and explaining classical, quantum, and hybrid machine learning approaches.

### Core Capabilities

- Biomedical tabular dataset analysis
- Lung CT image analysis
- Classical machine learning
- Variational Quantum Classification
- Hybrid quantum-classical learning
- Deep learning with ResNet-18
- Model benchmarking
- SHAP-based tabular explainability
- Grad-CAM imaging explainability
- Probability-based risk stratification
- Interactive analytical dashboard

The platform is intended for **research, education, and hackathon demonstration** rather than clinical diagnosis.

---

## System Architecture

```text
                               QuanDetect
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
             Tabular Pipeline              CT Imaging Pipeline
                    │                             │
                    ▼                             ▼
          Biomedical CSV Data              CT / PNG Imaging
                    │                             │
                    ▼                             ▼
           Data Preprocessing             Image Preprocessing
                    │                             │
                    ▼                             ▼
          Feature Selection + PCA          ROI Processing
                    │                             │
          ┌─────────┴─────────┐          ┌────────┴─────────┐
          │                   │          │                  │
          ▼                   ▼          ▼                  ▼
   Classical Models          VQC     ResNet-18             VQC
   RF / XGBoost / SVM         │        CNN                  │
          │                   │          │                  │
          │                   │          └────────┬─────────┘
          │                   │                   ▼
          │                   │          Hybrid Quantum-CNN
          │                   │                   │
          └──────────┬────────┘                   │
                     │                            │
                     ▼                            ▼
              Model Evaluation            Grad-CAM Explanation
                     │                            │
                     └─────────────┬──────────────┘
                                   ▼
                           Prediction & Risk
                                   │
                                   ▼
                         Interactive Dashboard
```

---

# 1. Tabular Disease Detection

The tabular pipeline accepts structured biomedical CSV datasets and applies a complete machine-learning workflow.

```text
Dataset Upload
      ↓
Target Detection
      ↓
Data Cleaning
      ↓
Missing Value Handling
      ↓
Feature Scaling
      ↓
Feature Selection
      ↓
PCA
      ↓
 ┌────┴─────────────┐
 │                  │
 ▼                  ▼
Classical ML       Quantum ML
 │                  │
 ├─ Random Forest   └─ VQC
 ├─ XGBoost
 └─ SVM
 │                  │
 └────────┬─────────┘
          ▼
    Model Comparison
          ↓
Prediction + Probability
          ↓
Risk Stratification
          ↓
SHAP Explainability
```

## Classical Models

### Random Forest

An ensemble of decision trees used as a robust classical classification baseline.

### XGBoost

A gradient-boosted decision-tree model used for high-performance tabular classification.

### Support Vector Machine

A margin-based classification algorithm used as an additional classical benchmark.

All classical models operate on the same prepared dataset so their results can be compared consistently.

---

# 2. Quantum Machine Learning

QuanDetect includes a **Variational Quantum Classifier (VQC)** implemented using **PennyLane and PyTorch**.

The quantum workflow uses classical preprocessing before sending compact feature representations into the quantum circuit.

```text
Biomedical Features
        ↓
Preprocessing
        ↓
Feature Selection
        ↓
PCA
        ↓
Angle Encoding
        ↓
Variational Quantum Circuit
        ↓
Quantum Measurements
        ↓
Classical Prediction Head
        ↓
Disease Probability
```

### Quantum Components

- PennyLane
- PyTorch
- `default.qubit`
- Angle embedding
- Variational quantum layers
- Strongly entangling layers
- Pauli-Z expectation measurements
- Classical output layer

The current implementation uses a **quantum simulator** and does not require physical quantum hardware.

### Why PCA Before the Quantum Circuit?

Quantum simulation becomes increasingly expensive as the number of qubits increases. PCA is therefore used to reduce the feature space to a compact representation suitable for the selected quantum circuit size.

For example, a four-qubit configuration has:

```text
2^4 = 16
```

computational basis states in simulation.

---

# 3. Medical Imaging — CT Lung Diagnostics

QuanDetect includes a dedicated medical-imaging module for lung CT analysis.

The dashboard provides an independent imaging workflow alongside the tabular ML pipeline.

```text
CT Image
   ↓
Image Preprocessing
   ↓
Lung / ROI Processing
   ↓
Feature Preparation
   ↓
Model Inference
   ↓
Prediction
   ↓
Explainability
   ↓
Benchmarking
```

## Imaging Workflow

The CT Lung Diagnostics interface supports:

- Curated clinical sample scans
- Custom CT upload workflow
- CT image inspection
- Lung-region / ROI processing
- Original CT visualization
- Segmented ROI visualization
- Grad-CAM heatmap visualization
- Model probability visualization
- Risk visualization
- Comparative model evaluation

The demonstration workflow includes **Normal, Benign, and Malignant** classes.

---

# 4. Imaging Models

## ResNet-18 — 2D CNN

A ResNet-18 based convolutional neural network is used as the classical deep-learning baseline for CT image classification.

```text
CT Image
   ↓
Preprocessing
   ↓
ResNet-18
   ↓
Classification
   ↓
Normal / Benign / Malignant
```

## Variational Quantum Classifier — Imaging

The imaging pipeline also includes a quantum classification workflow based on extracted image features.

```text
CT Image
   ↓
Feature Extraction
   ↓
Compact Feature Representation
   ↓
Quantum Encoding
   ↓
Variational Quantum Circuit
   ↓
Quantum Measurement
   ↓
Classification
```

## Hybrid Quantum-CNN

The Hybrid Quantum-CNN combines classical image feature extraction with a quantum classification component.

```text
CT Image
   ↓
CNN Feature Extraction
   ↓
Compact Feature Representation
   ↓
Quantum Encoding
   ↓
Variational Quantum Circuit
   ↓
Quantum Measurements
   ↓
Classification Head
   ↓
Prediction
```

This architecture allows the project to investigate how quantum components can be integrated into a conventional medical-imaging pipeline.

---

# 5. Explainable AI

Explainability is an important part of QuanDetect.

## SHAP — Tabular Pipeline

SHAP is used to analyze feature contributions to model predictions.

It provides an interpretable view of which input features contribute to an individual prediction or model output.

## Grad-CAM — CT Imaging

Grad-CAM is used to generate visual activation heatmaps for imaging predictions.

The imaging dashboard provides:

```text
Original CT
     ↓
Model Prediction
     ↓
Grad-CAM
     ↓
Visual Heatmap
```

Grad-CAM is treated as a model-interpretability mechanism. It is **not** a substitute for clinical image interpretation, definitive lesion segmentation, or radiological diagnosis.

---

# 6. Risk Stratification

QuanDetect provides probability-based risk visualization for demonstration purposes.

| Risk Level | Demonstration Probability |
|------------|---------------------------|
| Low | < 30% |
| Medium | 30% – 64.9% |
| High | ≥ 65% |

These thresholds are **demonstration values only** and are not validated clinical diagnostic thresholds.

---

# 7. Model Evaluation

QuanDetect provides comparative evaluation across classical, quantum, and hybrid approaches.

## Tabular Metrics

- Accuracy
- Sensitivity / Recall
- Specificity
- Precision
- F1-score

## Imaging Metrics

- Accuracy
- Sensitivity / Recall
- Specificity
- Precision
- F1-score
- ROC-AUC
- Inference latency
- Confusion matrix

Model results depend on the selected dataset, preprocessing configuration, train/test partition, model parameters, and execution environment.

---

# 8. Technology Stack

## Backend

- Python
- FastAPI
- Pandas
- NumPy
- scikit-learn
- XGBoost
- PyTorch
- PennyLane
- SHAP
- Torchvision
- OpenCV
- Pillow

## Frontend

- React
- Vite
- Tailwind CSS
- Recharts
- Framer Motion
- GSAP
- Lucide React

## Machine Learning

- Random Forest
- XGBoost
- Support Vector Machine
- PCA
- ANOVA F-test
- ResNet-18
- Variational Quantum Classifier
- Hybrid Quantum-CNN

## Explainability

- SHAP
- Grad-CAM

---

# 9. Project Structure

```text
QuanDetect/
│
├── README.md
├── .gitignore
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── preprocessing.py
│   │   ├── schemas.py
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── classical.py
│   │   │   └── quantum.py
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── data_utils.py
│   │       └── risk.py
│   │
│   ├── data/
│   │   ├── sample_breast_cancer.csv
│   │   └── survey_lung_cancer.csv
│   │
│   ├── imaging/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── preprocessing.py
│   │   ├── feature_extraction.py
│   │   ├── evaluation.py
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── classical_cnn.py
│   │   │   ├── quantum_imaging.py
│   │   │   └── hybrid_imaging.py
│   │   │
│   │   ├── explainability/
│   │   │   ├── __init__.py
│   │   │   └── gradcam.py
│   │   │
│   │   └── data/
│   │       ├── generate_samples.py
│   │       └── sample_ct_scans/
│   │
│   ├── test_imaging_pipeline.py
│   └── test_lung_cancer_tabular.py
│
└── frontend/
    ├── package.json
    ├── package-lock.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── index.html
    │
    └── src/
        ├── App.jsx
        ├── main.jsx
        ├── index.css
        │
        ├── components/
        │   ├── Sidebar.jsx
        │   ├── StatCard.jsx
        │   └── ui/
        │
        ├── lib/
        │   ├── api.js
        │   ├── imagingApi.js
        │   ├── utils.js
        │   └── AppStateContext.jsx
        │
        └── pages/
            ├── Landing.jsx
            ├── Upload.jsx
            ├── Training.jsx
            ├── Prediction.jsx
            ├── Comparison.jsx
            ├── Results.jsx
            │
            └── imaging/
                └── ImagingDashboard.jsx
```

---

# 10. Installation

## Prerequisites

- Python 3.12
- Node.js 18+
- npm
- Git

Python dependencies should be installed inside a virtual environment.

---

## Backend Setup

From the project root:

```bash
cd backend
```

Create a Python virtual environment:

```bash
python3.12 -m venv venv
```

Activate it on macOS / Linux:

```bash
source venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
python -m uvicorn main:app --reload --port 8000
```

Backend:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

---

## Frontend Setup

Open a second terminal:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Frontend:

```text
http://localhost:5173
```

For LAN/network-accessible development:

```bash
npm run dev -- --host 0.0.0.0
```

---

# 11. Environment Configuration

Environment-specific configuration should remain local and must not contain credentials in the repository.

Where environment configuration is required, use the provided example file:

```text
frontend/.env.example
```

Create the corresponding local environment file:

```text
frontend/.env
```

Do not commit:

- API keys
- Access tokens
- Passwords
- Private keys
- Cloud credentials
- Database credentials
- Other sensitive configuration

The repository `.gitignore` files exclude local environment files, virtual environments, runtime sessions, build output, and development artifacts.

---

# 12. API

## Tabular Pipeline

```text
POST /api/upload
POST /api/preprocess
POST /api/train/classical
POST /api/train/quantum
POST /api/predict
GET  /api/compare/{session_id}
POST /api/explain
GET  /api/sample-dataset-info
```

## Imaging Pipeline

The imaging module exposes dedicated routes for:

- CT image processing
- Image inference
- Model evaluation
- Explainability
- Imaging workflow operations

The exact route implementation is maintained within:

```text
backend/imaging/api/routes.py
```

---

# 13. Usage

## Tabular Pipeline

```text
1. Upload a biomedical CSV dataset
2. Select or detect the target column
3. Preprocess the dataset
4. Select relevant features
5. Train classical models
6. Train the quantum classifier
7. Generate predictions
8. Compare model performance
9. Inspect SHAP explanations
10. View probability and risk stratification
```

## CT Lung Diagnostics

```text
1. Open CT Lung Diagnostics
2. Select a curated CT sample or upload a supported CT image
3. Inspect the CT scan
4. Process the lung region / ROI
5. Run the available imaging models
6. Compare CNN, VQC and Hybrid outputs
7. Review prediction probabilities
8. Inspect Grad-CAM visualization
9. Review imaging benchmark metrics
```

---

# 14. Sample Data

The repository includes demonstration data for development and evaluation.

### Tabular Samples

```text
backend/data/sample_breast_cancer.csv
backend/data/survey_lung_cancer.csv
```

### Imaging Samples

```text
backend/imaging/data/sample_ct_scans/
```

The included sample data is intended for demonstration and development.

For research or real-world use, datasets should be independently validated, appropriately licensed, and processed according to applicable privacy and data-governance requirements.

---

# 15. Reproducibility and Version Control

Backend dependency versions are maintained in:

```text
backend/requirements.txt
```

Frontend dependency versions are locked through:

```text
frontend/package-lock.json
```

Git is used to track source-code changes.

Recommended workflow:

```bash
git status
git add .
git commit -m "Describe the change"
git push
```

Stable project milestones can be tagged:

```bash
git tag v1.0.0
git push origin v1.0.0
```

This allows future versions of the project to be tracked without losing previous stable states.

---

# 16. Testing

The backend includes pipeline-level test files for the major workflows:

```text
backend/test_imaging_pipeline.py
backend/test_lung_cancer_tabular.py
```

Tests can be executed from the backend environment using the project's configured Python test tooling.

---

# 17. Limitations

## Medical Limitations

- QuanDetect is not a certified medical device.
- Model predictions are not medical diagnoses.
- Demonstration risk thresholds are not validated clinical thresholds.
- Predictions depend on the dataset and model configuration.
- A single 2D CT slice does not represent complete patient-level clinical information.
- Cancer staging cannot be reliably inferred without appropriate staging labels and clinical information.
- Grad-CAM outputs are model-interpretability visualizations and not clinical evidence.

## Machine Learning Limitations

- Model performance depends on dataset quality and representativeness.
- Small or biased datasets can produce unreliable evaluation results.
- Metrics depend on train/test partitioning and preprocessing.
- Accuracy alone does not establish clinical usefulness.
- Quantum simulation results should not be interpreted as equivalent to execution on physical quantum hardware.

## Computational Limitations

- Quantum circuit simulation becomes increasingly expensive as qubit count increases.
- Deep-learning inference and training can require significant computational resources.
- Larger CT datasets require additional storage and processing infrastructure.

---

# 18. Responsible Use

> **QuanDetect is a research, educational, and Smart India Hackathon demonstration project. It is not a certified medical device and must not be used to diagnose, treat, or make clinical decisions about any individual. Model predictions, probability scores, risk levels, and visual explanations are experimental outputs and require appropriate clinical validation, regulatory review, and expert interpretation before any real-world medical application.**

---

# 19. Smart India Hackathon

**Problem Statement:** SIH26139

QuanDetect explores the use of hybrid quantum-classical machine learning for early and explainable disease detection.

The project brings together:

```text
Classical Machine Learning
          +
Deep Learning
          +
Quantum Machine Learning
          +
Medical Imaging
          +
Explainable AI
          +
Model Benchmarking
```

within a single research-oriented platform.

---

# 20. Future Scope

Potential extensions include:

- Execution on real quantum hardware
- Larger and more diverse biomedical datasets
- 3D CT volume analysis
- Advanced DICOM processing
- Improved lung segmentation
- Additional quantum architectures
- Transformer-based medical imaging models
- Automated hyperparameter optimization
- Independent external validation
- Probability calibration
- Cloud-based distributed training
- Model monitoring and drift detection
- Secure production-grade deployment
- Privacy-preserving learning approaches

---

# 21. Project Status

## Tabular Pipeline

- [x] Dataset upload
- [x] Automated target detection
- [x] Data preprocessing
- [x] Missing-value handling
- [x] Feature selection
- [x] PCA
- [x] Random Forest
- [x] XGBoost
- [x] SVM
- [x] Variational Quantum Classifier
- [x] Model comparison
- [x] Risk stratification
- [x] SHAP explainability

## Medical Imaging Pipeline

- [x] CT Lung Diagnostics dashboard
- [x] Curated CT sample scans
- [x] Custom CT upload workflow
- [x] CT inspection
- [x] Lung / ROI processing
- [x] ResNet-18 2D CNN
- [x] Imaging VQC
- [x] Hybrid Quantum-CNN
- [x] Grad-CAM explainability
- [x] Normal / Benign / Malignant workflow
- [x] Imaging model benchmarking
- [x] Confusion matrix visualization
- [x] Probability and risk visualization

---

# 22. License

This project is developed as a research and educational prototype for **Smart India Hackathon**.

Before redistribution, commercial use, or deployment with external datasets, verify the applicable licenses and usage requirements of all third-party libraries, datasets, pretrained models, and external resources used by the project.

---

## Acknowledgements

QuanDetect is built using open-source technologies including:

- FastAPI
- PyTorch
- PennyLane
- scikit-learn
- XGBoost
- SHAP
- Torchvision
- React
- Vite
- Tailwind CSS

The project also uses publicly available datasets and research resources where applicable.
