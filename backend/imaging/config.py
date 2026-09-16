"""
Configuration for the QuanDetect Medical Imaging Pipeline.

Controls all image resolutions, CT windowing levels, model hyperparameters,
and quantum circuit dimensions for the independent lung cancer CT module.
"""
import os

# --- Paths -------------------------------------------------------------------
IMAGING_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGING_DATA_DIR = os.path.join(IMAGING_BASE_DIR, "data")
SAMPLE_SCANS_DIR = os.path.join(IMAGING_DATA_DIR, "sample_ct_scans")
IMAGING_SESSION_DIR = os.path.join(IMAGING_BASE_DIR, "sessions")

os.makedirs(IMAGING_DATA_DIR, exist_ok=True)
os.makedirs(SAMPLE_SCANS_DIR, exist_ok=True)
os.makedirs(IMAGING_SESSION_DIR, exist_ok=True)

# --- CT Preprocessing & Normalization -----------------------------------------
IMAGE_SIZE = (224, 224)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Standard Hounsfield Unit (HU) Window for Lung CT
# Center = -600 HU, Width = 1500 HU -> range [-1350, 150] HU
LUNG_WINDOW_CENTER = -600
LUNG_WINDOW_WIDTH = 1500
HU_MIN = LUNG_WINDOW_CENTER - (LUNG_WINDOW_WIDTH // 2)
HU_MAX = LUNG_WINDOW_CENTER + (LUNG_WINDOW_WIDTH // 2)

# --- Diagnostic Classes -------------------------------------------------------
CLASSES = ["Normal", "Benign", "Malignant"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# --- Cancer Stage Disclaimer / Status -----------------------------------------
CAN_PREDICT_STAGE = False
STAGE_STATUS_MESSAGE = "Cannot be reliably predicted from 2D slice (dataset does not provide stage labels)"

# --- Quantum Classifier (VQC) -------------------------------------------------
N_QUBITS = 4
N_VQC_LAYERS = 3
VQC_EPOCHS = 20
VQC_LR = 0.05
RANDOM_STATE = 42

# --- Clinical Risk Stratification ---------------------------------------------
RISK_LOW_MAX = 0.30
RISK_MODERATE_MAX = 0.65
