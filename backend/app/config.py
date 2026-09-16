"""
Central configuration for the QuanDetect backend.

Keeping all tunables (paths, quantum circuit size, session storage, etc.)
in one place makes it easy to swap the sample dataset for a real
hospital / genomics CSV later without touching business logic.
"""
import os

# --- Paths -----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_DATASET_PATH = os.path.join(DATA_DIR, "survey_lung_cancer.csv")
SESSION_STORE_DIR = os.path.join(BASE_DIR, "sessions")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SESSION_STORE_DIR, exist_ok=True)

# --- Preprocessing -----------------------------------------------------------
DEFAULT_TARGET_COLUMN_CANDIDATES = [
    "lung_cancer",
    "lungcancer",
    "cancer",
    "target",
    "label",
    "diagnosis",
    "class",
    "outcome",
]
MISSING_VALUE_STRATEGY = "median"          # median | mean | most_frequent
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Quantum model (VQC) ------------------------------------------------------
# n_qubits also caps the number of PCA-reduced features fed to the circuit.
# default.qubit simulator scales roughly exponentially, so we keep this small
# for responsiveness in a demo/education environment.
N_QUBITS = 4
N_VQC_LAYERS = 3
VQC_EPOCHS = 25
VQC_LEARNING_RATE = 0.1
VQC_BATCH_SIZE = 16

# --- Risk stratification thresholds ------------------------------------------
RISK_LOW_MAX = 0.33
RISK_MEDIUM_MAX = 0.66
