"""
Module 1: Data Pre-processing & Feature Engineering.

Designed to be dataset-agnostic: any tabular biomedical CSV (hospital
records, genomics panels, the bundled Breast Cancer Wisconsin sample,
etc.) can be dropped in as long as it has a binary/multi-class target
column. Steps:

  1. Cleaning        -> drop fully-empty columns/rows, coerce dtypes
  2. Missing values   -> median / mean / most_frequent imputation
  3. Encoding         -> label-encode any categorical target/feature
  4. Scaling          -> StandardScaler for numeric features
  5. Feature selection-> ANOVA F-test (SelectKBest) to keep the most
                         informative features, which also keeps the
                         quantum circuit (limited qubit count) tractable
  6. Train/test split -> stratified, reproducible
"""
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from app.config import TEST_SIZE, RANDOM_STATE, DEFAULT_TARGET_COLUMN_CANDIDATES, N_QUBITS


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Best-effort guess of the target/label column for a new dataset."""
    for candidate in DEFAULT_TARGET_COLUMN_CANDIDATES:
        for col in df.columns:
            if col.strip().lower() == candidate:
                return col
    # Fall back: last column, if it looks binary/categorical (few unique values)
    last_col = df.columns[-1]
    if df[last_col].nunique() <= 10:
        return last_col
    return None


def missing_value_summary(df: pd.DataFrame) -> Dict[str, int]:
    return {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty columns, strip header whitespace, and attempt numeric coercion."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how="all")
    for col in df.columns:
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="coerce")
            # If most values convert cleanly, treat the column as numeric.
            if coerced.notna().sum() >= 0.9 * len(df):
                df[col] = coerced
    return df.reset_index(drop=True)


def encode_target(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    y_clean = y.astype(str).str.strip().str.upper()
    if set(y_clean.unique()).issubset({"YES", "NO"}):
        encoder.classes_ = np.array(["NO", "YES"])
        y_encoded = (y_clean == "YES").astype(int).values
        return y_encoded, encoder
    y_encoded = encoder.fit_transform(y_clean)
    return y_encoded, encoder


class PreprocessingPipeline:
    """Fits on training data, stores all transformers for later reuse
    during single-sample prediction so that inference-time features are
    processed identically to training-time features."""

    def __init__(
        self,
        target_column: str,
        drop_columns: Optional[List[str]] = None,
        missing_strategy: str = "median",
        scale: bool = True,
        feature_selection_k: Optional[int] = None,
    ):
        self.target_column = target_column
        self.drop_columns = drop_columns or []
        self.missing_strategy = missing_strategy
        self.scale = scale
        self.feature_selection_k = feature_selection_k

        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.selector: Optional[SelectKBest] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.pca_for_quantum: Optional[PCA] = None

        self.categorical_encoders: Dict[str, any] = {}
        self.original_feature_names: List[str] = []
        self.selected_feature_names: List[str] = []

    def fit_transform(self, df: pd.DataFrame):
        df = clean_dataframe(df)
        drop_cols = [c for c in self.drop_columns if c in df.columns and c != self.target_column]
        df = df.drop(columns=drop_cols, errors="ignore")

        feature_df = df.drop(columns=[self.target_column])
        
        # Categorical feature encoding (e.g. GENDER: M -> 1, F -> 0)
        self.categorical_encoders = {}
        encoded_cols = {}
        for col in feature_df.columns:
            if feature_df[col].dtype == object:
                vals = feature_df[col].astype(str).str.strip().str.upper()
                if set(vals.unique()).issubset({"M", "F"}):
                    self.categorical_encoders[col] = {"M": 1.0, "F": 0.0}
                    encoded_cols[col] = vals.map({"M": 1.0, "F": 0.0}).fillna(0.0).values
                else:
                    le = LabelEncoder()
                    encoded_cols[col] = le.fit_transform(vals).astype(float)
                    self.categorical_encoders[col] = le
            else:
                encoded_cols[col] = pd.to_numeric(feature_df[col], errors="coerce").values

        numeric_df = pd.DataFrame(encoded_cols)
        self.original_feature_names = numeric_df.columns.tolist()

        y_raw = df[self.target_column]
        y, self.label_encoder = encode_target(y_raw)

        # 1) Impute missing values
        self.imputer = SimpleImputer(strategy=self.missing_strategy)
        X = self.imputer.fit_transform(numeric_df.values)

        # 2) Scale
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # 3) Feature selection (ANOVA F-test) - keeps the most predictive subset
        k = self.feature_selection_k or min(10, X.shape[1])
        k = max(1, min(k, X.shape[1]))
        self.selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.selector.fit_transform(X, y)
        mask = self.selector.get_support()
        self.selected_feature_names = [f for f, m in zip(self.original_feature_names, mask) if m]

        # 4) Secondary PCA projection down to N_QUBITS dims for quantum angle-encoding
        n_components = min(N_QUBITS, X_selected.shape[1])
        self.pca_for_quantum = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_quantum = self.pca_for_quantum.fit_transform(X_selected)

        return X_selected, X_quantum, y

    def transform_single(self, feature_dict: Dict[str, any]) -> Tuple[np.ndarray, np.ndarray]:
        """Transform a single raw feature dict through the same fitted pipeline."""
        row_dict = {}
        for f in self.original_feature_names:
            raw_val = feature_dict.get(f, np.nan)
            if f in self.categorical_encoders:
                encoder = self.categorical_encoders[f]
                if isinstance(encoder, dict):
                    if isinstance(raw_val, str):
                        val = encoder.get(raw_val.strip().upper(), 0.0)
                    else:
                        val = float(raw_val) if not pd.isna(raw_val) else 0.0
                else:
                    try:
                        val = float(encoder.transform([str(raw_val).strip().upper()])[0])
                    except Exception:
                        val = 0.0
                row_dict[f] = val
            else:
                try:
                    row_dict[f] = float(raw_val) if not pd.isna(raw_val) else np.nan
                except (ValueError, TypeError):
                    row_dict[f] = np.nan

        row = pd.DataFrame([row_dict])
        X = self.imputer.transform(row.values)
        if self.scale:
            X = self.scaler.transform(X)
        X_selected = self.selector.transform(X)
        X_quantum = self.pca_for_quantum.transform(X_selected)
        return X_selected, X_quantum


def split_data(X, y, X_quantum=None):
    """Stratified split; returns matching splits for classical + quantum
    feature sets (and indices) so both model families train/test on the
    exact same rows."""
    indices = np.arange(len(y))
    if X_quantum is not None:
        X_train, X_test, Xq_train, Xq_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, X_quantum, y, indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        return X_train, X_test, Xq_train, Xq_test, y_train, y_test, idx_train, idx_test
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, idx_train, idx_test
