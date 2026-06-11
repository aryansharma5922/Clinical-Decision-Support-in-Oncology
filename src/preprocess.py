from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
DATA_PATH = DATA_DIR / "breast_cancer.csv"
UCI_WDBC_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
    "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]
COLUMNS = ["id", "diagnosis", *FEATURE_NAMES]
TARGET_COLUMN = "diagnosis_encoded"


def download_wdbc_dataset(force: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        return pd.read_csv(DATA_PATH)
    try:
        with urlopen(UCI_WDBC_URL, timeout=20) as response:
            raw = response.read().decode("utf-8")
        df = pd.read_csv(StringIO(raw), header=None, names=COLUMNS)
        source = "UCI Machine Learning Repository WDBC raw data"
    except Exception:
        from sklearn.datasets import load_breast_cancer

        bunch = load_breast_cancer(as_frame=True)
        df = bunch.frame.copy()
        df = df.rename(columns={old: new for old, new in zip(bunch.feature_names, FEATURE_NAMES)})
        df.insert(0, "id", np.arange(1, len(df) + 1))
        df.insert(1, "diagnosis", np.where(df["target"].eq(0), "M", "B"))
        df = df[["id", "diagnosis", *FEATURE_NAMES]]
        source = "scikit-learn bundled WDBC fallback"
    df.to_csv(DATA_PATH, index=False)
    (DATA_DIR / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "dataset": "Breast Cancer Wisconsin Diagnostic (WDBC)",
                "primary_source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
                "raw_data_url": UCI_WDBC_URL,
                "materialized_from": source,
                "samples": int(df.shape[0]),
                "diagnostic_features": len(FEATURE_NAMES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return df


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH) if DATA_PATH.exists() else download_wdbc_dataset()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.drop(columns=[c for c in df.columns if str(c).lower().startswith("unnamed")], errors="ignore")
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned["diagnosis"] = cleaned["diagnosis"].astype(str).str.upper().str.strip()
    cleaned[TARGET_COLUMN] = cleaned["diagnosis"].map({"M": 1, "B": 0})
    if cleaned[TARGET_COLUMN].isna().any():
        raise ValueError("Diagnosis contains values outside M/B.")
    return cleaned


def get_feature_target(df: pd.DataFrame):
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    X = df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def split_scale_pca(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA(n_components=0.95, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train, X_test, y_train, y_test, scaler, pca, X_train_pca, X_test_pca


def persist_preprocessors(scaler: StandardScaler, pca: PCA) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(pca, MODEL_DIR / "pca.pkl")
