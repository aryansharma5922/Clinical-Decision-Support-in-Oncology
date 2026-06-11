from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from src.preprocess import FEATURE_NAMES, MODEL_DIR, PROJECT_ROOT

HISTORY_PATH = PROJECT_ROOT / "reports" / "prediction_history.csv"


def _load_artifacts() -> dict:
    path = MODEL_DIR / "training_artifacts.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"best_model": "Voting Ensemble"}


def load_models() -> dict:
    return {
        "Random Forest": joblib.load(MODEL_DIR / "rf_model.pkl"),
        "SVM": joblib.load(MODEL_DIR / "svm_model.pkl"),
        "Voting Ensemble": joblib.load(MODEL_DIR / "voting_model.pkl"),
    }


def load_prediction_stack(model_name: str | None = None):
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    pca = joblib.load(MODEL_DIR / "pca.pkl")
    models = load_models()
    name = model_name or _load_artifacts().get("best_model", "Voting Ensemble")
    return scaler, pca, models.get(name, models["Voting Ensemble"]), name


def validate_features(values: dict[str, float]) -> pd.DataFrame:
    missing = [feature for feature in FEATURE_NAMES if feature not in values]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")
    row = pd.DataFrame([{feature: float(values[feature]) for feature in FEATURE_NAMES}])
    if row.isna().any().any():
        raise ValueError("All tumor feature values must be numeric.")
    return row


def validate_batch_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [feature for feature in FEATURE_NAMES if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    batch = df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    invalid = batch.columns[batch.isna().any()].tolist()
    if invalid:
        raise ValueError(f"These feature columns contain non-numeric or blank values: {', '.join(invalid)}")
    return batch


def predict_tumor(values: dict[str, float], model_name: str | None = None) -> dict:
    X = validate_features(values)
    scaler, pca, model, selected_model = load_prediction_stack(model_name)
    probability = float(model.predict_proba(pca.transform(scaler.transform(X)))[0, 1])
    prediction = int(probability >= 0.5)
    confidence = probability if prediction else 1 - probability
    risk = "High" if probability >= 0.70 else "Medium" if probability >= 0.40 else "Low"
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": selected_model,
        "prediction": "Malignant" if prediction else "Benign",
        "malignant_probability": probability,
        "confidence": confidence,
        "risk_level": risk,
        "inputs": values,
    }


def predict_batch(df: pd.DataFrame, model_name: str | None = None) -> pd.DataFrame:
    X = validate_batch_features(df)
    scaler, pca, model, selected_model = load_prediction_stack(model_name)
    probabilities = model.predict_proba(pca.transform(scaler.transform(X)))[:, 1]
    predictions = probabilities >= 0.5
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = df.copy()
    results["timestamp"] = now
    results["model"] = selected_model
    results["prediction"] = ["Malignant" if value else "Benign" for value in predictions]
    results["malignant_probability"] = probabilities
    results["confidence"] = [prob if pred else 1 - prob for prob, pred in zip(probabilities, predictions)]
    results["risk_level"] = ["High" if prob >= 0.70 else "Medium" if prob >= 0.40 else "Low" for prob in probabilities]
    result_columns = ["timestamp", "model", "prediction", "malignant_probability", "confidence", "risk_level"]
    return results[result_columns + [column for column in results.columns if column not in result_columns]]


def append_prediction_history(result: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": result["timestamp"],
        "model": result["model"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "malignant_probability": result["malignant_probability"],
        "risk_level": result["risk_level"],
        **result["inputs"],
    }
    df = pd.DataFrame([row])
    if HISTORY_PATH.exists():
        df = pd.concat([pd.read_csv(HISTORY_PATH), df], ignore_index=True)
    df.to_csv(HISTORY_PATH, index=False)


def append_batch_prediction_history(results: pd.DataFrame) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history_columns = [
        "timestamp",
        "model",
        "prediction",
        "confidence",
        "malignant_probability",
        "risk_level",
        *FEATURE_NAMES,
    ]
    rows = results[history_columns].copy()
    if HISTORY_PATH.exists():
        rows = pd.concat([pd.read_csv(HISTORY_PATH), rows], ignore_index=True)
    rows.to_csv(HISTORY_PATH, index=False)


def load_prediction_history() -> pd.DataFrame:
    return pd.read_csv(HISTORY_PATH) if HISTORY_PATH.exists() else pd.DataFrame()
