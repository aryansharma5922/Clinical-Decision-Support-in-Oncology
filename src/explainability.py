from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocess import FEATURE_NAMES, MODEL_DIR, PROJECT_ROOT, clean_dataset, get_feature_target, load_dataset

REPORT_DIR = PROJECT_ROOT / "reports"


def load_explainability_stack():
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    pca = joblib.load(MODEL_DIR / "pca.pkl")
    rf_model = joblib.load(MODEL_DIR / "rf_model.pkl")
    X, y = get_feature_target(clean_dataset(load_dataset()))
    X_pca = pca.transform(scaler.transform(X))
    pc_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
    return scaler, pca, rf_model, X, pd.DataFrame(X_pca, columns=pc_names), y


def pca_component_importance() -> pd.DataFrame:
    _, _, rf_model, _, X_pca, _ = load_explainability_stack()
    return pd.DataFrame({"component": X_pca.columns, "importance": rf_model.feature_importances_}).sort_values("importance", ascending=False)


def approximate_original_feature_importance() -> pd.DataFrame:
    _, pca, rf_model, _, _, _ = load_explainability_stack()
    contribution = np.abs(pca.components_.T).dot(rf_model.feature_importances_)
    contribution = contribution / contribution.sum()
    return pd.DataFrame({"feature": FEATURE_NAMES, "importance": contribution}).sort_values("importance", ascending=False)


def create_shap_summary_plot(max_samples: int = 200) -> Path | None:
    try:
        import shap
    except Exception:
        return None
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _, _, rf_model, _, X_pca, _ = load_explainability_stack()
    sample = X_pca.sample(min(max_samples, len(X_pca)), random_state=42)
    explainer = shap.TreeExplainer(rf_model)
    values = explainer.shap_values(sample)
    values = values[1] if isinstance(values, list) else values
    plt.figure(figsize=(8, 5))
    shap.summary_plot(values, sample, show=False)
    path = REPORT_DIR / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def create_shap_waterfall_plot(input_values: dict[str, float] | None = None) -> Path | None:
    try:
        import shap
    except Exception:
        return None
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scaler, pca, rf_model, X, X_pca, _ = load_explainability_stack()
    row = X.iloc[[0]] if input_values is None else pd.DataFrame([{f: float(input_values[f]) for f in FEATURE_NAMES}])
    row_pca = pd.DataFrame(pca.transform(scaler.transform(row)), columns=X_pca.columns)
    exp = shap.TreeExplainer(rf_model)(row_pca)[0]
    if len(exp.values.shape) > 1:
        exp = shap.Explanation(values=exp.values[:, 1], base_values=exp.base_values[1], data=exp.data, feature_names=exp.feature_names)
    shap.plots.waterfall(exp, show=False, max_display=10)
    path = REPORT_DIR / "shap_waterfall.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path
