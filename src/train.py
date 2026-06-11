from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

from src.evaluate import evaluate_models, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve, save_metrics
from src.preprocess import (
    FEATURE_NAMES, MODEL_DIR, clean_dataset, download_wdbc_dataset, get_feature_target,
    persist_preprocessors, split_scale_pca,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"


def train_models() -> dict:
    df = clean_dataset(download_wdbc_dataset())
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test, scaler, pca, X_train_pca, X_test_pca = split_scale_pca(X, y)

    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
    svm_model = SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")
    voting_model = VotingClassifier(estimators=[("rf", rf_model), ("svm", svm_model)], voting="soft")

    rf_model.fit(X_train_pca, y_train)
    svm_model.fit(X_train_pca, y_train)
    voting_model.fit(X_train_pca, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    persist_preprocessors(scaler, pca)
    joblib.dump(rf_model, MODEL_DIR / "rf_model.pkl")
    joblib.dump(svm_model, MODEL_DIR / "svm_model.pkl")
    joblib.dump(voting_model, MODEL_DIR / "voting_model.pkl")

    models = {"Random Forest": rf_model, "SVM": svm_model, "Voting Ensemble": voting_model}
    metrics_df = evaluate_models(models, X_test_pca, y_test)
    save_metrics(metrics_df)

    best_name = metrics_df.iloc[0]["model"]
    plot_confusion_matrix(models[best_name], X_test_pca, y_test, best_name)
    plot_roc_curve(models, X_test_pca, y_test)
    plot_precision_recall_curve(models, X_test_pca, y_test)

    artifacts = {
        "feature_names": FEATURE_NAMES,
        "feature_medians": X.median(numeric_only=True).to_dict(),
        "feature_min": X.min(numeric_only=True).to_dict(),
        "feature_max": X.max(numeric_only=True).to_dict(),
        "pca_components": int(pca.n_components_),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "best_model": best_name,
        "class_mapping": {"Benign": 0, "Malignant": 1},
    }
    (MODEL_DIR / "training_artifacts.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    X_train.assign(diagnosis_encoded=y_train.values).to_csv(REPORT_DIR / "train_split.csv", index=False)
    X_test.assign(diagnosis_encoded=y_test.values).to_csv(REPORT_DIR / "test_split.csv", index=False)
    return {"metrics": metrics_df, "best_model": best_name}


if __name__ == "__main__":
    result = train_models()
    print(result["metrics"].to_string(index=False))
    print(f"Best model: {result['best_model']}")
