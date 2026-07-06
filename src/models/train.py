"""
Model training for the visual quality inspection system.

Compares Logistic Regression, Random Forest, and SVM on HOG+LBP+intensity
features extracted from the synthetic surface image dataset, selects a
champion by F1 score (chosen over accuracy since a QA system's cost of a
missed defect and a false alarm are both operationally meaningful, and F1
balances precision/recall rather than optimizing either alone), and saves
the champion model + a fitted StandardScaler + metadata for serving.

Run:
    python -m src.models.train --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_candidate_models(cfg: dict) -> dict:
    m = cfg["model"]
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, random_state=42),
            {"clf__C": m["logistic_regression"]["C"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "clf__n_estimators": m["random_forest"]["n_estimators"],
                "clf__max_depth": m["random_forest"]["max_depth"],
                "clf__min_samples_leaf": m["random_forest"]["min_samples_leaf"],
            },
        ),
        "svm": (
            SVC(probability=True, random_state=42),
            {"clf__C": m["svm"]["C"], "clf__kernel": m["svm"]["kernel"]},
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision_defective": report["1"]["precision"],
        "recall_defective": report["1"]["recall"],
    }, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare defect classifiers.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--features", type=str, default="data/processed_features.npz")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    data = np.load(args.features, allow_pickle=True)
    X, y = data["X"], data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - cfg["data"]["train_test_split"], stratify=y, random_state=42
    )
    logger.info(
        "Train: %d, Test: %d (%.1f%% defective in test)",
        len(X_train),
        len(X_test),
        100 * y_test.mean(),
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = get_candidate_models(cfg)
    cv = StratifiedKFold(n_splits=cfg["model"]["cv_folds"], shuffle=True, random_state=42)
    primary_metric = cfg["model"]["primary_metric"]

    results = {}
    fitted_models = {}

    for name, (estimator, param_grid) in candidates.items():
        logger.info("=== Training candidate: %s ===", name)
        pipeline = Pipeline([("clf", estimator)])
        search = GridSearchCV(
            pipeline, param_grid, scoring=primary_metric, cv=cv, n_jobs=-1, refit=True
        )
        search.fit(X_train_scaled, y_train)
        metrics, y_pred = evaluate(search.best_estimator_, X_test_scaled, y_test)
        metrics["cv_best_score"] = float(search.best_score_)
        metrics["best_params"] = {k: str(v) for k, v in search.best_params_.items()}

        logger.info("%s test metrics: %s", name, metrics)
        results[name] = metrics
        fitted_models[name] = search.best_estimator_

    champion_name = max(results, key=lambda k: results[k][primary_metric])
    champion_model = fitted_models[champion_name]
    logger.info(
        "Champion model: %s (%s=%.4f)",
        champion_name,
        primary_metric,
        results[champion_name][primary_metric],
    )

    model_dir = Path(cfg["artifacts"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_model, cfg["artifacts"]["champion_model_path"])
    joblib.dump(scaler, cfg["artifacts"]["scaler_path"])

    metadata = {
        "champion_model": champion_name,
        "primary_metric": primary_metric,
        "all_results": results,
        "feature_dim": int(X.shape[1]),
        "decision_threshold": cfg["api"]["decision_threshold"],
    }
    with open(cfg["artifacts"]["metadata_path"], "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Saved champion model, scaler, and metadata to %s", model_dir)

    reports_dir = Path(cfg["artifacts"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    _, y_pred_champion = evaluate(champion_model, X_test_scaled, y_test)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_champion,
        display_labels=["good", "defective"],
        ax=ax,
        cmap="Blues",
    )
    ax.set_title(f"Confusion Matrix - {champion_name}")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=130)
    plt.close()

    with open(reports_dir / "model_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
