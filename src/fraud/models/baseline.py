"""Baseline Logistic Regression model with class weighting."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.fraud.data.features import FraudFeatureEngineer
from src.fraud.models.metrics import evaluate_predictions, find_optimal_cost_threshold

logger = logging.getLogger(__name__)


def build_baseline_pipeline(
    class_weight: str = "balanced", random_state: int = 42
) -> Pipeline:
    """Build Logistic Regression pipeline with feature extraction and scaling."""
    return Pipeline(
        [
            (
                "features",
                FraudFeatureEngineer(include_hour=True, include_log_amount=True),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight,
                    random_state=random_state,
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_and_eval_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[Pipeline, float, dict[str, float]]:
    """Train baseline Logistic Regression, determine optimal threshold, and evaluate on val set."""
    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"].to_numpy()

    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"].to_numpy()
    val_amounts = val_df["Amount"].to_numpy()

    pipeline = build_baseline_pipeline(
        class_weight="balanced", random_state=random_state
    )
    logger.info("Training baseline LogisticRegression...")
    pipeline.fit(X_train, y_train)

    val_probs = pipeline.predict_proba(X_val)[:, 1]
    optimal_thresh, _min_cost, _breakdown = find_optimal_cost_threshold(
        y_true=y_val, y_prob=val_probs, amounts=val_amounts
    )

    metrics = evaluate_predictions(
        y_true=y_val,
        y_prob=val_probs,
        amounts=val_amounts,
        threshold=optimal_thresh,
    )

    logger.info(
        "Baseline validation results: PR-AUC=%.4f, Cost=%.2f, Threshold=%.3f",
        metrics["pr_auc"],
        metrics["total_cost"],
        optimal_thresh,
    )
    return pipeline, optimal_thresh, metrics
