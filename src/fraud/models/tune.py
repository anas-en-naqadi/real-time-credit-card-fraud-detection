"""Hyperparameter optimization using Optuna."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from src.fraud.data.features import FraudFeatureEngineer

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_lightgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_trials: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run Optuna study to optimize LightGBM hyperparameters for PR-AUC."""
    feat_eng = FraudFeatureEngineer(include_hour=True, include_log_amount=True)

    X_train = feat_eng.fit_transform(train_df.drop(columns=["Class"]))
    y_train = train_df["Class"].to_numpy()

    X_val = feat_eng.transform(val_df.drop(columns=["Class"]))
    y_val = val_df["Class"].to_numpy()

    scale_pos = float((len(y_train) - sum(y_train)) / max(sum(y_train), 1))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 250),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", 1.0, scale_pos, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": random_state,
            "verbosity": -1,
            "n_jobs": -1,
        }

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        val_probs = clf.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val, val_probs))

    logger.info("Starting Optuna optimization for LightGBM (%d trials)...", n_trials)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(
        "Optuna best trial PR-AUC: %.4f with params: %s",
        study.best_value,
        study.best_params,
    )
    return {
        "best_params": study.best_params,
        "best_pr_auc": study.best_value,
        "study": study,
    }
