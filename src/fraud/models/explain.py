"""SHAP Explainability module for global and local fraud interpretations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def explain_model_shap(
    classifier: Any,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    raw_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calculate global and case-specific local SHAP values.

    Extracts:
    1. Global top feature importances based on mean absolute SHAP values.
    2. Local analysis:
       - Detected fraud (True Positive)
       - Missed fraud (False Negative)
       - False alarm (False Positive)
    """
    logger.info("Computing SHAP values with TreeExplainer...")
    explainer = shap.TreeExplainer(classifier)
    shap_vals = explainer.shap_values(feature_matrix)

    # In binary classification, some versions return list [shap_neg, shap_pos]
    if isinstance(shap_vals, list):
        shap_vals_pos = shap_vals[-1]
    elif getattr(shap_vals, "ndim", 0) == 3:
        shap_vals_pos = shap_vals[:, :, 1]
    else:
        shap_vals_pos = shap_vals

    # Global feature importances
    mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)
    global_ranking = sorted(
        zip(feature_names, [float(v) for v in mean_abs_shap]),
        key=lambda item: item[1],
        reverse=True,
    )

    y_pred = (y_prob >= threshold).astype(int)

    def extract_local_explanation(index: int) -> dict[str, Any]:
        row_shap = shap_vals_pos[index]
        top_contribs = sorted(
            zip(feature_names, [float(v) for v in row_shap]),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5]
        return {
            "index": int(index),
            "true_class": int(y_true[index]),
            "predicted_prob": round(float(y_prob[index]), 4),
            "amount": round(float(raw_df.iloc[index]["Amount"]), 2),
            "top_shap_features": [
                {"feature": f, "shap_value": round(val, 4)} for f, val in top_contribs
            ],
        }

    # Find representative cases
    tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]

    detected_case = (
        extract_local_explanation(tp_indices[0]) if len(tp_indices) else None
    )
    missed_case = extract_local_explanation(fn_indices[0]) if len(fn_indices) else None
    false_alarm_case = (
        extract_local_explanation(fp_indices[0]) if len(fp_indices) else None
    )

    return {
        "global_importance": [
            {"feature": f, "mean_abs_shap": round(val, 5)} for f, val in global_ranking
        ],
        "local_cases": {
            "detected_fraud_tp": detected_case,
            "missed_fraud_fn": missed_case,
            "false_alarm_fp": false_alarm_case,
        },
    }
