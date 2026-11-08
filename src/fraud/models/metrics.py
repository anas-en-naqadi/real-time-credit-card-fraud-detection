"""Evaluation metrics and business cost function for fraud detection.

Implements the Anas <-> Medo contract cost matrix:
- Missed fraud (FN) = transaction amount
- False alarm (FP) = 5.0 (verification cost)
- Successful detection (TP) = 5.0 (investigation cost)
- Legitimate passage (TN) = 0.0
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_financial_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    verification_cost: float = 5.0,
) -> tuple[float, dict[str, float]]:
    """Compute total operational financial cost given a decision threshold.

    Cost formula:
        Cost = sum(Amount_i for i in False Negatives)
             + verification_cost * (False Positives + True Positives)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    amounts = np.asarray(amounts, dtype=float)

    y_pred = (y_prob >= threshold).astype(int)

    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tn_mask = (y_true == 0) & (y_pred == 0)

    tp_count = int(np.sum(tp_mask))
    fp_count = int(np.sum(fp_mask))
    fn_count = int(np.sum(fn_mask))
    tn_count = int(np.sum(tn_mask))

    fn_amount_loss = float(np.sum(amounts[fn_mask]))
    investigation_cost = verification_cost * (tp_count + fp_count)
    total_cost = fn_amount_loss + investigation_cost

    breakdown = {
        "total_cost": total_cost,
        "missed_fraud_loss": fn_amount_loss,
        "investigation_cost": investigation_cost,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "tn": tn_count,
    }
    return total_cost, breakdown


def find_optimal_cost_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    verification_cost: float = 5.0,
    num_steps: int = 200,
) -> tuple[float, float, dict[str, float]]:
    """Find the decision threshold that minimizes financial loss on validation data.

    Returns:
        Tuple of (optimal_threshold, min_cost, breakdown_at_optimum).
    """
    thresholds = np.linspace(0.01, 0.99, num_steps)
    best_threshold = 0.5
    min_cost = float("inf")
    best_breakdown: dict[str, float] = {}

    for thresh in thresholds:
        cost, breakdown = calculate_financial_cost(
            y_true=y_true,
            y_prob=y_prob,
            amounts=amounts,
            threshold=float(thresh),
            verification_cost=verification_cost,
        )
        if cost < min_cost:
            min_cost = cost
            best_threshold = float(thresh)
            best_breakdown = breakdown

    return best_threshold, min_cost, best_breakdown


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray | None = None,
    threshold: float = 0.5,
    verification_cost: float = 5.0,
) -> dict[str, Any]:
    """Compute comprehensive classification and business cost metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = (
        float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    )
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    metrics: dict[str, Any] = {
        "pr_auc": round(pr_auc, 5),
        "roc_auc": round(roc_auc, 5),
        "recall": round(rec, 5),
        "precision": round(prec, 5),
        "f1": round(f1, 5),
        "threshold": round(threshold, 4),
    }

    if amounts is not None:
        cost, breakdown = calculate_financial_cost(
            y_true, y_prob, amounts, threshold, verification_cost
        )
        metrics["total_cost"] = round(cost, 2)
        metrics["missed_fraud_loss"] = round(breakdown["missed_fraud_loss"], 2)
        metrics["investigation_cost"] = round(breakdown["investigation_cost"], 2)
        metrics["tp"] = breakdown["tp"]
        metrics["fp"] = breakdown["fp"]
        metrics["fn"] = breakdown["fn"]
        metrics["tn"] = breakdown["tn"]

    return metrics
