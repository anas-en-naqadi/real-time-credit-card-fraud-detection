"""Unit tests for financial cost evaluation and threshold optimization."""

import numpy as np

from src.fraud.models.metrics import (
    calculate_financial_cost,
    evaluate_predictions,
    find_optimal_cost_threshold,
)


def test_calculate_financial_cost():
    # 3 samples:
    # 0: True Legit, Prob 0.1 -> TN -> cost 0
    # 1: True Fraud, Prob 0.8 -> TP -> cost 5 (verification)
    # 2: True Fraud, Prob 0.2 -> FN -> cost = Amount (100)
    # 3: True Legit, Prob 0.9 -> FP -> cost 5 (verification)
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.2, 0.9])
    amounts = np.array([50.0, 200.0, 100.0, 20.0])

    cost, breakdown = calculate_financial_cost(
        y_true, y_prob, amounts, threshold=0.5, verification_cost=5.0
    )

    # Missed fraud loss = 100.0
    # Investigation cost = 5.0 * (1 TP + 1 FP) = 10.0
    # Total = 110.0
    assert cost == 110.0
    assert breakdown["tp"] == 1
    assert breakdown["fp"] == 1
    assert breakdown["fn"] == 1
    assert breakdown["tn"] == 1
    assert breakdown["missed_fraud_loss"] == 100.0


def test_find_optimal_cost_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.15, 0.6, 0.9])
    amounts = np.array([10.0, 20.0, 500.0, 300.0])

    best_thresh, min_cost, breakdown = find_optimal_cost_threshold(
        y_true, y_prob, amounts, verification_cost=5.0
    )

    # Threshold around 0.15-0.6 detects both frauds (cost 10) with 0 false positives and 0 missed frauds
    assert 0.15 <= best_thresh <= 0.6
    assert min_cost == 10.0
    assert breakdown["tp"] == 2
    assert breakdown["fn"] == 0
    assert breakdown["fp"] == 0


def test_evaluate_predictions_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2])
    amounts = np.array([10.0, 100.0, 200.0, 50.0])

    metrics = evaluate_predictions(y_true, y_prob, amounts=amounts, threshold=0.5)
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["total_cost"] == 10.0
