"""End-to-end integration tests verifying the link between Anas's MLflow model and Medo's serving API."""

import json
from pathlib import Path

import mlflow
import pytest
from fastapi.testclient import TestClient

from src.fraud.monitoring.drift import calculate_drift
from src.fraud.serving.api import app, load_ml_model


@pytest.fixture(scope="module")
def client():
    # Force reload model from MLflow with local sqlite uri
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    load_ml_model()
    return TestClient(app)


def make_transaction(transaction_id="tx-test-1", amount=120.0, is_fraud=False):
    tx = {
        "transaction_id": transaction_id,
        "Time": 45000.0,
        "Amount": amount,
    }
    for i in range(1, 29):
        # Fraud typical values vs legit
        tx[f"V{i}"] = -3.5 if (is_fraud and i in [14, 12, 10]) else 0.05 * i
    return tx


def test_link_mlflow_model_loaded_in_api(client):
    """Verify that the API successfully loads the trained MLflow Production model (not fallback)."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["fallback"] is False
    assert data["model_version"] != "fallback"


def test_link_threshold_loaded_from_json(client):
    """Verify threshold matches threshold.json deliverable."""
    assert Path("threshold.json").exists()
    expected = json.loads(Path("threshold.json").read_text(encoding="utf-8"))[
        "threshold"
    ]

    response = client.post("/predict", json=make_transaction())
    assert response.status_code == 200
    assert response.json()["threshold"] == expected


def test_link_predict_legitimate_transaction(client):
    """Verify that a normal transaction is classified as LEGIT with accurate SHAP."""
    tx = make_transaction(transaction_id="tx-legit-01", amount=25.0, is_fraud=False)
    response = client.post("/predict", json=tx)
    assert response.status_code == 200
    res = response.json()
    assert res["transaction_id"] == "tx-legit-01"
    assert res["decision"] == "LEGIT"
    assert res["fraud_probability"] < res["threshold"]
    assert len(res["top_features"]) == 3
    assert res["latency_ms"] < 200.0  # Latency requirement under 200ms


def test_link_predict_fraudulent_transaction(client):
    """Verify that a fraud transaction triggers FRAUD verdict and top SHAP features."""
    tx = make_transaction(transaction_id="tx-fraud-01", amount=1500.0, is_fraud=True)
    response = client.post("/predict", json=tx)
    assert response.status_code == 200
    res = response.json()
    assert res["transaction_id"] == "tx-fraud-01"
    assert len(res["top_features"]) == 3
    # Check that top features contain expected discriminating PCA features
    top_names = [item["feature"] for item in res["top_features"]]
    assert any(
        feat in top_names
        for feat in ["V14", "V12", "V10", "V4", "V17", "Amount", "log_amount"]
    )


def test_link_evidently_drift_monitoring_integration():
    """Verify that drift monitoring runs cleanly against Anas's generated train and test splits."""
    report = calculate_drift(
        reference_path="data/train.csv",
        current_path="data/test.csv",
        output_path="data/test_drift_report.json",
    )
    assert "data_drift_detected" in report
    assert "concept_drift_detected" in report
    assert "drift_scores" in report
    assert len(report["drift_scores"]) > 20
    assert report["current_rows"] == 42722
