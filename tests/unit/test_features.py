"""Unit tests for feature engineering transformers."""

import numpy as np
import pandas as pd
import pytest

from src.fraud.data.features import FraudFeatureEngineer


@pytest.fixture
def sample_transaction_df():
    data = {
        "Time": [0.0, 3600.0, 72000.0, 86400.0],
        "Amount": [0.0, 10.0, 100.0, 1000.0],
    }
    for i in range(1, 29):
        data[f"V{i}"] = [float(i) * 0.1] * 4
    return pd.DataFrame(data)


def test_feature_engineer_output_shape(sample_transaction_df):
    transformer = FraudFeatureEngineer(include_hour=True, include_log_amount=True)
    transformed = transformer.fit_transform(sample_transaction_df)

    # 28 PCA features + Hour + log_amount + Amount = 31 columns
    assert transformed.shape == (4, 31)
    feature_names = transformer.get_feature_names_out()
    assert "Hour" in feature_names
    assert "log_amount" in feature_names
    assert "Amount" in feature_names
    assert len(feature_names) == 31


def test_feature_engineer_hour_values(sample_transaction_df):
    transformer = FraudFeatureEngineer(include_hour=True, include_log_amount=True)
    transformed = transformer.fit_transform(sample_transaction_df)
    names = transformer.get_feature_names_out()
    hour_idx = names.index("Hour")

    expected_hours = [0.0, 1.0, 20.0, 0.0]
    np.testing.assert_allclose(transformed[:, hour_idx], expected_hours)


def test_feature_engineer_log_amount(sample_transaction_df):
    transformer = FraudFeatureEngineer(include_hour=True, include_log_amount=True)
    transformed = transformer.fit_transform(sample_transaction_df)
    names = transformer.get_feature_names_out()
    log_amt_idx = names.index("log_amount")

    expected_log = np.log1p([0.0, 10.0, 100.0, 1000.0])
    np.testing.assert_allclose(transformed[:, log_amt_idx], expected_log, rtol=1e-5)
