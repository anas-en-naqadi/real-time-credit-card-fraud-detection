"""Unsupervised models: Isolation Forest for anomaly detection and K-Means clustering for profiling."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.fraud.data.features import FraudFeatureEngineer
from src.fraud.models.metrics import evaluate_predictions

logger = logging.getLogger(__name__)


def train_isolation_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    random_state: int = 42,
    contamination: float = 0.002,
) -> dict[str, Any]:
    """Train Isolation Forest on training transactions and evaluate on validation data."""
    transformer = FraudFeatureEngineer(include_hour=True, include_log_amount=True)
    scaler = StandardScaler()

    X_train_raw = train_df.drop(columns=["Class"])
    X_train_feats = scaler.fit_transform(transformer.fit_transform(X_train_raw))

    X_val_raw = val_df.drop(columns=["Class"])
    X_val_feats = scaler.transform(transformer.transform(X_val_raw))
    y_val = val_df["Class"].to_numpy()
    val_amounts = val_df["Amount"].to_numpy()

    logger.info("Fitting IsolationForest with contamination=%.4f...", contamination)
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso_forest.fit(X_train_feats)

    # Score samples: lower score means more anomalous. Invert for fraud probability proxy.
    scores = iso_forest.score_samples(X_val_feats)
    # Normalize inverted scores to [0, 1]
    inv_scores = -scores
    min_s, max_s = inv_scores.min(), inv_scores.max()
    probs = (inv_scores - min_s) / (max_s - min_s + 1e-9)

    # Predictions where anomaly label == -1
    y_pred = (iso_forest.predict(X_val_feats) == -1).astype(int)

    metrics = evaluate_predictions(y_val, probs, amounts=val_amounts, threshold=0.5)
    return {
        "model": iso_forest,
        "metrics": metrics,
        "raw_anomaly_count": int(np.sum(y_pred)),
    }


def train_kmeans_profiling(
    train_df: pd.DataFrame,
    n_clusters: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Profile transaction patterns using K-Means clustering.

    Analyzes transaction distribution and fraud concentration per cluster.
    """
    transformer = FraudFeatureEngineer(include_hour=True, include_log_amount=True)
    scaler = StandardScaler()

    X_train_raw = train_df.drop(columns=["Class"])
    X_train_feats = scaler.fit_transform(transformer.fit_transform(X_train_raw))

    logger.info("Clustering transactions into %d profiles with K-Means...", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_train_feats)

    profile_df = train_df[["Time", "Amount", "Class"]].copy()
    profile_df["Hour"] = (profile_df["Time"] // 3600) % 24
    profile_df["Cluster"] = clusters

    cluster_stats = []
    for c in range(n_clusters):
        subset = profile_df[profile_df["Cluster"] == c]
        total_tx = len(subset)
        fraud_tx = int(subset["Class"].sum())
        cluster_stats.append(
            {
                "cluster": c,
                "count": total_tx,
                "mean_amount": round(float(subset["Amount"].mean()), 2),
                "median_amount": round(float(subset["Amount"].median()), 2),
                "mean_hour": round(float(subset["Hour"].mean()), 1),
                "fraud_count": fraud_tx,
                "fraud_rate": round(float(fraud_tx / total_tx) if total_tx else 0.0, 5),
            }
        )

    return {
        "kmeans": kmeans,
        "cluster_stats": cluster_stats,
    }
