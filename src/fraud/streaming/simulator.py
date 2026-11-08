"""Real-time transaction streaming simulator.

Streams transactions from data/test.csv to the FastAPI serving API
and persists predictions to data/fraud.db for the Streamlit dashboard.
Can be used standalone in local mode without Redpanda/Kafka.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.fraud.storage.db import save_prediction

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fraud.simulator")

API_URL = os.getenv("FRAUD_API_URL", "http://localhost:8000/predict")
TEST_DATA_PATH = Path(os.getenv("TEST_DATA_PATH", "data/test.csv"))
DATABASE = os.getenv("FRAUD_DB", "data/fraud.db")
DEFAULT_SPEED = float(os.getenv("STREAM_SPEED", "3"))  # transactions per second


def stream_transactions(
    speed: float = DEFAULT_SPEED,
    max_count: int | None = None,
    burst_first: int = 30,
) -> None:
    """Stream transactions continuously to the API and save predictions in SQLite.

    Args:
        speed: Rate in transactions/second.
        max_count: Maximum number of transactions to stream (None = infinite loop).
        burst_first: Number of initial transactions to send immediately without delay.
    """
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found at {TEST_DATA_PATH}")

    df = pd.read_csv(TEST_DATA_PATH)
    logger.info("Loaded %d test transactions from %s", len(df), TEST_DATA_PATH)

    # Sort to bring some fraud transactions into the early stream for demo visibility
    frauds = df[df["Class"] == 1].copy()
    legits = df[df["Class"] == 0].copy()

    # Interleave 1 fraud every 15-20 transactions for live demonstration
    interleaved_rows = []
    fraud_idx = 0
    legit_idx = 0
    while legit_idx < len(legits):
        # 15 legit transactions
        for _ in range(15):
            if legit_idx < len(legits):
                interleaved_rows.append(legits.iloc[legit_idx].to_dict())
                legit_idx += 1
        # 1 fraud transaction
        if fraud_idx < len(frauds):
            interleaved_rows.append(frauds.iloc[fraud_idx].to_dict())
            fraud_idx += 1

    total_stream = len(interleaved_rows)
    logger.info(
        "Prepared interleaved stream (%d items). Streaming at %.1f tx/s...",
        total_stream,
        speed,
    )

    delay = 1.0 / max(speed, 0.1)
    count = 0

    for i, row in enumerate(interleaved_rows):
        if max_count is not None and count >= max_count:
            break

        tx_id = f"tx-{int(time.time() * 1000)}-{i:05d}"
        payload: dict[str, Any] = {
            "transaction_id": tx_id,
            "Time": float(row["Time"]),
            "Amount": float(row["Amount"]),
        }
        for v in range(1, 29):
            payload[f"V{v}"] = float(row[f"V{v}"])

        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            if resp.status_code == 200:
                pred = resp.json()
                prob = float(pred["fraud_probability"])
                decision = str(pred["decision"])
                latency = float(pred["latency_ms"])
                now_iso = datetime.now(UTC).isoformat()

                top_features = pred.get("top_features", [])
                top_feat = top_features[0]["feature"] if top_features else "-"
                shap_val = top_features[0]["shap_value"] if top_features else 0.0

                save_prediction(
                    transaction_id=tx_id,
                    probability=prob,
                    decision=decision,
                    latency_ms=latency,
                    timestamp=now_iso,
                    database=DATABASE,
                    amount=float(payload["Amount"]),
                    top_driver=f"{top_feat} ({shap_val:+.4f})",
                    shap_json=json.dumps(top_features),
                )

                if decision == "FRAUD":
                    logger.warning(
                        "🚨 [FRAUD ALERT] %s | Prob: %.4f | Amt: %.2f € | Driver: %s (shap: %.4f) | Latency: %.2f ms",
                        tx_id,
                        prob,
                        payload["Amount"],
                        top_feat,
                        shap_val,
                        latency,
                    )
                else:
                    logger.info(
                        "✅ [LEGIT] %s | Prob: %.4f | Amt: %.2f € | Driver: %s | Latency: %.2f ms",
                        tx_id,
                        prob,
                        payload["Amount"],
                        top_feat,
                        latency,
                    )

                count += 1
            else:
                logger.error("API Error %d: %s", resp.status_code, resp.text)
        except requests.RequestException as exc:
            logger.error("Connection failed: %s", exc)

        # Apply delay unless we're in the initial burst
        if i >= burst_first:
            time.sleep(delay)


if __name__ == "__main__":
    stream_transactions()
