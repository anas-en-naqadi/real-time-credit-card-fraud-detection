import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    transaction_id TEXT PRIMARY KEY,
    probability REAL NOT NULL,
    decision TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    timestamp TEXT NOT NULL,
    amount REAL,
    top_driver TEXT,
    shap_json TEXT
);
"""


def connect(database: str = "data/fraud.db") -> sqlite3.Connection:
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(SCHEMA)
    # Ensure migration if table existed previously with only 5 columns
    cur = connection.cursor()
    cur.execute("PRAGMA table_info(predictions)")
    cols = [r[1] for r in cur.fetchall()]
    if "amount" not in cols:
        cur.execute("ALTER TABLE predictions ADD COLUMN amount REAL")
    if "top_driver" not in cols:
        cur.execute("ALTER TABLE predictions ADD COLUMN top_driver TEXT")
    if "shap_json" not in cols:
        cur.execute("ALTER TABLE predictions ADD COLUMN shap_json TEXT")
    connection.commit()
    return connection


def save_prediction(
    transaction_id: str,
    probability: float,
    decision: str,
    latency_ms: float,
    timestamp: str,
    database: str = "data/fraud.db",
    amount: float | None = None,
    top_driver: str | None = None,
    shap_json: str | None = None,
) -> bool:
    """Insert once per transaction ID; return False for duplicates."""
    connection = connect(database)
    try:
        result = connection.execute(
            """INSERT OR IGNORE INTO predictions
            (transaction_id, probability, decision, latency_ms, timestamp, amount, top_driver, shap_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                transaction_id,
                probability,
                decision,
                latency_ms,
                timestamp,
                amount,
                top_driver,
                shap_json,
            ),
        )
        connection.commit()
        return result.rowcount == 1
    finally:
        connection.close()
