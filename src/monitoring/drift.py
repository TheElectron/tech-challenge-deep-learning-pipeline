"""ML monitoring via Evidently AI.

Two responsibilities:
  1. Log every live prediction to a JSONL file so that drift can be measured
     over time against a reference distribution.
  2. Generate HTML drift reports comparing current predictions to the reference.

Reference dataset
-----------------
The reference is built from model predictions on the held-out test set (all
known tickers).  Run ``scripts/build_reference.py`` once after each champion
promotion to regenerate it.

Drift report
-----------
Call ``generate_drift_report()`` to produce a timestamped HTML report in
``reports/drift/``.  This can be triggered via the ``POST /monitoring/report``
API endpoint or run on a schedule externally (cron, Airflow, etc.).

Features tracked
----------------
  last_close      — the most recent input price in the window
  predicted_close — the model's output (denormalized)
  price_mean      — mean of the 60-step input window (volatility proxy)
  price_range     — max-min of the 60-step window (spread proxy)

Tracking these four features captures both the input distribution and the
output distribution, so Evidently can flag:
  * The API is receiving requests for assets in a very different price range
    than the training distribution (input drift).
  * The model's predictions have shifted in distribution (output drift).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PREDICTIONS_LOG = Path("data/predictions_log.jsonl")
REFERENCE_PATH = Path("data/reference_predictions.csv")
REPORTS_DIR = Path("reports/drift")

DRIFT_FEATURES = ["last_close", "predicted_close", "price_mean", "price_range"]


# ---------------------------------------------------------------------------
# Prediction logging (called on every /predict request)
# ---------------------------------------------------------------------------


def log_prediction(
    ticker: str,
    close_prices: list[float],
    predicted_close: float,
) -> None:
    """Append one prediction record to the JSONL log (non-blocking, best-effort)."""
    try:
        PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "last_close": close_prices[-1],
            "predicted_close": predicted_close,
            "price_mean": sum(close_prices) / len(close_prices),
            "price_range": max(close_prices) - min(close_prices),
        }
        with open(PREDICTIONS_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not log prediction: %s", exc)


# ---------------------------------------------------------------------------
# Reference dataset (built from test-set predictions)
# ---------------------------------------------------------------------------


def build_reference(tickers: list[str], model, device) -> pd.DataFrame:
    """Generate reference predictions from the test split of each ticker.

    The resulting CSV is used as the stable reference distribution for
    all subsequent Evidently drift comparisons.
    """
    import numpy as np
    import torch

    from src.data.preprocess import load_scaler, load_split

    records = []
    for ticker in tickers:
        try:
            X, _ = load_split(ticker, "test")
            scaler = load_scaler(ticker)
        except FileNotFoundError:
            logger.warning("Skipping %s — processed data not found", ticker)
            continue

        model.eval()
        with torch.no_grad():
            preds_scaled = model(torch.from_numpy(X).to(device)).cpu().numpy()

        for i, (window_scaled, pred_scaled) in enumerate(zip(X, preds_scaled)):
            prices_scaled = window_scaled[:, 0].reshape(-1, 1)
            prices = scaler.inverse_transform(prices_scaled)[:, 0]
            pred_price = float(scaler.inverse_transform([[pred_scaled[0]]])[0, 0])

            records.append({
                "ticker": ticker,
                "last_close": float(prices[-1]),
                "predicted_close": pred_price,
                "price_mean": float(prices.mean()),
                "price_range": float(prices.max() - prices.min()),
            })

    df = pd.DataFrame(records)
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REFERENCE_PATH, index=False)
    logger.info("Reference dataset saved → %s (%d rows)", REFERENCE_PATH, len(df))
    return df


# ---------------------------------------------------------------------------
# Drift report generation
# ---------------------------------------------------------------------------


def generate_drift_report(min_current_rows: int = 10) -> Path:
    """Compare live predictions against the reference and write an HTML report.

    Returns the path to the generated report file.
    """
    from evidently import Dataset, DataDefinition
    from evidently.presets import DataDriftPreset
    from evidently import Report

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference not found at {REFERENCE_PATH}. "
            "Run `python scripts/build_reference.py` first."
        )
    if not PREDICTIONS_LOG.exists():
        raise FileNotFoundError("No live predictions logged yet.")

    reference_df = pd.read_csv(REFERENCE_PATH)[DRIFT_FEATURES]
    current_df = pd.read_json(PREDICTIONS_LOG, lines=True)[DRIFT_FEATURES]

    if len(current_df) < min_current_rows:
        raise ValueError(
            f"Only {len(current_df)} live predictions logged; "
            f"need at least {min_current_rows} for a meaningful report."
        )

    data_definition = DataDefinition(numerical_columns=DRIFT_FEATURES)
    ref_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
    cur_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"drift_{timestamp}.html"
    result.save_html(str(report_path))

    logger.info("Drift report saved → %s", report_path)
    return report_path


def load_recent_predictions(n: int = 100) -> list[dict]:
    """Return the last *n* records from the prediction log."""
    if not PREDICTIONS_LOG.exists():
        return []
    lines = PREDICTIONS_LOG.read_text().strip().splitlines()
    return [json.loads(line) for line in lines[-n:]]
