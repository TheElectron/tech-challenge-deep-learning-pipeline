"""Data pipeline entry point.

Orchestrates download → preprocess → MLflow logging for all configured assets.

Usage
-----
    python -m src.data.pipeline
"""

import json
import logging
import sys
from pathlib import Path

import mlflow

from src import config as cfg
from src.data.ingest import download, load_raw
from src.data.preprocess import PROCESSED_DIR, process_ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def run(
    tickers: list[str],
    start: str,
    end: str,
    features: list[str],
    window_size: int,
    split_ratios: tuple[float, float, float],
) -> dict:
    """Execute the full data pipeline and log results to MLflow.

    Returns the metadata dict written to ``data/processed/metadata.json``.
    """
    logger.info("Starting data pipeline for %d ticker(s): %s", len(tickers), tickers)

    with mlflow.start_run(run_name="data_pipeline"):
        mlflow.log_params(
            {
                "tickers": ",".join(tickers),
                "start_date": start,
                "end_date": end,
                "features": ",".join(features),
                "window_size": window_size,
                "split_ratios": str(list(split_ratios)),
            }
        )

        raw_data = download(tickers, start=start, end=end)
        if not raw_data:
            logger.error("No data was downloaded. Aborting pipeline.")
            return {}

        metadata: dict = {
            "window_size": window_size,
            "features": features,
            "split_ratios": list(split_ratios),
            "start_date": start,
            "end_date": end,
            "tickers": [],
        }

        for ticker in raw_data:
            df = load_raw(ticker)
            info = process_ticker(
                df,
                ticker=ticker,
                features=features,
                window_size=window_size,
                split_ratios=split_ratios,
            )
            metadata["tickers"].append(info)

            mlflow.log_metrics(
                {
                    f"{ticker}_train_samples": info["train_samples"],
                    f"{ticker}_val_samples": info["val_samples"],
                    f"{ticker}_test_samples": info["test_samples"],
                }
            )

        metadata_path = PROCESSED_DIR / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Log lightweight artifacts (metadata + scalers only; numpy arrays stay local)
        mlflow.log_artifact(str(metadata_path))
        for scaler_path in PROCESSED_DIR.glob("*/scaler.pkl"):
            mlflow.log_artifact(
                str(scaler_path),
                artifact_path=f"scalers/{scaler_path.parent.name}",
            )

        logger.info("Pipeline complete. Metadata → %s", metadata_path)
        return metadata


if __name__ == "__main__":
    conf = cfg.load()
    data_conf = conf["data"]
    mlflow_conf = conf["mlflow"]

    mlflow.set_tracking_uri(mlflow_conf["tracking_uri"])
    mlflow.set_experiment(mlflow_conf["experiment_name"])

    run(
        tickers=data_conf["tickers"],
        start=data_conf["start_date"],
        end=data_conf["end_date"],
        features=data_conf["features"],
        window_size=data_conf["window_size"],
        split_ratios=tuple(data_conf["split_ratios"]),
    )
