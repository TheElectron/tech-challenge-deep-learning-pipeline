"""Phase 2 pipeline: train both LSTM configurations, evaluate, and compare.

Model configurations
--------------------

Config A — StackedLSTM
    Two sequential LSTM layers (128 → 64 hidden units) with dropout.
    ~100 k trainable parameters.

Config B — AttentionLSTM
    A 2-layer LSTM (128 units) followed by 4-head self-attention,
    residual connection, and LayerNorm.
    ~350 k trainable parameters.

Recommended choice: StackedLSTM (Config A)
-------------------------------------------
The primary reasons are directly tied to the constraints of this dataset:

1. Limited training data — with ~744 samples per asset (or ~4 000 across
   all six assets), Config B's 3.5× parameter count is a significant
   overfitting risk.  Dropout alone is rarely enough to regularise a
   model that is substantially over-parameterised relative to the data.

2. Short sequences reduce attention's advantage — multi-head attention
   shines when sequences are long (>200 steps) and only a sparse subset of
   positions is informative.  A 60-step window of daily closes is short
   enough that the LSTM hidden state captures temporal dependencies without
   needing an explicit weighting mechanism.

3. Normalisation already removes scale — because every asset is independently
   scaled to [0, 1] before training, the 60-step sequences the model sees
   are already compact and uniform.  The second LSTM layer's compression is
   sufficient for this representation.

4. Generalisation across assets — a simpler inductive bias generalises
   better when the model must handle assets with very different volatility
   profiles (e.g. AAPL vs PETR4.SA) without overfitting to the training set.

5. Production cost — Config A has 3.5× fewer parameters, smaller artifacts,
   and faster inference with no accuracy cost on this problem size.

Config B would be preferred if:
  * Sequences were > 200 steps and attention interpretability was needed.
  * Training data was 10× larger (e.g. minute-level data or 50+ assets).
  * The goal shifted to explaining *which* historical days drove predictions.

Usage
-----
    python -m src.model.pipeline
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import mlflow

from src import config as cfg
from src.model.evaluate import evaluate_all
from src.model.registry import register
from src.model.train import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

CONFIGS: dict[str, dict] = {
    "stacked_lstm": {
        "model_kwargs": {"input_size": 1, "hidden_sizes": (128, 64), "dropout": 0.2},
    },
    "attention_lstm": {
        "model_kwargs": {"input_size": 1, "hidden_size": 128, "num_heads": 4, "dropout": 0.2},
    },
}


def _print_comparison(results: dict[str, dict[str, dict[str, float]]]) -> None:
    header = f"{'Model':<20} {'Ticker':<12} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'R²':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for model_name, ticker_metrics in results.items():
        for ticker, m in ticker_metrics.items():
            print(
                f"{model_name:<20} {ticker:<12} "
                f"{m['rmse']:>8.4f} {m['mae']:>8.4f} "
                f"{m['mape']:>7.2f}% {m['r2']:>7.4f}"
            )
        print("-" * len(header))
    print()


def run(
    tickers: list[str],
    training_cfg: dict,
) -> dict[str, dict]:
    all_results: dict[str, dict] = {}

    for model_name, model_cfg in CONFIGS.items():
        logger.info("=" * 60)
        logger.info("Starting run: %s", model_name)
        logger.info("=" * 60)

        with mlflow.start_run(run_name=model_name) as active_run:
            run_id = active_run.info.run_id
            model = train(
                model_name=model_name,
                tickers=tickers,
                model_kwargs=model_cfg["model_kwargs"],
                **training_cfg,
            )
            results = evaluate_all(model, model_name, tickers, split="test")
            mean_rmse = float(np.mean([m["rmse"] for m in results.values()]))
            mlflow.log_metric("mean_test_rmse", mean_rmse)
            all_results[model_name] = results

        version = register(run_id)
        logger.info("Registry: '%s' → v%d", model_name, version)

    _print_comparison(all_results)

    report_path = Path("reports/comparison.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Comparison saved → %s", report_path)

    return all_results


if __name__ == "__main__":
    conf = cfg.load()

    mlflow.set_tracking_uri(conf["mlflow"]["tracking_uri"])
    mlflow.set_experiment(conf["mlflow"]["experiment_name"])

    with open("data/processed/metadata.json") as f:
        tickers = [t["ticker"] for t in json.load(f)["tickers"]]

    training_cfg = {
        "epochs": conf["model"]["training"]["epochs"],
        "batch_size": conf["model"]["training"]["batch_size"],
        "lr": conf["model"]["training"]["lr"],
        "patience": conf["model"]["training"]["patience"],
    }

    run(tickers=tickers, training_cfg=training_cfg)
