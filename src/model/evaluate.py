"""Model evaluation: metrics (RMSE, MAE, MAPE, R²) and prediction plots."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.preprocess import load_scaler, load_split
from src.model.train import DEVICE

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("reports/figures")


def _denormalize(values: np.ndarray, scaler) -> np.ndarray:
    """Inverse-transform scaled Close values back to original price units."""
    flat = values.ravel()
    dummy = np.zeros((len(flat), scaler.n_features_in_))
    dummy[:, 0] = flat  # Close is always the first (and often only) feature
    return scaler.inverse_transform(dummy)[:, 0]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_ticker(
    model: torch.nn.Module,
    ticker: str,
    model_name: str,
    split: str = "test",
    figures_dir: Path = FIGURES_DIR,
) -> dict[str, float]:
    """Evaluate model on one ticker's split; log metrics + plot to MLflow."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    X, y_scaled = load_split(ticker, split)
    scaler = load_scaler(ticker)

    with torch.no_grad():
        y_pred_scaled = model(torch.from_numpy(X).to(DEVICE)).cpu().numpy()

    y_true = _denormalize(y_scaled, scaler)
    y_pred = _denormalize(y_pred_scaled, scaler)
    metrics = compute_metrics(y_true, y_pred)

    logger.info(
        "[%s] %s %s | RMSE=%.4f  MAE=%.4f  MAPE=%.2f%%  R²=%.4f",
        model_name, ticker, split,
        metrics["rmse"], metrics["mae"], metrics["mape"], metrics["r2"],
    )

    # Prediction plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true, label="Actual", linewidth=1.5)
    ax.plot(y_pred, label="Predicted", linewidth=1.5, linestyle="--", alpha=0.85)
    ax.set_title(f"{model_name} | {ticker} — Actual vs Predicted ({split})")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Close price")
    ax.legend()
    fig.tight_layout()

    plot_path = figures_dir / f"{model_name}_{ticker}_{split}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    mlflow.log_artifact(str(plot_path), artifact_path="plots")
    mlflow.log_metrics({f"{ticker}_{k}": v for k, v in metrics.items()})
    return metrics


def evaluate_all(
    model: torch.nn.Module,
    model_name: str,
    tickers: list[str],
    split: str = "test",
) -> dict[str, dict[str, float]]:
    """Evaluate a model across all tickers and return a results dict."""
    return {
        ticker: evaluate_ticker(model, ticker, model_name, split)
        for ticker in tickers
    }
