"""Preprocessing: normalization, sequence generation, and chronological splitting.

Design decisions
----------------
* The MinMaxScaler is fit **only on the training portion** of each asset to
  prevent any future information from leaking into the model.
* One scaler is saved per asset so that predictions can be denormalized at
  inference time without knowing the original price range upfront.
* Sequences are built independently for each split, so no window ever straddles
  the train/val or val/test boundary.
* ``y`` is always the **next-day Close price** (scaled), regardless of how many
  input features are used. This is the value the LSTM must learn to predict.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chronological_split(
    series: np.ndarray,
    ratios: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a time series array without shuffling."""
    n = len(series)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    return series[:train_end], series[train_end:val_end], series[val_end:]


def _make_sequences(
    data: np.ndarray,
    window_size: int,
    target_col: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 2-D scaled array into (X, y) sliding-window pairs.

    Parameters
    ----------
    data:       shape (T, n_features)
    window_size: number of past time steps fed as input
    target_col: column index of the Close price used as the prediction target

    Returns
    -------
    X: shape (N, window_size, n_features)  — input sequences
    y: shape (N, 1)                        — next-step Close values
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size, target_col])
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32).reshape(-1, 1),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_ticker(
    df: pd.DataFrame,
    ticker: str,
    features: list[str],
    window_size: int,
    split_ratios: tuple[float, float, float],
    output_dir: Path = PROCESSED_DIR,
) -> dict:
    """Run the full preprocessing chain for a single asset.

    Saves ``X_{split}.npy``, ``y_{split}.npy``, and ``scaler.pkl`` under
    ``output_dir / ticker /``.

    Returns a summary dict suitable for logging to MLflow.
    """
    ticker_dir = output_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: columns {missing} not found in DataFrame")

    target_col = features.index("Close")
    series = df[features].values  # (T, n_features)

    train_raw, val_raw, test_raw = _chronological_split(series, split_ratios)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    splits = {
        "train": _make_sequences(train_scaled, window_size, target_col),
        "val": _make_sequences(val_scaled, window_size, target_col),
        "test": _make_sequences(test_scaled, window_size, target_col),
    }

    for split_name, (X, y) in splits.items():
        np.save(ticker_dir / f"X_{split_name}.npy", X)
        np.save(ticker_dir / f"y_{split_name}.npy", y)

    scaler_path = ticker_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    X_train, _ = splits["train"]
    logger.info(
        "%s processed — train=%d val=%d test=%d input_shape=%s",
        ticker,
        len(splits["train"][0]),
        len(splits["val"][0]),
        len(splits["test"][0]),
        X_train.shape[1:],
    )

    return {
        "ticker": ticker,
        "train_samples": int(len(splits["train"][0])),
        "val_samples": int(len(splits["val"][0])),
        "test_samples": int(len(splits["test"][0])),
        "input_shape": list(X_train.shape[1:]),
        "scaler_path": str(scaler_path),
    }


def load_split(
    ticker: str,
    split: str,
    data_dir: Path = PROCESSED_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed (X, y) arrays for a given ticker and split."""
    ticker_dir = data_dir / ticker
    X = np.load(ticker_dir / f"X_{split}.npy")
    y = np.load(ticker_dir / f"y_{split}.npy")
    return X, y


def load_scaler(ticker: str, data_dir: Path = PROCESSED_DIR) -> MinMaxScaler:
    """Load the per-asset MinMaxScaler used during preprocessing."""
    scaler_path = data_dir / ticker / "scaler.pkl"
    with open(scaler_path, "rb") as f:
        return pickle.load(f)  # noqa: S301
