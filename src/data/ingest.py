"""Download raw OHLCV data from Yahoo Finance and persist to data/raw/."""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")


def download(
    tickers: list[str],
    start: str,
    end: str,
    output_dir: Path = RAW_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV history for each ticker and save a CSV per asset.

    Returns a dict mapping ticker → DataFrame for successfully fetched assets.
    Failed tickers are logged and skipped rather than raising.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            df = (
                yf.Ticker(ticker)
                .history(start=start, end=end, auto_adjust=True)
                .drop(columns=["Dividends", "Stock Splits"], errors="ignore")
            )
            if df.empty:
                logger.warning("No data returned for %s — skipping", ticker)
                continue

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"
            df = df.sort_index()

            path = output_dir / f"{ticker}.csv"
            df.to_csv(path)
            results[ticker] = df
            logger.info("Downloaded %d rows for %s → %s", len(df), ticker, path)

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to download %s: %s", ticker, exc)

    return results


def load_raw(ticker: str, data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load a previously downloaded CSV for *ticker*."""
    path = data_dir / f"{ticker}.csv"
    return pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
