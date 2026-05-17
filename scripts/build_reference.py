"""Build the reference prediction dataset for Evidently drift monitoring.

Run this once after each champion promotion:

    python scripts/build_reference.py

The script loads the @champion model, runs it against the test split of every
known ticker, and writes the results to data/reference_predictions.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import mlflow
import torch

import src.config as cfg
from src.api import state as app_state
from src.monitoring.drift import build_reference

conf = cfg.load()

model_dir = Path("models")
if (model_dir / "champion").exists():
    # Use the pre-exported model (same as Docker)
    app_state._load_from_dir(model_dir)
else:
    mlflow.set_tracking_uri(conf["mlflow"]["tracking_uri"])
    app_state._load_from_registry(conf["mlflow"]["tracking_uri"])

model = app_state.get_model()
device = next(model.parameters()).device

with open("data/processed/metadata.json") as f:
    tickers = [t["ticker"] for t in json.load(f)["tickers"]]

print(f"Building reference from test splits: {tickers}")
df = build_reference(tickers, model, device)
print(f"Reference saved → data/reference_predictions.csv ({len(df)} rows)")
print(df.describe().to_string())
