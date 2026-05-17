"""Shared application state: champion model and its registry metadata.

Two loading modes are supported, selected by the MODEL_DIR environment variable:

  MODEL_DIR unset  →  load from the MLflow registry (development / local use)
  MODEL_DIR set    →  load from a pre-exported local directory (Docker / production)

The local-directory mode avoids any MLflow server dependency at inference time.
Run ``scripts/export_champion.py`` to populate the directory before starting
the container.
"""

import json
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_model = None
_model_info: dict = {}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def load(tracking_uri: str) -> None:
    """Load the champion model using whichever mode is configured."""
    model_dir = os.getenv("MODEL_DIR")
    if model_dir:
        _load_from_dir(Path(model_dir))
    else:
        _load_from_registry(tracking_uri)


def get_model():
    return _model


def get_info() -> dict:
    return _model_info


# ---------------------------------------------------------------------------
# Loading strategies
# ---------------------------------------------------------------------------


def _load_from_dir(model_dir: Path) -> None:
    """Load from a pre-exported directory (used inside Docker containers)."""
    global _model, _model_info

    import mlflow.pytorch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    champion_path = model_dir / "champion"

    if not champion_path.exists():
        raise FileNotFoundError(
            f"Exported champion not found at {champion_path}. "
            "Run `python scripts/export_champion.py` first."
        )

    _model = mlflow.pytorch.load_model(str(champion_path), map_location=device).to(device)
    _model.eval()

    meta_path = model_dir / "champion_meta.json"
    _model_info = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "name": "lstm-stock-predictor",
        "alias": "champion",
        "version": "exported",
        "model_type": "unknown",
        "rmse": 0.0,
    }
    logger.info(
        "Model loaded from local dir: %s (v%s, rmse=%.4f)",
        champion_path, _model_info.get("version"), _model_info.get("rmse"),
    )


def _load_from_registry(tracking_uri: str) -> None:
    """Load from the MLflow Model Registry (development mode)."""
    global _model, _model_info

    import mlflow
    from mlflow import MlflowClient
    from src.model.registry import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, load_champion

    mlflow.set_tracking_uri(tracking_uri)
    _model = load_champion()
    _model.eval()

    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    _model_info = {
        "name": REGISTERED_MODEL_NAME,
        "version": str(mv.version),
        "alias": CHAMPION_ALIAS,
        "model_type": mv.tags.get("model_type", "unknown"),
        "rmse": float(mv.tags.get("rmse", 0.0)),
    }
    logger.info(
        "Champion model loaded from registry: v%s | type=%s | rmse=%.4f",
        mv.version, _model_info["model_type"], _model_info["rmse"],
    )
