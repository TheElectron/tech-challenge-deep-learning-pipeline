"""API route handlers."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from sklearn.preprocessing import MinMaxScaler

from src.api import state as app_state
from src.api.schemas import (
    WINDOW_SIZE,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.monitoring.drift import (
    generate_drift_report,
    load_recent_predictions,
    log_prediction,
)
from src.monitoring.metrics import PREDICTION_VALUE, PREDICTIONS_TOTAL

router = APIRouter()


# ---------------------------------------------------------------------------
# Ops endpoints
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness check — confirms the API is up and the model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=app_state.get_model() is not None,
    )


@router.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded champion model."""
    info = app_state.get_info()
    if not info:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return ModelInfoResponse(**info)


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------


@router.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(request: PredictRequest) -> PredictResponse:
    """Predict the next closing price for a given asset.

    Accepts at least 60 historical closing prices (most recent last).
    Normalisation is performed on-the-fly so any asset is accepted —
    the model is not limited to tickers seen during training.
    """
    model = app_state.get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    window = np.array(request.close_prices[-WINDOW_SIZE:], dtype=np.float32).reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    window_scaled = scaler.fit_transform(window)
    X = torch.from_numpy(window_scaled.reshape(1, WINDOW_SIZE, 1))

    device = next(model.parameters()).device
    with torch.no_grad():
        pred_scaled = model(X.to(device)).cpu().numpy()

    pred_price = float(scaler.inverse_transform(pred_scaled)[0, 0])

    # Prometheus instrumentation
    PREDICTIONS_TOTAL.labels(ticker=request.ticker).inc()
    PREDICTION_VALUE.observe(pred_price)

    # Drift monitoring log
    log_prediction(
        ticker=request.ticker,
        close_prices=request.close_prices,
        predicted_close=pred_price,
    )

    info = app_state.get_info()
    return PredictResponse(
        ticker=request.ticker,
        predicted_close=round(pred_price, 4),
        model_version=info.get("version", "unknown"),
        model_alias=info.get("alias", "champion"),
        prediction_timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Monitoring endpoints
# ---------------------------------------------------------------------------


@router.get("/monitoring/predictions", tags=["monitoring"])
def recent_predictions(n: int = 50) -> list[dict]:
    """Return the last *n* logged prediction records."""
    return load_recent_predictions(n)


@router.post("/monitoring/report", tags=["monitoring"])
def drift_report() -> dict:
    """Generate an Evidently data-drift report and return its path.

    Requires at least 10 live predictions to have been logged
    and a reference dataset built via ``scripts/build_reference.py``.
    """
    try:
        path = generate_drift_report()
        return {"status": "ok", "report": str(path)}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
