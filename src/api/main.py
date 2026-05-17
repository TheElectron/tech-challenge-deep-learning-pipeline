"""FastAPI application entry point.

Usage
-----
    uvicorn src.api.main:app --reload

Interactive docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
Prometheus metrics:
    http://localhost:8000/metrics
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import src.config as cfg
from src.api import state as app_state
from src.api.routes import router
from src.monitoring.metrics import MODEL_RMSE, REQUEST_COUNT, REQUEST_LATENCY


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model on startup and publish its RMSE to Prometheus."""
    conf = cfg.load()
    app_state.load(tracking_uri=conf["mlflow"]["tracking_uri"])
    rmse = app_state.get_info().get("rmse", 0.0)
    MODEL_RMSE.set(rmse)
    yield


app = FastAPI(
    title="LSTM Stock Price Predictor",
    description=(
        "RESTful API for multi-asset stock closing price prediction. "
        "The model is a StackedLSTM trained on normalised daily Close sequences "
        "and versioned via the MLflow Model Registry."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Record request count and latency for every route except /metrics itself."""
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path

    if path != "/metrics":
        duration = time.perf_counter() - start
        REQUEST_LATENCY.labels(path=path).observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            path=path,
            status_code=str(response.status_code),
        ).inc()

    return response


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(router)
