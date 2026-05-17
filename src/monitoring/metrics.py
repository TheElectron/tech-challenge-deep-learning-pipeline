"""Prometheus metrics exposed by the prediction API.

All metrics are module-level singletons — safe to import from anywhere.
The FastAPI middleware in main.py populates REQUEST_COUNT and REQUEST_LATENCY
automatically for every request.  PREDICTIONS_TOTAL, PREDICTION_VALUE, and
MODEL_RMSE are updated explicitly inside the route handlers.
"""

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "End-to-end HTTP request latency in seconds",
    ["path"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total successful predictions served",
    ["ticker"],
)

PREDICTION_VALUE = Histogram(
    "prediction_value_usd",
    "Distribution of predicted closing prices (USD / local currency)",
    buckets=[10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 2000, 5000],
)

MODEL_RMSE = Gauge(
    "model_rmse",
    "Test-set RMSE of the currently loaded champion model",
)
