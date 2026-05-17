"""Pydantic request and response schemas for the prediction API."""

from datetime import datetime
from pydantic import BaseModel, field_validator

# Must match config.yaml data.window_size
WINDOW_SIZE = 60


class PredictRequest(BaseModel):
    ticker: str
    close_prices: list[float]

    @field_validator("close_prices")
    @classmethod
    def enough_prices(cls, v: list[float]) -> list[float]:
        if len(v) < WINDOW_SIZE:
            raise ValueError(
                f"At least {WINDOW_SIZE} closing prices required, got {len(v)}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ticker": "AAPL",
                    "close_prices": [round(150 + i * 0.5, 2) for i in range(60)],
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    ticker: str
    predicted_close: float
    model_version: str
    model_alias: str
    prediction_timestamp: datetime


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    alias: str
    model_type: str
    rmse: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
