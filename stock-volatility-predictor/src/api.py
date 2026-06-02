"""FastAPI service for NASDAQ volatility risk predictions."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .predict import DEFAULT_MODEL_PATH, predict_latest

MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))


class PredictionRequest(BaseModel):
    """A batch of NASDAQ ticker symbols entered in the dashboard."""

    tickers: list[str] = Field(min_length=1, max_length=20)


class PredictionError(BaseModel):
    ticker: str
    message: str


class PredictionResponse(BaseModel):
    predictions: list[dict]
    errors: list[PredictionError]


app = FastAPI(
    title="NASDAQ Volatility Risk API",
    description="Predict high-volatility risk over the next five trading days.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Report whether the API and saved model file are available."""
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Return the latest prediction for every valid, unique ticker."""
    predictions = []
    errors = []
    seen = set()

    for raw_ticker in request.tickers:
        ticker = raw_ticker.strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        try:
            predictions.append(predict_latest(ticker, MODEL_PATH))
        except Exception as exc:  # Keep valid ticker results when one fetch fails.
            errors.append(PredictionError(ticker=ticker, message=str(exc)))

    return PredictionResponse(predictions=predictions, errors=errors)
