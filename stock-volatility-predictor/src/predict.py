"""Predict current high-volatility risk from a saved model."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib

from .data_loader import fetch_stock_data
from .features import create_feature_frame

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models/lightgbm_volatility.joblib"


@lru_cache(maxsize=4)
def load_model_bundle(model_path: str) -> dict:
    """Load and cache a model bundle so web requests do not reload it."""
    return joblib.load(model_path)


def predict_latest(ticker: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Return a risk probability for the most recent Stooq trading day."""
    ticker = ticker.strip().upper()
    bundle = load_model_bundle(str(Path(model_path).resolve()))
    raw_data = fetch_stock_data(ticker)
    featured = create_feature_frame(raw_data, require_outcome=False)
    latest = featured.iloc[-1]
    latest_features = featured[bundle["features"]].tail(1)
    probability = float(
        bundle["model"].predict_proba(latest_features)[0, 1]
    )
    return {
        "ticker": ticker.upper(),
        "date": str(latest.name.date()),
        "high_volatility_probability": probability,
        "predicted_high_volatility": int(probability >= 0.5),
        "risk_level": "High risk" if probability >= 0.5 else "Normal risk",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    print(json.dumps(predict_latest(**vars(parser.parse_args())), indent=2))
