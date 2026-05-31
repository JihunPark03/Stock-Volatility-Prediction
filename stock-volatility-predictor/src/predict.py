"""Predict current high-volatility risk from a saved model."""

from __future__ import annotations

import argparse
import json

import joblib

from .data_loader import fetch_stock_data
from .features import create_feature_frame


def predict_latest(ticker: str, model_path: str) -> dict:
    """Return a risk probability for the most recent Stooq trading day."""
    bundle = joblib.load(model_path)
    raw_data = fetch_stock_data(ticker)
    featured = create_feature_frame(raw_data, require_outcome=False)
    latest = featured.iloc[-1]
    probability = float(
        bundle["model"].predict_proba(latest[bundle["features"]].to_frame().T)[0, 1]
    )
    return {
        "ticker": ticker.upper(),
        "date": str(latest.name.date()),
        "high_volatility_probability": probability,
        "predicted_high_volatility": int(probability >= 0.5),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--model-path", default="models/lightgbm_volatility.joblib")
    print(json.dumps(predict_latest(**vars(parser.parse_args())), indent=2))

