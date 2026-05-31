"""Create model features and forward-looking volatility outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "daily_return",
    "volatility_5d",
    "volatility_20d",
    "ma_5d_ratio",
    "ma_20d_ratio",
    "volume_change",
    "high_low_range",
    "open_close_range",
]
OUTCOME_COLUMN = "future_volatility_5d"
TARGET_COLUMN = "high_volatility"


def _add_features_for_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features for one ticker using current and historical rows only."""
    frame = frame.sort_index().copy()
    returns = frame["Close"].pct_change()

    frame["daily_return"] = returns
    frame["volatility_5d"] = returns.rolling(5).std()
    frame["volatility_20d"] = returns.rolling(20).std()
    frame["ma_5d_ratio"] = frame["Close"] / frame["Close"].rolling(5).mean() - 1
    frame["ma_20d_ratio"] = frame["Close"] / frame["Close"].rolling(20).mean() - 1
    frame["volume_change"] = frame["Volume"].pct_change()
    frame["high_low_range"] = (frame["High"] - frame["Low"]) / frame["Close"]
    frame["open_close_range"] = (frame["Close"] - frame["Open"]) / frame["Open"]

    # At date t this uses returns from t+1 through t+5. It is an outcome only.
    frame[OUTCOME_COLUMN] = returns.rolling(5).std().shift(-5)
    return frame


def create_feature_frame(data: pd.DataFrame, require_outcome: bool = True) -> pd.DataFrame:
    """Create features ticker-by-ticker so rolling windows never cross symbols."""
    ticker_frames = []
    for ticker, ticker_frame in data.groupby("Ticker"):
        featured_ticker = _add_features_for_ticker(ticker_frame)
        featured_ticker["Ticker"] = ticker
        ticker_frames.append(featured_ticker)
    featured = pd.concat(ticker_frames).replace([np.inf, -np.inf], np.nan)
    required = FEATURE_COLUMNS + ([OUTCOME_COLUMN] if require_outcome else [])
    return featured.dropna(subset=required).sort_index()


def add_target(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Label future volatility using a threshold learned from training data."""
    labeled = frame.copy()
    labeled[TARGET_COLUMN] = (labeled[OUTCOME_COLUMN] >= threshold).astype(int)
    return labeled
