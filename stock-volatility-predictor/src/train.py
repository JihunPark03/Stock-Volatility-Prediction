"""Train a LightGBM model for high-volatility risk prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier

from .data_loader import fetch_multiple_stocks
from .evaluate import evaluate_classifier
from .explain import save_shap_importance
from .features import (
    FEATURE_COLUMNS,
    OUTCOME_COLUMN,
    TARGET_COLUMN,
    add_target,
    create_feature_frame,
)


def chronological_split(frame, train_ratio: float = 0.8):
    """Split each ticker chronologically, preserving every ticker in both sets."""
    train_parts = []
    test_parts = []
    for _, ticker_frame in frame.groupby("Ticker"):
        ticker_frame = ticker_frame.sort_index()
        split_at = int(len(ticker_frame) * train_ratio)
        if split_at == 0 or split_at == len(ticker_frame):
            raise ValueError("Not enough rows to create chronological train/test splits.")
        train_parts.append(ticker_frame.iloc[:split_at])
        test_parts.append(ticker_frame.iloc[split_at:])
    return (
        pd.concat(train_parts).sort_index(),
        pd.concat(test_parts).sort_index(),
    )


def train_model(
    tickers: list[str],
    start: str,
    end: str | None,
    model_path: str | Path,
    report_dir: str | Path,
    train_ratio: float = 0.8,
) -> dict:
    """Fetch data, train the model, evaluate it, explain it, and persist it."""
    raw_data = fetch_multiple_stocks(tickers, start=start, end=end)
    featured = create_feature_frame(raw_data)
    train_frame, test_frame = chronological_split(featured, train_ratio=train_ratio)

    # Determine the target cutoff without using any test-period distribution data.
    threshold = float(train_frame[OUTCOME_COLUMN].quantile(0.75))
    train_frame = add_target(train_frame, threshold)
    test_frame = add_target(test_frame, threshold)

    x_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame[TARGET_COLUMN]
    x_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame[TARGET_COLUMN]

    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=20,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)

    metrics = evaluate_classifier(model, x_test, y_test, report_dir)
    importance = save_shap_importance(model, x_test, report_dir)
    metrics.update(
        {
            "tickers": tickers,
            "training_rows": len(train_frame),
            "test_rows": len(test_frame),
            "training_end": str(train_frame.index.max().date()),
            "test_start": str(test_frame.index.min().date()),
            "volatility_threshold": threshold,
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
            "top_shap_features": importance.head(5).to_dict(orient="records"),
        }
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "volatility_threshold": threshold,
        "tickers": tickers,
    }
    joblib.dump(bundle, model_path)

    report_dir = Path(report_dir)
    with (report_dir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=["AAPL"])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--model-path", default="models/lightgbm_volatility.joblib")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train_model(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        model_path=args.model_path,
        report_dir=args.report_dir,
        train_ratio=args.train_ratio,
    )
    print(json.dumps(result, indent=2))
