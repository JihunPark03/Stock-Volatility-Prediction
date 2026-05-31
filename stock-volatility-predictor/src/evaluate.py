"""Evaluate a binary volatility risk classifier."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(model, x_test, y_test, report_dir: str | Path) -> dict:
    """Calculate metrics and save a confusion matrix image."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    display = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    display.ax_.set_title("High-volatility risk confusion matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close(display.figure_)
    return metrics

