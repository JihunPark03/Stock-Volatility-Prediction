"""Create SHAP-based feature importance reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap


def save_shap_importance(model, x_test: pd.DataFrame, report_dir: str | Path) -> pd.DataFrame:
    """Save a SHAP bar chart and a CSV with mean absolute importance values."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    values = explainer(x_test)
    shap.plots.bar(values, max_display=len(x_test.columns), show=False)
    plt.title("SHAP feature importance")
    plt.tight_layout()
    plt.savefig(report_dir / "shap_feature_importance.png", dpi=150)
    plt.close()

    importance = pd.DataFrame(
        {"feature": x_test.columns, "mean_abs_shap": abs(values.values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(report_dir / "shap_feature_importance.csv", index=False)
    return importance

