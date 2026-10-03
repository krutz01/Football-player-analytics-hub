"""Explicit, leakage-aware feature definitions and data-driven diagnostics."""

from __future__ import annotations

import pandas as pd

from .preprocessing import (
    SCORING_CLASSIFICATION_FEATURES,
    CLUSTERING_FEATURES,
    GOAL_REGRESSION_FEATURES,
)


def correlation_report(data: pd.DataFrame) -> pd.DataFrame:
    """Return absolute pairwise numeric correlations for feature review."""
    numeric = data.select_dtypes(include="number")
    correlations = numeric.corr().stack().rename("correlation").reset_index()
    correlations.columns = ["feature_1", "feature_2", "correlation"]
    correlations = correlations[correlations.feature_1 < correlations.feature_2]
    correlations["absolute_correlation"] = correlations.correlation.abs()
    return correlations.sort_values("absolute_correlation", ascending=False)


def feature_lists() -> dict[str, list[str]]:
    return {
        "regression": list(GOAL_REGRESSION_FEATURES),
        "classification": list(SCORING_CLASSIFICATION_FEATURES),
        "clustering": list(CLUSTERING_FEATURES),
    }
