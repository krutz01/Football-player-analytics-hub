"""Linear regression to predict season goal count."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import MIN_MINUTES, GOAL_REGRESSION_FEATURES
from .visualization import save_scatter_with_trend


def _pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def run_regression(data: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Train Linear Regression to predict season goals and evaluate on held-out test set."""
    for folder in ("models", "tables", "plots"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    
    eligible = data.loc[data.Min >= MIN_MINUTES].dropna(subset=["Gls"]).copy()
    X = eligible[GOAL_REGRESSION_FEATURES]
    y = eligible["Gls"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Single model: Linear Regression
    model = _pipeline(LinearRegression())
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring={
            "mse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    
    predictions = pd.DataFrame(
        {
            "Player": eligible.loc[X_test.index, "Player"],
            "Squad": eligible.loc[X_test.index, "Squad"],
            "actual_goals": y_test,
            "predicted_goals": predicted,
        },
        index=X_test.index,
    )
    
    metrics_row = {
        "model": "Linear Regression",
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "test_mae": float(mean_absolute_error(y_test, predicted)),
        "test_r2": float(r2_score(y_test, predicted)),
        "cv_rmse_mean": float(np.sqrt(-scores["test_mse"]).mean()),
        "cv_mae_mean": float(-scores["test_mae"].mean()),
        "cv_r2_mean": float(scores["test_r2"].mean()),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    metrics = pd.DataFrame([metrics_row])
    metrics.to_csv(output / "tables" / "regression_metrics.csv", index=False)
    predictions.to_csv(output / "tables" / "regression_test_predictions.csv", index=False)
    joblib.dump(model, output / "models" / "regression_linear.joblib")
    
    # Plot: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, predicted, alpha=0.5, color="#2878b5")
    limit = max(float(y_test.max()), float(predicted.max()))
    ax.plot([0, limit], [0, limit], "--", color="black")
    ax.set(xlabel="Actual goals", ylabel="Predicted goals", title="Actual vs predicted (Linear Regression)")
    fig.tight_layout()
    fig.savefig(output / "plots" / "regression_actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    
    # Plot: Residuals
    residuals = y_test - predicted
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(predicted, residuals, alpha=0.5, color="#2878b5")
    ax.axhline(0, color="black", linestyle="--")
    ax.set(xlabel="Predicted goals", ylabel="Residual (actual - predicted)", title="Regression residuals")
    fig.tight_layout()
    fig.savefig(output / "plots" / "regression_residuals.png", dpi=150)
    plt.close(fig)
    
    # Permutation importance
    importance = permutation_importance(
        model, X_test, y_test, n_repeats=15, random_state=42, scoring="neg_mean_absolute_error"
    )
    pd.DataFrame(
        {"feature": GOAL_REGRESSION_FEATURES, "importance_mae_decrease": importance.importances_mean}
    ).sort_values("importance_mae_decrease", ascending=False).to_csv(
        output / "tables" / "regression_permutation_importance.csv", index=False
    )
    
    # Scatter plots
    save_scatter_with_trend(eligible, "Sh", "Gls", output / "plots" / "shots_vs_goals.png", "Shots vs goals")
    save_scatter_with_trend(eligible, "Min", "Gls", output / "plots" / "minutes_vs_goals.png", "Minutes vs goals")
    save_scatter_with_trend(eligible, "Ast", "Gls", output / "plots" / "assists_vs_goals.png", "Assists vs goals")
    
    return metrics
