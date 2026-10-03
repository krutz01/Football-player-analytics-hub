"""Goal-count regression with leakage-conscious inputs and held-out evaluation."""

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
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import MIN_MINUTES, REGRESSION_FEATURES
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
    for folder in ("models", "tables", "plots"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    eligible = data.loc[data.Min >= MIN_MINUTES].dropna(subset=["Gls"]).copy()
    X = eligible[REGRESSION_FEATURES]
    y = eligible["Gls"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    estimators = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Lasso Regression": Lasso(alpha=0.05, max_iter=10000),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    predictions = pd.DataFrame(
        {
            "Player": eligible.loc[X_test.index, "Player"],
            "Squad": eligible.loc[X_test.index, "Squad"],
            "actual_goals": y_test,
        },
        index=X_test.index,
    )
    fitted = {}
    for name, estimator in estimators.items():
        model = _pipeline(estimator)
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "mse": "neg_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
        )
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        predictions[name] = predicted
        rows.append(
            {
                "model": name,
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
                "test_mae": float(mean_absolute_error(y_test, predicted)),
                "test_r2": float(r2_score(y_test, predicted)),
                "cv_rmse_mean": float(np.sqrt(-scores["test_mse"]).mean()),
                "cv_mae_mean": float(-scores["test_mae"].mean()),
                "cv_r2_mean": float(scores["test_r2"].mean()),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )
        fitted[name] = model
        joblib.dump(model, output / "models" / f"regression_{name.split()[0].lower()}.joblib")

    metrics = pd.DataFrame(rows).sort_values("test_rmse")
    metrics.to_csv(output / "tables" / "regression_metrics.csv", index=False)
    predictions.to_csv(output / "tables" / "regression_test_predictions.csv", index=False)

    best_name = metrics.iloc[0]["model"]
    best_model = fitted[best_name]
    best_predictions = predictions[best_name]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, best_predictions, alpha=0.5)
    limit = max(float(y_test.max()), float(best_predictions.max()))
    ax.plot([0, limit], [0, limit], "--", color="black")
    ax.set(xlabel="Actual goals", ylabel="Predicted goals", title=f"Actual vs predicted ({best_name})")
    fig.tight_layout()
    fig.savefig(output / "plots" / "regression_actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    residuals = y_test - best_predictions
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(best_predictions, residuals, alpha=0.5)
    ax.axhline(0, color="black", linestyle="--")
    ax.set(xlabel="Predicted goals", ylabel="Residual (actual - predicted)", title="Regression residuals")
    fig.tight_layout()
    fig.savefig(output / "plots" / "regression_residuals.png", dpi=150)
    plt.close(fig)

    metrics.set_index("model")[["test_rmse", "test_mae", "test_r2"]].plot(
        kind="bar", subplots=True, layout=(1, 3), legend=False, figsize=(12, 4), color="#2878b5"
    )
    plt.tight_layout()
    plt.savefig(output / "plots" / "regression_model_comparison.png", dpi=150)
    plt.close()

    importance = permutation_importance(
        best_model, X_test, y_test, n_repeats=15, random_state=42, scoring="neg_mean_absolute_error"
    )
    pd.DataFrame(
        {"feature": REGRESSION_FEATURES, "importance_mae_decrease": importance.importances_mean}
    ).sort_values("importance_mae_decrease", ascending=False).to_csv(
        output / "tables" / "regression_permutation_importance.csv", index=False
    )
    save_scatter_with_trend(eligible, "Sh", "Gls", output / "plots" / "shots_vs_goals.png", "Shots vs goals")
    save_scatter_with_trend(eligible, "Min", "Gls", output / "plots" / "minutes_vs_goals.png", "Minutes vs goals")
    save_scatter_with_trend(eligible, "Ast", "Gls", output / "plots" / "assists_vs_goals.png", "Assists vs goals (xAG unavailable)")
    return metrics
