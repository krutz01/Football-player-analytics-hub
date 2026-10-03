"""Four-class position prediction from performance statistics, not position fields."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .preprocessing import CLASSIFICATION_FEATURES, MIN_MINUTES


def _pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def run_classification(data: pd.DataFrame, output: Path) -> pd.DataFrame:
    for folder in ("models", "tables", "plots"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    eligible = data.loc[
        (data.Min >= MIN_MINUTES) & data.Position_Category.notna()
    ].copy()
    X = eligible[CLASSIFICATION_FEATURES]
    y = eligible["Position_Category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Support Vector Machine": SVC(C=2.0, kernel="rbf", class_weight="balanced"),
        "Multi-Layer Perceptron": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", alpha=0.01,
            max_iter=1200, early_stopping=False, random_state=42,
        ),
    }
    labels = ["GK", "DEF", "MID", "FWD"]
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, estimator in estimators.items():
        model = _pipeline(estimator)
        cv_scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        )
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, predicted, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, predicted, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_test, predicted, labels=labels)
        specificity = []
        for i in range(len(labels)):
            true_positive = cm[i, i]
            false_negative = cm[i, :].sum() - true_positive
            false_positive = cm[:, i].sum() - true_positive
            true_negative = cm.sum() - true_positive - false_negative - false_positive
            specificity.append(true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0)
        rows.append(
            {
                "model": name,
                "test_accuracy": float(accuracy_score(y_test, predicted)),
                "test_precision_macro": float(precision_macro),
                "test_recall_macro": float(recall_macro),
                "test_f1_macro": float(f1_macro),
                "test_specificity_macro": float(np.mean(specificity)),
                "test_precision_weighted": float(precision_weighted),
                "test_recall_weighted": float(recall_weighted),
                "test_f1_weighted": float(f1_weighted),
                "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
                "cv_f1_macro_mean": float(cv_scores["test_f1_macro"].mean()),
            }
        )
        joblib.dump(model, output / "models" / f"classification_{name.split()[0].lower()}.joblib")
        pd.DataFrame(
            classification_report(
                y_test, predicted, labels=labels, output_dict=True, zero_division=0
            )
        ).transpose().to_csv(
            output / "tables" / f"classification_{name.split()[0].lower()}_report.csv"
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(cm, cmap="Blues")
        ax.set(
            xticks=range(len(labels)), yticks=range(len(labels)),
            xticklabels=labels, yticklabels=labels,
            xlabel="Predicted position", ylabel="True position",
            title=f"Confusion matrix — {name}",
        )
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output / "plots" / f"confusion_matrix_{name.split()[0].lower()}.png", dpi=150)
        plt.close(fig)

    metrics = pd.DataFrame(rows).sort_values("test_f1_macro", ascending=False)
    metrics.to_csv(output / "tables" / "classification_metrics.csv", index=False)
    metrics.set_index("model")[["test_accuracy", "test_f1_macro", "test_specificity_macro"]].plot(
        kind="bar", figsize=(9, 5), ylim=(0, 1), rot=0
    )
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(output / "plots" / "classification_model_comparison.png", dpi=150)
    plt.close()
    eligible["Position_Category"].value_counts().reindex(["GK", "DEF", "MID", "FWD"]).plot(
        kind="bar", color="#2878b5", title="Position class distribution"
    )
    plt.ylabel("Player-seasons")
    plt.tight_layout()
    plt.savefig(output / "plots" / "classification_class_distribution.png", dpi=150)
    plt.close()
    return metrics
