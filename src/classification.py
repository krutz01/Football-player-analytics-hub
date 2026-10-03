"""Binary classification: will this player score in the next match?

Uses season-level per-90 statistics to predict scoring likelihood.
Target: goals_per_match >= 0.1 → 'Likely to Score', else 'Unlikely'.
Models: Logistic Regression and Support Vector Machine.
"""

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
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .preprocessing import MIN_MINUTES, SCORING_CLASSIFICATION_FEATURES, SCORING_THRESHOLD


def _pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def run_classification(data: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Train Logistic Regression and SVM to predict next-match scoring."""
    for folder in ("models", "tables", "plots"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    # Prepare data: outfield players with enough minutes
    eligible = data.loc[
        (data.Min >= MIN_MINUTES)
        & (data.Position_Category != "GK")
        & data.Position_Category.notna()
    ].copy()
    eligible["goals_per_match"] = eligible["Gls"] / eligible["MP"].replace(0, np.nan)
    eligible["will_score"] = (eligible["goals_per_match"] >= SCORING_THRESHOLD).astype(int)
    eligible = eligible.dropna(subset=["will_score"])

    X = eligible[SCORING_CLASSIFICATION_FEATURES]
    y = eligible["will_score"]
    labels = [0, 1]
    label_names = ["Unlikely", "Likely to Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Support Vector Machine": SVC(
            C=2.0, kernel="rbf", class_weight="balanced", probability=True
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for name, estimator in estimators.items():
        model = _pipeline(estimator)
        cv_scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["accuracy", "f1", "precision", "recall", "roc_auc"],
        )
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predicted, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_test, predicted, labels=labels)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        rows.append({
            "model": name,
            "test_accuracy": float(accuracy_score(y_test, predicted)),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "test_specificity": float(specificity),
            "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
            "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
            "cv_f1_mean": float(cv_scores["test_f1"].mean()),
            "cv_roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
        })

        # Save model
        tag = name.split()[0].lower()
        joblib.dump(model, output / "models" / f"classification_{tag}.joblib")

        # Classification report
        report = classification_report(
            y_test, predicted, labels=labels, target_names=label_names,
            output_dict=True, zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(
            output / "tables" / f"classification_{tag}_report.csv"
        )

        # Confusion matrix plot
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(cm, cmap="Blues")
        ax.set(
            xticks=range(len(label_names)), yticks=range(len(label_names)),
            xticklabels=label_names, yticklabels=label_names,
            xlabel="Predicted", ylabel="Actual",
            title=f"Confusion matrix \u2014 {name}",
        )
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output / "plots" / f"confusion_matrix_{tag}.png", dpi=150)
        plt.close(fig)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probabilities):.3f})")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title=f"ROC Curve \u2014 {name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "plots" / f"roc_curve_{tag}.png", dpi=150)
        plt.close(fig)

    metrics = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    metrics.to_csv(output / "tables" / "classification_metrics.csv", index=False)

    # Model comparison bar chart
    metrics.set_index("model")[["test_accuracy", "test_f1", "test_roc_auc"]].plot(
        kind="bar", figsize=(9, 5), ylim=(0, 1), rot=0, color=["#2878b5", "#2a9d8f", "#e76f51"]
    )
    plt.ylabel("Score")
    plt.title("Scoring prediction — model comparison")
    plt.tight_layout()
    plt.savefig(output / "plots" / "classification_model_comparison.png", dpi=150)
    plt.close()

    # Class distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    y.value_counts().reindex([0, 1]).plot(
        kind="bar", color=["#e76f51", "#2a9d8f"], ax=ax
    )
    ax.set_xticklabels(label_names, rotation=0)
    ax.set(title="Scoring class distribution (outfield, 450+ min)", ylabel="Player-seasons")
    fig.tight_layout()
    fig.savefig(output / "plots" / "classification_class_distribution.png", dpi=150)
    plt.close(fig)

    return metrics
