"""Unsupervised Gaussian-mixture/EM archetypes and PCA visualization."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .preprocessing import CLUSTERING_FEATURES, MIN_MINUTES


def run_clustering(data: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    for folder in ("models", "tables", "plots"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    eligible = data.loc[data.Min >= MIN_MINUTES].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(eligible[CLUSTERING_FEATURES])
    X_scaled = scaler.fit_transform(X_imputed)
    pca = PCA(n_components=0.90, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled)
    pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(output / "tables" / "pca_explained_variance.csv", index=False)

    candidate_rows = []
    fitted_models = {}
    for k in range(2, 13):
        model = GaussianMixture(
            n_components=k, covariance_type="diag", reg_covar=1e-5,
            n_init=3, max_iter=500, random_state=42,
        )
        model.fit(X_pca)
        labels = model.predict(X_pca)
        score = silhouette_score(X_pca, labels, sample_size=min(1500, len(labels)), random_state=42)
        candidate_rows.append(
            {"k": k, "bic": model.bic(X_pca), "aic": model.aic(X_pca), "silhouette": score}
        )
        fitted_models[k] = model
    selection = pd.DataFrame(candidate_rows)
    # K=2 mostly separates keeper vs outfield statistical profiles; choose among
    # richer archetype sets using silhouette, then inspect BIC/AIC and profiles.
    interpretable_candidates = selection.loc[selection.k >= 3]
    chosen_k = int(interpretable_candidates.sort_values(
        ["silhouette", "bic"], ascending=[False, True]
    ).iloc[0]["k"])
    selection["selected"] = selection.k == chosen_k
    selection["selection_note"] = (
        "Chosen from K>=3 by silhouette (BIC/AIC and post-fit profile interpretation are supporting evidence); "
        "K=2 is a coarse keeper/outfield split."
    )
    selection.to_csv(output / "tables" / "cluster_model_selection.csv", index=False)
    fig, left_axis = plt.subplots(figsize=(8, 5))
    left_axis.plot(selection.k, selection.bic, marker="o", label="BIC")
    left_axis.plot(selection.k, selection.aic, marker="s", label="AIC")
    left_axis.set(xlabel="Number of mixture components (K)", ylabel="Information criterion (lower is better)")
    right_axis = left_axis.twinx()
    right_axis.plot(selection.k, selection.silhouette, marker="^", color="#2a9d8f", label="Silhouette")
    right_axis.set_ylabel("Silhouette (higher is better)")
    handles, labels_for_legend = left_axis.get_legend_handles_labels()
    other_handles, other_labels = right_axis.get_legend_handles_labels()
    left_axis.legend(handles + other_handles, labels_for_legend + other_labels, loc="best")
    left_axis.set_title("Gaussian-mixture cluster-count diagnostics")
    fig.tight_layout()
    fig.savefig(output / "plots" / "cluster_model_selection.png", dpi=150)
    plt.close(fig)
    model = fitted_models[chosen_k]
    labels = model.predict(X_pca)
    probabilities = model.predict_proba(X_pca).max(axis=1)

    result = eligible[["Player", "Squad", "Comp", "Pos", "Min", "Position_Category"]].copy()
    result["cluster"] = labels
    result["assignment_probability"] = probabilities
    result["PC1"] = X_pca[:, 0]
    result["PC2"] = X_pca[:, 1] if X_pca.shape[1] > 1 else 0.0
    result.to_csv(output / "tables" / "player_clusters.csv", index=False)

    # Interpret labels only after fitting; position is excluded from every fit/selection input.
    profiles = eligible[CLUSTERING_FEATURES].copy()
    profiles["cluster"] = labels
    profile = profiles.groupby("cluster").mean()
    profile["player_seasons"] = pd.Series(labels).value_counts().reindex(profile.index)
    profile.to_csv(output / "tables" / "cluster_profiles.csv")
    interpretation = []
    outfield = profile.drop(index=[
        cluster_id for cluster_id, row in profile.iterrows()
        if pd.notna(row["Saves_per90"]) or pd.notna(row["Save%"])
    ], errors="ignore")
    shot_cluster = outfield["Sh_per90"].idxmax() if not outfield.empty else None
    cross_cluster = outfield["Crs_per90"].idxmax() if not outfield.empty else None
    defensive_cluster = (
        outfield[["TklW_per90", "Int_per90"]].mean(axis=1).idxmax()
        if not outfield.empty else None
    )
    for cluster_id, row in profile[CLUSTERING_FEATURES].iterrows():
        leaders = row.sort_values(ascending=False).head(3).index.tolist()
        if pd.notna(profile.loc[cluster_id, "Saves_per90"]) or pd.notna(profile.loc[cluster_id, "Save%"]):
            name = "Goalkeeper statistical profile"
        elif cluster_id == shot_cluster and cluster_id == cross_cluster:
            name = "High attacking/wide-involvement profile"
        elif cluster_id == shot_cluster:
            name = "Shot-oriented attacking profile"
        elif cluster_id == cross_cluster:
            name = "Cross-heavy wide-involvement profile"
        elif cluster_id == defensive_cluster:
            name = "Defensive-action profile"
        else:
            name = "Mixed lower-volume outfield profile"
        interpretation.append(
            {"cluster": int(cluster_id), "archetype_interpretation": name,
             "top_characteristics": ", ".join(leaders),
             "interpretation_note": "Post-hoc statistical description; not a label supplied to clustering"}
        )
    pd.DataFrame(interpretation).to_csv(output / "tables" / "cluster_interpretations.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(result.PC1, result.PC2, c=labels, cmap="tab10", alpha=0.65, s=24)
    ax.set(xlabel="PC1", ylabel="PC2", title=f"Player archetypes from Gaussian mixture (K={chosen_k})")
    fig.colorbar(scatter, ax=ax, label="Discovered cluster")
    fig.tight_layout()
    fig.savefig(output / "plots" / "player_clusters_pca.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker="o")
    ax.axhline(0.90, color="grey", linestyle="--", label="90% variance threshold")
    ax.set(xlabel="Principal component", ylabel="Cumulative explained variance", ylim=(0, 1.02), title="PCA cumulative explained variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "plots" / "pca_cumulative_variance.png", dpi=150)
    plt.close(fig)

    joblib.dump(
        {"imputer": imputer, "scaler": scaler, "pca": pca, "gmm": model,
         "features": CLUSTERING_FEATURES, "chosen_k": chosen_k},
        output / "models" / "clustering_pca_gmm.joblib",
    )
    return selection, profile
