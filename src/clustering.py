"""K-Means clustering to discover player archetypes from per-90 statistics."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .preprocessing import CLUSTERING_FEATURES, MIN_MINUTES


def run_clustering(data: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run K-Means clustering on per-90 player stats and visualize with PCA."""
    for folder in ("models", "tables", "plots"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    eligible = data.loc[data.Min >= MIN_MINUTES].copy()

    # Preprocessing: impute missing values and scale
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(eligible[CLUSTERING_FEATURES])
    X_scaled = scaler.fit_transform(X_imputed)

    # Find best K using elbow method + silhouette score
    candidate_rows = []
    fitted_models = {}
    for k in range(2, 11):
        model = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
        labels = model.fit_predict(X_scaled)
        inertia = model.inertia_
        score = silhouette_score(X_scaled, labels, sample_size=min(1500, len(labels)), random_state=42)
        candidate_rows.append({"k": k, "inertia": inertia, "silhouette": score})
        fitted_models[k] = model

    selection = pd.DataFrame(candidate_rows)

    # Choose K with best silhouette among K >= 3 (K=2 is too coarse)
    interpretable = selection.loc[selection.k >= 3]
    chosen_k = int(interpretable.sort_values("silhouette", ascending=False).iloc[0]["k"])
    selection["selected"] = selection.k == chosen_k
    selection.to_csv(output / "tables" / "cluster_model_selection.csv", index=False)

    # Elbow + silhouette plot
    fig, left_axis = plt.subplots(figsize=(8, 5))
    left_axis.plot(selection.k, selection.inertia, marker="o", label="Inertia (elbow)")
    left_axis.set(xlabel="Number of clusters (K)", ylabel="Inertia (lower is better)")
    right_axis = left_axis.twinx()
    right_axis.plot(selection.k, selection.silhouette, marker="^", color="#2a9d8f", label="Silhouette")
    right_axis.set_ylabel("Silhouette (higher is better)")
    left_axis.axvline(x=chosen_k, color="red", linestyle="--", alpha=0.5, label=f"Chosen K={chosen_k}")
    handles, leg_labels = left_axis.get_legend_handles_labels()
    other_handles, other_labels = right_axis.get_legend_handles_labels()
    left_axis.legend(handles + other_handles, leg_labels + other_labels, loc="best")
    left_axis.set_title("K-Means cluster selection")
    fig.tight_layout()
    fig.savefig(output / "plots" / "cluster_model_selection.png", dpi=150)
    plt.close(fig)

    # Apply chosen model
    model = fitted_models[chosen_k]
    labels = model.predict(X_scaled)

    # PCA for 2D visualization only
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Save PCA variance info
    pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
    }).to_csv(output / "tables" / "pca_explained_variance.csv", index=False)

    # Player cluster assignments
    result = eligible[["Player", "Squad", "Comp", "Pos", "Min", "Position_Category"]].copy()
    result["cluster"] = labels
    result["PC1"] = X_pca[:, 0]
    result["PC2"] = X_pca[:, 1]
    result.to_csv(output / "tables" / "player_clusters.csv", index=False)

    # Cluster profiles (mean stats per cluster)
    profiles = eligible[CLUSTERING_FEATURES].copy()
    profiles["cluster"] = labels
    profile = profiles.groupby("cluster").mean()
    profile["player_seasons"] = pd.Series(labels).value_counts().reindex(profile.index)
    profile.to_csv(output / "tables" / "cluster_profiles.csv")

    # Interpret clusters based on dominant stats
    interpretation = []
    for cluster_id, row in profile[CLUSTERING_FEATURES].iterrows():
        leaders = row.sort_values(ascending=False).head(3).index.tolist()
        # A goalkeeper cluster has near-zero goals and very high saves
        is_gk = (row.get("Gls_per90", 0) < 0.01 and row.get("Saves_per90", 0) > 2.5)
        if is_gk:
            name = "Goalkeeper profile"
        elif row.get("Gls_per90", 0) > profile["Gls_per90"].mean():
            if row.get("Ast_per90", 0) > profile["Ast_per90"].mean():
                name = "Goal scorer + playmaker"
            else:
                name = "Goal scorer"
        elif row.get("TklW_per90", 0) > profile["TklW_per90"].mean() and row.get("Int_per90", 0) > profile["Int_per90"].mean():
            name = "Defensive player"
        elif row.get("Crs_per90", 0) > profile["Crs_per90"].mean():
            name = "Wide / creative player"
        else:
            name = "All-round outfield player"
        interpretation.append({
            "cluster": int(cluster_id),
            "archetype": name,
            "top_characteristics": ", ".join(leaders),
            "note": "Post-hoc interpretation based on cluster means",
        })
    pd.DataFrame(interpretation).to_csv(output / "tables" / "cluster_interpretations.csv", index=False)

    # PCA scatter plot colored by cluster
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.65, s=24)
    ax.set(xlabel="PC1", ylabel="PC2", title=f"Player clusters (K-Means, K={chosen_k})")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(output / "plots" / "player_clusters_pca.png", dpi=160)
    plt.close(fig)

    # Save model bundle
    joblib.dump(
        {"imputer": imputer, "scaler": scaler, "pca": pca, "kmeans": model,
         "features": CLUSTERING_FEATURES, "chosen_k": chosen_k},
        output / "models" / "clustering_kmeans.joblib",
    )

    return selection, profile
