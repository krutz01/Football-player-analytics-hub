"""Run the complete, reproducible dataset inspection and modeling workflow."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .classification import run_classification
from .clustering import run_clustering
from .feature_selection import correlation_report
from .preprocessing import (
    CLASSIFICATION_FEATURES,
    CLUSTERING_FEATURES,
    DEFAULT_DATA,
    DEFAULT_LIGHT_DATA,
    MIN_MINUTES,
    REGRESSION_FEATURES,
    clean_dataset,
    inspect_dataset,
    save_data_dictionary,
)
from .regression import run_regression
from .visualization import save_correlation_heatmap, save_scatter_with_trend


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for folder in ("models", "report", "data/processed", "plots", "tables"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    source = DEFAULT_DATA
    light_source = DEFAULT_LIGHT_DATA
    raw, dictionary = inspect_dataset(source, light_source)
    clean = clean_dataset(raw)
    save_data_dictionary(dictionary, root / "report" / "data_dictionary.csv")
    clean.to_csv(root / "data" / "processed" / "players_cleaned.csv", index=False)

    duplicates = int(raw.duplicated().sum())
    player_team_duplicates = int(raw.duplicated(["Player", "Squad", "Comp"]).sum())
    repeated_player_names = int(raw["Player"].duplicated(keep=False).sum())
    missing_pct = (raw.isna().mean() * 100).sort_values(ascending=False)
    numeric = raw.select_dtypes(include="number")
    correlations = correlation_report(raw)
    correlations.to_csv(root / "report" / "numeric_correlations.csv", index=False)
    high_correlations = correlations.loc[
        correlations.absolute_correlation >= 0.95
    ].head(50)
    min_distribution = raw["Min"].quantile([0, .1, .25, .5, .75, .9, .95, .99, 1])
    summary = {
        "source_files": [
            str(source.relative_to(root)),
            str(light_source.relative_to(root)),
        ],
        "source_join_key": ["Player", "Squad", "Comp"],
        "source_join_validation": (
            "One-to-one left merge preserving full-source row order, with both-way key coverage verified; "
            "all 2,839 player/team/competition records match, "
            "and all 53 overlapping light-source columns agree exactly with the full source."
        ),
        "rows": int(raw.shape[0]),
        "columns": int(raw.shape[1]),
        "numeric_columns": int(len(numeric.columns)),
        "categorical_columns": int(raw.shape[1] - len(numeric.columns)),
        "exact_duplicate_rows": duplicates,
        "duplicate_player_team_competition_records": player_team_duplicates,
        "rows_with_repeated_player_name": repeated_player_names,
        "unique_positions": sorted(raw.Pos.dropna().unique().tolist()),
        "unique_leagues": sorted(raw.Comp.dropna().unique().tolist()),
        "columns_missing_over_60_percent": missing_pct[missing_pct > 60].index.tolist(),
        "missing_value_pct": {str(k): round(float(v), 3) for k, v in missing_pct.items()},
        "minute_quantiles": {str(k): float(v) for k, v in min_distribution.items()},
        "minutes_threshold": MIN_MINUTES,
        "rows_meeting_minutes_threshold": int((clean.Min >= MIN_MINUTES).sum()),
        "rows_below_threshold": int((clean.Min < MIN_MINUTES).sum()),
        "threshold_rationale": (
            "450 minutes is five full matches: it limits rate-stat instability while retaining "
            "1,999 of 2,839 records (70.4%). Low-minute rows remain in the cleaned dataset."
        ),
        "available_required_fields": {
            "xG": "Absent; no xG-based experiment or xG plot is claimed.",
            "xAG": "Absent; assists are used as the closest observed creative-output alternative.",
            "key_passes": "Absent.",
            "progressive_passes_carries_receptions": "Absent.",
            "passing_volume_completion": "Absent.",
            "available_shooting": ["Gls", "Ast", "Sh", "SoT"],
            "available_defensive": ["TklW", "Int"],
            "available_misc": ["Fls", "Fld", "Off", "Crs", "CrdY", "CrdR"],
            "available_goalkeeper": ["GA90", "SoTA", "Saves", "Save%", "CS%"],
        },
        "final_feature_lists": {
            "regression": REGRESSION_FEATURES,
            "classification": CLASSIFICATION_FEATURES,
            "clustering": CLUSTERING_FEATURES,
        },
        "high_correlation_pairs_ge_0_95": high_correlations[
            ["feature_1", "feature_2", "correlation"]
        ].to_dict(orient="records"),
        "feature_selection_notes": [
            "Regression target is Gls. G+A, G-PK, G+A-PK, Gls_stats_shooting, G/Sh, G/SoT, and goals-per-90 are excluded to avoid direct target leakage.",
            "xG is absent; Sh is retained as the closest available shot-volume feature. SoT is excluded from regression because it overlaps strongly with Sh (r=0.954).",
            "Repeated FBref table columns such as Min_stats_playing_time and Gls_stats_shooting are not independent features.",
            "Identifiers, rank, nationality, squad, league, birth year, and position text are not model inputs.",
            "Classification uses observed performance fields only; Pos supplies the target through its first listed position, not as an input.",
            "Clustering uses per-90 rates for playing statistics and observed goalkeeper rates; Position_Category is never used for fitting, choosing K, or PCA.",
            "Extreme but valid players are retained; IQR outliers are not deleted because they may be genuine high-performance seasons.",
        ],
    }
    (root / "report" / "dataset_inspection.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # EDA: cohort context, outcome distributions, and observed performance relationships.
    clean["Position_Category"].value_counts().sort_index().plot(
        kind="bar", color="#2878b5", title="Players by broad position"
    )
    plt.ylabel("Player-seasons")
    plt.tight_layout()
    plt.savefig(root / "plots" / "eda_position_distribution.png", dpi=150)
    plt.close()
    clean["Comp"].value_counts().sort_values().plot(
        kind="barh", color="#2a9d8f", title="Players by league"
    )
    plt.xlabel("Player-seasons")
    plt.tight_layout()
    plt.savefig(root / "plots" / "eda_league_distribution.png", dpi=150)
    plt.close()
    clean["Age"].dropna().plot(kind="hist", bins=20, color="#2878b5", title="Age distribution")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.savefig(root / "plots" / "eda_age_distribution.png", dpi=150)
    plt.close()
    for column, title in (("Gls", "Goals"), ("Ast", "Assists"), ("Min", "Minutes")):
        clean[column].plot(kind="hist", bins=30, color="#2878b5", title=f"{title} distribution")
        plt.xlabel(column)
        plt.tight_layout()
        plt.savefig(root / "plots" / f"eda_{column.lower()}_distribution.png", dpi=150)
        plt.close()
    for x, y, title, filename in [
        ("Gls", "Sh", "Goals vs shots", "eda_goals_vs_shots.png"),
        ("Gls", "SoT", "Goals vs shots on target", "eda_goals_vs_sot.png"),
        ("Ast", "Crs", "Assists vs crosses", "eda_assists_vs_crosses.png"),
        ("Age", "Gls_per90", "Age vs goals per 90", "eda_age_vs_goals_per90.png"),
    ]:
        save_scatter_with_trend(clean, x, y, root / "plots" / filename, title)
    save_correlation_heatmap(
        clean, ["Gls", "Ast", "Sh", "SoT", "Min", "Starts", "Crs", "TklW", "Int", "Fls"],
        root / "plots" / "eda_correlation_heatmap.png",
    )
    clean.loc[clean.Min >= MIN_MINUTES].groupby("Position_Category")[
        ["Gls_per90", "Ast_per90", "Sh_per90", "TklW_per90"]
    ].mean().to_csv(root / "tables" / "position_performance_summary.csv")

    regression_metrics = run_regression(clean, root)
    classification_metrics = run_classification(clean, root)
    cluster_selection, cluster_profile = run_clustering(clean, root)
    dashboard_info = {
        "players": int(len(clean)),
        "leagues": int(clean.Comp.nunique()),
        "average_goals": float(clean.Gls.mean()),
        "average_assists": float(clean.Ast.mean()),
        "average_age": float(clean.Age.mean()),
        "minutes_threshold": MIN_MINUTES,
        "regression_best_model": str(regression_metrics.iloc[0]["model"]),
        "classification_best_model": str(classification_metrics.iloc[0]["model"]),
        "cluster_count": int(cluster_selection.loc[cluster_selection.selected, "k"].iloc[0]),
        "pca_components": int(len(pd.read_csv(root / "tables" / "pca_explained_variance.csv"))),
        "xg_available": False,
        "xag_available": False,
    }
    (root / "report" / "run_summary.json").write_text(
        json.dumps(dashboard_info, indent=2), encoding="utf-8"
    )
    print(json.dumps(dashboard_info, indent=2))
    print(f"Saved {len(dictionary)} data dictionary entries and {len(cluster_profile)} cluster profiles.")


if __name__ == "__main__":
    main()
