"""Streamlit football analytics dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    MIN_MINUTES,
    REGRESSION_FEATURES,
    clean_dataset,
    load_source_dataset,
)

st.set_page_config(page_title="Football Player Analytics", page_icon="⚽", layout="wide")
st.title("⚽ Football Player Analytics & Performance Prediction")
st.caption("2025–2026 top-five European leagues | descriptive analysis, supervised learning, and unsupervised profiling")


@st.cache_data
def load_players() -> pd.DataFrame:
    cleaned = ROOT / "data" / "processed" / "players_cleaned.csv"
    return pd.read_csv(cleaned) if cleaned.exists() else clean_dataset(load_source_dataset())


@st.cache_data
def load_clusters() -> pd.DataFrame:
    path = ROOT / "tables" / "player_clusters.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_table(relative_path: str) -> pd.DataFrame:
    path = ROOT / relative_path
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


players = load_players()
metrics = load_table("tables/regression_metrics.csv")
class_metrics = load_table("tables/classification_metrics.csv")
summary_path = ROOT / "report" / "run_summary.json"
summary = pd.read_json(summary_path, typ="series") if summary_path.exists() else pd.Series(dtype=object)
tabs = st.tabs(["Overview & EDA", "Player Analysis", "Goal Prediction", "Position Classification", "Player Profiling"])

with tabs[0]:
    st.subheader("Season overview")
    cards = st.columns(5)
    values = [
        ("Player-seasons", len(players)),
        ("Leagues", players.Comp.nunique()),
        ("Average goals", f"{players.Gls.mean():.2f}"),
        ("Average assists", f"{players.Ast.mean():.2f}"),
        ("Average age", f"{players.Age.mean():.1f}"),
    ]
    for card, (label, value) in zip(cards, values):
        card.metric(label, value)
    left, right = st.columns(2)
    for column, title, target in [
        (left, "Goals vs shots", "plots/eda_goals_vs_shots.png"),
        (right, "Goals vs shots on target", "plots/eda_goals_vs_sot.png"),
    ]:
        column.markdown(f"**{title}**")
        image = ROOT / target
        if image.exists():
            column.image(str(image), width="stretch")
    st.markdown("**Players by league**")
    st.bar_chart(players.Comp.value_counts())
    st.info(
        "This source has no xG, xAG, key-pass, progressive-action, or passing-volume fields. "
        "Charts and models use available fields only; absent values are not fabricated."
    )

with tabs[1]:
    st.subheader("Player analysis")
    choices = players.sort_values(["Player", "Squad"])
    lookup = {
        f"{row.Player} — {row.Squad} ({row.Comp})": idx
        for idx, row in choices.iterrows()
    }
    selected = st.selectbox("Select player-season", list(lookup))
    player = players.loc[lookup[selected]]
    player_clusters = load_clusters()
    cluster_record = player_clusters.loc[
        (player_clusters.Player == player.Player)
        & (player_clusters.Squad == player.Squad)
        & (player_clusters.Comp == player.Comp)
    ] if not player_clusters.empty else pd.DataFrame()
    info = st.columns(6)
    cluster_id = int(cluster_record.iloc[0].cluster) if not cluster_record.empty else None
    interpretations = load_table("tables/cluster_interpretations.csv")
    archetype = "Below modeling minutes threshold" if cluster_id is None else "Statistical profile"
    if cluster_id is not None and not interpretations.empty:
        match = interpretations.loc[interpretations.cluster == cluster_id]
        if not match.empty:
            archetype = match.iloc[0].archetype_interpretation
    for card, label, value in zip(
        info, ["Age", "Position", "League", "Minutes", "Cluster", "Archetype"],
        [player.Age, player.Pos, player.Comp, int(player.Min),
         "Not profiled" if cluster_id is None else cluster_id, archetype],
    ):
        card.metric(label, value)
    st.write(
        pd.DataFrame(
            {
                "Statistic": ["Goals", "Assists", "Shots", "Shots on target", "Crosses", "Tackles won", "Interceptions"],
                "Value": [player.Gls, player.Ast, player.Sh, player.SoT, player.Crs, player.TklW, player.Int],
            }
        ).set_index("Statistic")
    )
    st.caption("xG and xAG are not present in the supplied dataset.")

with tabs[2]:
    st.subheader("Predict season goal count")
    model_files = {
        "Linear Regression": "models/regression_linear.joblib",
        "Ridge Regression": "models/regression_ridge.joblib",
        "Lasso Regression": "models/regression_lasso.joblib",
    }
    available = [name for name, path in model_files.items() if (ROOT / path).exists()]
    if not available:
        st.warning("Run `python -m src.run_analysis` to train the prediction models.")
    else:
        player_options = ["Enter values manually"] + list(lookup)
        choice = st.selectbox("Use an existing player or enter values", player_options)
        source_player = players.loc[lookup[choice]] if choice != player_options[0] else None
        selected_model = st.selectbox("Regression model", available)
        model_metrics = metrics.loc[metrics.model == selected_model] if not metrics.empty else pd.DataFrame()
        with st.form("goal_prediction"):
            entered = {}
            columns = st.columns(3)
            for index, feature in enumerate(REGRESSION_FEATURES):
                default = float(source_player[feature]) if source_player is not None else float(players[feature].median())
                entered[feature] = columns[index % 3].number_input(
                    feature, min_value=0.0 if feature != "Age" else 14.0,
                    value=default, key=f"reg_{feature}",
                )
            submitted = st.form_submit_button("Predict goals")
        if submitted:
            model = joblib.load(ROOT / model_files[selected_model])
            prediction = max(0.0, float(model.predict(pd.DataFrame([entered]))[0]))
            st.metric("Predicted goals", f"{prediction:.1f}")
            st.caption(f"Model: {selected_model}; minimum training minutes: {MIN_MINUTES}.")
            if entered["Min"] < MIN_MINUTES:
                st.warning("This input is below the model's 450-minute training threshold; treat the prediction as an extrapolation.")
            if not model_metrics.empty:
                row = model_metrics.iloc[0]
                st.write(f"Test RMSE: {row.test_rmse:.2f} · MAE: {row.test_mae:.2f} · R²: {row.test_r2:.3f}")
        if not metrics.empty:
            st.markdown("**Holdout and cross-validation model comparison**")
            st.dataframe(metrics, width="stretch", hide_index=True)
        plot_left, plot_right = st.columns(2)
        for column, image_name, caption in [
            (plot_left, "regression_actual_vs_predicted.png", "Actual vs predicted goals"),
            (plot_right, "regression_residuals.png", "Residual diagnostics"),
        ]:
            image_path = ROOT / "plots" / image_name
            if image_path.exists():
                column.image(str(image_path), caption=caption, width="stretch")
        comparison_path = ROOT / "plots" / "regression_model_comparison.png"
        if comparison_path.exists():
            st.image(str(comparison_path), caption="Regression model comparison", width="stretch")

with tabs[3]:
    st.subheader("Broad-position classification")
    if class_metrics.empty:
        st.warning("Run `python -m src.run_analysis` to train the classifiers.")
    else:
        st.dataframe(class_metrics, width="stretch", hide_index=True)
        class_model = st.selectbox("Confusion matrix model", class_metrics.model.tolist())
        matrix_name = class_model.split()[0].lower()
        matrix_path = ROOT / "plots" / f"confusion_matrix_{matrix_name}.png"
        if matrix_path.exists():
            st.image(str(matrix_path), width="stretch")
        image = ROOT / "plots" / "classification_model_comparison.png"
        if image.exists():
            st.image(str(image), width="stretch")
        st.caption("Classes are GK, DEF, MID, and FWD. The detailed position text is never used as a model input.")

with tabs[4]:
    st.subheader("Unsupervised player profiling")
    clusters = load_clusters()
    if clusters.empty:
        st.warning("Run `python -m src.run_analysis` to discover player clusters.")
    else:
        cluster_choices = {
            f"{row.Player} — {row.Squad} ({row.Comp})": idx
            for idx, row in clusters.iterrows()
        }
        clustered_player = st.selectbox("Choose a profiled player-season", list(cluster_choices))
        record = clusters.loc[cluster_choices[clustered_player]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Discovered cluster", int(record.cluster))
        c2.metric("PCA coordinate PC1", f"{record.PC1:.2f}")
        c3.metric("PCA coordinate PC2", f"{record.PC2:.2f}")
        c4, = st.columns(1)
        c4.metric("Assignment probability", f"{record.assignment_probability:.0%}")
        image = ROOT / "plots" / "player_clusters_pca.png"
        if image.exists():
            st.image(str(image), width="stretch")
        criteria_image = ROOT / "plots" / "cluster_model_selection.png"
        if criteria_image.exists():
            st.image(str(criteria_image), width="stretch")
        profile = load_table("tables/cluster_profiles.csv")
        if not profile.empty and "cluster" in profile.columns:
            profile = profile.set_index("cluster")
            if int(record.cluster) in profile.index:
                st.write(profile.loc[[int(record.cluster)]].T.rename(columns={int(record.cluster): "Cluster mean"}))
        peers = clusters.loc[clusters.cluster == record.cluster, ["Player", "Squad", "Comp", "Pos", "Min"]]
        peers = peers.loc[
            ~((peers.Player == record.Player) & (peers.Squad == record.Squad) & (peers.Comp == record.Comp))
        ].head(10)
        st.markdown("**Other player-seasons in the same statistical group**")
        st.dataframe(peers, width="stretch", hide_index=True)
        st.caption(
            "Position labels were excluded from imputation, scaling, PCA, cluster selection, and model fitting. "
            "Cluster names/characteristics are post-hoc interpretations."
        )
