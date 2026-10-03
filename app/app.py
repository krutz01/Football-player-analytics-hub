"""Streamlit Football Analytics Dashboard — Pro-Level Sports Intelligence UI."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    MIN_MINUTES,
    GOAL_REGRESSION_FEATURES,
    SCORING_CLASSIFICATION_FEATURES,
    SCORING_THRESHOLD,
    clean_dataset,
    load_source_dataset,
)

# ── Page Config ──
st.set_page_config(
    page_title="Football Player Analytics & Performance Hub",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Archetype Mappings ──
ARCHETYPE_MAP = {
    0: {
        "name": "Goal Scorer & Playmaker",
        "badge_color": "#16C784",
        "bg_color": "rgba(22, 199, 132, 0.1)",
        "desc": "High offensive impact with frequent shots, penalty box entry, creative crossing, and attacking pressure.",
    },
    1: {
        "name": "Defensive & Work-Rate Specialist",
        "badge_color": "#3B82F6",
        "bg_color": "rgba(59, 130, 246, 0.1)",
        "desc": "Robust defensive contributions featuring high tackle recovery, pass interceptions, and defensive awareness.",
    },
    2: {
        "name": "Goalkeeper Profile",
        "badge_color": "#8B5CF6",
        "bg_color": "rgba(139, 92, 246, 0.1)",
        "desc": "Specialized shot-stopping profile focused on saves per 90, save percentages, and box protection.",
    },
    3: {
        "name": "All-Round Outfield Engine",
        "badge_color": "#F59E0B",
        "bg_color": "rgba(245, 158, 11, 0.1)",
        "desc": "Balanced transition player supporting midfield build-up, spatial coverage, and team continuity.",
    },
}

# ── Load football pitch background image and encode as base64 ──
_pitch_bg_path = Path(__file__).parent / "football_pitch_bg.jpg"
if _pitch_bg_path.exists():
    _pitch_b64 = base64.b64encode(_pitch_bg_path.read_bytes()).decode()
    _bg_image_css = f"""
    .stApp {{
        background-image: url("data:image/jpeg;base64,{_pitch_b64}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }}
    """
else:
    _bg_image_css = """
    .stApp {
        background-color: #F5F7FA !important;
    }
    """

# ── Custom CSS Theme ──
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* Global Reset & Theme */
html, body, [class*="css"], .stApp {{
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #0B1F33;
}}

/* Background with football pitch image */
{_bg_image_css}

/* ═══════════════════════════════════════════════
   TEXT VISIBILITY — high contrast dark text on light elements
   ═══════════════════════════════════════════════ */

/* Page content text */
.stApp,
.stApp .stMarkdown,
.stApp .stMarkdown p,
.stApp .stMarkdown span,
.stApp .stMarkdown li,
.stApp .stText {{
    color: #0B1F33 !important;
}}

/* Form widget labels */
.stApp label,
.stApp label p,
.stApp label span,
.stApp div[data-testid="stWidgetLabel"],
.stApp div[data-testid="stWidgetLabel"] p,
.stApp div[data-testid="stWidgetLabel"] span {{
    color: #0B1F33 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}}

/* Radio button options */
.stApp .stRadio div[role="radiogroup"] label,
.stApp .stRadio div[role="radiogroup"] label p,
.stApp .stRadio div[role="radiogroup"] label span {{
    color: #0B1F33 !important;
    font-weight: 600 !important;
}}

/* Selectbox and multiselect tags & options */
.stApp div[data-baseweb="select"] span,
.stApp div[data-baseweb="select"] div,
.stApp div[data-baseweb="select"] input {{
    color: #0B1F33 !important;
}}
.stApp div[data-baseweb="tag"] {{
    background-color: #E2E8F0 !important;
}}
.stApp div[data-baseweb="tag"] span {{
    color: #0B1F33 !important;
    font-weight: 600 !important;
}}

/* Number and text inputs */
.stApp input[type="number"],
.stApp input[type="text"],
.stApp .stNumberInput input,
.stApp .stTextInput input {{
    color: #0B1F33 !important;
    background-color: #FFFFFF !important;
    font-weight: 600 !important;
}}

/* Captions and helper subtitles */
.stApp .stCaption, 
.stApp .stCaption p {{
    color: #475569 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}}

/* Notification alerts */
.stApp .stAlert p, 
.stApp .stAlert span,
.stApp div[data-testid="stNotification"] p,
.stApp div[data-testid="stNotification"] span {{
    color: #0B1F33 !important;
}}

/* Expanders */
.stApp div[data-testid="stExpander"] summary span,
.stApp div[data-testid="stExpander"] summary p {{
    color: #0B1F33 !important;
    font-weight: 700 !important;
}}
.stApp div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] * {{
    color: #0B1F33 !important;
}}

/* Built-in Metrics */
.stApp div[data-testid="stMetric"] label {{
    color: #475569 !important;
}}
.stApp div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    color: #0B1F33 !important;
}}

/* Headings */
h1, h2, h3, .bebas-header {{
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.04em !important;
    color: #0B1F33 !important;
}}
h4, h5, h6 {{
    font-family: 'Space Grotesk', sans-serif !important;
    color: #0B1F33 !important;
    font-weight: 700 !important;
}}

.stApp strong, .stApp b {{
    color: #0B1F33 !important;
}}

/* ═══════════════════════════════════════════════
   BRAND HEADER BAR — high specificity for white text
   ═══════════════════════════════════════════════ */
.brand-header {{
    background: #0B1F33 !important;
    padding: 24px 32px !important;
    border-radius: 16px !important;
    margin-bottom: 24px !important;
    box-shadow: 0 10px 25px rgba(11, 31, 51, 0.15) !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
}}
.brand-header,
.brand-header * {{
    color: #FFFFFF !important;
}}
.brand-header .brand-title {{
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2.5rem !important;
    margin: 0 !important;
    line-height: 1 !important;
    color: #FFFFFF !important;
    letter-spacing: 0.05em !important;
}}
.brand-header .brand-subtitle {{
    font-size: 0.95rem !important;
    color: #CBD5E1 !important;
    margin-top: 6px !important;
    font-weight: 500 !important;
}}
.brand-header .header-badge {{
    background: rgba(22, 199, 132, 0.2) !important;
    border: 1px solid #16C784 !important;
    color: #16C784 !important;
    padding: 6px 16px !important;
    border-radius: 999px !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}}

/* ═══════════════════════════════════════════════
   METRIC CARDS
   ═══════════════════════════════════════════════ */
.card-metric {{
    background: rgba(255, 255, 255, 0.94) !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 14px !important;
    padding: 20px 22px !important;
    box-shadow: 0 4px 14px rgba(11, 31, 51, 0.05) !important;
    margin-bottom: 16px !important;
    backdrop-filter: blur(8px) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}}
.card-metric:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(11, 31, 51, 0.09) !important;
}}
.card-metric .val {{
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2.5rem !important;
    line-height: 1 !important;
    color: #0B1F33 !important;
}}
.card-metric .val.green {{ color: #16C784 !important; }}
.card-metric .val.blue {{ color: #3B82F6 !important; }}
.card-metric .val.amber {{ color: #F59E0B !important; }}
.card-metric .lbl {{
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-top: 6px !important;
}}
.card-metric .sub {{
    font-size: 0.8rem !important;
    color: #64748B !important;
    margin-top: 4px !important;
}}

/* ═══════════════════════════════════════════════
   PLAYER PROFILE HEADER CARD
   ═══════════════════════════════════════════════ */
.player-profile-card {{
    background: rgba(255, 255, 255, 0.94) !important;
    border: 1px solid #CBD5E1 !important;
    border-left: 6px solid #16C784 !important;
    border-radius: 14px !important;
    padding: 24px 28px !important;
    box-shadow: 0 4px 16px rgba(11, 31, 51, 0.06) !important;
    margin-bottom: 24px !important;
    backdrop-filter: blur(8px) !important;
}}
.player-name-large {{
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2.8rem !important;
    line-height: 1 !important;
    color: #0B1F33 !important;
    margin: 0 !important;
}}
.player-tags {{
    margin-top: 10px !important;
    display: flex !important;
    gap: 10px !important;
    flex-wrap: wrap !important;
    align-items: center !important;
}}
.tag-pill {{
    background: #F1F5F9 !important;
    color: #0B1F33 !important;
    padding: 5px 14px !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    border: 1px solid #CBD5E1 !important;
}}
.tag-pill.green {{
    background: rgba(22, 199, 132, 0.15) !important;
    color: #0D9488 !important;
    border-color: rgba(22, 199, 132, 0.4) !important;
}}

/* ═══════════════════════════════════════════════
   PREDICTION & LIKELIHOOD BANNERS
   ═══════════════════════════════════════════════ */
.prediction-banner {{
    background: rgba(255, 255, 255, 0.95) !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 16px !important;
    padding: 32px !important;
    text-align: center !important;
    box-shadow: 0 6px 20px rgba(11, 31, 51, 0.07) !important;
    margin: 20px 0 !important;
    position: relative !important;
    overflow: hidden !important;
    backdrop-filter: blur(8px) !important;
}}
.prediction-banner.likely {{
    border-top: 6px solid #16C784 !important;
}}
.prediction-banner.unlikely {{
    border-top: 6px solid #3B82F6 !important;
}}
.prediction-banner .title {{
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}}
.prediction-banner .number {{
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 4.5rem !important;
    line-height: 1 !important;
    margin: 12px 0 6px 0 !important;
}}
.prediction-banner.likely .number {{ color: #16C784 !important; }}
.prediction-banner.unlikely .number {{ color: #0B1F33 !important; }}
.prediction-banner .expl {{
    font-size: 1rem !important;
    color: #334155 !important;
    font-weight: 500 !important;
}}

/* Reliability Pill */
.reliability-badge {{
    display: inline-block !important;
    padding: 6px 16px !important;
    border-radius: 999px !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    margin-top: 14px !important;
}}
.reliability-badge.high {{
    background: rgba(22, 199, 132, 0.15) !important;
    color: #0D9488 !important;
    border: 1px solid rgba(22, 199, 132, 0.4) !important;
}}
.reliability-badge.low {{
    background: rgba(245, 158, 11, 0.15) !important;
    color: #D97706 !important;
    border: 1px solid rgba(245, 158, 11, 0.4) !important;
}}

/* ═══════════════════════════════════════════════
   TABS — ensure active tab has white text, inactive has dark text
   ═══════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px !important;
    background-color: rgba(255,255,255,0.95) !important;
    padding: 8px 12px !important;
    border-radius: 14px !important;
    border: 1px solid #CBD5E1 !important;
    box-shadow: 0 2px 10px rgba(11, 31, 51, 0.05) !important;
    backdrop-filter: blur(8px) !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px !important;
    padding: 10px 20px !important;
    background-color: transparent !important;
    border: none !important;
}}
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span,
.stTabs [data-baseweb="tab"] div {{
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}}
.stTabs [aria-selected="true"] {{
    background-color: #0B1F33 !important;
    box-shadow: 0 4px 12px rgba(11, 31, 51, 0.2) !important;
}}
.stTabs [aria-selected="true"] *,
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] div {{
    color: #FFFFFF !important;
    font-weight: 700 !important;
}}

/* ═══════════════════════════════════════════════
   SECTION TITLES & DESCRIPTIONS
   ═══════════════════════════════════════════════ */
.section-title {{
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.8rem !important;
    color: #0B1F33 !important;
    margin: 28px 0 8px 0 !important;
    letter-spacing: 0.03em !important;
}}
.section-desc {{
    font-size: 0.95rem !important;
    color: #475569 !important;
    margin-bottom: 20px !important;
    margin-top: -6px !important;
    font-weight: 500 !important;
}}

/* Expanders for Technical Details */
div[data-testid="stExpander"] {{
    background-color: rgba(255,255,255,0.95) !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(11, 31, 51, 0.04) !important;
    backdrop-filter: blur(8px) !important;
}}

/* Buttons */
.stButton > button {{
    background-color: #0B1F33 !important;
    border: 1px solid #0B1F33 !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 12px rgba(11, 31, 51, 0.15) !important;
    transition: all 0.2s ease !important;
}}
.stButton > button,
.stButton > button *,
.stButton > button p,
.stButton > button span,
.stButton > button div {{
    color: #FFFFFF !important;
    font-weight: 700 !important;
}}
.stButton > button:hover {{
    background-color: #16C784 !important;
    border-color: #16C784 !important;
    transform: translateY(-1px) !important;
}}
.stButton > button:hover * {{
    color: #FFFFFF !important;
}}

/* ═══════════════════════════════════════════════
   PLOTLY CHART CARD & TEXT STYLING
   ═══════════════════════════════════════════════ */
.stPlotlyChart {{
    background: #FFFFFF !important;
    border-radius: 14px !important;
    padding: 8px !important;
    border: 1px solid #CBD5E1 !important;
    box-shadow: 0 4px 14px rgba(11, 31, 51, 0.05) !important;
}}
.stPlotlyChart .xtick text,
.stPlotlyChart .ytick text,
.stPlotlyChart .xtitle text,
.stPlotlyChart .ytitle text,
.stPlotlyChart .legendtext,
.stPlotlyChart .gtitle text,
.stPlotlyChart .polargrid text,
.stPlotlyChart .axis-title text {{
    fill: #0B1F33 !important;
    color: #0B1F33 !important;
    font-weight: 600 !important;
}}
</style>
""",
    unsafe_allow_html=True,
)


def apply_chart_theme(
    fig,
    height: int = 420,
    show_legend: bool = True,
    is_polar: bool = False,
    is_gauge: bool = False,
):
    """Enforce high-contrast, fully visible dark fonts, axis titles, and ticks across all Plotly charts."""
    font_family = "Space Grotesk, sans-serif"
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=35, r=30, t=30, b=35),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(family=font_family, color="#0B1F33", size=12),
        showlegend=show_legend,
    )
    if not is_polar and not is_gauge:
        fig.update_xaxes(
            title_font=dict(family=font_family, color="#0B1F33", size=13),
            tickfont=dict(family=font_family, color="#0B1F33", size=11),
            gridcolor="#E2E8F0",
            zerolinecolor="#CBD5E1",
            linecolor="#CBD5E1",
            showline=True,
            showticklabels=True,
        )
        fig.update_yaxes(
            title_font=dict(family=font_family, color="#0B1F33", size=13),
            tickfont=dict(family=font_family, color="#0B1F33", size=11),
            gridcolor="#E2E8F0",
            zerolinecolor="#CBD5E1",
            linecolor="#CBD5E1",
            showline=True,
            showticklabels=True,
        )
    elif is_polar:
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            polar=dict(
                bgcolor="#FFFFFF",
                radialaxis=dict(
                    visible=True,
                    showticklabels=True,
                    tickfont=dict(family=font_family, color="#0B1F33", size=10),
                    gridcolor="#E2E8F0",
                    linecolor="#CBD5E1",
                ),
                angularaxis=dict(
                    tickfont=dict(family=font_family, color="#0B1F33", size=11),
                    gridcolor="#E2E8F0",
                    linecolor="#CBD5E1",
                ),
            ),
        )
    elif is_gauge:
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )

    if show_legend:
        fig.update_layout(
            legend=dict(
                font=dict(family=font_family, color="#0B1F33", size=11),
                title_font=dict(family=font_family, color="#0B1F33", size=12),
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="#E2E8F0",
                borderwidth=1,
            )
        )
    return fig


# ── Helper Component Functions ──
def render_metric_card(val: str | int | float, label: str, sub: str = "", accent: str = "") -> str:
    color_class = f" {accent}" if accent else ""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f"""
    <div class="card-metric">
        <div class="val{color_class}">{val}</div>
        <div class="lbl">{label}</div>
        {sub_html}
    </div>
    """


def render_player_header(name: str, squad: str, comp: str, pos: str, age: str | float | int) -> str:
    age_str = f"Age {int(age)}" if pd.notna(age) else "Age N/A"
    return f"""
    <div class="player-profile-card">
        <div class="player-name-large">{name}</div>
        <div class="player-tags">
            <span class="tag-pill green">⚽ {squad}</span>
            <span class="tag-pill">🏆 {comp}</span>
            <span class="tag-pill">📌 Position: {pos}</span>
            <span class="tag-pill">🎂 {age_str}</span>
        </div>
    </div>
    """


# ── Data Loading with Caching ──
@st.cache_data
def load_players() -> pd.DataFrame:
    cleaned = ROOT / "data" / "processed" / "players_cleaned.csv"
    if cleaned.exists():
        return pd.read_csv(cleaned)
    return clean_dataset(load_source_dataset())


@st.cache_data
def load_clusters() -> pd.DataFrame:
    path = ROOT / "tables" / "player_clusters.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_table(relative_path: str) -> pd.DataFrame:
    path = ROOT / relative_path
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


players = load_players()
clusters = load_clusters()
reg_metrics = load_table("tables/regression_metrics.csv")
class_metrics = load_table("tables/classification_metrics.csv")
reg_predictions = load_table("tables/regression_test_predictions.csv")
cluster_profiles = load_table("tables/cluster_profiles.csv")
cluster_selection = load_table("tables/cluster_model_selection.csv")
perm_importance = load_table("tables/regression_permutation_importance.csv")

# Build player lookup dictionary
choices = players.sort_values(["Player", "Squad"])
lookup = {
    f"{row.Player} — {row.Squad} ({row.Comp})": idx for idx, row in choices.iterrows()
}

# ── Header Banner ──
st.markdown(
    """
<div class="brand-header">
    <div>
        <div class="brand-title">⚽ FOOTBALL PLAYER ANALYTICS HUB</div>
        <div class="brand-subtitle">Pro-level performance statistics, goal scoring prediction, and player profile intelligence</div>
    </div>
    <div class="header-badge">2025–2026 Season · Top 5 Leagues</div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Navigation Tabs ──
tabs = st.tabs(
    [
        "📊 Overview",
        "👤 Player Analysis",
        "⚽ Goal Prediction",
        "🎯 Will Score?",
        "🧩 Player Types",
    ]
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━ TAB 1: OVERVIEW ━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[0]:
    st.markdown('<div class="section-title">Season Overview & Key Figures</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Key performance benchmarks aggregated across Europe\'s top 5 football leagues.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(
        render_metric_card(f"{len(players):,}", "Players Analyzed", "2,839 player records analyzed", "blue"),
        unsafe_allow_html=True,
    )
    c2.markdown(
        render_metric_card(players.Comp.nunique(), "Leagues Covered", "Premier League, La Liga, etc.", "green"),
        unsafe_allow_html=True,
    )
    c3.markdown(
        render_metric_card(f"{players.Gls.mean():.1f}", "Average Goals", "Goals per player season", "amber"),
        unsafe_allow_html=True,
    )
    c4.markdown(
        render_metric_card(f"{players.Ast.mean():.1f}", "Average Assists", "Assists per player season"),
        unsafe_allow_html=True,
    )
    c5.markdown(
        render_metric_card(f"{players.Age.mean():.1f}", "Average Age", "Squad player age average"),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Interactive Player Performance Visuals</div>', unsafe_allow_html=True)

    # Interactive Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_leagues = st.multiselect(
            "Filter by League",
            options=sorted(players.Comp.dropna().unique().tolist()),
            default=sorted(players.Comp.dropna().unique().tolist()),
            key="overview_league_filter",
        )
    with col_f2:
        selected_positions = st.multiselect(
            "Filter by Position Category",
            options=sorted(players.Position_Category.dropna().unique().tolist()),
            default=sorted(players.Position_Category.dropna().unique().tolist()),
            key="overview_pos_filter",
        )

    filtered_players = players[
        players.Comp.isin(selected_leagues) & players.Position_Category.isin(selected_positions)
    ]

    left, right = st.columns(2)

    # Plot 1: Goals vs Shots
    with left:
        st.markdown("### Do More Shots Lead to More Goals?")
        st.caption("Each point represents a player season. Hover over points to explore individual stats.")

        fig_shots = px.scatter(
            filtered_players,
            x="Sh",
            y="Gls",
            color="Position_Category",
            size=filtered_players["Min"].clip(lower=200),
            hover_name="Player",
            hover_data={
                "Squad": True,
                "Comp": True,
                "Pos": True,
                "Gls": True,
                "Sh": True,
                "Ast": True,
                "Min": True,
            },
            labels={
                "Sh": "Total Shots Attempted",
                "Gls": "Goals Scored",
                "Position_Category": "Position Group",
            },
            color_discrete_map={
                "FWD": "#16C784",
                "MID": "#3B82F6",
                "DEF": "#F59E0B",
                "GK": "#8B5CF6",
            },
        )
        apply_chart_theme(fig_shots, height=420)
        st.plotly_chart(fig_shots, use_container_width=True, theme=None)

    # Plot 2: Goals vs Shots on Target
    with right:
        st.markdown("### How Does Accuracy Relate to Goals?")
        st.caption("Compare total goals scored with shots that actually reached the target.")

        fig_sot = px.scatter(
            filtered_players,
            x="SoT",
            y="Gls",
            color="Position_Category",
            hover_name="Player",
            hover_data={
                "Squad": True,
                "Comp": True,
                "Gls": True,
                "SoT": True,
                "Sh": True,
                "Min": True,
            },
            labels={
                "SoT": "Shots on Target",
                "Gls": "Goals Scored",
                "Position_Category": "Position Group",
            },
            color_discrete_map={
                "FWD": "#16C784",
                "MID": "#3B82F6",
                "DEF": "#F59E0B",
                "GK": "#8B5CF6",
            },
        )
        apply_chart_theme(fig_sot, height=420)
        st.plotly_chart(fig_sot, use_container_width=True, theme=None)

    col_b1, col_b2 = st.columns(2)

    # Plot 3: Players by League
    with col_b1:
        st.markdown("### Where Are Our Players Coming From?")
        st.caption("Player-season count breakdown across Europe's top 5 domestic leagues.")

        league_counts = filtered_players["Comp"].value_counts().reset_index()
        league_counts.columns = ["League", "Count"]

        fig_league = px.bar(
            league_counts,
            x="League",
            y="Count",
            color="League",
            text="Count",
            color_discrete_sequence=["#0B1F33", "#16C784", "#3B82F6", "#F59E0B", "#8B5CF6"],
        )
        apply_chart_theme(fig_league, height=360, show_legend=False)
        fig_league.update_traces(textposition="outside")
        st.plotly_chart(fig_league, use_container_width=True, theme=None)

    # Plot 4: Goals by Position Boxplot
    with col_b2:
        st.markdown("### Goal Production Across Positions")
        st.caption("Distribution of goals scored across outfield and goalkeeper positions.")

        fig_box = px.box(
            filtered_players,
            x="Position_Category",
            y="Gls",
            color="Position_Category",
            points="outliers",
            labels={"Position_Category": "Position", "Gls": "Goals Scored"},
            color_discrete_map={
                "FWD": "#16C784",
                "MID": "#3B82F6",
                "DEF": "#F59E0B",
                "GK": "#8B5CF6",
            },
        )
        apply_chart_theme(fig_box, height=360, show_legend=False)
        st.plotly_chart(fig_box, use_container_width=True, theme=None)

    # Technical Details Expander
    with st.expander("🔬 Technical Details & Dataset Architecture"):
        st.markdown(
            """
        **Dataset Ingestion & Preprocessing Summary:**
        - **Raw Sources**: `players_data-2025_2026.csv` (102 features) merged with `players_data_light-2025_2026.csv` (53 features).
        - **Record Coverage**: 2,839 unique player-team-competition records. Zero exact duplicate rows.
        - **Minutes Qualification Threshold**: Minimum 450 minutes (5 full matches) required for supervised modeling to filter extreme rate noise (1,999 qualifying records).
        - **Derived Per-90 Metrics**: Per-90 metrics calculated as `(Stat / Min) * 90` for fair player comparisons.
        """
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━ TAB 2: PLAYER ANALYSIS ━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[1]:
    st.markdown('<div class="section-title">Individual Player Profile</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Search and select any player to inspect their key stats, scoring rates, and performance footprint.</div>',
        unsafe_allow_html=True,
    )

    selected_player_key = st.selectbox("Select Player", list(lookup), key="player_tab_select")
    player_row = players.loc[lookup[selected_player_key]]

    # Render Header Card
    st.markdown(
        render_player_header(
            player_row.Player,
            player_row.Squad,
            player_row.Comp,
            player_row.Pos,
            player_row.Age,
        ),
        unsafe_allow_html=True,
    )

    # Primary Metric Summary Cards
    gpm = player_row.Gls / max(player_row.MP, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        render_metric_card(int(player_row.Gls), "Goals Scored", "Season total", "green"),
        unsafe_allow_html=True,
    )
    c2.markdown(
        render_metric_card(int(player_row.Ast), "Assists", "Goal setups", "blue"),
        unsafe_allow_html=True,
    )
    c3.markdown(
        render_metric_card(int(player_row.Sh), "Shots Attempted", "Total goal attempts"),
        unsafe_allow_html=True,
    )
    c4.markdown(
        render_metric_card(int(player_row.SoT), "Shots on Target", "On-frame attempts"),
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        render_metric_card(f"{int(player_row.Min):,}", "Minutes Played", "Total pitch duration"),
        unsafe_allow_html=True,
    )
    c2.markdown(
        render_metric_card(int(player_row.MP), "Matches Played", "Appearances this season"),
        unsafe_allow_html=True,
    )

    # Friendly rate wording
    rate_desc = (
        f"Averages 1 goal every {1/gpm:.1f} matches"
        if gpm > 0
        else "No goals recorded yet this season"
    )
    c3.markdown(
        render_metric_card(f"{gpm:.2f}", "Goals Per Match", rate_desc, "green" if gpm >= 0.25 else ""),
        unsafe_allow_html=True,
    )

    shots_per_game = player_row.Sh / max(player_row.MP, 1)
    c4.markdown(
        render_metric_card(f"{shots_per_game:.1f}", "Shots Per Match", "Shooting frequency"),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Performance Footprint (Per 90 Metrics)</div>', unsafe_allow_html=True)
    st.caption("Compare this player's per-90 statistics against the overall league average.")

    # Radar Chart of Per-90 Statistics
    p90_metrics = {
        "Goals / 90": player_row.get("Gls_per90", (player_row.Gls / max(player_row.Min, 1)) * 90),
        "Assists / 90": player_row.get("Ast_per90", (player_row.Ast / max(player_row.Min, 1)) * 90),
        "Shots / 90": player_row.get("Sh_per90", (player_row.Sh / max(player_row.Min, 1)) * 90),
        "On Target / 90": player_row.get("SoT_per90", (player_row.SoT / max(player_row.Min, 1)) * 90),
        "Crosses / 90": player_row.get("Crs_per90", (player_row.Crs / max(player_row.Min, 1)) * 90),
        "Tackles Won / 90": player_row.get("TklW_per90", (player_row.TklW / max(player_row.Min, 1)) * 90),
        "Interceptions / 90": player_row.get("Int_per90", (player_row.Int / max(player_row.Min, 1)) * 90),
        "Fouls Drawn / 90": player_row.get("Fld_per90", (player_row.Fld / max(player_row.Min, 1)) * 90),
    }

    avg_p90_metrics = {
        "Goals / 90": players["Gls_per90"].mean() if "Gls_per90" in players.columns else 0.1,
        "Assists / 90": players["Ast_per90"].mean() if "Ast_per90" in players.columns else 0.1,
        "Shots / 90": players["Sh_per90"].mean() if "Sh_per90" in players.columns else 1.0,
        "On Target / 90": players["SoT_per90"].mean() if "SoT_per90" in players.columns else 0.4,
        "Crosses / 90": players["Crs_per90"].mean() if "Crs_per90" in players.columns else 0.8,
        "Tackles Won / 90": players["TklW_per90"].mean() if "TklW_per90" in players.columns else 0.8,
        "Interceptions / 90": players["Int_per90"].mean() if "Int_per90" in players.columns else 0.6,
        "Fouls Drawn / 90": players["Fld_per90"].mean() if "Fld_per90" in players.columns else 0.8,
    }

    categories = list(p90_metrics.keys())
    player_vals = [float(p90_metrics[k]) for k in categories]
    avg_vals = [float(avg_p90_metrics[k]) for k in categories]

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=player_vals + [player_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=player_row.Player,
            line_color="#16C784",
            fillcolor="rgba(22, 199, 132, 0.2)",
        )
    )
    fig_radar.add_trace(
        go.Scatterpolar(
            r=avg_vals + [avg_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="League Average",
            line_color="#94A3B8",
            fillcolor="rgba(148, 163, 184, 0.15)",
        )
    )
    apply_chart_theme(fig_radar, height=450, is_polar=True)
    st.plotly_chart(fig_radar, use_container_width=True, theme=None)

    # Detailed Action Breakdown
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        st.markdown("### 🎯 Attacking Actions")
        att_df = pd.DataFrame(
            [
                {"Metric": "Goals", "Value": int(player_row.Gls)},
                {"Metric": "Assists", "Value": int(player_row.Ast)},
                {"Metric": "Shots", "Value": int(player_row.Sh)},
                {"Metric": "Shots on Target", "Value": int(player_row.SoT)},
                {"Metric": "Crosses", "Value": int(player_row.Crs)},
                {"Metric": "Offsides", "Value": int(player_row.Off)},
            ]
        )
        st.dataframe(att_df, use_container_width=True, hide_index=True)

    with col_act2:
        st.markdown("### 🛡️ Defensive & Physical Actions")
        def_df = pd.DataFrame(
            [
                {"Metric": "Tackles Won", "Value": int(player_row.TklW)},
                {"Metric": "Interceptions", "Value": int(player_row.Int)},
                {"Metric": "Fouls Drawn", "Value": int(player_row.Fld)},
                {"Metric": "Fouls Committed", "Value": int(player_row.Fls)},
                {"Metric": "Yellow Cards", "Value": int(player_row.CrdY)},
                {"Metric": "Red Cards", "Value": int(player_row.CrdR)},
            ]
        )
        st.dataframe(def_df, use_container_width=True, hide_index=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━ TAB 3: GOAL PREDICTION ━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[2]:
    st.markdown('<div class="section-title">Expected Season Goals Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Forecast how many goals a player will score in a full season based on their performance metrics and playing time.</div>',
        unsafe_allow_html=True,
    )

    model_path = ROOT / "models" / "regression_linear.joblib"
    if not model_path.exists():
        st.warning("⚠️ Machine learning models are currently building. Run `python -m src.run_analysis` to generate model files.")
    else:
        # Sample Ronaldo stats for quick testing
        RONALDO_STATS = {
            "Age": 39.0,
            "Min": 2880.0,
            "Starts": 32.0,
            "Sh": 120.0,
            "Ast": 3.0,
            "Crs": 15.0,
            "TklW": 5.0,
            "Int": 4.0,
            "Fld": 45.0,
            "Fls": 30.0,
            "Off": 18.0,
            "CrdY": 4.0,
            "CrdR": 0.0,
        }

        input_mode = st.radio(
            "Select Prediction Mode",
            ["Select a Player", "Manual Stat Input", "🧪 Test with Cristiano Ronaldo's Data"],
            horizontal=True,
            key="reg_input_mode_v2",
        )

        if input_mode == "Select a Player":
            reg_player_sel = st.selectbox("Choose Player", list(lookup), key="reg_player_sel_v2")
            source_p = players.loc[lookup[reg_player_sel]]
            defaults = {f: float(source_p[f]) for f in GOAL_REGRESSION_FEATURES}
        elif input_mode == "🧪 Test with Cristiano Ronaldo's Data":
            st.info("📋 **Cristiano Ronaldo** — Estimated season statistics (2024-25 sample)")
            defaults = RONALDO_STATS.copy()
            reg_player_sel = "Ronaldo"
        else:
            defaults = {f: float(players[f].median()) for f in GOAL_REGRESSION_FEATURES}
            reg_player_sel = "Custom"

        st.markdown("### Player Performance Inputs")

        # Organize inputs into intuitive cards
        col_in1, col_in2, col_in3 = st.columns(3)

        entered = {}
        with col_in1:
            st.markdown("#### ⏱️ Playing Time")
            entered["Min"] = st.number_input("Minutes Played", min_value=0.0, value=defaults["Min"], step=90.0)
            entered["Starts"] = st.number_input("Matches Started", min_value=0.0, value=defaults["Starts"], step=1.0)
            entered["Age"] = st.number_input("Player Age", min_value=15.0, max_value=45.0, value=defaults["Age"], step=1.0)

        with col_in2:
            st.markdown("#### 🎯 Attacking Volume")
            entered["Sh"] = st.number_input("Shots Attempted", min_value=0.0, value=defaults["Sh"], step=1.0)
            entered["Ast"] = st.number_input("Assists", min_value=0.0, value=defaults["Ast"], step=1.0)
            entered["Crs"] = st.number_input("Crosses", min_value=0.0, value=defaults["Crs"], step=1.0)
            entered["Off"] = st.number_input("Offsides", min_value=0.0, value=defaults["Off"], step=1.0)

        with col_in3:
            st.markdown("#### 🛡️ Defense & Discipline")
            entered["TklW"] = st.number_input("Tackles Won", min_value=0.0, value=defaults["TklW"], step=1.0)
            entered["Int"] = st.number_input("Interceptions", min_value=0.0, value=defaults["Int"], step=1.0)
            entered["Fld"] = st.number_input("Fouls Drawn", min_value=0.0, value=defaults["Fld"], step=1.0)
            entered["Fls"] = st.number_input("Fouls Committed", min_value=0.0, value=defaults["Fls"], step=1.0)
            entered["CrdY"] = st.number_input("Yellow Cards", min_value=0.0, value=defaults["CrdY"], step=1.0)
            entered["CrdR"] = st.number_input("Red Cards", min_value=0.0, value=defaults["CrdR"], step=1.0)

        # Load Model and Predict — reorder columns to match exact training feature order
        reg_model = joblib.load(model_path)
        ordered_entered = {f: entered[f] for f in GOAL_REGRESSION_FEATURES}
        input_df = pd.DataFrame([ordered_entered])
        predicted_goals = max(0.0, float(reg_model.predict(input_df)[0]))

        # Large Prominent Result Banner
        reliability_class = "high" if entered["Min"] >= MIN_MINUTES else "low"
        reliability_text = (
            "✅ High Prediction Reliability — 450+ minutes of active match data"
            if entered["Min"] >= MIN_MINUTES
            else "⚠️ Low Minutes Warning — under 450 minutes played, prediction has higher variance"
        )

        st.markdown(
            f"""
        <div class="prediction-banner likely">
            <div class="title">EXPECTED SEASON GOALS</div>
            <div class="number">{predicted_goals:.1f} GOALS</div>
            <div class="expl">Forecasted output based on current performance stats across {int(entered['Min'])} active match minutes.</div>
            <div class="reliability-badge {reliability_class}">{reliability_text}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Interactive Model Accuracy Scatter Chart
        st.markdown('<div class="section-title">Prediction Accuracy Across Dataset</div>', unsafe_allow_html=True)
        st.caption("How closely predicted goal counts match actual observed goals across 400 held-out test players.")

        if not reg_predictions.empty:
            fig_acc = px.scatter(
                reg_predictions,
                x="actual_goals",
                y="predicted_goals",
                hover_name="Player",
                hover_data={"Squad": True, "actual_goals": True, "predicted_goals": True},
                labels={"actual_goals": "Actual Season Goals", "predicted_goals": "Predicted Season Goals"},
                color_discrete_sequence=["#3B82F6"],
            )

            # Add 1:1 Identity Reference Line
            max_val = max(reg_predictions["actual_goals"].max(), reg_predictions["predicted_goals"].max())
            fig_acc.add_shape(
                type="line",
                line=dict(dash="dash", color="#16C784", width=2),
                x0=0,
                y0=0,
                x1=max_val,
                y1=max_val,
            )

            apply_chart_theme(fig_acc, height=420, show_legend=False)
            st.plotly_chart(fig_acc, use_container_width=True, theme=None)

        # Expandable Technical Details Section
        with st.expander("🔬 Technical Details & Model Validation"):
            st.markdown(
                """
            #### Model Architecture & Evaluation
            - **Algorithm**: Ordinary Least Squares Linear Regression (with Ridge & Lasso regularization benchmarks).
            - **Target Variable**: Season Total Goals (`Gls`).
            - **Leakage Prevention**: Excludes xG (absent in dataset), penalty goal totals, derived ratios (`G+A`, `G-PK`), and collinear shot-on-target duplicates (`SoT`).
            """
            )

            if not reg_metrics.empty:
                m_row = reg_metrics.iloc[0]
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("Test RMSE (Goal Error)", f"{m_row.test_rmse:.2f} goals")
                tc2.metric("Test MAE (Mean Absolute Error)", f"{m_row.test_mae:.2f} goals")
                tc3.metric("Test R² Variance Explained", f"{m_row.test_r2:.3f}")

            if not perm_importance.empty:
                st.markdown("#### Feature Permutation Importance")
                st.dataframe(perm_importance, use_container_width=True, hide_index=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━ TAB 4: WILL SCORE? ━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[3]:
    st.markdown('<div class="section-title">Next Match Scoring Probability</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Assess the likelihood of a player scoring in an upcoming fixture based on their per-90 rate profile.</div>',
        unsafe_allow_html=True,
    )

    model_files = {
        "Support Vector Machine (Recommended)": "models/classification_support.joblib",
        "Logistic Regression": "models/classification_logistic.joblib",
    }
    available_models = [n for n, p in model_files.items() if (ROOT / p).exists()]

    if not available_models:
        st.warning("⚠️ Classification models not found. Run `python -m src.run_analysis` to build model binaries.")
    else:
        col_s1, col_s2 = st.columns([3, 1])
        with col_s1:
            score_player_sel = st.selectbox("Select Player", list(lookup), key="scoring_player_select_v2")
        with col_s2:
            selected_model_name = st.selectbox("Prediction Engine", available_models, key="scoring_model_select_v2")

        player_data = players.loc[lookup[score_player_sel]]

        # Player Header
        st.markdown(
            render_player_header(
                player_data.Player,
                player_data.Squad,
                player_data.Comp,
                player_data.Pos,
                player_data.Age,
            ),
            unsafe_allow_html=True,
        )

        # Key Rates Summary
        gpm_val = player_data.Gls / max(player_data.MP, 1)
        sh90_val = player_data.get("Sh_per90", (player_data.Sh / max(player_data.Min, 1)) * 90)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            render_metric_card(int(player_data.Gls), "Season Goals", "Goal tally", "green"),
            unsafe_allow_html=True,
        )
        c2.markdown(
            render_metric_card(f"{gpm_val:.2f}", "Goals Per Match", "Scoring rate per appearance"),
            unsafe_allow_html=True,
        )
        c3.markdown(
            render_metric_card(f"{sh90_val:.2f}", "Shots Per 90", "Attempts per 90 mins"),
            unsafe_allow_html=True,
        )
        c4.markdown(
            render_metric_card(f"{int(player_data.Min):,}", "Minutes Played", "Sample duration"),
            unsafe_allow_html=True,
        )

        # Load Model & Predict
        clf_model = joblib.load(ROOT / model_files[selected_model_name])
        features_dict = {}
        for feat in SCORING_CLASSIFICATION_FEATURES:
            val = player_data.get(feat, np.nan)
            features_dict[feat] = float(val) if pd.notna(val) else np.nan

        clf_input = pd.DataFrame([features_dict])
        pred_class = clf_model.predict(clf_input)[0]
        pred_probs = clf_model.predict_proba(clf_input)[0]
        score_prob = pred_probs[1] * 100

        # Scoring Verdict & Radial Gauge
        st.markdown('<div class="section-title">Scoring Likelihood Verdict</div>', unsafe_allow_html=True)

        col_v1, col_v2 = st.columns([1, 1])

        with col_v1:
            verdict_text = "LIKELY TO SCORE" if pred_class == 1 else "UNLIKELY TO SCORE"
            verdict_class = "likely" if pred_class == 1 else "unlikely"
            verdict_color = "#16C784" if pred_class == 1 else "#3B82F6"
            expl_text = (
                "High statistical probability of scoring based on active shooting output and goal rate."
                if pred_class == 1
                else "Lower probability of scoring — profile leans toward buildup, defense, or lower shot volume."
            )

            st.markdown(
                f"""
            <div class="prediction-banner {verdict_class}">
                <div class="title">SCORING VERDICT</div>
                <div class="number" style="color: {verdict_color};">{verdict_text}</div>
                <div class="expl">{expl_text}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col_v2:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score_prob,
                    number={"suffix": "%", "font": {"family": "Bebas Neue", "size": 60, "color": verdict_color}},
                    title={"text": "Scoring Probability", "font": {"size": 16, "family": "Space Grotesk"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": verdict_color},
                        "steps": [
                            {"range": [0, 30], "color": "#F1F5F9"},
                            {"range": [30, 60], "color": "#E2E8F0"},
                            {"range": [60, 100], "color": "rgba(22, 199, 132, 0.15)"},
                        ],
                    },
                )
            )
            apply_chart_theme(fig_gauge, height=260, show_legend=False, is_gauge=True)
            st.plotly_chart(fig_gauge, use_container_width=True, theme=None)

        # Performance Drivers
        st.markdown("### Key Performance Drivers")
        drivers = []
        if gpm_val >= 0.25:
            drivers.append("✅ **Strong Goal Rate**: Averages ≥ 0.25 goals per match.")
        else:
            drivers.append("ℹ️ **Moderate Goal Rate**: Below 0.25 goals per match benchmark.")

        if sh90_val >= 2.0:
            drivers.append("✅ **High Shooting Volume**: Generates over 2.0 shots per 90 minutes.")
        else:
            drivers.append("ℹ️ **Lower Shot Frequency**: Under 2.0 shots per 90 minutes.")

        if player_data.Min >= MIN_MINUTES:
            drivers.append("✅ **Established Playing Time**: Sufficient minutes sample (≥ 450 mins).")
        else:
            drivers.append("⚠️ **Limited Minutes**: Under 450 minutes played — prediction has higher variance.")

        for d in drivers:
            st.markdown(d)

        # Technical Details Expander
        with st.expander("🔬 Technical Details & Model Validation"):
            st.markdown(
                """
            #### Classification Benchmarks & Validation
            - **Target**: Binary indicator (`Gls/MP >= 0.10` vs `< 0.10`).
            - **Stratified Split**: 80/20 train/test holdout with 5-fold stratified cross-validation on training data.
            """
            )

            if not class_metrics.empty:
                st.dataframe(class_metrics, use_container_width=True, hide_index=True)

            col_cm1, col_cm2 = st.columns(2)
            with col_cm1:
                cm_p = ROOT / "plots" / "confusion_matrix_support.png"
                if cm_p.exists():
                    st.image(str(cm_p), caption="SVM Confusion Matrix", use_container_width=True)
            with col_cm2:
                roc_p = ROOT / "plots" / "roc_curve_support.png"
                if roc_p.exists():
                    st.image(str(roc_p), caption="SVM ROC Curve", use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━ TAB 5: PLAYER TYPES ━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[4]:
    st.markdown('<div class="section-title">Player Style Profiles & Similar Players</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Discover distinct playing style profiles discovered through performance metrics, and match similar players across European leagues.</div>',
        unsafe_allow_html=True,
    )

    if clusters.empty:
        st.warning("⚠️ Player style profiles not found. Run `python -m src.run_analysis` to execute clustering.")
    else:
        cluster_lookup = {
            f"{row.Player} — {row.Squad} ({row.Comp})": idx for idx, row in clusters.iterrows()
        }
        selected_cluster_player = st.selectbox(
            "Select Player to Profile", list(cluster_lookup), key="cluster_player_select_v2"
        )
        record = clusters.loc[cluster_lookup[selected_cluster_player]]

        cluster_id = int(record.cluster)
        archetype_info = ARCHETYPE_MAP.get(
            cluster_id,
            {"name": f"Profile Group {cluster_id}", "badge_color": "#0B1F33", "desc": "Performance group"},
        )

        # Player Header Card
        st.markdown(
            render_player_header(record.Player, record.Squad, record.Comp, record.Pos, getattr(record, "Age", np.nan)),
            unsafe_allow_html=True,
        )

        # Profile Archetype Banner
        st.markdown(
            f"""
        <div class="prediction-banner" style="border-top: 6px solid {archetype_info['badge_color']};">
            <div class="title">ASSIGNED PLAYER TYPE</div>
            <div class="number" style="color: {archetype_info['badge_color']};">{archetype_info['name']}</div>
            <div class="expl">{archetype_info['desc']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Interactive 2D PCA Style Landscape Plot
        st.markdown('<div class="section-title">Player Style Landscape (Performance Map)</div>', unsafe_allow_html=True)
        st.caption("2D performance space constructed via Principal Component Analysis (PCA). Selected player is highlighted.")

        # Map cluster IDs to named Archetypes in DataFrame for Plotly legend
        clusters_plot = clusters.copy()
        clusters_plot["Player_Type"] = clusters_plot["cluster"].map(lambda c: ARCHETYPE_MAP.get(int(c), {}).get("name", f"Type {c}"))

        fig_pca = px.scatter(
            clusters_plot,
            x="PC1",
            y="PC2",
            color="Player_Type",
            hover_name="Player",
            hover_data={"Squad": True, "Comp": True, "Pos": True, "Min": True, "Player_Type": True},
            color_discrete_map={info["name"]: info["badge_color"] for info in ARCHETYPE_MAP.values()},
            labels={"PC1": "Attacking vs Defensive Axis (PC1)", "PC2": "Activity & Work-Rate Axis (PC2)", "Player_Type": "Player Type"},
        )

        # Highlight selected player with larger star marker
        fig_pca.add_trace(
            go.Scatter(
                x=[record.PC1],
                y=[record.PC2],
                mode="markers+text",
                marker=dict(symbol="star", size=22, color="#0B1F33", line=dict(color="#FFFFFF", width=2)),
                text=[f"⭐ {record.Player}"],
                textposition="top center",
                name=f"Selected: {record.Player}",
            )
        )

        apply_chart_theme(fig_pca, height=500)
        st.plotly_chart(fig_pca, use_container_width=True, theme=None)

        # Similar Players Section
        st.markdown('<div class="section-title">Players with Similar Playing Styles</div>', unsafe_allow_html=True)
        st.caption(f"Other players classified under the **{archetype_info['name']}** style profile.")

        peers = clusters.loc[
            clusters.cluster == cluster_id, ["Player", "Squad", "Comp", "Pos", "Min"]
        ]
        peers = peers.loc[
            ~((peers.Player == record.Player) & (peers.Squad == record.Squad) & (peers.Comp == record.Comp))
        ].head(10)

        st.dataframe(peers, use_container_width=True, hide_index=True)

        # Technical Details Expander
        with st.expander("🔬 Technical Details & Clustering Methodology"):
            st.markdown(
                """
            #### Unsupervised Machine Learning Pipeline
            - **Feature Standardization**: 14 per-90 performance metrics standardized using `StandardScaler`.
            - **Dimensionality Reduction**: Principal Component Analysis (PCA) retaining 90%+ cumulative explained variance.
            - **Clustering Algorithm**: Gaussian Mixture Models (GMM) optimized with the Expectation-Maximization (EM) algorithm.
            - **Model Selection Criteria**: K=4 selected based on Bayesian Information Criterion (BIC), Akaike Information Criterion (AIC), and Silhouette score.
            """
            )

            if not cluster_profiles.empty:
                st.markdown("#### Mean Feature Profile Matrix by Cluster")
                st.dataframe(cluster_profiles, use_container_width=True)

            if not cluster_selection.empty:
                st.markdown("#### Cluster Model Selection Criteria (K=2 to K=8)")
                st.dataframe(cluster_selection, use_container_width=True)
