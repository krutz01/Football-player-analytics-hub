"""Dataset inspection, data dictionary generation, and reproducible cleaning."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw" / "players_data-2025_2026.csv"
DEFAULT_LIGHT_DATA = ROOT / "data" / "raw" / "players_data_light-2025_2026.csv"
MIN_MINUTES = 450
SOURCE_KEYS = ["Player", "Squad", "Comp"]

POSITION_MAP = {
    "GK": "GK",
    "DF": "DEF",
    "MF": "MID",
    "FW": "FWD",
}

GOAL_REGRESSION_FEATURES = [
    "Age", "Min", "Starts", "Sh", "Ast", "Crs", "TklW", "Int",
    "Fld", "Fls", "Off", "CrdY", "CrdR",
]
SCORING_CLASSIFICATION_FEATURES = [
    "Sh_per90", "SoT_per90", "Ast_per90", "Off_per90", "Fld_per90", "Crs_per90",
    "TklW_per90", "Int_per90", "Fls_per90", "Age", "Starts"
]
CLUSTERING_FEATURES = [
    "Gls_per90", "Ast_per90", "Sh_per90", "SoT_per90", "Crs_per90",
    "TklW_per90", "Int_per90", "Fld_per90", "Fls_per90", "Off_per90",
    "Saves_per90", "GA90", "Save%", "CS%",
]

SCORING_THRESHOLD = 0.1  # goals per match — players above this are 'Likely to Score'

DESCRIPTIONS = {
    "Rk": "FBref rank in the source table; not a performance feature.",
    "Player": "Player name as reported by FBref.",
    "Nation": "Player nationality (country code and nationality text).",
    "Pos": "Detailed FBref position label; comma-separated positions are ordered.",
    "Squad": "Club/team name for this player-season record.",
    "Comp": "Competition / league name.",
    "Age": "Age during the season; source may encode fractional years.",
    "Born": "Birth year.",
    "MP": "Matches played.",
    "Starts": "Matches started.",
    "Min": "Minutes played.",
    "90s": "Estimated 90-minute periods played (minutes divided by 90).",
    "Gls": "Goals scored; regression target.",
    "Ast": "Assists.",
    "G+A": "Goals plus assists; derived from Gls + Ast.",
    "G-PK": "Non-penalty goals; derived from Gls - PK.",
    "PK": "Penalty-kick goals.",
    "PKatt": "Penalty kicks attempted (or attempts faced when repeated in the keeper table).",
    "CrdY": "Yellow cards.",
    "CrdR": "Red cards.",
    "G+A-PK": "Non-penalty goals plus assists; derived from G-PK + Ast.",
    "GA": "Goals conceded while the player was goalkeeper.",
    "GA90": "Goals conceded per 90 minutes as goalkeeper.",
    "SoTA": "Shots on target faced by a goalkeeper.",
    "Saves": "Goalkeeper saves.",
    "Save%": "Goalkeeper save percentage.",
    "W": "Goalkeeper wins.",
    "D": "Goalkeeper draws.",
    "L": "Goalkeeper losses.",
    "CS": "Goalkeeper clean sheets.",
    "CS%": "Goalkeeper clean-sheet percentage.",
    "PKA": "Penalty kicks scored against a goalkeeper.",
    "PKsv": "Penalty kicks saved.",
    "PKm": "Penalty kicks missed by opponents.",
    "Sh": "Shots total.",
    "SoT": "Shots on target.",
    "SoT%": "Percentage of shots on target.",
    "Sh/90": "Shots per 90 minutes.",
    "SoT/90": "Shots on target per 90 minutes.",
    "G/Sh": "Goals per shot; derived shooting efficiency.",
    "G/SoT": "Goals per shot on target; derived shooting efficiency.",
    "Mn/MP": "Minutes per appearance.",
    "Min%": "Share of available team minutes played.",
    "Mn/Start": "Minutes per start.",
    "Compl": "Full 90-minute appearances.",
    "Subs": "Substitute appearances.",
    "Mn/Sub": "Minutes per substitute appearance.",
    "unSub": "Unused substitute appearances.",
    "PPM": "Team points per match with the player appearing.",
    "onG": "Goals scored by the team while the player was on the pitch.",
    "onGA": "Goals conceded by the team while the player was on the pitch.",
    "+/-": "Team goal difference while the player was on the pitch.",
    "+/-90": "Team goal difference per 90 minutes while on the pitch.",
    "On-Off": "Team goal-difference change per 90 on versus off the pitch.",
    "2CrdY": "Second yellow cards.",
    "Fls": "Fouls committed.",
    "Fld": "Fouls drawn.",
    "Off": "Offsides.",
    "Crs": "Crosses.",
    "Int": "Interceptions.",
    "TklW": "Tackles won.",
    "OG": "Own goals.",
}

DERIVED_COLUMNS = {
    "Rk", "G+A", "G-PK", "G+A-PK", "90s", "Min%", "Mn/MP",
    "Mn/Start", "Mn/Sub", "PPM", "GA90", "Save%", "CS%", "SoT%",
    "Sh/90", "SoT/90", "G/Sh", "G/SoT", "+/-90", "On-Off",
}

TASK_USE = {
    "Gls": "Regression target; classification/clustering input",
    "Position": "Classification target only; never an ML input",
    "Regression": "Regression input",
    "Classification": "Classification input",
    "Clustering": "Unsupervised clustering/PCA input",
    "No": "Not used as an ML feature",
}


def _description(column: str) -> str:
    base = column
    suffix = ""
    for table in ("keeper", "shooting", "playing_time", "misc"):
        marker = f"_stats_{table}"
        if column.endswith(marker):
            base = column[: -len(marker)]
            suffix = f" Repeated in the FBref {table.replace('_', ' ')} table."
            break
    if base in DESCRIPTIONS:
        return DESCRIPTIONS[base] + suffix
    label = re.sub(r"([a-z])([A-Z])", r"\1 \2", base).replace("_", " ")
    return f"FBref {label.lower()} statistic." + suffix


def _is_derived(column: str) -> bool:
    return column in DERIVED_COLUMNS


def _feature_use(column: str) -> str:
    if column == "Pos":
        return "Classification target source only; never an input"
    if column in {"Player", "Nation", "Squad", "Comp", "Born"}:
        return "No — identifier/context only"
    if column in {
        "Rk", "G+A", "G-PK", "G+A-PK", "Gls_stats_shooting",
        "Sh/90", "SoT/90", "G/Sh", "G/SoT", "90s", "Min%",
    } or "_stats_" in column:
        return "No — duplicate, derived, or table-repeat field"
    uses = []
    if column in GOAL_REGRESSION_FEATURES:
        uses.append("Regression")
    if column in SCORING_CLASSIFICATION_FEATURES:
        uses.append("Classification")
    if column in CLUSTERING_FEATURES:
        uses.append("Clustering/PCA")
    if column == "Gls":
        uses.append("Regression target")
    return ", ".join(uses) if uses else "No — not selected (unavailable/inappropriate/redundant)"


def load_source_dataset(
    full_path: Path = DEFAULT_DATA, light_path: Path = DEFAULT_LIGHT_DATA
) -> pd.DataFrame:
    """Join and validate the full and light source CSVs without duplicating fields."""
    full = pd.read_csv(full_path)
    light = pd.read_csv(light_path)
    for label, source in (("full", full), ("light", light)):
        missing_keys = sorted(set(SOURCE_KEYS) - set(source.columns))
        if missing_keys:
            raise ValueError(f"The {label} source is missing join columns: {missing_keys}")
        duplicate_keys = source.duplicated(SOURCE_KEYS)
        if duplicate_keys.any():
            examples = source.loc[duplicate_keys, SOURCE_KEYS].head(5).to_dict("records")
            raise ValueError(f"The {label} source has duplicate player/team/competition keys: {examples}")

    unknown_light_columns = sorted(set(light.columns) - set(full.columns))
    if unknown_light_columns:
        raise ValueError(
            "The light source contains fields absent from the full source; "
            f"review schema before merging: {unknown_light_columns}"
        )

    merged = full.merge(
        light,
        on=SOURCE_KEYS,
        how="left",
        sort=False,
        suffixes=("", "__light"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched_full = merged.loc[merged["_merge"] != "both", SOURCE_KEYS].head(5)
    unmatched_light = light.merge(
        full[SOURCE_KEYS],
        on=SOURCE_KEYS,
        how="left",
        sort=False,
        indicator=True,
        validate="one_to_one",
    )
    unmatched_light = unmatched_light.loc[
        unmatched_light["_merge"] != "both", SOURCE_KEYS
    ].head(5)
    if not unmatched_full.empty or not unmatched_light.empty:
        raise ValueError(
            "The full and light sources do not contain the same player/team/competition records: "
            f"full-only={unmatched_full.to_dict('records')}, "
            f"light-only={unmatched_light.to_dict('records')}"
        )

    mismatches = []
    for column in light.columns:
        if column in SOURCE_KEYS:
            continue
        left = merged[column]
        right = merged[f"{column}__light"]
        equal = left.eq(right) | (left.isna() & right.isna())
        if not equal.all():
            mismatches.append((column, int((~equal).sum())))
    if mismatches:
        raise ValueError(
            "Overlapping fields differ between the full and light sources "
            f"(column, mismatched rows): {mismatches}"
        )

    result_columns = list(full.columns) + [c for c in light.columns if c not in full.columns]
    return merged[result_columns].copy()


def inspect_dataset(
    path: Path = DEFAULT_DATA, light_path: Path = DEFAULT_LIGHT_DATA
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inspect both source files and return the combined frame and column dictionary."""
    frame = load_source_dataset(path, light_path)
    light_columns = set(pd.read_csv(light_path, nrows=0).columns)
    dictionary = pd.DataFrame(
        {
            "column": frame.columns,
            "data_type": [str(frame[c].dtype) for c in frame.columns],
            "description": [_description(c) for c in frame.columns],
            "missing_pct": [frame[c].isna().mean() * 100 for c in frame.columns],
            "derived": [_is_derived(c) for c in frame.columns],
            "source_files": [
                "full + light (matched and cross-checked)" if c in light_columns
                else "full"
                for c in frame.columns
            ],
            "used": [
                not _feature_use(c).startswith("No —")
                for c in frame.columns
            ],
            "ml_task": [_feature_use(c) for c in frame.columns],
        }
    )
    return frame, dictionary


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean source rows, standardize positions, and retain auditable raw columns."""
    data = frame.copy()
    data = data.drop_duplicates()
    data = data.drop_duplicates(subset=["Player", "Squad", "Comp"], keep="first")
    for column in data.select_dtypes(include="object"):
        data[column] = data[column].str.strip()
    data["Position_Category"] = data["Pos"].str.split(",").str[0].map(POSITION_MAP)
    numeric_columns = data.columns.difference(
        ["Player", "Nation", "Pos", "Squad", "Comp", "Position_Category"]
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    rates = {
        f"{source}_per90": data[source].div(data["90s"].replace(0, np.nan))
        for source in ("Gls", "Ast", "Sh", "SoT", "Crs", "TklW", "Int", "Fld", "Fls", "Off", "Saves")
    }
    data = pd.concat([data, pd.DataFrame(rates, index=data.index)], axis=1)
    return data.reset_index(drop=True)


def save_data_dictionary(dictionary: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    dictionary.to_csv(destination, index=False, float_format="%.2f")
