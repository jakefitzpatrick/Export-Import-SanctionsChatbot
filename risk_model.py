#!/usr/bin/env python3
"""Utilities for computing and consuming governance risk scores."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from logger import setup_logger

logger = setup_logger(__name__)

RISK_THRESHOLDS = {
    "Low": 33.0,
    "Medium": 66.0,
    "High": 100.0,
}

RISK_COLOR_MAP = {
    "Low": "#1f9d55",
    "Medium": "#f1c40f",
    "High": "#c0392b",
}

FALLBACK_COUNTRY_RISK = {
    "Cameroon": {"score": 73.79, "year": 2025},
    "Russia": {"score": 91.2, "year": 2025},
    "China": {"score": 65.4, "year": 2025},
    "Iran": {"score": 95.1, "year": 2025},
    "Germany": {"score": 12.3, "year": 2025},
    "Canada": {"score": 8.7, "year": 2025},
    "Brazil": {"score": 48.5, "year": 2025},
    "Nigeria": {"score": 78.2, "year": 2025},
    "France": {"score": 15.1, "year": 2025},
    "India": {"score": 52.3, "year": 2025},
}

# ✅ FIXED FILE PATH
INPUT_PATH = Path("data/vdem_risk_subset_CLEANED.csv")
OUTPUT_PATH = Path("vdem_risk_scored.csv")

INDICATORS = [
    "v2excrptps",
    "v2exthftps",
    "v2cltrnslw",
    "v2clrspct",
    "v2stcritrecadm",
]

WEIGHTS = {
    "risk_v2excrptps": 0.25,
    "risk_v2exthftps": 0.15,
    "risk_v2cltrnslw": 0.20,
    "risk_v2clrspct": 0.20,
    "risk_v2stcritrecadm": 0.20,
}

RISK_METHOD_SUMMARY = (
    "Governance risk scores are derived from five V-Dem indicators: corrupt exchanges (v2excrptps), "
    "public theft (v2exthftps), transparent laws (v2cltrnslw), respect for civil administration (v2clrspct), "
    "and meritocratic state appointments (v2stcritrecadm). Each indicator is inverted and normalized to a 0–100 "
    "scale using risk = ((max - value) / (max - min)) * 100 so that higher numbers always mean higher risk. "
    "Those normalized columns are then combined into a composite score using fixed weights "
    "(25%, 15%, 20%, 20%, 20% respectively). The resulting `risk_score` stays between 0 and 100 and is "
    "bucketed using shared thresholds: <33 = Low (#1f9d55), 33–66 = Medium (#f1c40f), >66 = High (#c0392b)."
)


def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV dataset and handle file-not-found gracefully."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input file not found: {path}") from exc


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Ensure all required indicators are present."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def normalize_indicator(series: pd.Series) -> pd.Series:
    """Min-max normalize and invert so higher values represent more risk."""
    minimum = series.min(skipna=True)
    maximum = series.max(skipna=True)

    # Handle edge cases
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(0, index=series.index)

    # Inverted min-max normalization → higher = more risk
    scaled = (maximum - series) / (maximum - minimum)
    return scaled * 100


def categorize_risk(score: float) -> str:
    if not 0 <= score <= 100:
        raise ValueError("risk score must be between 0 and 100")
    if score < RISK_THRESHOLDS["Low"]:
        return "Low"
    if score < RISK_THRESHOLDS["Medium"]:
        return "Medium"
    return "High"


def get_risk_color(score: float) -> tuple[str, str]:
    level = categorize_risk(score)
    return RISK_COLOR_MAP[level], level


def load_country_risk_df(csv_path: Path | str = OUTPUT_PATH) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        return _fallback_country_risk_df()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return _fallback_country_risk_df()
    country_col = "country_name" if "country_name" in df.columns else "country"
    if country_col not in df.columns or "risk_score" not in df.columns:
        logger.warning("Risk CSV missing required columns; using fallback data.")
        return _fallback_country_risk_df()
    df = df.rename(columns={country_col: "country"})
    df = df.dropna(subset=["country", "risk_score"])
    df["score"] = df["risk_score"].astype(float)
    if "year" not in df.columns:
        df["year"] = datetime.now().year
    df["level"] = df["score"].apply(categorize_risk)
    df["color"] = df["level"].map(RISK_COLOR_MAP)
    columns = ["country", "score", "year", "color", "level"]
    return df[columns].drop_duplicates(subset=["country"]).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def get_risk_df() -> pd.DataFrame:
    """Streamlit-friendly wrapper that loads the scored CSV once per process."""
    return load_country_risk_df()


def _fallback_country_risk_df() -> pd.DataFrame:
    rows = []
    for name, meta in FALLBACK_COUNTRY_RISK.items():
        color, level = get_risk_color(meta["score"])
        rows.append(
            {
                "country": name,
                "score": meta["score"],
                "year": meta.get("year", datetime.now().year),
                "color": color,
                "level": level,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    # Load and validate data
    df = load_data(INPUT_PATH)
    ensure_columns(df, INDICATORS)

    # Create normalized risk columns
    for indicator in INDICATORS:
        risk_col = f"risk_{indicator}"
        df[risk_col] = normalize_indicator(df[indicator])

    # Compute weighted risk score
    df["risk_score"] = sum(df[col] * weight for col, weight in WEIGHTS.items())

    # Create categorical risk levels using shared thresholds
    df["risk_level"] = df["risk_score"].apply(categorize_risk)

    # Identify country column
    id_col = "country_name" if "country_name" in df.columns else "country"
    if id_col not in df.columns:
        raise ValueError("Neither 'country_name' nor 'country' exists in the dataset.")

    # Sort by risk score (highest risk first)
    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # Save full dataset
    df.to_csv(OUTPUT_PATH, index=False)

    # Log preview of results and summary statistics
    preview_columns = [id_col, "year", "risk_score", "risk_level"]
    logger.info("Top risk entries:\n%s", df.loc[:, preview_columns].head(10).to_string(index=False))
    logger.info("Risk score summary:\n%s", df["risk_score"].describe().to_string())

    # Explanation for write-up
    logger.debug(
        "Normalization formula: risk = ((max_value - observed_value) / (max_value - min_value)) * 100 "
        "- inverts governance indicators so higher governance quality becomes lower risk."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logger.exception("Unhandled exception in risk_model")
        sys.exit(1)
