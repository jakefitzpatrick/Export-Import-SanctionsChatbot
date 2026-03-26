#!/usr/bin/env python3
"""
Create country-level trade risk scores from V-Dem governance indicators.
"""

import sys
from pathlib import Path

import pandas as pd

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

    # Create categorical risk levels
    bins = [0, 25, 50, 75, 100.000001]
    labels = ["Low", "Moderate", "High", "Very High"]
    df["risk_level"] = pd.cut(df["risk_score"], bins=bins, labels=labels, right=False)

    # Identify country column
    id_col = "country_name" if "country_name" in df.columns else "country"
    if id_col not in df.columns:
        raise ValueError("Neither 'country_name' nor 'country' exists in the dataset.")

    # Sort by risk score (highest risk first)
    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # Save full dataset
    df.to_csv(OUTPUT_PATH, index=False)

    # Print preview
    preview_columns = [id_col, "year", "risk_score", "risk_level"]
    print(df.loc[:, preview_columns].head(10).to_string(index=False))

    print("\nRisk score summary:")
    print(df["risk_score"].describe().to_string())

    # Print explanation for your project write-up
    print(
        "\nNormalization formula used:\n"
        "risk = ((max_value - observed_value) / (max_value - min_value)) * 100\n"
        "This inverts governance indicators so higher governance quality becomes lower risk.\n"
        "The worst observed value maps to 100 (highest risk) and the best maps to 0 (lowest risk)."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
