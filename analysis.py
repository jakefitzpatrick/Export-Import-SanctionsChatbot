
"""Analysis pipeline helpers for ImportInsight AI.

This module contains the correlation/risk analysis helpers used by the Streamlit
app. UI components call into these functions; this file does not render any UI
on its own.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime
import uuid
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Iterable, Tuple

import openai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from duty_parser import DutyRate, parse_general_duty
from special_rates import parse_special_duty
from chat import append_message
from session import compute_selection_signature
from logger import setup_logger

logger = setup_logger(__name__)

# Static latitude/longitude map reused by multiple visualisations; defined once to
# avoid rebuilding the same literal dictionary on every Streamlit rerun.
COUNTRY_COORDINATES = {
    "Afghanistan": (33.9, 67.7),
    "Albania": (41.1, 20.2),
    "Algeria": (28.0, 1.7),
    "Angola": (-11.2, 17.9),
    "Argentina": (-38.4, -63.6),
    "Armenia": (40.1, 45.0),
    "Australia": (-25.3, 133.8),
    "Austria": (47.5, 14.6),
    "Azerbaijan": (40.1, 47.6),
    "Bangladesh": (23.7, 90.4),
    "Belgium": (50.8, 4.5),
    "Bolivia": (-16.3, -63.6),
    "Brazil": (-14.2, -51.9),
    "Bulgaria": (42.7, 25.5),
    "Cambodia": (12.6, 104.9),
    "Cameroon": (7.4, 12.4),
    "Canada": (56.1, -106.3),
    "Chad": (15.5, 18.7),
    "Chile": (-35.7, -71.5),
    "China": (35.9, 104.2),
    "Colombia": (4.6, -74.3),
    "Denmark": (56.3, 9.5),
    "Ecuador": (-1.8, -78.2),
    "Egypt": (26.8, 30.8),
    "Eritrea": (15.2, 39.8),
    "Ethiopia": (9.1, 40.5),
    "Finland": (61.9, 25.7),
    "France": (46.2, 2.2),
    "Germany": (51.2, 10.5),
    "Ghana": (7.9, -1.0),
    "Greece": (39.1, 21.8),
    "India": (20.6, 79.0),
    "Indonesia": (-0.8, 113.9),
    "Iran": (32.4, 53.7),
    "Iraq": (33.2, 43.7),
    "Italy": (41.9, 12.6),
    "Japan": (36.2, 138.3),
    "Jordan": (30.6, 36.2),
    "Kazakhstan": (48.0, 66.9),
    "Kenya": (-0.0, 37.9),
    "Libya": (26.3, 17.2),
    "Malaysia": (4.2, 108.0),
    "Mali": (17.6, -2.0),
    "Mexico": (23.6, -102.6),
    "Morocco": (31.8, -7.1),
    "Netherlands": (52.1, 5.3),
    "Nicaragua": (12.9, -85.2),
    "Nigeria": (9.1, 8.7),
    "Norway": (60.5, 8.5),
    "Pakistan": (30.4, 69.3),
    "Peru": (-9.2, -75.0),
    "Philippines": (12.9, 121.8),
    "Poland": (51.9, 19.1),
    "Portugal": (39.4, -8.2),
    "Romania": (45.9, 24.9),
    "Russia": (61.5, 105.3),
    "Saudi Arabia": (23.9, 45.1),
    "Senegal": (14.5, -14.5),
    "Singapore": (1.4, 103.8),
    "Somalia": (5.2, 46.2),
    "South Africa": (-30.6, 22.9),
    "South Korea": (35.9, 127.8),
    "South Sudan": (6.9, 31.3),
    "Spain": (40.5, -3.7),
    "Sri Lanka": (7.9, 80.8),
    "Sudan": (12.9, 30.2),
    "Sweden": (60.1, 18.6),
    "Switzerland": (46.8, 8.2),
    "Syria": (34.8, 38.8),
    "Taiwan": (23.7, 121.0),
    "Tanzania": (-6.4, 34.9),
    "Thailand": (15.9, 100.9),
    "Turkey": (38.9, 35.2),
    "Uganda": (1.4, 32.3),
    "Ukraine": (48.4, 31.2),
    "United Arab Emirates": (23.4, 53.8),
    "United Kingdom": (55.4, -3.4),
    "United States": (37.1, -95.7),
    "Uruguay": (-32.5, -55.8),
    "Venezuela": (6.4, -66.6),
    "Vietnam": (14.1, 108.3),
    "Yemen": (15.6, 48.5),
    "Zimbabwe": (-19.0, 29.2),
}

SUMMARY_SAMPLE_LIMIT = 60
DISPLAY_COLUMN_MAP = [
    ("country", "Country"),
    ("corruption_score", "Corruption Score"),
    ("hts_code", "HTS Code"),
    ("product_description", "Product"),
    ("general_duty_rate_text", "Base Rate"),
    ("ad_valorem_rate", "Effective Rate (%)"),
    ("ch99_delta", "Ch.99 Δ"),
    ("ch99_tradeprogram", "Trade Program"),
     ("ch99_specific_surcharge", "Ch.99 Specific"),
    ("rate_source", "Rate Source"),
    ("plotted", "Plotted"),
]


def _format_percent(value: float | None) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    formatted = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{formatted}%"


def _normalize_unit(unit: str | None) -> str:
    if not unit:
        return ""
    cleaned = unit.strip()
    if cleaned and not cleaned.startswith("/"):
        cleaned = f"/{cleaned}"
    return cleaned


def _format_decimal_preserving(value: float) -> str:
    try:
        dec_value = Decimal(str(value)).normalize()
    except (InvalidOperation, ValueError):
        return str(value)
    text = format(dec_value, "f")
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def _format_specific(value: float | None, unit: str | None, raw: str | None) -> str | None:
    cleaned_raw = ""
    if isinstance(raw, str):
        cleaned_raw = raw.strip()
    elif raw is not None:
        # Pandas often converts empty strings to NaN/NA scalars, so drop them silently.
        if pd.isna(raw):
            cleaned_raw = ""
        else:
            cleaned_raw = str(raw).strip()
    if cleaned_raw:
        return cleaned_raw
    if value is None:
        return None
    unit_fmt = _normalize_unit(unit) or "/unit"
    amount = _format_decimal_preserving(value)
    return f"{amount}{unit_fmt}"


def _hts_ancestor_codes(code: str) -> list[str]:
    """Return parent HTS codes by truncating dotted segments."""
    if not code:
        return []
    sanitized = code.strip()
    if not sanitized:
        return []
    parts = sanitized.split(".")
    ancestors: list[str] = []
    while len(parts) > 1:
        parts = parts[:-1]
        ancestors.append(".".join(parts))
    return ancestors


def _normalize_denominator(unit_text: str | None) -> str | None:
    if not unit_text:
        return None
    cleaned = unit_text.lower().replace(" per ", "/")
    for token in ("$", "usd", "dollars", "dollar", "cents", "cent", "¢"):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.strip()
    if "/" in cleaned:
        cleaned = cleaned.split("/", 1)[1]
    cleaned = cleaned.lstrip("/").strip()
    return cleaned or None


def _convert_specific_value(value: float | None, unit_hint: str | None, raw_hint: str | None = None) -> float | None:
    """Convert specific duty amounts to USD based on currency hint."""
    if value is None:
        return None
    tokens = (unit_hint or "") + " " + (raw_hint or "")
    lower = tokens.lower()
    if "cent" in lower or "¢" in lower:
        return value / 100.0
    return value


def _decorate_specific_raw(amount: float | None, base_raw: str | None, unit: str | None) -> str | None:
    if amount is None:
        return None
    unit_suffix = (unit or "").strip()
    if unit_suffix and not unit_suffix.startswith("/"):
        unit_suffix = f"/{unit_suffix}"
    symbol = ""
    lower_raw = (base_raw or "").lower()
    if "$" in (base_raw or ""):
        symbol = "$"
        formatted_value = _format_decimal_preserving(amount)
    elif "¢" in lower_raw or "cent" in lower_raw:
        symbol = "¢"
        formatted_value = _format_decimal_preserving(amount * 100.0)
    else:
        formatted_value = _format_decimal_preserving(amount)
    return f"{symbol}{formatted_value}{unit_suffix}".strip()


def _resolve_general_duty_rate(
    conn: sqlite3.Connection,
    code: str,
    cache: dict[str, tuple[str, str | None]],
) -> tuple[str, str | None]:
    """Return (duty_text, source_code) for the first ancestor with a general duty rate."""
    cached = cache.get(code)
    if cached is not None:
        return cached
    row = conn.execute("SELECT general_duty_rate FROM hts WHERE hts_code = ?", (code,)).fetchone()
    text = ""
    if row:
        text = (row[0] or "").strip()
    if text:
        cache[code] = (text, code)
        return cache[code]
    for ancestor in _hts_ancestor_codes(code):
        candidate = _resolve_general_duty_rate(conn, ancestor, cache)
        if candidate[0]:
            cache[code] = candidate
            return candidate
    cache[code] = ("", None)
    return cache[code]


def fetch_ch99_for_codes_and_countries(
    conn: sqlite3.Connection,
    selected_codes: list[str],
    selected_countries: list[str],
) -> pd.DataFrame:
    """Return ALL Chapter 99 rows for each (hts_code, queried_country/Global) pair.

    Queries the raw chapter_99 table with the multi-format JOIN so that every
    applicable rule (e.g. a higher-priority CAFTA-DR row AND a lower-priority
    Section 122 additive row) is visible to _best_ch99_rule for proper combination.
    """
    if not selected_codes or not selected_countries:
        return pd.DataFrame()

    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM chapter_99 LIMIT 1")
    except sqlite3.Error as exc:
        logger.warning("Chapter 99 table missing; skipping overrides", exc_info=exc)
        return pd.DataFrame()

    code_ph = ",".join(["?"] * len(selected_codes))
    country_ph = ",".join(["?"] * len(selected_countries))
    query = f"""
        SELECT
            m.hts_code                                                      AS hts_code,
            c.COUNTRY                                                       AS ch99_country,
            CAST(NULLIF(c.NEWRATE_CLEAN, '') AS REAL)                       AS ch99_newrate,
            c.RATE_MODIFIER                                                 AS ch99_rate_modifier,
            COALESCE(CAST(NULLIF(c.ADDITIONAL_DUTY_PCT, '') AS REAL), 0.0) AS ch99_additional_pct,
            CAST(NULLIF(c.ADDITIONAL_VALUE, '') AS REAL)                    AS ch99_additional_value,
            c.ADDITIONAL_VALUE                                             AS ch99_additional_value_raw,
            c.ADDITIONAL_UNIT                                              AS ch99_additional_unit,
            c.TRADEPROGRAM                                                  AS ch99_tradeprogram,
            CAST(c.MATCH_PRIORITY AS INTEGER)                               AS ch99_match_priority
        FROM hts AS m
        JOIN chapter_99 AS c
            ON (
                c.HTS_MASTER_CODE = m.hts_code
                OR (LENGTH(c.HTS_MASTER_CODE) = 10
                    AND c.HTS_MASTER_CODE = SUBSTR(m.hts_code, 1, 10))
                OR (LENGTH(c.HTS_MASTER_CODE) = 12
                    AND SUBSTR(c.HTS_MASTER_CODE, 1, 10) || '.' || SUBSTR(c.HTS_MASTER_CODE, 11) = m.hts_code)
                OR (LENGTH(c.HTS_MASTER_CODE) = 2
                    AND c.HTS_MASTER_CODE = SUBSTR(m.hts_code, 1, 2))
                OR c.HTS_MASTER_CODE = 'ALL'
            )
        WHERE m.hts_code IN ({code_ph})
          AND (c.COUNTRY IN ({country_ph}) OR c.COUNTRY = 'Global')
    """
    params = selected_codes + selected_countries
    df = pd.read_sql_query(query, conn, params=params)
    return df


def _build_ch99_lookup(ch99_df: pd.DataFrame | None) -> dict[tuple[str, str], pd.DataFrame]:
    """Group Chapter 99 rows by (hts_code, country) for fast lookup."""
    if ch99_df is None or ch99_df.empty:
        return {}
    grouped: dict[tuple[str, str], pd.DataFrame] = {}
    for (hts_code, country), frame in ch99_df.groupby(["hts_code", "ch99_country"], sort=False):
        grouped[(hts_code, country)] = frame
    return grouped


def _best_ch99_rule(
    ch99_lookup: dict[tuple[str, str], pd.DataFrame],
    hts_code: str,
    country: str,
) -> dict | None:
    """Synthesize the net Chapter 99 effect for a (hts_code, country) pair.

    Because multiple rules from different trade programs can apply simultaneously
    (e.g. a CAFTA-DR-specific row at P10 and a Section 122 additive row at P30),
    this function:
      1. Finds the best (lowest MATCH_PRIORITY) Floor rule → sets the floor rate.
      2. Sums all additive ADDITIONAL_DUTY_PCT values from non-Floor, non-replacement rows.
      3. Finds the best replacement NEWRATE_CLEAN (non-Floor) row if one exists.
      4. Returns a synthetic rule dict combining Floor + additive so that
         apply_ch99_to_duty can compute max(base + additive, floor).
    """
    if not ch99_lookup:
        return None

    frames: list[pd.DataFrame] = []
    exact = ch99_lookup.get((hts_code, country))
    if exact is not None:
        frames.append(exact)
    global_rows = ch99_lookup.get((hts_code, "Global"))
    if global_rows is not None:
        frames.append(global_rows)
    if not frames:
        return None
    candidates = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    candidates = candidates.sort_values("ch99_match_priority")

    is_floor = candidates["ch99_rate_modifier"] == "Floor"
    floor_rows = candidates[is_floor]
    non_floor = candidates[~is_floor]

    # Best replacement rate (non-Floor with an actual NEWRATE_CLEAN number)
    replacement_rows = non_floor[non_floor["ch99_newrate"].notna()]
    if not replacement_rows.empty:
        return replacement_rows.iloc[0].to_dict()

    # Best Floor rate (lowest priority = highest precedence)
    best_floor_rate: float | None = None
    best_floor_program: str = ""
    if not floor_rows.empty:
        best_floor_row = floor_rows.iloc[0]
        v = best_floor_row["ch99_newrate"]
        try:
            f = float(v)
            best_floor_rate = None if f != f else f
        except (TypeError, ValueError):
            best_floor_rate = None
        best_floor_program = best_floor_row.get("ch99_tradeprogram", "")

    # Sum additive surcharges from non-Floor, non-replacement rows.
    #
    # Two exclusions:
    # 1. Deduplicate by (tradeprogram, country, match_priority, additional_pct) so that
    #    multiple NEWCODE rows from the same program (e.g. Nicaragua's 9903.01.49 and
    #    9903.02.47 both at P30 Section 122 0.18) are not double-counted.
    # 2. If a Floor row exists at the same (country, priority) level as an additive row,
    #    the Floor is mutually exclusive with the additive — exclude the additive.
    #    (e.g. USMCA Blocker Floor at P30/Canada supersedes Transshipment-Evasion +40%
    #    additive also at P30/Canada; they are alternative enforcement paths, not stackable.)
    floor_country_priorities: set[tuple] = {
        (r["ch99_country"], r["ch99_match_priority"])
        for _, r in floor_rows.iterrows()
    }
    raw_additive = non_floor[non_floor["ch99_newrate"].isna()]
    non_superseded_additive = raw_additive[
        ~raw_additive.apply(
            lambda r: (r["ch99_country"], r["ch99_match_priority"]) in floor_country_priorities,
            axis=1,
        )
    ].drop_duplicates(
        subset=["ch99_tradeprogram", "ch99_country", "ch99_match_priority", "ch99_additional_pct"]
    )
    additive_rows = non_superseded_additive
    total_additive: float = 0.0
    specific_components: list[str] = []
    for _, row in additive_rows.iterrows():
        v = row.get("ch99_additional_pct", 0.0)
        try:
            f = float(v)
            if f == f:  # not NaN
                total_additive += f
        except (TypeError, ValueError):
            pass

        raw_specific = row.get("ch99_additional_value_raw")
        specific_val = row.get("ch99_additional_value")
        specific_unit = (row.get("ch99_additional_unit") or "").strip()
        if raw_specific or specific_unit:
            normalized_amount = (
                _convert_specific_value(specific_val, specific_unit, raw_specific)
                if specific_val is not None
                else None
            )
            if raw_specific and (specific_val is None or specific_val != specific_val):
                amount_text = raw_specific.strip()
            elif specific_val is not None and specific_val == specific_val:
                amount_text = _format_decimal_preserving(specific_val)
            else:
                amount_text = ""
            text = amount_text
            if specific_unit:
                text = f"{text} {specific_unit}".strip()
            label = row.get("ch99_tradeprogram") or ""
            if label:
                text = f"{text} ({label})" if text else label
            specific_components.append(
                {
                    "amount": normalized_amount,
                    "denominator": _normalize_denominator(specific_unit or raw_specific),
                    "display": text.strip(),
                }
            )

    # Determine the representative trade program label
    if total_additive > 0 and not additive_rows.empty:
        tradeprogram = additive_rows.iloc[0].get("ch99_tradeprogram", "")
    elif best_floor_rate is not None:
        tradeprogram = best_floor_program
    elif not candidates.empty:
        tradeprogram = candidates.iloc[0].get("ch99_tradeprogram", "")
    else:
        tradeprogram = ""

    specific_text = "; ".join(
        comp["display"] for comp in specific_components if comp.get("display")
    )

    if best_floor_rate is None and total_additive == 0.0 and not specific_text:
        return None  # no usable rate content

    return {
        "hts_code": hts_code,
        "ch99_country": country,
        "ch99_newrate": best_floor_rate,
        "ch99_rate_modifier": "Floor" if best_floor_rate is not None else "",
        "ch99_additional_pct": total_additive,
        "ch99_tradeprogram": tradeprogram,
        "ch99_match_priority": int(candidates.iloc[0]["ch99_match_priority"]),
        "ch99_specific_surcharge": specific_text,
        "ch99_specific_components": specific_components,
    }


def apply_ch99_to_duty(
    base_rate: DutyRate,
    ch99: dict,
) -> float | None:
    """Calculate the effective ad-valorem duty rate after applying Chapter 99 logic.

    Scale conventions in the source CSV:
      - base_rate.ad_valorem_rate : percentage points  (5.0  = 5%)
      - NEWRATE_CLEAN (Floor rows): percentage points  (15.0 = 15%)
      - NEWRATE_CLEAN (non-Floor) : decimal fraction   (0.072 = 7.2%) → multiply by 100
      - ADDITIONAL_DUTY_PCT       : decimal fraction   (0.1  = 10%)  → multiply by 100

    Rules (in order):
      1. RATE_MODIFIER = 'Floor'  → duty = max(base + additive, floor_rate)
                                    both values are in percentage-point scale
      2. NEWRATE_CLEAN is a number → duty = NEWRATE_CLEAN * 100  (replaces base)
      3. Otherwise                 → duty = base_rate + ADDITIONAL_DUTY_PCT * 100
    """
    if base_rate is None or base_rate.ad_valorem_rate is None:
        return None

    base_av = base_rate.ad_valorem_rate  # percentage points; None for non-ad-valorem rates
    modifier = (ch99.get("ch99_rate_modifier") or "").strip()

    raw_newrate = ch99.get("ch99_newrate")
    raw_additional = ch99.get("ch99_additional_pct")

    def _to_float_or_none(v: object) -> float | None:
        if v is None:
            return None
        try:
            f = float(v)
            return None if f != f else f  # NaN check: NaN != NaN
        except (TypeError, ValueError):
            return None

    newrate = _to_float_or_none(raw_newrate)
    try:
        additional = float(raw_additional) if raw_additional is not None else 0.0
    except (TypeError, ValueError):
        additional = 0.0

    if modifier == "Floor":
        # Floor: effective = max(base + additive, floor_rate)
        # Both base_av and newrate are in percentage-point scale; additional is decimal fraction.
        if base_av is not None:
            base_with_additive = base_av + (additional * 100)
            if newrate is not None:
                return max(base_with_additive, newrate)
            return base_with_additive
        return base_av

    if newrate is not None:
        # Non-Floor NEWRATE_CLEAN is a decimal fraction — scale to percentage points
        return newrate * 100

    if base_av is not None:
        # ADDITIONAL_DUTY_PCT is a decimal fraction — scale before adding
        return base_av + (additional * 100)

    return base_av


def _format_rate_columns(row: pd.Series) -> Tuple[str, str]:
    percent = _format_percent(row.get("ad_valorem_rate"))
    specific = _format_specific(row.get("specific_amount"), row.get("specific_unit"), row.get("specific_raw"))
    general_text = (row.get("general_duty_rate_text") or "").strip()

    # When a duty has both ad valorem and specific components, the percent drives
    # the chart/primary cell while the full mixed string remains in Base Rate.
    if row.get("chart_eligible"):
        primary = percent or (general_text if general_text else specific or "")
        secondary = specific if (row.get("has_specific") and specific and percent) else ""
    elif row.get("has_specific") and specific:
        primary = specific
        secondary = ""
    elif general_text:
        primary = general_text
        secondary = ""
    else:
        primary = ""
        secondary = ""

    return primary or "", secondary or ""


def _format_non_ad_summary_for_text(summary: dict) -> str | None:
    if not summary or not summary.get("count"):
        return None
    parts = [
        f"{summary['count']} selection(s) have duties without a percentage component, so they were omitted from the scatter plot."
    ]
    by_kind = summary.get("by_kind") or {}
    if by_kind:
        kind_bits = ", ".join(f"{count} {kind}" for kind, count in by_kind.items())
        parts.append(f"Breakdown by duty kind: {kind_bits}.")
    samples = summary.get("samples") or []
    if samples:
        sample_text = "; ".join(samples)
        parts.append(f"Representative duty strings: {sample_text}.")
    if summary.get("mixed_present"):
        parts.append("Mixed duties are plotted using their % component only; the paired specific amounts remain visible in the table.")
    return " ".join(parts)


def _build_display_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    target_columns = [src for src, _ in DISPLAY_COLUMN_MAP]
    available = [col for col in target_columns if col in df.columns]
    column_headers = [label for src, label in DISPLAY_COLUMN_MAP if src in available]
    if df.empty or not available:
        return pd.DataFrame(columns=column_headers), column_headers
    display_df = df[available].copy().rename(
        columns={src: label for src, label in DISPLAY_COLUMN_MAP if src in available}
    )
    if "Corruption Score" in display_df.columns:
        display_df["Corruption Score"] = display_df["Corruption Score"].apply(
            lambda v: f"{float(v):.1f}" if pd.notna(v) else "—"
        )
    if "Effective Rate (%)" in display_df.columns:
        display_df["Effective Rate (%)"] = display_df["Effective Rate (%)"].apply(
            lambda v: f"{float(v):.2f}%" if pd.notna(v) else "—"
        )
    if "Ch.99 Δ" in display_df.columns:
        def _format_delta(val):
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return "—"
            sign = "+" if val >= 0 else ""
            return f"{sign}{val:.2f}%"

        display_df["Ch.99 Δ"] = display_df["Ch.99 Δ"].apply(_format_delta)
    if "Trade Program" in display_df.columns:
        display_df["Trade Program"] = display_df["Trade Program"].fillna("—")
    if "Rate Source" in display_df.columns:
        display_df["Rate Source"] = display_df["Rate Source"].fillna("General")
    if "Plotted" in display_df.columns:
        display_df["Plotted"] = display_df["Plotted"].fillna("No")
    return display_df, column_headers


def _summarize_specific_effects(df: pd.DataFrame) -> str | None:
    if df.empty or "ch99_specific_surcharge" not in df.columns:
        return None
    subset = df[
        df["ch99_specific_surcharge"].fillna("").astype(str).str.strip() != ""
    ]
    if subset.empty:
        return None
    bits: list[str] = []
    for _, row in subset.iterrows():
        specific_text = _format_specific(row.get("specific_amount"), row.get("specific_unit"), row.get("specific_raw"))
        label = row.get("ch99_specific_surcharge")
        descriptor = specific_text or label
        if descriptor:
            bits.append(f"{row.get('country')} {row.get('hts_code')}: {descriptor}")
    if not bits:
        return None
    return "Ch.99 specific surcharges now fix the per-unit rates at " + "; ".join(bits) + "."


def fetch_tariffs_for_codes(
    conn,
    selected_codes: list[str],
) -> pd.DataFrame:
    """Load and parse general duty rates for the selected HTS codes."""
    if not selected_codes:
        logger.info("No product codes selected; skipping tariff fetch")
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(selected_codes))
    query = (
        "SELECT hts_code, description, general_duty_rate, special_duty_rate "
        "FROM hts WHERE hts_code IN (" + placeholders + ")"
    )
    logger.info("Fetching tariff rows for selected products", extra={"product_count": len(selected_codes)})
    df = pd.read_sql_query(query, conn, params=selected_codes)
    if df.empty:
        logger.warning("Tariff lookup returned no rows for selected products")
        return df
    df = df.rename(columns={"general_duty_rate": "general_duty_rate_text"})
    df["general_duty_rate_text"] = df["general_duty_rate_text"].fillna("").astype(str)
    df["original_general_duty_rate_text"] = df["general_duty_rate_text"]
    df["general_duty_rate_source_code"] = df["hts_code"]
    inheritance_cache: dict[str, tuple[str, str | None]] = {}
    missing_mask = df["general_duty_rate_text"].str.strip() == ""
    if missing_mask.any():
        for idx, row in df[missing_mask].iterrows():
            fallback, source = _resolve_general_duty_rate(conn, row["hts_code"], inheritance_cache)
            if fallback:
                df.at[idx, "general_duty_rate_text"] = fallback
                df.at[idx, "general_duty_rate_source_code"] = source or row["hts_code"]
            else:
                logger.warning(
                    "No general duty rate found for %s or its ancestors", row["hts_code"]
                )
    df["special_duty_rate"] = df["special_duty_rate"].fillna("").astype(str)
    parsed_rates = df["general_duty_rate_text"].apply(parse_general_duty)

    def _summarize_components(rate) -> str:
        if not rate.components:
            return rate.raw_text
        return " + ".join(
            comp.raw_segment for comp in rate.components if comp.raw_segment
        ) or rate.raw_text

    df["duty_kind"] = [rate.kind for rate in parsed_rates]
    df["has_ad_valorem"] = [rate.has_ad_valorem for rate in parsed_rates]
    df["has_specific"] = [rate.has_specific for rate in parsed_rates]
    df["ad_valorem_rate"] = [rate.ad_valorem_rate for rate in parsed_rates]
    df["specific_amount"] = [rate.specific_amount for rate in parsed_rates]
    df["specific_unit"] = [rate.specific_unit for rate in parsed_rates]
    df["specific_raw"] = [
        next((comp.raw_segment for comp in rate.components if comp.type == "specific"), None)
        for rate in parsed_rates
    ]
    df["duty_notes"] = [rate.notes for rate in parsed_rates]
    df["duty_component_summary"] = [_summarize_components(rate) for rate in parsed_rates]
    df["parsed_duty_rate"] = parsed_rates
    df["special_duty_rule"] = [
        parse_special_duty(text) if text else None for text in df["special_duty_rate"]
    ]
    logger.info(
        "Parsed tariff duty structures",
        extra={"rows": len(df), "ad_valorem_rows": int(df["has_ad_valorem"].sum())},
    )
    return df


def _summarize_non_ad_rows(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"count": 0, "by_kind": {}, "samples": []}
    by_kind = Counter(df["duty_kind"].fillna("unknown"))
    samples = df["general_duty_rate_text"].dropna().head(5).tolist()
    return {
        "count": int(len(df)),
        "by_kind": dict(by_kind),
        "samples": samples,
        "mixed_present": bool((df["duty_kind"] == "mixed").any()),
    }


def build_correlation_dataframe(
    selected_countries: Iterable[str],
    tariff_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    ch99_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, list[dict], bool]:
    """Construct the chart dataframe, combined table, and metadata summaries."""
    selected_countries = list(selected_countries)
    logger.info(
        "Building correlation dataframe",
        extra={
            "countries": len(selected_countries),
            "tariff_rows": len(tariff_df),
        },
    )
    ch99_lookup = _build_ch99_lookup(ch99_df)
    ch99_available = bool(ch99_lookup)
    if not selected_countries or tariff_df.empty:
        empty_cols = [
            "country",
            "corruption_score",
            "risk_level",
            "hts_code",
            "product_description",
            "general_duty_rate_text",
            "rate_source",
            "special_program_code",
            "special_program_label",
            "duty_kind",
            "ad_valorem_rate",
            "chart_eligible",
        ]
        logger.info("Correlation dataframe empty due to missing selections or tariff rows")
        return (
            pd.DataFrame(),
            pd.DataFrame(columns=empty_cols),
            {"count": 0, "by_kind": {}, "samples": []},
            {"n_adjusted": 0, "n_total": 0, "programs": []},
            [],
            False,
        )

    country_subset = risk_df[risk_df["country"].isin(selected_countries)].copy()
    if country_subset.empty:
        empty_cols = [
            "country",
            "corruption_score",
            "risk_level",
            "hts_code",
            "product_description",
            "general_duty_rate_text",
            "rate_source",
            "special_program_code",
            "special_program_label",
            "duty_kind",
            "ad_valorem_rate",
            "chart_eligible",
        ]
        logger.info("Correlation dataframe empty because selected countries lack risk data")
        return (
            pd.DataFrame(),
            pd.DataFrame(columns=empty_cols),
            {"count": 0, "by_kind": {}, "samples": []},
            {"n_adjusted": 0, "n_total": 0, "programs": []},
            [],
            False,
        )

    country_records = country_subset.to_dict("records")
    product_records = tariff_df.to_dict("records")

    combined_rows: list[dict] = []
    special_override_count = 0
    unresolved_codes: set[str] = set()
    ch99_programs: set[str] = set()
    ch99_summary = {"n_adjusted": 0, "n_total": 0, "programs": []}
    for country_meta in country_records:
        for product_meta in product_records:
            country_name = country_meta["country"]
            special_rule = product_meta.get("special_duty_rule")
            rate_source = "General"
            program_code_summary = ""
            program_label_summary = ""
            applied_special = False
            ch99_specific_text = ""

            base_general_text = product_meta["general_duty_rate_text"]
            duty_component_summary = product_meta["duty_component_summary"]
            base_specific_amount = product_meta["specific_amount"]
            base_specific_unit = product_meta["specific_unit"]
            base_specific_raw = product_meta.get("specific_raw")
            effective_specific_amount = base_specific_amount
            effective_specific_unit = base_specific_unit
            effective_specific_raw = base_specific_raw
            duty_kind = product_meta["duty_kind"]
            has_specific = product_meta["has_specific"]

            base_duty_rate_obj = product_meta.get("parsed_duty_rate")
            if base_duty_rate_obj is None:
                base_duty_rate_obj = parse_general_duty(base_general_text)
            if base_duty_rate_obj:
                duty_kind = base_duty_rate_obj.kind or duty_kind
                if base_duty_rate_obj.specific_amount is not None:
                    effective_specific_amount = base_duty_rate_obj.specific_amount
                if base_duty_rate_obj.specific_unit:
                    effective_specific_unit = base_duty_rate_obj.specific_unit
                has_specific = has_specific or base_duty_rate_obj.has_specific

            base_has_ad_valorem = bool(
                base_duty_rate_obj and base_duty_rate_obj.has_ad_valorem and base_duty_rate_obj.ad_valorem_rate is not None
            )
            base_duty_pct = base_duty_rate_obj.ad_valorem_rate if base_has_ad_valorem else None
            effective_rate = base_duty_pct
            ch99_applied = False
            ch99_tradeprogram = None
            ch99_delta = None

            if special_rule:
                if special_rule.applies_to(country_name):
                    matching_codes = special_rule.codes_for_country(country_name)
                    applied_special = True
                    program_code_summary = ", ".join(matching_codes) if matching_codes else ""
                    program_label_summary = special_rule.format_labels(matching_codes or [])
                    rate_source = (
                        f"Special — {program_code_summary}"
                        if program_code_summary
                        else "Special — eligible"
                    )
                elif special_rule.has_dynamic_codes:
                    rate_source = f"Special (Unresolved — {', '.join(special_rule.dynamic_codes)})"
                    program_code_summary = ", ".join(special_rule.dynamic_codes)
                    program_label_summary = special_rule.format_labels(special_rule.dynamic_codes)
                    unresolved_codes.update(special_rule.dynamic_codes)

            if applied_special and special_rule:
                special_override_count += 1
                duty_rate = special_rule.duty_rate
                if duty_rate:
                    if duty_rate.ad_valorem_rate is not None:
                        effective_rate = duty_rate.ad_valorem_rate
                    if duty_rate.has_specific and duty_rate.specific_amount is not None:
                        effective_specific_amount = duty_rate.specific_amount
                        effective_specific_unit = duty_rate.specific_unit
                        effective_specific_raw = duty_rate.raw_text
                        has_specific = True
                    duty_kind = duty_rate.kind or duty_kind

            if base_has_ad_valorem:
                ch99_summary["n_total"] += 1

            rule = _best_ch99_rule(ch99_lookup, product_meta["hts_code"], country_name) if ch99_available else None
            if rule is not None and base_has_ad_valorem:
                adjusted_rate = apply_ch99_to_duty(base_duty_rate_obj, rule)
                if adjusted_rate is not None:
                    effective_rate = adjusted_rate
                    base_val = base_duty_pct or 0.0
                    has_change = base_duty_pct is None or not math.isclose(adjusted_rate, base_val, rel_tol=1e-6)
                    if has_change:
                        ch99_applied = True
                        ch99_delta = adjusted_rate - base_val if base_duty_pct is not None else adjusted_rate
                        ch99_summary["n_adjusted"] += 1
                    ch99_tradeprogram = rule.get("ch99_tradeprogram") or None
                    if ch99_tradeprogram:
                        ch99_programs.add(ch99_tradeprogram)

            if rule is not None:
                ch99_specific_text = rule.get("ch99_specific_surcharge") or ""
                specific_components = rule.get("ch99_specific_components") or []
                if specific_components:
                    base_denom = _normalize_denominator(effective_specific_unit or base_specific_raw)
                    total_increment = 0.0
                    for comp in specific_components:
                        inc = comp.get("amount")
                        denom = comp.get("denominator")
                        if inc is None:
                            continue
                        if denom and base_denom and denom != base_denom:
                            logger.warning(
                                "Skipped Ch.99 specific adder due to unit mismatch",
                                extra={"hts_code": product_meta["hts_code"], "country": country_name, "denominator": denom},
                            )
                            continue
                        if not base_denom and denom:
                            base_denom = denom
                        total_increment += inc
                    if total_increment:
                        effective_specific_amount = (effective_specific_amount or 0.0) + total_increment
                        has_specific = True
                        if not effective_specific_unit and base_denom:
                            effective_specific_unit = f"/{base_denom}"
                        effective_specific_raw = _decorate_specific_raw(
                            effective_specific_amount,
                            base_specific_raw,
                            effective_specific_unit,
                        )
                if ch99_specific_text and not ch99_applied:
                    ch99_tradeprogram = ch99_tradeprogram or rule.get("ch99_tradeprogram")
                    ch99_label = f"Ch.99 ({ch99_tradeprogram})" if ch99_tradeprogram else "Ch.99 adjusted"
                    if not rate_source or rate_source == "General":
                        rate_source = ch99_label
                    elif "Ch.99" not in rate_source:
                        rate_source = f"{rate_source} · {ch99_label}"

            if ch99_applied:
                rate_source = "Ch.99 adjusted"
                if ch99_tradeprogram:
                    rate_source = f"Ch.99 ({ch99_tradeprogram})"

            chart_eligible = bool(effective_rate is not None)
            combined_rows.append(
                {
                    "country": country_name,
                    "corruption_score": country_meta["score"],
                    "risk_level": country_meta["level"],
                    "country_color": country_meta["color"],
                    "hts_code": product_meta["hts_code"],
                    "product_description": product_meta["description"],
                    "base_duty_pct": base_duty_pct,
                    "ad_valorem_rate": effective_rate,
                    "ch99_applied": ch99_applied,
                    "ch99_tradeprogram": ch99_tradeprogram,
                    "ch99_delta": ch99_delta,
                    "ch99_specific_surcharge": ch99_specific_text,
                    "general_duty_rate_text": base_general_text,
                    "duty_kind": duty_kind,
                    "duty_component_summary": duty_component_summary,
                    "specific_amount": effective_specific_amount,
                    "specific_unit": effective_specific_unit,
                    "specific_raw": effective_specific_raw,
                    "has_specific": has_specific,
                    "rate_source": rate_source,
                    "special_program_code": program_code_summary,
                    "special_program_label": program_label_summary,
                    "chart_eligible": chart_eligible,
                }
            )

    combined_df = pd.DataFrame(combined_rows)
    has_specific_data = bool(not combined_df.empty and combined_df["specific_amount"].notna().any())
    if special_override_count:
        logger.info(
            "Applied special duty overrides",
            extra={"override_rows": special_override_count},
        )
    if unresolved_codes:
        logger.warning(
            "Encountered unresolved special duty programs",
            extra={"codes": sorted(unresolved_codes)},
        )
    if combined_df.empty:
        logger.info("Combined dataframe empty; returning placeholders")
        empty_cols = [
            "country",
            "corruption_score",
            "risk_level",
            "hts_code",
            "product_description",
            "general_duty_rate_text",
            "duty_kind",
            "ad_valorem_rate",
            "chart_eligible",
        ]
        return (
            pd.DataFrame(),
            pd.DataFrame(columns=empty_cols),
            {"count": 0, "by_kind": {}, "samples": []},
            ch99_summary,
            [],
            has_specific_data,
        )

    corr_df_full = combined_df[combined_df["chart_eligible"]].copy()
    logger.info(
        "Built correlation dataframe",
        extra={"pairs": len(corr_df_full), "chart_exclusions": int((~combined_df["chart_eligible"]).sum())},
    )
    non_ad_summary = _summarize_non_ad_rows(combined_df[~combined_df["chart_eligible"]])
    duty_exclusions = combined_df[~combined_df["chart_eligible"]][
        ["country", "hts_code", "general_duty_rate_text", "duty_kind"]
    ].to_dict("records")

    rate_columns = combined_df.apply(_format_rate_columns, axis=1)
    combined_df = combined_df.assign(
        rate_primary=[primary for primary, _ in rate_columns],
        rate_secondary=[secondary for _, secondary in rate_columns],
        plotted=["Yes" if eligible else "No" for eligible in combined_df["chart_eligible"]],
    )
    ch99_summary["programs"] = sorted(ch99_programs)
    return corr_df_full, combined_df, non_ad_summary, ch99_summary, duty_exclusions, has_specific_data


def build_risk_snapshot(risk_df: pd.DataFrame, selected_countries: list[str]) -> list[dict]:
    """Small summary of risk metadata used for UI pills."""
    if not selected_countries:
        return []
    subset = risk_df[risk_df["country"].isin(selected_countries)]
    if subset.empty:
        return []
    return subset[["country", "score", "level", "color", "year"]].to_dict("records")


def render_correlation_chart(df: pd.DataFrame, mode: str = "ad_valorem") -> go.Figure | None:
    """Return a Plotly scatter figure for the correlation pairs.

    mode='ad_valorem' — Y axis = effective ad valorem rate (%, ch99-adjusted where applicable)
    mode='specific'   — Y axis = specific duty amount ($/kg, ¢/dozen, etc.)
    """
    if df.empty:
        return None

    df = df.copy()
    df["Trade Program"] = df["ch99_tradeprogram"].fillna("—") if "ch99_tradeprogram" in df.columns else "—"
    df["Base Rate"] = df["general_duty_rate_text"] if "general_duty_rate_text" in df.columns else ""

    if mode == "specific":
        plot_df = df[df["specific_amount"].notna()].copy() if "specific_amount" in df.columns else pd.DataFrame()
        if plot_df.empty:
            return None

        units = plot_df["specific_unit"].dropna().unique() if "specific_unit" in plot_df.columns else []
        unit_label = units[0].strip() if len(units) == 1 else "unit"

        fig = go.Figure()
        countries = plot_df["country"].unique()
        colors = ["#0B2A4A", "#4F6D7A", "#7AAFDE", "#B5D4F4"]
        for i, country in enumerate(countries):
            row = plot_df[plot_df["country"] == country]
            fig.add_trace(go.Bar(
                name=country,
                x=[country],
                y=row["specific_amount"].values,
                marker_color=colors[i % len(colors)],
                text=[f"{v:.4f} {unit_label}" for v in row["specific_amount"].values],
                textposition="outside",
            ))
        fig.update_layout(
            yaxis_title=f"Specific Duty Rate ({unit_label})",
            template="plotly_white",
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
            bargap=0.35,
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#F0F2F5"),
        )
        return fig

    # --- ad_valorem mode ---
    plot_df = df[df["ad_valorem_rate"].notna()].copy() if "ad_valorem_rate" in df.columns else pd.DataFrame()
    if plot_df.empty:
        return None

    def _delta_label(row) -> str:
        if not row.get("ch99_applied"):
            return "—"
        base = row.get("base_duty_pct")
        eff = row.get("ad_valorem_rate")
        if base is None or eff is None:
            return "—"
        delta = eff - base
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.2f}%"

    plot_df["Rate Source"] = plot_df["ch99_applied"].map({True: "Ch.99 adjusted", False: "Base rate"}) if "ch99_applied" in plot_df.columns else "Base rate"
    plot_df["Ch.99 Δ"] = plot_df.apply(_delta_label, axis=1)

    has_ch99 = "ch99_applied" in plot_df.columns
    agg_dict = {
        "ad_valorem_rate": ("ad_valorem_rate", "first"),
        "corruption_score": ("corruption_score", "first"),
        "risk_level": ("risk_level", "first"),
    }
    if has_ch99:
        agg_dict["ch99_applied"] = ("ch99_applied", "first")
    summary = plot_df.groupby("country", as_index=False).agg(**agg_dict)

    palette = ["#0B2A4A", "#4F6D7A", "#7AAFDE", "#B5D4F4", "#378ADD"]

    fig = go.Figure()

    for i, row in summary.iterrows():
        ch99 = bool(row.get("ch99_applied", False)) if has_ch99 else False
        color = palette[i % len(palette)]
        level = row.get("risk_level", "")

        fig.add_trace(go.Scatter(
            x=[row["corruption_score"]],
            y=[row["ad_valorem_rate"]],
            mode="markers+text",
            name=row["country"],
            text=[row["country"]],
            textposition="top center",
            textfont=dict(size=13, color=color),
            marker=dict(
                size=48,
                color=color,
                line=dict(width=2.5, color="white"),
                symbol="circle",
            ),
            hovertemplate=(
                f"<b>{row['country']}</b><br>"
                f"Corruption score: {row['corruption_score']:.1f} / 100<br>"
                f"Effective duty: {row['ad_valorem_rate']:.2f}%<br>"
                f"Risk level: {level}<br>"
                f"{'⚠ Ch.99 adjusted' if ch99 else 'Base rate'}"
                "<extra></extra>"
            ),
        ))

        fig.add_annotation(
            x=row["corruption_score"],
            y=row["ad_valorem_rate"],
            text=f"{row['ad_valorem_rate']:.1f}%",
            showarrow=False,
            font=dict(size=11, color="white", family="monospace"),
            yshift=0,
        )

    max_risk = summary["corruption_score"].max()
    max_duty = summary["ad_valorem_rate"].max()



    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=50),
        plot_bgcolor="white",
        showlegend=False,
        xaxis=dict(
            title="Country corruption score (lower = safer)",
            gridcolor="#F0F2F5",
            zeroline=False,
            range=[-2, max_risk * 1.3],
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Effective duty rate (%)",
            gridcolor="#F0F2F5",
            zeroline=False,
            ticksuffix="%",
            range=[-0.5, max_duty * 1.35],
            tickfont=dict(size=12),
        ),
    )
    if "ch99_applied" in plot_df.columns:
        adjusted = plot_df[plot_df["ch99_applied"]]
        if not adjusted.empty:
            fig.add_trace(
                go.Scatter(
                    x=adjusted["corruption_score"],
                    y=adjusted["ad_valorem_rate"],
                    mode="markers",
                    marker=dict(
                        size=20,
                        symbol="circle-open",
                        line=dict(color="rgba(255,140,0,0.85)", width=2.5),
                        color="rgba(0,0,0,0)",
                    ),
                    name="Ch.99 adjusted",
                    hoverinfo="skip",
                )
            )
    return fig


def _build_summary_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "count_pairs": 0,
            "countries": [],
            "hts_codes": [],
            "risk_min": None,
            "risk_max": None,
            "duty_min": None,
            "duty_max": None,
        }
    def _round_one(value: float | None) -> float | None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return round(float(value), 1)

    return {
        "count_pairs": len(df),
        "countries": sorted(df["country"].unique().tolist()),
        "hts_codes": sorted(df["hts_code"].unique().tolist()),
        "risk_min": _round_one(df["corruption_score"].min()),
        "risk_max": _round_one(df["corruption_score"].max()),
        "duty_min": float(df["ad_valorem_rate"].min()),
        "duty_max": float(df["ad_valorem_rate"].max()),
    }


def _round_display(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(round(float(value), 2)).rstrip("0").rstrip(".") or "0"


def _derive_headline(
    df: pd.DataFrame,
    selected_countries: Iterable[str],
    selected_products: Iterable[str],
) -> str:
    country_text = ", ".join(selected_countries) if selected_countries else "your selections"
    product_text = ", ".join(selected_products) if selected_products else "the chosen HTS codes"

    if df.empty:
        return f"No overlapping tariff-risk data for {country_text} on {product_text}."

    unique_pairs = df[["country", "hts_code"]]
    if len(unique_pairs) == 1:
        row = df.iloc[0]
        duty_text = row.get("general_duty_rate_text") or (
            f"{_round_display(row.get('ad_valorem_rate'))}% ad valorem"
            if pd.notna(row.get("ad_valorem_rate"))
            else "the posted duty"
        )
        risk_level = (row.get("risk_level") or "Unknown").lower()
        return f"{row['country']} is your only {risk_level} option for HTS {row['hts_code']} with {duty_text}."

    duties = df["ad_valorem_rate"]
    corruption_scores = df["corruption_score"]
    if duties.nunique(dropna=False) == 1 and corruption_scores.nunique(dropna=False) == 1:
        duty_val = _round_display(duties.iloc[0])
        risk_val = round(float(corruption_scores.iloc[0]), 1)
        return (
            f"All {len(unique_pairs)} selections share the same profile: {duty_val}% duty and risk ≈ {risk_val}."
        )

    min_duty = duties.min()
    best_by_duty = df[duties == min_duty]
    min_risk = best_by_duty["corruption_score"].min()
    best_rows = best_by_duty[best_by_duty["corruption_score"] == min_risk]
    best_countries = sorted(best_rows["country"].unique().tolist())
    best_codes = sorted(best_rows["hts_code"].unique().tolist())
    duty_display = _round_display(min_duty)
    risk_display = round(float(min_risk), 1)
    country_phrase = ", ".join(best_countries)
    code_phrase = ", ".join(best_codes)
    return (
        f"{country_phrase} offer(s) the lowest duty ({duty_display}%) while staying around risk {risk_display} for HTS {code_phrase}."
    )


def stream_analysis_to_placeholder(
    df: pd.DataFrame,
    deployment_id: str,
    placeholder: st.delta_generator.DeltaGenerator | None,
    headline: str | None = None,
) -> str:
    """LLM-generated narrative summary of the correlation DataFrame."""
    subset = df.head(SUMMARY_SAMPLE_LIMIT)
    stats = _build_summary_stats(df)
    payload = {
        "stats": stats,
        "sample_rows": subset.to_dict("records"),
    }
    headline_note = ""
    if headline:
        headline_note = (
            f"The user already saw this headline: \"{headline}\". "
            "Do not restate it verbatim or contradict it; elaborate with supporting context instead."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a trade intelligence assistant with the mindset of a "
                "pragmatic trade manager. You help users understand tariff exposure "
                "and country risk for specific HTS codes.\n\n"
                "Your goals:\n"
                "- Give a medium-length answer: a short paragraph plus 1–3 concise, "
                "practical suggestions.\n"
                "- Highlight notable patterns — e.g. countries where duty and risk "
                "diverge, or where a trade program drives the effective rate.\n"
                "- Keep recommendations business-focused and specific to the countries "
                "and HTS codes in view.\n\n"
                "Constraints:\n"
                "- Speak as a knowledgeable trade advisor, not as someone analyzing a "
                "dataset or inspecting rows. Never reference 'the data', 'the dataset', "
                "'sample rows', 'sample size', 'supplied data', or 'the provided rows'.\n"
                "- Do not invent numbers or countries beyond what you have been given.\n"
                "- This is not legal advice; if your answer could be interpreted that "
                "way, remind the user to consult their customs or trade counsel.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "Summarize the relationship between country risk and ad valorem duty rates "
                "for the countries and HTS code(s) below. Highlight any extremes or "
                "interesting patterns, then give 1–3 actionable, business-focused "
                "suggestions. Speak as a trade advisor — do not reference datasets, rows, "
                "or sample sizes.\n\n"
                f"{headline_note}\n\n"
                f"{json.dumps(payload)}"
            ),
        },
    ]
    stream = openai.chat.completions.create(
        model=deployment_id,
        messages=messages,
        temperature=1,
        stream=True,
    )
    full_text = ""
    headline_html = (
        f"<p class='analysis-headline'><strong>{headline}</strong></p>" if headline else ""
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if not delta:
            continue
        full_text += delta
        if placeholder:
            placeholder.markdown(
                f"<div class='bubble-bot'>{headline_html}{full_text}▌</div>",
                unsafe_allow_html=True,
            )
    if placeholder:
        placeholder.markdown(
            f"<div class='bubble-bot'>{headline_html}{full_text}</div>",
            unsafe_allow_html=True,
        )
    return full_text.strip()


def queue_analysis_request(
    selected_countries: list[str],
    selected_products: list[str],
) -> bool:
    """Store a pending analysis request in session state."""
    signature = compute_selection_signature(selected_countries, selected_products)
    if not signature:
        logger.warning("Analysis request skipped; selections incomplete")
        return False
    st.session_state["analysis_request"] = {
        "countries": list(selected_countries),
        "products": list(selected_products),
        "signature": signature,
    }
    # Reset chart mode so new analyses always open in ad_valorem view
    st.session_state["chart_mode"] = "ad_valorem"
    logger.info(
        "Queued analysis request",
        extra={"countries": selected_countries, "products": selected_products},
    )
    return True


def maybe_run_analysis(
    conn,
    deployment_id: str,
    risk_df: pd.DataFrame,
    placeholder: st.delta_generator.DeltaGenerator | None,
) -> None:
    """If an analysis request is queued, execute it and emit chat + charts."""
    request = st.session_state.get("analysis_request")
    if not request or st.session_state.get("analysis_inflight"):
        return

    countries = request.get("countries", [])
    products = request.get("products", [])
    signature = request.get("signature")
    if not countries or not products:
        st.session_state["analysis_request"] = None
        logger.warning("Analysis request discarded due to missing selections")
        return

    run_id = uuid.uuid4().hex[:8]
    st.session_state["analysis_request"] = None
    st.session_state["analysis_inflight"] = True
    st.session_state["analysis_active_run"] = run_id
    success = False
    logger.info(
        "Starting analysis run",
        extra={"analysis_run": run_id, "countries": countries, "products": products},
    )

    try:
        def ll_to_xy(lat, lon, w=500, h=250):
            x = round((lon + 180) * (w / 360), 1)
            y = round((90 - lat) * (h / 180), 1)
            return x, y

        pts = []
        for ctry in countries:
            if ctry in COUNTRY_COORDINATES:
                lat, lon = COUNTRY_COORDINATES[ctry]
                x, y = ll_to_xy(lat, lon)
                pts.append({"name": ctry, "x": x, "y": y})

        paths_svg = ""
        travelers_svg = ""
        labels_svg = ""
        style_css = ""

        for i in range(len(pts) - 1):
            x1, y1 = pts[i]["x"], pts[i]["y"]
            x2, y2 = pts[i+1]["x"], pts[i+1]["y"]
            mx = round((x1+x2)/2, 1)
            my = round(min(y1,y2) - 40, 1)
            paths_svg += f'<path id="p{i}" d="M{x1},{y1} Q{mx},{my} {x2},{y2}" fill="none" stroke="#B5D4F4" stroke-width="1.5" stroke-dasharray="5 3" opacity="0.8"/>'
            delay = round(i * 0.6, 1)
            style_css += f".t{i}{{offset-path:path('M{x1},{y1} Q{mx},{my} {x2},{y2}');animation:mv{i} 2.4s ease-in-out {delay}s infinite;}}"
            style_css += f"@keyframes mv{i}{{0%{{offset-distance:0%;opacity:1}}85%{{offset-distance:100%;opacity:1}}100%{{offset-distance:100%;opacity:0}}}}"
            travelers_svg += f'<circle class="t{i}" r="5" fill="#378ADD" stroke="white" stroke-width="1.5"/>'

        for pt in pts:
            paths_svg += f'<circle cx="{pt["x"]}" cy="{pt["y"]}" r="6" fill="#0B2A4A" stroke="white" stroke-width="2"/>'
            labels_svg += f'<text x="{pt["x"]}" y="{pt["y"]-10}" text-anchor="middle" font-size="9" font-family="sans-serif" fill="#0B2A4A" font-weight="bold">{pt["name"]}</text>'

        country_list = ", ".join(countries)
        globe_html = f"""<div style="background:#EAF2FB;border-radius:12px;padding:18px 20px;border:0.5px solid #B5D4F4;margin:8px 0;">
<p style="font-size:13px;color:#0B2A4A;font-weight:500;margin-bottom:10px;text-align:center;">Analyzing {country_list}…</p>
<svg viewBox="0 0 500 250" width="100%" style="border-radius:8px;background:#EAF2FB;">
<style>{style_css}</style>
<rect width="500" height="250" fill="#E1EDF8" rx="8"/>
<line x1="0" y1="125" x2="500" y2="125" stroke="#C5D9ED" stroke-width="0.5"/>
<line x1="250" y1="0" x2="250" y2="250" stroke="#C5D9ED" stroke-width="0.5"/>
{paths_svg}{travelers_svg}{labels_svg}
</svg>
<p style="font-size:11px;color:#4F6D7A;text-align:center;margin-top:8px;">Querying HTS database &middot; Scoring V-Dem indicators &middot; Generating analysis</p>
</div>"""

        pts_3d = []
        for ctry in countries:
            if ctry in COUNTRY_COORDINATES:
                lat, lon = COUNTRY_COORDINATES[ctry]
                pts_3d.append({"name": ctry, "lat": lat, "lon": lon})

        if placeholder:
            thinking_steps = [
                (f"Identifying selected countries: <strong>{', '.join(countries)}</strong>", 0.8),
                (f"Querying HTS SQLite database for: <strong>{', '.join(products)}</strong>", 1.0),
                ("Parsing general duty rates and checking Chapter 99 tariff adjustments...", 1.0),
                ("Loading V-Dem indicators and computing corruption scores per country...", 1.0),
                ("Building correlation matrix of corruption scores vs effective duty rates...", 0.8),
                ("Sending data to <strong>Azure OpenAI GPT-5-mini</strong> for compliance narrative...", 0.8),
                ("<em style='color:#4F6D7A;'>Generating actionable insights and recommendations...</em>", 0.5),
            ]

            def render_thinking(steps_so_far):
                rows = ""
                for idx, (text, _) in enumerate(steps_so_far):
                    num = str(idx + 1).zfill(2)
                    rows += f"<div style='display:flex;gap:10px;align-items:flex-start;margin-bottom:8px;'><span style='color:#0B2A4A;font-weight:600;font-size:11px;min-width:20px;margin-top:2px;'>{num}</span><span style='font-size:13px;color:#374151;line-height:1.6;'>{text}</span></div>"
                placeholder.markdown(f"""
<div style='background:#F8FAFC;border:0.5px solid #E2E8F0;border-radius:12px;padding:18px 20px;margin:8px 0;'>
  <div style='display:flex;align-items:center;gap:8px;margin-bottom:14px;'>
    <div style='width:8px;height:8px;border-radius:50%;background:#0B2A4A;'></div>
    <span style='font-size:13px;font-weight:500;color:#0B2A4A;'>Thinking...</span>
  </div>
  {rows}
</div>""", unsafe_allow_html=True)

            import time
            shown = []
            for step in thinking_steps:
                shown.append(step)
                render_thinking(shown)
                time.sleep(0.4)

        with st.spinner("Running analysis..."):
            tariff_df = fetch_tariffs_for_codes(conn, products)
            ch99_df = fetch_ch99_for_codes_and_countries(conn, products, countries)
            (
                corr_df_full,
                corr_df_table,
                non_ad_summary,
                ch99_summary,
                duty_exclusions,
                has_specific_data,
            ) = build_correlation_dataframe(countries, tariff_df, risk_df, ch99_df)
            headline_source_df = corr_df_full if not corr_df_full.empty else corr_df_table
            headline_text = _derive_headline(headline_source_df, countries, products)
            timestamp = datetime.now().strftime("%I:%M %p")
            risk_snapshot = build_risk_snapshot(risk_df, countries)
            non_ad_text = _format_non_ad_summary_for_text(non_ad_summary)
            duty_exclusion_message = None
            if duty_exclusions:
                preview_labels = [
                    f"{item['hts_code']} ({item['general_duty_rate_text']})"
                    for item in duty_exclusions[:3]
                ]
                preview = ", ".join(preview_labels)
                extra = len(duty_exclusions) - len(preview_labels)
                if extra > 0:
                    preview = f"{preview}, +{extra} more"
                duty_exclusion_message = (
                    f"Skipped {len(duty_exclusions)} product-country pair(s) with non-percentage duties: {preview}."
                )
            if non_ad_summary.get("count"):
                if non_ad_text:
                    st.info(non_ad_text)
                if duty_exclusion_message:
                    st.warning(duty_exclusion_message)
                logger.info(
                    "Analysis run has non-ad duties",
                    extra={"analysis_run": run_id, "non_ad_count": non_ad_summary.get("count")},
                )
            elif duty_exclusion_message:
                st.warning(duty_exclusion_message)
            headline_html = (
                f"<p class='analysis-headline'><strong>{headline_text}</strong></p>" if headline_text else ""
            )

            if corr_df_full.empty:
                if len(corr_df_table) and non_ad_summary.get("count"):
                    summary_text = (
                        "All selected products currently rely on quantity- or rule-based duty rates, so no ad valorem analysis is available."
                    )
                    if non_ad_text:
                        summary_text = f"{summary_text} {non_ad_text}"
                else:
                    summary_text = "No overlapping tariff-risk data for the current selections."
                specific_sentence = _summarize_specific_effects(corr_df_table)
                if specific_sentence:
                    summary_text = f"{summary_text}\n\n{specific_sentence}"
                if placeholder:
                    placeholder.markdown(
                        f"<div class='bubble-bot'>{headline_html}{summary_text}</div>",
                        unsafe_allow_html=True,
                    )
                display_table, display_columns = _build_display_table(corr_df_table)
                append_message(
                    {
                        "role": "assistant",
                        "content": summary_text,
                        "time": timestamp,
                        "type": "analysis",
                        "chart_data": display_table.to_dict("records"),
                        "chart_columns": display_columns,
                        "risk_snapshot": risk_snapshot,
                        "selections": {
                            "countries": countries,
                            "products": products,
                        },
                        "non_ad_summary": non_ad_summary,
                        "non_ad_summary_text": non_ad_text,
                        "headline": headline_text,
                        "ch99_summary": ch99_summary,
                        "duty_exclusions": duty_exclusions,
                        "duty_exclusion_message": duty_exclusion_message,
                        "has_specific_data": has_specific_data,
                    }
                )
            else:
                fig = render_correlation_chart(corr_df_full)
                summary_text = stream_analysis_to_placeholder(
                    corr_df_full, deployment_id, placeholder, headline=headline_text
                )
                if non_ad_text:
                    summary_text = f"{summary_text}\n\n{non_ad_text}"
                    if placeholder:
                        placeholder.markdown(
                            f"<div class='bubble-bot'>{headline_html}{summary_text}</div>",
                            unsafe_allow_html=True,
                        )
                specific_sentence = _summarize_specific_effects(corr_df_table)
                if specific_sentence:
                    summary_text = f"{summary_text}\n\n{specific_sentence}"
                    if placeholder:
                        placeholder.markdown(
                            f"<div class='bubble-bot'>{headline_html}{summary_text}</div>",
                            unsafe_allow_html=True,
                        )
                display_table, display_columns = _build_display_table(corr_df_table)
                append_message(
                    {
                        "role": "assistant",
                        "content": summary_text,
                        "time": timestamp,
                        "type": "analysis",
                        "plotly_fig": fig.to_dict() if fig else None,
                        # raw_chart_data/raw_chart_columns: full corr_df_full for toggle re-rendering
                        "raw_chart_data": corr_df_full.to_dict("records"),
                        "raw_chart_columns": corr_df_full.columns.tolist(),
                        # chart_data/chart_columns: formatted display table
                        "chart_data": display_table.to_dict("records"),
                        "chart_columns": display_columns,
                        "risk_snapshot": risk_snapshot,
                        "selections": {
                            "countries": countries,
                            "products": products,
                        },
                        "non_ad_summary": non_ad_summary,
                        "non_ad_summary_text": non_ad_text,
                        "headline": headline_text,
                        "ch99_summary": ch99_summary,
                        "duty_exclusions": duty_exclusions,
                        "duty_exclusion_message": duty_exclusion_message,
                        "has_specific_data": has_specific_data,
                    }
                )
            st.session_state["correlation_signature"] = signature
            success = True
            logger.info(
                "Analysis run completed",
                extra={
                    "analysis_run": run_id,
                    "pairs": len(corr_df_full),
                    "table_rows": len(corr_df_table),
                    "non_ad_rows": non_ad_summary.get("count", 0),
                    "ch99_adjusted": ch99_summary.get("n_adjusted"),
                },
            )
    except Exception as exc:  # pragma: no cover - Streamlit handles UI errors
        logger.exception("Correlation analysis failed", extra={"analysis_run": run_id})
        if placeholder:
            placeholder.markdown(
                f"<div class='bubble-bot'>Error: {exc}</div>",
                unsafe_allow_html=True,
            )
        st.error(f"Correlation analysis failed: {exc}")
    finally:
        if st.session_state.get("analysis_active_run") == run_id:
            st.session_state["analysis_inflight"] = False
            st.session_state["analysis_active_run"] = None
        if success:
            st.rerun()
=======
"""Analysis pipeline helpers for ImportInsight AI.

This module contains the correlation/risk analysis helpers used by the Streamlit
app. UI components call into these functions; this file does not render any UI
on its own.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime
import uuid
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Iterable, Tuple

import openai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from duty_parser import DutyRate, parse_general_duty
from special_rates import parse_special_duty
from chat import append_message
from session import compute_selection_signature
from logger import setup_logger

logger = setup_logger(__name__)

# Static latitude/longitude map reused by multiple visualisations; defined once to
# avoid rebuilding the same literal dictionary on every Streamlit rerun.
COUNTRY_COORDINATES = {
    "Afghanistan": (33.9, 67.7),
    "Albania": (41.1, 20.2),
    "Algeria": (28.0, 1.7),
    "Angola": (-11.2, 17.9),
    "Argentina": (-38.4, -63.6),
    "Armenia": (40.1, 45.0),
    "Australia": (-25.3, 133.8),
    "Austria": (47.5, 14.6),
    "Azerbaijan": (40.1, 47.6),
    "Bangladesh": (23.7, 90.4),
    "Belgium": (50.8, 4.5),
    "Bolivia": (-16.3, -63.6),
    "Brazil": (-14.2, -51.9),
    "Bulgaria": (42.7, 25.5),
    "Cambodia": (12.6, 104.9),
    "Cameroon": (7.4, 12.4),
    "Canada": (56.1, -106.3),
    "Chad": (15.5, 18.7),
    "Chile": (-35.7, -71.5),
    "China": (35.9, 104.2),
    "Colombia": (4.6, -74.3),
    "Denmark": (56.3, 9.5),
    "Ecuador": (-1.8, -78.2),
    "Egypt": (26.8, 30.8),
    "Eritrea": (15.2, 39.8),
    "Ethiopia": (9.1, 40.5),
    "Finland": (61.9, 25.7),
    "France": (46.2, 2.2),
    "Germany": (51.2, 10.5),
    "Ghana": (7.9, -1.0),
    "Greece": (39.1, 21.8),
    "India": (20.6, 79.0),
    "Indonesia": (-0.8, 113.9),
    "Iran": (32.4, 53.7),
    "Iraq": (33.2, 43.7),
    "Italy": (41.9, 12.6),
    "Japan": (36.2, 138.3),
    "Jordan": (30.6, 36.2),
    "Kazakhstan": (48.0, 66.9),
    "Kenya": (-0.0, 37.9),
    "Libya": (26.3, 17.2),
    "Malaysia": (4.2, 108.0),
    "Mali": (17.6, -2.0),
    "Mexico": (23.6, -102.6),
    "Morocco": (31.8, -7.1),
    "Netherlands": (52.1, 5.3),
    "Nicaragua": (12.9, -85.2),
    "Nigeria": (9.1, 8.7),
    "Norway": (60.5, 8.5),
    "Pakistan": (30.4, 69.3),
    "Peru": (-9.2, -75.0),
    "Philippines": (12.9, 121.8),
    "Poland": (51.9, 19.1),
    "Portugal": (39.4, -8.2),
    "Romania": (45.9, 24.9),
    "Russia": (61.5, 105.3),
    "Saudi Arabia": (23.9, 45.1),
    "Senegal": (14.5, -14.5),
    "Singapore": (1.4, 103.8),
    "Somalia": (5.2, 46.2),
    "South Africa": (-30.6, 22.9),
    "South Korea": (35.9, 127.8),
    "South Sudan": (6.9, 31.3),
    "Spain": (40.5, -3.7),
    "Sri Lanka": (7.9, 80.8),
    "Sudan": (12.9, 30.2),
    "Sweden": (60.1, 18.6),
    "Switzerland": (46.8, 8.2),
    "Syria": (34.8, 38.8),
    "Taiwan": (23.7, 121.0),
    "Tanzania": (-6.4, 34.9),
    "Thailand": (15.9, 100.9),
    "Turkey": (38.9, 35.2),
    "Uganda": (1.4, 32.3),
    "Ukraine": (48.4, 31.2),
    "United Arab Emirates": (23.4, 53.8),
    "United Kingdom": (55.4, -3.4),
    "United States": (37.1, -95.7),
    "Uruguay": (-32.5, -55.8),
    "Venezuela": (6.4, -66.6),
    "Vietnam": (14.1, 108.3),
    "Yemen": (15.6, 48.5),
    "Zimbabwe": (-19.0, 29.2),
}

SUMMARY_SAMPLE_LIMIT = 60
DISPLAY_COLUMN_MAP = [
    ("country", "Country"),
    ("corruption_score", "Corruption Score"),
    ("hts_code", "HTS Code"),
    ("product_description", "Product"),
    ("general_duty_rate_text", "Base Rate"),
    ("ad_valorem_rate", "Effective Rate (%)"),
    ("ch99_delta", "Ch.99 Δ"),
    ("ch99_tradeprogram", "Trade Program"),
     ("ch99_specific_surcharge", "Ch.99 Specific"),
    ("rate_source", "Rate Source"),
    ("plotted", "Plotted"),
]


def _format_percent(value: float | None) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    formatted = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{formatted}%"


def _normalize_unit(unit: str | None) -> str:
    if not unit:
        return ""
    cleaned = unit.strip()
    if cleaned and not cleaned.startswith("/"):
        cleaned = f"/{cleaned}"
    return cleaned


def _format_decimal_preserving(value: float) -> str:
    try:
        dec_value = Decimal(str(value)).normalize()
    except (InvalidOperation, ValueError):
        return str(value)
    text = format(dec_value, "f")
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def _format_specific(value: float | None, unit: str | None, raw: str | None) -> str | None:
    cleaned_raw = ""
    if isinstance(raw, str):
        cleaned_raw = raw.strip()
    elif raw is not None:
        # Pandas often converts empty strings to NaN/NA scalars, so drop them silently.
        if pd.isna(raw):
            cleaned_raw = ""
        else:
            cleaned_raw = str(raw).strip()
    if cleaned_raw:
        return cleaned_raw
    if value is None:
        return None
    unit_fmt = _normalize_unit(unit) or "/unit"
    amount = _format_decimal_preserving(value)
    return f"{amount}{unit_fmt}"


def _hts_ancestor_codes(code: str) -> list[str]:
    """Return parent HTS codes by truncating dotted segments."""
    if not code:
        return []
    sanitized = code.strip()
    if not sanitized:
        return []
    parts = sanitized.split(".")
    ancestors: list[str] = []
    while len(parts) > 1:
        parts = parts[:-1]
        ancestors.append(".".join(parts))
    return ancestors


def _normalize_denominator(unit_text: str | None) -> str | None:
    if not unit_text:
        return None
    cleaned = unit_text.lower().replace(" per ", "/")
    for token in ("$", "usd", "dollars", "dollar", "cents", "cent", "¢"):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.strip()
    if "/" in cleaned:
        cleaned = cleaned.split("/", 1)[1]
    cleaned = cleaned.lstrip("/").strip()
    return cleaned or None


def _convert_specific_value(value: float | None, unit_hint: str | None, raw_hint: str | None = None) -> float | None:
    """Convert specific duty amounts to USD based on currency hint."""
    if value is None:
        return None
    tokens = (unit_hint or "") + " " + (raw_hint or "")
    lower = tokens.lower()
    if "cent" in lower or "¢" in lower:
        return value / 100.0
    return value


def _decorate_specific_raw(amount: float | None, base_raw: str | None, unit: str | None) -> str | None:
    if amount is None:
        return None
    unit_suffix = (unit or "").strip()
    if unit_suffix and not unit_suffix.startswith("/"):
        unit_suffix = f"/{unit_suffix}"
    symbol = ""
    lower_raw = (base_raw or "").lower()
    if "$" in (base_raw or ""):
        symbol = "$"
        formatted_value = _format_decimal_preserving(amount)
    elif "¢" in lower_raw or "cent" in lower_raw:
        symbol = "¢"
        formatted_value = _format_decimal_preserving(amount * 100.0)
    else:
        formatted_value = _format_decimal_preserving(amount)
    return f"{symbol}{formatted_value}{unit_suffix}".strip()


def _resolve_general_duty_rate(
    conn: sqlite3.Connection,
    code: str,
    cache: dict[str, tuple[str, str | None]],
) -> tuple[str, str | None]:
    """Return (duty_text, source_code) for the first ancestor with a general duty rate."""
    cached = cache.get(code)
    if cached is not None:
        return cached
    row = conn.execute("SELECT general_duty_rate FROM hts WHERE hts_code = ?", (code,)).fetchone()
    text = ""
    if row:
        text = (row[0] or "").strip()
    if text:
        cache[code] = (text, code)
        return cache[code]
    for ancestor in _hts_ancestor_codes(code):
        candidate = _resolve_general_duty_rate(conn, ancestor, cache)
        if candidate[0]:
            cache[code] = candidate
            return candidate
    cache[code] = ("", None)
    return cache[code]


def fetch_ch99_for_codes_and_countries(
    conn: sqlite3.Connection,
    selected_codes: list[str],
    selected_countries: list[str],
) -> pd.DataFrame:
    """Return ALL Chapter 99 rows for each (hts_code, queried_country/Global) pair.

    Queries the raw chapter_99 table with the multi-format JOIN so that every
    applicable rule (e.g. a higher-priority CAFTA-DR row AND a lower-priority
    Section 122 additive row) is visible to _best_ch99_rule for proper combination.
    """
    if not selected_codes or not selected_countries:
        return pd.DataFrame()

    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM chapter_99 LIMIT 1")
    except sqlite3.Error as exc:
        logger.warning("Chapter 99 table missing; skipping overrides", exc_info=exc)
        return pd.DataFrame()

    code_ph = ",".join(["?"] * len(selected_codes))
    country_ph = ",".join(["?"] * len(selected_countries))
    query = f"""
        SELECT
            m.hts_code                                                      AS hts_code,
            c.COUNTRY                                                       AS ch99_country,
            CAST(NULLIF(c.NEWRATE_CLEAN, '') AS REAL)                       AS ch99_newrate,
            c.RATE_MODIFIER                                                 AS ch99_rate_modifier,
            COALESCE(CAST(NULLIF(c.ADDITIONAL_DUTY_PCT, '') AS REAL), 0.0) AS ch99_additional_pct,
            CAST(NULLIF(c.ADDITIONAL_VALUE, '') AS REAL)                    AS ch99_additional_value,
            c.ADDITIONAL_VALUE                                             AS ch99_additional_value_raw,
            c.ADDITIONAL_UNIT                                              AS ch99_additional_unit,
            c.TRADEPROGRAM                                                  AS ch99_tradeprogram,
            CAST(c.MATCH_PRIORITY AS INTEGER)                               AS ch99_match_priority
        FROM hts AS m
        JOIN chapter_99 AS c
            ON (
                c.HTS_MASTER_CODE = m.hts_code
                OR (LENGTH(c.HTS_MASTER_CODE) = 10
                    AND c.HTS_MASTER_CODE = SUBSTR(m.hts_code, 1, 10))
                OR (LENGTH(c.HTS_MASTER_CODE) = 12
                    AND SUBSTR(c.HTS_MASTER_CODE, 1, 10) || '.' || SUBSTR(c.HTS_MASTER_CODE, 11) = m.hts_code)
                OR (LENGTH(c.HTS_MASTER_CODE) = 2
                    AND c.HTS_MASTER_CODE = SUBSTR(m.hts_code, 1, 2))
                OR c.HTS_MASTER_CODE = 'ALL'
            )
        WHERE m.hts_code IN ({code_ph})
          AND (c.COUNTRY IN ({country_ph}) OR c.COUNTRY = 'Global')
    """
    params = selected_codes + selected_countries
    df = pd.read_sql_query(query, conn, params=params)
    return df


def _build_ch99_lookup(ch99_df: pd.DataFrame | None) -> dict[tuple[str, str], pd.DataFrame]:
    """Group Chapter 99 rows by (hts_code, country) for fast lookup."""
    if ch99_df is None or ch99_df.empty:
        return {}
    grouped: dict[tuple[str, str], pd.DataFrame] = {}
    for (hts_code, country), frame in ch99_df.groupby(["hts_code", "ch99_country"], sort=False):
        grouped[(hts_code, country)] = frame
    return grouped


def _best_ch99_rule(
    ch99_lookup: dict[tuple[str, str], pd.DataFrame],
    hts_code: str,
    country: str,
) -> dict | None:
    """Synthesize the net Chapter 99 effect for a (hts_code, country) pair.

    Because multiple rules from different trade programs can apply simultaneously
    (e.g. a CAFTA-DR-specific row at P10 and a Section 122 additive row at P30),
    this function:
      1. Finds the best (lowest MATCH_PRIORITY) Floor rule → sets the floor rate.
      2. Sums all additive ADDITIONAL_DUTY_PCT values from non-Floor, non-replacement rows.
      3. Finds the best replacement NEWRATE_CLEAN (non-Floor) row if one exists.
      4. Returns a synthetic rule dict combining Floor + additive so that
         apply_ch99_to_duty can compute max(base + additive, floor).
    """
    if not ch99_lookup:
        return None

    frames: list[pd.DataFrame] = []
    exact = ch99_lookup.get((hts_code, country))
    if exact is not None:
        frames.append(exact)
    global_rows = ch99_lookup.get((hts_code, "Global"))
    if global_rows is not None:
        frames.append(global_rows)
    if not frames:
        return None
    candidates = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    candidates = candidates.sort_values("ch99_match_priority")

    is_floor = candidates["ch99_rate_modifier"] == "Floor"
    floor_rows = candidates[is_floor]
    non_floor = candidates[~is_floor]

    # Best replacement rate (non-Floor with an actual NEWRATE_CLEAN number)
    replacement_rows = non_floor[non_floor["ch99_newrate"].notna()]
    if not replacement_rows.empty:
        return replacement_rows.iloc[0].to_dict()

    # Best Floor rate (lowest priority = highest precedence)
    best_floor_rate: float | None = None
    best_floor_program: str = ""
    if not floor_rows.empty:
        best_floor_row = floor_rows.iloc[0]
        v = best_floor_row["ch99_newrate"]
        try:
            f = float(v)
            best_floor_rate = None if f != f else f
        except (TypeError, ValueError):
            best_floor_rate = None
        best_floor_program = best_floor_row.get("ch99_tradeprogram", "")

    # Sum additive surcharges from non-Floor, non-replacement rows.
    #
    # Two exclusions:
    # 1. Deduplicate by (tradeprogram, country, match_priority, additional_pct) so that
    #    multiple NEWCODE rows from the same program (e.g. Nicaragua's 9903.01.49 and
    #    9903.02.47 both at P30 Section 122 0.18) are not double-counted.
    # 2. If a Floor row exists at the same (country, priority) level as an additive row,
    #    the Floor is mutually exclusive with the additive — exclude the additive.
    #    (e.g. USMCA Blocker Floor at P30/Canada supersedes Transshipment-Evasion +40%
    #    additive also at P30/Canada; they are alternative enforcement paths, not stackable.)
    floor_country_priorities: set[tuple] = {
        (r["ch99_country"], r["ch99_match_priority"])
        for _, r in floor_rows.iterrows()
    }
    raw_additive = non_floor[non_floor["ch99_newrate"].isna()]
    non_superseded_additive = raw_additive[
        ~raw_additive.apply(
            lambda r: (r["ch99_country"], r["ch99_match_priority"]) in floor_country_priorities,
            axis=1,
        )
    ].drop_duplicates(
        subset=["ch99_tradeprogram", "ch99_country", "ch99_match_priority", "ch99_additional_pct"]
    )
    additive_rows = non_superseded_additive
    total_additive: float = 0.0
    specific_components: list[str] = []
    for _, row in additive_rows.iterrows():
        v = row.get("ch99_additional_pct", 0.0)
        try:
            f = float(v)
            if f == f:  # not NaN
                total_additive += f
        except (TypeError, ValueError):
            pass

        raw_specific = row.get("ch99_additional_value_raw")
        specific_val = row.get("ch99_additional_value")
        specific_unit = (row.get("ch99_additional_unit") or "").strip()
        if raw_specific or specific_unit:
            normalized_amount = (
                _convert_specific_value(specific_val, specific_unit, raw_specific)
                if specific_val is not None
                else None
            )
            if raw_specific and (specific_val is None or specific_val != specific_val):
                amount_text = raw_specific.strip()
            elif specific_val is not None and specific_val == specific_val:
                amount_text = _format_decimal_preserving(specific_val)
            else:
                amount_text = ""
            text = amount_text
            if specific_unit:
                text = f"{text} {specific_unit}".strip()
            label = row.get("ch99_tradeprogram") or ""
            if label:
                text = f"{text} ({label})" if text else label
            specific_components.append(
                {
                    "amount": normalized_amount,
                    "denominator": _normalize_denominator(specific_unit or raw_specific),
                    "display": text.strip(),
                }
            )

    # Determine the representative trade program label
    if total_additive > 0 and not additive_rows.empty:
        tradeprogram = additive_rows.iloc[0].get("ch99_tradeprogram", "")
    elif best_floor_rate is not None:
        tradeprogram = best_floor_program
    elif not candidates.empty:
        tradeprogram = candidates.iloc[0].get("ch99_tradeprogram", "")
    else:
        tradeprogram = ""

    specific_text = "; ".join(
        comp["display"] for comp in specific_components if comp.get("display")
    )

    if best_floor_rate is None and total_additive == 0.0 and not specific_text:
        return None  # no usable rate content

    return {
        "hts_code": hts_code,
        "ch99_country": country,
        "ch99_newrate": best_floor_rate,
        "ch99_rate_modifier": "Floor" if best_floor_rate is not None else "",
        "ch99_additional_pct": total_additive,
        "ch99_tradeprogram": tradeprogram,
        "ch99_match_priority": int(candidates.iloc[0]["ch99_match_priority"]),
        "ch99_specific_surcharge": specific_text,
        "ch99_specific_components": specific_components,
    }


def apply_ch99_to_duty(
    base_rate: DutyRate,
    ch99: dict,
) -> float | None:
    """Calculate the effective ad-valorem duty rate after applying Chapter 99 logic.

    Scale conventions in the source CSV:
      - base_rate.ad_valorem_rate : percentage points  (5.0  = 5%)
      - NEWRATE_CLEAN (Floor rows): percentage points  (15.0 = 15%)
      - NEWRATE_CLEAN (non-Floor) : decimal fraction   (0.072 = 7.2%) → multiply by 100
      - ADDITIONAL_DUTY_PCT       : decimal fraction   (0.1  = 10%)  → multiply by 100

    Rules (in order):
      1. RATE_MODIFIER = 'Floor'  → duty = max(base + additive, floor_rate)
                                    both values are in percentage-point scale
      2. NEWRATE_CLEAN is a number → duty = NEWRATE_CLEAN * 100  (replaces base)
      3. Otherwise                 → duty = base_rate + ADDITIONAL_DUTY_PCT * 100
    """
    if base_rate is None or base_rate.ad_valorem_rate is None:
        return None

    base_av = base_rate.ad_valorem_rate  # percentage points; None for non-ad-valorem rates
    modifier = (ch99.get("ch99_rate_modifier") or "").strip()

    raw_newrate = ch99.get("ch99_newrate")
    raw_additional = ch99.get("ch99_additional_pct")

    def _to_float_or_none(v: object) -> float | None:
        if v is None:
            return None
        try:
            f = float(v)
            return None if f != f else f  # NaN check: NaN != NaN
        except (TypeError, ValueError):
            return None

    newrate = _to_float_or_none(raw_newrate)
    try:
        additional = float(raw_additional) if raw_additional is not None else 0.0
    except (TypeError, ValueError):
        additional = 0.0

    if modifier == "Floor":
        # Floor: effective = max(base + additive, floor_rate)
        # Both base_av and newrate are in percentage-point scale; additional is decimal fraction.
        if base_av is not None:
            base_with_additive = base_av + (additional * 100)
            if newrate is not None:
                return max(base_with_additive, newrate)
            return base_with_additive
        return base_av

    if newrate is not None:
        # Non-Floor NEWRATE_CLEAN is a decimal fraction — scale to percentage points
        return newrate * 100

    if base_av is not None:
        # ADDITIONAL_DUTY_PCT is a decimal fraction — scale before adding
        return base_av + (additional * 100)

    return base_av


def _format_rate_columns(row: pd.Series) -> Tuple[str, str]:
    percent = _format_percent(row.get("ad_valorem_rate"))
    specific = _format_specific(row.get("specific_amount"), row.get("specific_unit"), row.get("specific_raw"))
    general_text = (row.get("general_duty_rate_text") or "").strip()

    # When a duty has both ad valorem and specific components, the percent drives
    # the chart/primary cell while the full mixed string remains in Base Rate.
    if row.get("chart_eligible"):
        primary = percent or (general_text if general_text else specific or "")
        secondary = specific if (row.get("has_specific") and specific and percent) else ""
    elif row.get("has_specific") and specific:
        primary = specific
        secondary = ""
    elif general_text:
        primary = general_text
        secondary = ""
    else:
        primary = ""
        secondary = ""

    return primary or "", secondary or ""


def _format_non_ad_summary_for_text(summary: dict) -> str | None:
    if not summary or not summary.get("count"):
        return None
    parts = [
        f"{summary['count']} selection(s) have duties without a percentage component, so they were omitted from the scatter plot."
    ]
    by_kind = summary.get("by_kind") or {}
    if by_kind:
        kind_bits = ", ".join(f"{count} {kind}" for kind, count in by_kind.items())
        parts.append(f"Breakdown by duty kind: {kind_bits}.")
    samples = summary.get("samples") or []
    if samples:
        sample_text = "; ".join(samples)
        parts.append(f"Representative duty strings: {sample_text}.")
    if summary.get("mixed_present"):
        parts.append("Mixed duties are plotted using their % component only; the paired specific amounts remain visible in the table.")
    return " ".join(parts)


def _build_display_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    target_columns = [src for src, _ in DISPLAY_COLUMN_MAP]
    available = [col for col in target_columns if col in df.columns]
    column_headers = [label for src, label in DISPLAY_COLUMN_MAP if src in available]
    if df.empty or not available:
        return pd.DataFrame(columns=column_headers), column_headers
    display_df = df[available].copy().rename(
        columns={src: label for src, label in DISPLAY_COLUMN_MAP if src in available}
    )
    if "Corruption Score" in display_df.columns:
        display_df["Corruption Score"] = display_df["Corruption Score"].apply(
            lambda v: f"{float(v):.1f}" if pd.notna(v) else "—"
        )
    if "Effective Rate (%)" in display_df.columns:
        display_df["Effective Rate (%)"] = display_df["Effective Rate (%)"].apply(
            lambda v: f"{float(v):.2f}%" if pd.notna(v) else "—"
        )
    if "Ch.99 Δ" in display_df.columns:
        def _format_delta(val):
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return "—"
            sign = "+" if val >= 0 else ""
            return f"{sign}{val:.2f}%"

        display_df["Ch.99 Δ"] = display_df["Ch.99 Δ"].apply(_format_delta)
    if "Trade Program" in display_df.columns:
        display_df["Trade Program"] = display_df["Trade Program"].fillna("—")
    if "Rate Source" in display_df.columns:
        display_df["Rate Source"] = display_df["Rate Source"].fillna("General")
    if "Plotted" in display_df.columns:
        display_df["Plotted"] = display_df["Plotted"].fillna("No")
    return display_df, column_headers


def _summarize_specific_effects(df: pd.DataFrame) -> str | None:
    if df.empty or "ch99_specific_surcharge" not in df.columns:
        return None
    subset = df[
        df["ch99_specific_surcharge"].fillna("").astype(str).str.strip() != ""
    ]
    if subset.empty:
        return None
    bits: list[str] = []
    for _, row in subset.iterrows():
        specific_text = _format_specific(row.get("specific_amount"), row.get("specific_unit"), row.get("specific_raw"))
        label = row.get("ch99_specific_surcharge")
        descriptor = specific_text or label
        if descriptor:
            bits.append(f"{row.get('country')} {row.get('hts_code')}: {descriptor}")
    if not bits:
        return None
    return "Ch.99 specific surcharges now fix the per-unit rates at " + "; ".join(bits) + "."


def fetch_tariffs_for_codes(
    conn,
    selected_codes: list[str],
) -> pd.DataFrame:
    """Load and parse general duty rates for the selected HTS codes."""
    if not selected_codes:
        logger.info("No product codes selected; skipping tariff fetch")
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(selected_codes))
    query = (
        "SELECT hts_code, description, general_duty_rate, special_duty_rate "
        "FROM hts WHERE hts_code IN (" + placeholders + ")"
    )
    logger.info("Fetching tariff rows for selected products", extra={"product_count": len(selected_codes)})
    df = pd.read_sql_query(query, conn, params=selected_codes)
    if df.empty:
        logger.warning("Tariff lookup returned no rows for selected products")
        return df
    df = df.rename(columns={"general_duty_rate": "general_duty_rate_text"})
    df["general_duty_rate_text"] = df["general_duty_rate_text"].fillna("").astype(str)
    df["original_general_duty_rate_text"] = df["general_duty_rate_text"]
    df["general_duty_rate_source_code"] = df["hts_code"]
    inheritance_cache: dict[str, tuple[str, str | None]] = {}
    missing_mask = df["general_duty_rate_text"].str.strip() == ""
    if missing_mask.any():
        for idx, row in df[missing_mask].iterrows():
            fallback, source = _resolve_general_duty_rate(conn, row["hts_code"], inheritance_cache)
            if fallback:
                df.at[idx, "general_duty_rate_text"] = fallback
                df.at[idx, "general_duty_rate_source_code"] = source or row["hts_code"]
            else:
                logger.warning(
                    "No general duty rate found for %s or its ancestors", row["hts_code"]
                )
    df["special_duty_rate"] = df["special_duty_rate"].fillna("").astype(str)
    parsed_rates = df["general_duty_rate_text"].apply(parse_general_duty)

    def _summarize_components(rate) -> str:
        if not rate.components:
            return rate.raw_text
        return " + ".join(
            comp.raw_segment for comp in rate.components if comp.raw_segment
        ) or rate.raw_text

    df["duty_kind"] = [rate.kind for rate in parsed_rates]
    df["has_ad_valorem"] = [rate.has_ad_valorem for rate in parsed_rates]
    df["has_specific"] = [rate.has_specific for rate in parsed_rates]
    df["ad_valorem_rate"] = [rate.ad_valorem_rate for rate in parsed_rates]
    df["specific_amount"] = [rate.specific_amount for rate in parsed_rates]
    df["specific_unit"] = [rate.specific_unit for rate in parsed_rates]
    df["specific_raw"] = [
        next((comp.raw_segment for comp in rate.components if comp.type == "specific"), None)
        for rate in parsed_rates
    ]
    df["duty_notes"] = [rate.notes for rate in parsed_rates]
    df["duty_component_summary"] = [_summarize_components(rate) for rate in parsed_rates]
    df["parsed_duty_rate"] = parsed_rates
    df["special_duty_rule"] = [
        parse_special_duty(text) if text else None for text in df["special_duty_rate"]
    ]
    logger.info(
        "Parsed tariff duty structures",
        extra={"rows": len(df), "ad_valorem_rows": int(df["has_ad_valorem"].sum())},
    )
    return df


def _summarize_non_ad_rows(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"count": 0, "by_kind": {}, "samples": []}
    by_kind = Counter(df["duty_kind"].fillna("unknown"))
    samples = df["general_duty_rate_text"].dropna().head(5).tolist()
    return {
        "count": int(len(df)),
        "by_kind": dict(by_kind),
        "samples": samples,
        "mixed_present": bool((df["duty_kind"] == "mixed").any()),
    }


def build_correlation_dataframe(
    selected_countries: Iterable[str],
    tariff_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    ch99_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, list[dict], bool]:
    """Construct the chart dataframe, combined table, and metadata summaries."""
    selected_countries = list(selected_countries)
    logger.info(
        "Building correlation dataframe",
        extra={
            "countries": len(selected_countries),
            "tariff_rows": len(tariff_df),
        },
    )
    ch99_lookup = _build_ch99_lookup(ch99_df)
    ch99_available = bool(ch99_lookup)
    if not selected_countries or tariff_df.empty:
        empty_cols = [
            "country",
            "corruption_score",
            "risk_level",
            "hts_code",
            "product_description",
            "general_duty_rate_text",
            "rate_source",
            "special_program_code",
            "special_program_label",
            "duty_kind",
            "ad_valorem_rate",
            "chart_eligible",
        ]
        logger.info("Correlation dataframe empty due to missing selections or tariff rows")
        return (
            pd.DataFrame(),
            pd.DataFrame(columns=empty_cols),
            {"count": 0, "by_kind": {}, "samples": []},
            {"n_adjusted": 0, "n_total": 0, "programs": []},
            [],
            False,
        )

    country_subset = risk_df[risk_df["country"].isin(selected_countries)].copy()
    if country_subset.empty:
        empty_cols = [
            "country",
            "corruption_score",
            "risk_level",
            "hts_code",
            "product_description",
            "general_duty_rate_text",
            "rate_source",
            "special_program_code",
            "special_program_label",
            "duty_kind",
            "ad_valorem_rate",
            "chart_eligible",
        ]
        logger.info("Correlation dataframe empty because selected countries lack risk data")
        return (
            pd.DataFrame(),
            pd.DataFrame(columns=empty_cols),
            {"count": 0, "by_kind": {}, "samples": []},
            {"n_adjusted": 0, "n_total": 0, "programs": []},
            [],
            False,
        )

    country_records = country_subset.to_dict("records")
    product_records = tariff_df.to_dict("records")

    combined_rows: list[dict] = []
    special_override_count = 0
    unresolved_codes: set[str] = set()
    ch99_programs: set[str] = set()
    ch99_summary = {"n_adjusted": 0, "n_total": 0, "programs": []}
    for country_meta in country_records:
        for product_meta in product_records:
            country_name = country_meta["country"]
            special_rule = product_meta.get("special_duty_rule")
            rate_source = "General"
            program_code_summary = ""
            program_label_summary = ""
            applied_special = False
            ch99_specific_text = ""

            base_general_text = product_meta["general_duty_rate_text"]
            duty_component_summary = product_meta["duty_component_summary"]
            base_specific_amount = product_meta["specific_amount"]
            base_specific_unit = product_meta["specific_unit"]
            base_specific_raw = product_meta.get("specific_raw")
            effective_specific_amount = base_specific_amount
            effective_specific_unit = base_specific_unit
            effective_specific_raw = base_specific_raw
            duty_kind = product_meta["duty_kind"]
            has_specific = product_meta["has_specific"]

            base_duty_rate_obj = product_meta.get("parsed_duty_rate")
            if base_duty_rate_obj is None:
                base_duty_rate_obj = parse_general_duty(base_general_text)
            if base_duty_rate_obj:
                duty_kind = base_duty_rate_obj.kind or duty_kind
                if base_duty_rate_obj.specific_amount is not None:
                    effective_specific_amount = base_duty_rate_obj.specific_amount
                if base_duty_rate_obj.specific_unit:
                    effective_specific_unit = base_duty_rate_obj.specific_unit
                has_specific = has_specific or base_duty_rate_obj.has_specific

            base_has_ad_valorem = bool(
                base_duty_rate_obj and base_duty_rate_obj.has_ad_valorem and base_duty_rate_obj.ad_valorem_rate is not None
            )
            base_duty_pct = base_duty_rate_obj.ad_valorem_rate if base_has_ad_valorem else None
            effective_rate = base_duty_pct
            ch99_applied = False
            ch99_tradeprogram = None
            ch99_delta = None

            if special_rule:
                if special_rule.applies_to(country_name):
                    matching_codes = special_rule.codes_for_country(country_name)
                    applied_special = True
                    program_code_summary = ", ".join(matching_codes) if matching_codes else ""
                    program_label_summary = special_rule.format_labels(matching_codes or [])
                    rate_source = (
                        f"Special — {program_code_summary}"
                        if program_code_summary
                        else "Special — eligible"
                    )
                elif special_rule.has_dynamic_codes:
                    rate_source = f"Special (Unresolved — {', '.join(special_rule.dynamic_codes)})"
                    program_code_summary = ", ".join(special_rule.dynamic_codes)
                    program_label_summary = special_rule.format_labels(special_rule.dynamic_codes)
                    unresolved_codes.update(special_rule.dynamic_codes)

            if applied_special and special_rule:
                special_override_count += 1
                duty_rate = special_rule.duty_rate
                if duty_rate:
                    if duty_rate.ad_valorem_rate is not None:
                        effective_rate = duty_rate.ad_valorem_rate
                    if duty_rate.has_specific and duty_rate.specific_amount is not None:
                        effective_specific_amount = duty_rate.specific_amount
                        effective_specific_unit = duty_rate.specific_unit
                        effective_specific_raw = duty_rate.raw_text
                        has_specific = True
                    duty_kind = duty_rate.kind or duty_kind

            if base_has_ad_valorem:
                ch99_summary["n_total"] += 1

            rule = _best_ch99_rule(ch99_lookup, product_meta["hts_code"], country_name) if ch99_available else None
            if rule is not None and base_has_ad_valorem:
                adjusted_rate = apply_ch99_to_duty(base_duty_rate_obj, rule)
                if adjusted_rate is not None:
                    effective_rate = adjusted_rate
                    base_val = base_duty_pct or 0.0
                    has_change = base_duty_pct is None or not math.isclose(adjusted_rate, base_val, rel_tol=1e-6)
                    if has_change:
                        ch99_applied = True
                        ch99_delta = adjusted_rate - base_val if base_duty_pct is not None else adjusted_rate
                        ch99_summary["n_adjusted"] += 1
                    ch99_tradeprogram = rule.get("ch99_tradeprogram") or None
                    if ch99_tradeprogram:
                        ch99_programs.add(ch99_tradeprogram)

            if rule is not None:
                ch99_specific_text = rule.get("ch99_specific_surcharge") or ""
                specific_components = rule.get("ch99_specific_components") or []
                if specific_components:
                    base_denom = _normalize_denominator(effective_specific_unit or base_specific_raw)
                    total_increment = 0.0
                    for comp in specific_components:
                        inc = comp.get("amount")
                        denom = comp.get("denominator")
                        if inc is None:
                            continue
                        if denom and base_denom and denom != base_denom:
                            logger.warning(
                                "Skipped Ch.99 specific adder due to unit mismatch",
                                extra={"hts_code": product_meta["hts_code"], "country": country_name, "denominator": denom},
                            )
                            continue
                        if not base_denom and denom:
                            base_denom = denom
                        total_increment += inc
                    if total_increment:
                        effective_specific_amount = (effective_specific_amount or 0.0) + total_increment
                        has_specific = True
                        if not effective_specific_unit and base_denom:
                            effective_specific_unit = f"/{base_denom}"
                        effective_specific_raw = _decorate_specific_raw(
                            effective_specific_amount,
                            base_specific_raw,
                            effective_specific_unit,
                        )
                if ch99_specific_text and not ch99_applied:
                    ch99_tradeprogram = ch99_tradeprogram or rule.get("ch99_tradeprogram")
                    ch99_label = f"Ch.99 ({ch99_tradeprogram})" if ch99_tradeprogram else "Ch.99 adjusted"
                    if not rate_source or rate_source == "General":
                        rate_source = ch99_label
                    elif "Ch.99" not in rate_source:
                        rate_source = f"{rate_source} · {ch99_label}"

            if ch99_applied:
                rate_source = "Ch.99 adjusted"
                if ch99_tradeprogram:
                    rate_source = f"Ch.99 ({ch99_tradeprogram})"

            chart_eligible = bool(effective_rate is not None)
            combined_rows.append(
                {
                    "country": country_name,
                    "corruption_score": country_meta["score"],
                    "risk_level": country_meta["level"],
                    "country_color": country_meta["color"],
                    "hts_code": product_meta["hts_code"],
                    "product_description": product_meta["description"],
                    "base_duty_pct": base_duty_pct,
                    "ad_valorem_rate": effective_rate,
                    "ch99_applied": ch99_applied,
                    "ch99_tradeprogram": ch99_tradeprogram,
                    "ch99_delta": ch99_delta,
                    "ch99_specific_surcharge": ch99_specific_text,
                    "general_duty_rate_text": base_general_text,
                    "duty_kind": duty_kind,
                    "duty_component_summary": duty_component_summary,
                    "specific_amount": effective_specific_amount,
                    "specific_unit": effective_specific_unit,
                    "specific_raw": effective_specific_raw,
                    "has_specific": has_specific,
                    "rate_source": rate_source,
                    "special_program_code": program_code_summary,
                    "special_program_label": program_label_summary,
                    "chart_eligible": chart_eligible,
                }
            )

    combined_df = pd.DataFrame(combined_rows)
    has_specific_data = bool(not combined_df.empty and combined_df["specific_amount"].notna().any())
    if special_override_count:
        logger.info(
            "Applied special duty overrides",
            extra={"override_rows": special_override_count},
        )
    if unresolved_codes:
        logger.warning(
            "Encountered unresolved special duty programs",
            extra={"codes": sorted(unresolved_codes)},
        )
    if combined_df.empty:
        logger.info("Combined dataframe empty; returning placeholders")
        empty_cols = [
            "country",
            "corruption_score",
            "risk_level",
            "hts_code",
            "product_description",
            "general_duty_rate_text",
            "duty_kind",
            "ad_valorem_rate",
            "chart_eligible",
        ]
        return (
            pd.DataFrame(),
            pd.DataFrame(columns=empty_cols),
            {"count": 0, "by_kind": {}, "samples": []},
            ch99_summary,
            [],
            has_specific_data,
        )

    corr_df_full = combined_df[combined_df["chart_eligible"]].copy()
    logger.info(
        "Built correlation dataframe",
        extra={"pairs": len(corr_df_full), "chart_exclusions": int((~combined_df["chart_eligible"]).sum())},
    )
    non_ad_summary = _summarize_non_ad_rows(combined_df[~combined_df["chart_eligible"]])
    duty_exclusions = combined_df[~combined_df["chart_eligible"]][
        ["country", "hts_code", "general_duty_rate_text", "duty_kind"]
    ].to_dict("records")

    rate_columns = combined_df.apply(_format_rate_columns, axis=1)
    combined_df = combined_df.assign(
        rate_primary=[primary for primary, _ in rate_columns],
        rate_secondary=[secondary for _, secondary in rate_columns],
        plotted=["Yes" if eligible else "No" for eligible in combined_df["chart_eligible"]],
    )
    ch99_summary["programs"] = sorted(ch99_programs)
    return corr_df_full, combined_df, non_ad_summary, ch99_summary, duty_exclusions, has_specific_data


def build_risk_snapshot(risk_df: pd.DataFrame, selected_countries: list[str]) -> list[dict]:
    """Small summary of risk metadata used for UI pills."""
    if not selected_countries:
        return []
    subset = risk_df[risk_df["country"].isin(selected_countries)]
    if subset.empty:
        return []
    return subset[["country", "score", "level", "color", "year"]].to_dict("records")


def _jitter_positions(values: list[float], min_gap: float) -> list[float]:
    """Spread values that are closer than min_gap apart so markers don't overlap."""
    if len(values) <= 1:
        return values
    result = list(values)
    result.sort()
    for i in range(1, len(result)):
        if result[i] - result[i - 1] < min_gap:
            result[i] = result[i - 1] + min_gap
    # Re-center around the original mean
    shift = sum(values) / len(values) - sum(result) / len(result)
    return [v + shift for v in result]


def _text_positions_for(n: int) -> list[str]:
    """Cycle text positions to reduce label overlap for small point counts."""
    pool = ["top center", "bottom center", "top right", "bottom left"]
    return [pool[i % len(pool)] for i in range(n)]


def render_correlation_chart(df: pd.DataFrame, mode: str = "ad_valorem") -> go.Figure | None:
    """Return a Plotly scatter figure for the correlation pairs.

    mode='ad_valorem' — Y axis = effective ad valorem rate (%, ch99-adjusted where applicable)
    mode='specific'   — Y axis = specific duty amount ($/kg, ¢/dozen, etc.)
    """
    if df.empty:
        return None

    df = df.copy()
    df["Trade Program"] = df["ch99_tradeprogram"].fillna("—") if "ch99_tradeprogram" in df.columns else "—"
    df["Base Rate"] = df["general_duty_rate_text"] if "general_duty_rate_text" in df.columns else ""

    if mode == "specific":
        plot_df = df[df["specific_amount"].notna()].copy() if "specific_amount" in df.columns else pd.DataFrame()
        if plot_df.empty or "corruption_score" not in plot_df.columns:
            return None

        units = plot_df["specific_unit"].dropna().unique() if "specific_unit" in plot_df.columns else []
        unit_label = units[0].strip() if len(units) == 1 else "unit"

        palette = ["#0B2A4A", "#4F6D7A", "#7AAFDE", "#B5D4F4", "#378ADD"]
        fig = go.Figure()

        # Spread x positions so overlapping points are readable
        x_raw = plot_df["corruption_score"].tolist()
        x_jittered = _jitter_positions(x_raw, min_gap=6.0)
        text_pos_list = _text_positions_for(len(plot_df))

        for i, (_, row) in enumerate(plot_df.iterrows()):
            color = palette[i % len(palette)]
            level = row.get("risk_level", "")
            amount = row["specific_amount"]
            formatted = f"{amount:.4f}".rstrip("0").rstrip(".")
            fig.add_trace(go.Scatter(
                x=[x_jittered[i]],
                y=[amount],
                mode="markers+text",
                name=row["country"],
                text=[row["country"]],
                textposition=text_pos_list[i],
                textfont=dict(size=13, color=color),
                marker=dict(
                    size=48,
                    color=color,
                    line=dict(width=2.5, color="white"),
                    symbol="circle",
                ),
                hovertemplate=(
                    f"<b>{row['country']}</b><br>"
                    f"Corruption score: {row['corruption_score']:.1f} / 100<br>"
                    f"Specific duty: {formatted} {unit_label}<br>"
                    f"Risk level: {level}"
                    "<extra></extra>"
                ),
            ))
            fig.add_annotation(
                x=x_jittered[i],
                y=amount,
                text=f"{formatted} {unit_label}",
                showarrow=False,
                font=dict(size=11, color="white", family="monospace"),
                yshift=0,
            )

        fig.update_layout(
            xaxis_title="Corruption / Political Risk Score",
            yaxis_title=f"Specific Duty Rate ({unit_label})",
            template="plotly_white",
            margin=dict(l=10, r=10, t=40, b=40),
            showlegend=False,
            plot_bgcolor="white",
            xaxis=dict(range=[-5, 105], gridcolor="#F0F2F5"),
            yaxis=dict(gridcolor="#F0F2F5"),
        )
        return fig

    # --- ad_valorem mode ---
    plot_df = df[df["ad_valorem_rate"].notna()].copy() if "ad_valorem_rate" in df.columns else pd.DataFrame()
    if plot_df.empty:
        return None

    def _delta_label(row) -> str:
        if not row.get("ch99_applied"):
            return "—"
        base = row.get("base_duty_pct")
        eff = row.get("ad_valorem_rate")
        if base is None or eff is None:
            return "—"
        delta = eff - base
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.2f}%"

    plot_df["Rate Source"] = plot_df["ch99_applied"].map({True: "Ch.99 adjusted", False: "Base rate"}) if "ch99_applied" in plot_df.columns else "Base rate"
    plot_df["Ch.99 Δ"] = plot_df.apply(_delta_label, axis=1)

    has_ch99 = "ch99_applied" in plot_df.columns
    agg_dict = {
        "ad_valorem_rate": ("ad_valorem_rate", "first"),
        "corruption_score": ("corruption_score", "first"),
        "risk_level": ("risk_level", "first"),
    }
    if has_ch99:
        agg_dict["ch99_applied"] = ("ch99_applied", "first")
    summary = plot_df.groupby("country", as_index=False).agg(**agg_dict)

    palette = ["#0B2A4A", "#4F6D7A", "#7AAFDE", "#B5D4F4", "#378ADD"]

    fig = go.Figure()

    x_raw_av = summary["corruption_score"].tolist()
    x_jittered_av = _jitter_positions(x_raw_av, min_gap=6.0)
    text_pos_av = _text_positions_for(len(summary))

    for i, row in summary.iterrows():
        ch99 = bool(row.get("ch99_applied", False)) if has_ch99 else False
        color = palette[i % len(palette)]
        level = row.get("risk_level", "")

        fig.add_trace(go.Scatter(
            x=[x_jittered_av[i]],
            y=[row["ad_valorem_rate"]],
            mode="markers+text",
            name=row["country"],
            text=[row["country"]],
            textposition=text_pos_av[i],
            textfont=dict(size=13, color=color),
            marker=dict(
                size=48,
                color=color,
                line=dict(width=2.5, color="white"),
                symbol="circle",
            ),
            hovertemplate=(
                f"<b>{row['country']}</b><br>"
                f"Corruption score: {row['corruption_score']:.1f} / 100<br>"
                f"Effective duty: {row['ad_valorem_rate']:.2f}%<br>"
                f"Risk level: {level}<br>"
                f"{'⚠ Ch.99 adjusted' if ch99 else 'Base rate'}"
                "<extra></extra>"
            ),
        ))

        fig.add_annotation(
            x=x_jittered_av[i],
            y=row["ad_valorem_rate"],
            text=f"{row['ad_valorem_rate']:.1f}%",
            showarrow=False,
            font=dict(size=11, color="white", family="monospace"),
            yshift=0,
        )

    max_risk = summary["corruption_score"].max()
    max_duty = summary["ad_valorem_rate"].max()



    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=50),
        plot_bgcolor="white",
        showlegend=False,
        xaxis=dict(
            title="Country corruption score (lower = safer)",
            gridcolor="#F0F2F5",
            zeroline=False,
            range=[-2, max_risk * 1.3],
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Effective duty rate (%)",
            gridcolor="#F0F2F5",
            zeroline=False,
            ticksuffix="%",
            range=[-0.5, max_duty * 1.35],
            tickfont=dict(size=12),
        ),
    )
    if "ch99_applied" in plot_df.columns:
        adjusted = plot_df[plot_df["ch99_applied"]]
        if not adjusted.empty:
            fig.add_trace(
                go.Scatter(
                    x=adjusted["corruption_score"],
                    y=adjusted["ad_valorem_rate"],
                    mode="markers",
                    marker=dict(
                        size=20,
                        symbol="circle-open",
                        line=dict(color="rgba(255,140,0,0.85)", width=2.5),
                        color="rgba(0,0,0,0)",
                    ),
                    name="Ch.99 adjusted",
                    hoverinfo="skip",
                )
            )
    return fig


def _build_summary_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "count_pairs": 0,
            "countries": [],
            "hts_codes": [],
            "risk_min": None,
            "risk_max": None,
            "duty_min": None,
            "duty_max": None,
        }
    def _round_one(value: float | None) -> float | None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return round(float(value), 1)

    return {
        "count_pairs": len(df),
        "countries": sorted(df["country"].unique().tolist()),
        "hts_codes": sorted(df["hts_code"].unique().tolist()),
        "risk_min": _round_one(df["corruption_score"].min()),
        "risk_max": _round_one(df["corruption_score"].max()),
        "duty_min": float(df["ad_valorem_rate"].min()),
        "duty_max": float(df["ad_valorem_rate"].max()),
    }


def _round_display(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(round(float(value), 2)).rstrip("0").rstrip(".") or "0"


def _derive_headline(
    df: pd.DataFrame,
    selected_countries: Iterable[str],
    selected_products: Iterable[str],
    specific_only_countries: list[str] | None = None,
) -> str:
    country_text = ", ".join(selected_countries) if selected_countries else "your selections"
    product_text = ", ".join(selected_products) if selected_products else "the chosen HTS codes"

    if df.empty:
        return f"No overlapping tariff-risk data for {country_text} on {product_text}."

    unique_pairs = df[["country", "hts_code"]]
    if len(unique_pairs) == 1:
        row = df.iloc[0]
        duty_text = row.get("general_duty_rate_text") or (
            f"{_round_display(row.get('ad_valorem_rate'))}% ad valorem"
            if pd.notna(row.get("ad_valorem_rate"))
            else "the posted duty"
        )
        risk_level = (row.get("risk_level") or "Unknown").lower()
        return f"{row['country']} is your only {risk_level} option for HTS {row['hts_code']} with {duty_text}."

    duties = df["ad_valorem_rate"]
    corruption_scores = df["corruption_score"]
    if duties.nunique(dropna=False) == 1 and corruption_scores.nunique(dropna=False) == 1:
        duty_val = _round_display(duties.iloc[0])
        risk_val = round(float(corruption_scores.iloc[0]), 1)
        return (
            f"All {len(unique_pairs)} selections share the same profile: {duty_val}% duty and risk ≈ {risk_val}."
        )

    min_duty = duties.min()
    best_by_duty = df[duties == min_duty]
    min_risk = best_by_duty["corruption_score"].min()
    best_rows = best_by_duty[best_by_duty["corruption_score"] == min_risk]
    best_countries = sorted(best_rows["country"].unique().tolist())
    best_codes = sorted(best_rows["hts_code"].unique().tolist())
    duty_display = _round_display(min_duty)
    risk_display = round(float(min_risk), 1)
    country_phrase = ", ".join(best_countries)
    code_phrase = ", ".join(best_codes)
    headline = (
        f"{country_phrase} offer(s) the lowest duty ({duty_display}%) while staying around risk {risk_display} for HTS {code_phrase}."
    )
    if specific_only_countries:
        headline += f" Note: {', '.join(specific_only_countries)} carry specific (per-unit) duties — see the rate table."
    return headline


def stream_analysis_to_placeholder(
    df: pd.DataFrame,
    deployment_id: str,
    placeholder: st.delta_generator.DeltaGenerator | None,
    headline: str | None = None,
) -> str:
    """LLM-generated narrative summary of the correlation DataFrame."""
    subset = df.head(SUMMARY_SAMPLE_LIMIT)
    stats = _build_summary_stats(df)
    payload = {
        "stats": stats,
        "sample_rows": subset.to_dict("records"),
    }
    headline_note = ""
    if headline:
        headline_note = (
            f"The user already saw this headline: \"{headline}\". "
            "Do not restate it verbatim or contradict it; elaborate with supporting context instead."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a trade intelligence assistant with the mindset of a "
                "pragmatic trade manager. You help users understand tariff exposure "
                "and country risk for specific HTS codes.\n\n"
                "Your goals:\n"
                "- Give a medium-length answer: a short paragraph plus 1–3 concise, "
                "practical suggestions.\n"
                "- Highlight notable patterns — e.g. countries where duty and risk "
                "diverge, or where a trade program drives the effective rate.\n"
                "- Keep recommendations business-focused and specific to the countries "
                "and HTS codes in view.\n\n"
                "Constraints:\n"
                "- Speak as a knowledgeable trade advisor, not as someone analyzing a "
                "dataset or inspecting rows. Never reference 'the data', 'the dataset', "
                "'sample rows', 'sample size', 'supplied data', or 'the provided rows'.\n"
                "- Do not invent numbers or countries beyond what you have been given.\n"
                "- This is not legal advice; if your answer could be interpreted that "
                "way, remind the user to consult their customs or trade counsel.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "Summarize the relationship between country risk and ad valorem duty rates "
                "for the countries and HTS code(s) below. Highlight any extremes or "
                "interesting patterns, then give 1–3 actionable, business-focused "
                "suggestions. Speak as a trade advisor — do not reference datasets, rows, "
                "or sample sizes.\n\n"
                f"{headline_note}\n\n"
                f"{json.dumps(payload)}"
            ),
        },
    ]
    stream = openai.chat.completions.create(
        model=deployment_id,
        messages=messages,
        temperature=1,
        stream=True,
    )
    full_text = ""
    headline_html = (
        f"<p class='analysis-headline'><strong>{headline}</strong></p>" if headline else ""
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if not delta:
            continue
        full_text += delta
        if placeholder:
            placeholder.markdown(
                f"<div class='bubble-bot'>{headline_html}{full_text}▌</div>",
                unsafe_allow_html=True,
            )
    if placeholder:
        placeholder.markdown(
            f"<div class='bubble-bot'>{headline_html}{full_text}</div>",
            unsafe_allow_html=True,
        )
    return full_text.strip()


def queue_analysis_request(
    selected_countries: list[str],
    selected_products: list[str],
) -> bool:
    """Store a pending analysis request in session state."""
    signature = compute_selection_signature(selected_countries, selected_products)
    if not signature:
        logger.warning("Analysis request skipped; selections incomplete")
        return False
    st.session_state["analysis_request"] = {
        "countries": list(selected_countries),
        "products": list(selected_products),
        "signature": signature,
    }
    # Reset chart mode so new analyses always open in ad_valorem view
    st.session_state["chart_mode"] = "ad_valorem"
    logger.info(
        "Queued analysis request",
        extra={"countries": selected_countries, "products": selected_products},
    )
    return True


def maybe_run_analysis(
    conn,
    deployment_id: str,
    risk_df: pd.DataFrame,
    placeholder: st.delta_generator.DeltaGenerator | None,
) -> None:
    """If an analysis request is queued, execute it and emit chat + charts."""
    request = st.session_state.get("analysis_request")
    if not request or st.session_state.get("analysis_inflight"):
        return

    countries = request.get("countries", [])
    products = request.get("products", [])
    signature = request.get("signature")
    if not countries or not products:
        st.session_state["analysis_request"] = None
        logger.warning("Analysis request discarded due to missing selections")
        return

    run_id = uuid.uuid4().hex[:8]
    st.session_state["analysis_request"] = None
    st.session_state["analysis_inflight"] = True
    st.session_state["analysis_active_run"] = run_id
    success = False
    logger.info(
        "Starting analysis run",
        extra={"analysis_run": run_id, "countries": countries, "products": products},
    )

    try:
        def ll_to_xy(lat, lon, w=500, h=250):
            x = round((lon + 180) * (w / 360), 1)
            y = round((90 - lat) * (h / 180), 1)
            return x, y

        pts = []
        for ctry in countries:
            if ctry in COUNTRY_COORDINATES:
                lat, lon = COUNTRY_COORDINATES[ctry]
                x, y = ll_to_xy(lat, lon)
                pts.append({"name": ctry, "x": x, "y": y})

        paths_svg = ""
        travelers_svg = ""
        labels_svg = ""
        style_css = ""

        for i in range(len(pts) - 1):
            x1, y1 = pts[i]["x"], pts[i]["y"]
            x2, y2 = pts[i+1]["x"], pts[i+1]["y"]
            mx = round((x1+x2)/2, 1)
            my = round(min(y1,y2) - 40, 1)
            paths_svg += f'<path id="p{i}" d="M{x1},{y1} Q{mx},{my} {x2},{y2}" fill="none" stroke="#B5D4F4" stroke-width="1.5" stroke-dasharray="5 3" opacity="0.8"/>'
            delay = round(i * 0.6, 1)
            style_css += f".t{i}{{offset-path:path('M{x1},{y1} Q{mx},{my} {x2},{y2}');animation:mv{i} 2.4s ease-in-out {delay}s infinite;}}"
            style_css += f"@keyframes mv{i}{{0%{{offset-distance:0%;opacity:1}}85%{{offset-distance:100%;opacity:1}}100%{{offset-distance:100%;opacity:0}}}}"
            travelers_svg += f'<circle class="t{i}" r="5" fill="#378ADD" stroke="white" stroke-width="1.5"/>'

        for pt in pts:
            paths_svg += f'<circle cx="{pt["x"]}" cy="{pt["y"]}" r="6" fill="#0B2A4A" stroke="white" stroke-width="2"/>'
            labels_svg += f'<text x="{pt["x"]}" y="{pt["y"]-10}" text-anchor="middle" font-size="9" font-family="sans-serif" fill="#0B2A4A" font-weight="bold">{pt["name"]}</text>'

        country_list = ", ".join(countries)
        globe_html = f"""<div style="background:#EAF2FB;border-radius:12px;padding:18px 20px;border:0.5px solid #B5D4F4;margin:8px 0;">
<p style="font-size:13px;color:#0B2A4A;font-weight:500;margin-bottom:10px;text-align:center;">Analyzing {country_list}…</p>
<svg viewBox="0 0 500 250" width="100%" style="border-radius:8px;background:#EAF2FB;">
<style>{style_css}</style>
<rect width="500" height="250" fill="#E1EDF8" rx="8"/>
<line x1="0" y1="125" x2="500" y2="125" stroke="#C5D9ED" stroke-width="0.5"/>
<line x1="250" y1="0" x2="250" y2="250" stroke="#C5D9ED" stroke-width="0.5"/>
{paths_svg}{travelers_svg}{labels_svg}
</svg>
<p style="font-size:11px;color:#4F6D7A;text-align:center;margin-top:8px;">Querying HTS database &middot; Scoring V-Dem indicators &middot; Generating analysis</p>
</div>"""

        pts_3d = []
        for ctry in countries:
            if ctry in COUNTRY_COORDINATES:
                lat, lon = COUNTRY_COORDINATES[ctry]
                pts_3d.append({"name": ctry, "lat": lat, "lon": lon})

        if placeholder:
            import time
            thinking_steps = [
                (f"Identifying selected countries: <strong>{', '.join(countries)}</strong>", 0.8),
                (f"Querying HTS SQLite database for: <strong>{', '.join(products)}</strong>", 1.0),
                ("Parsing general duty rates and checking Chapter 99 tariff adjustments...", 1.0),
                ("Loading V-Dem indicators and computing corruption scores per country...", 1.0),
                ("Building correlation matrix of corruption scores vs effective duty rates...", 0.8),
                ("Sending data to <strong>Azure OpenAI GPT-5-mini</strong> for compliance narrative...", 0.8),
                ("<em style='color:#4F6D7A;'>Generating actionable insights and recommendations...</em>", 0.5),
            ]

            def render_thinking(steps_so_far):
                rows = ""
                for idx, (text, _) in enumerate(steps_so_far):
                    num = str(idx + 1).zfill(2)
                    rows += f"<div style='display:flex;gap:10px;align-items:flex-start;margin-bottom:8px;'><span style='color:#0B2A4A;font-weight:600;font-size:11px;min-width:20px;margin-top:2px;'>{num}</span><span style='font-size:13px;color:#374151;line-height:1.6;'>{text}</span></div>"
                placeholder.markdown(f"""
<div style='background:#F8FAFC;border:0.5px solid #E2E8F0;border-radius:12px;padding:18px 20px;margin:8px 0;'>
  <div style='display:flex;align-items:center;gap:8px;margin-bottom:14px;'>
    <div style='width:8px;height:8px;border-radius:50%;background:#0B2A4A;'></div>
    <span style='font-size:13px;font-weight:500;color:#0B2A4A;'>Thinking...</span>
  </div>
  {rows}
</div>""", unsafe_allow_html=True)

            shown = []
            for step in thinking_steps:
                shown.append(step)
                render_thinking(shown)
                time.sleep(step[1])

        with st.spinner("Running analysis..."):
            tariff_df = fetch_tariffs_for_codes(conn, products)
            ch99_df = fetch_ch99_for_codes_and_countries(conn, products, countries)
            (
                corr_df_full,
                corr_df_table,
                non_ad_summary,
                ch99_summary,
                duty_exclusions,
                has_specific_data,
            ) = build_correlation_dataframe(countries, tariff_df, risk_df, ch99_df)
            # Countries with specific duties but no ad-valorem rate
            _specific_only = []
            if "specific_amount" in corr_df_table.columns and "chart_eligible" in corr_df_table.columns:
                _mask = corr_df_table["specific_amount"].notna() & ~corr_df_table["chart_eligible"]
                _specific_only = sorted(corr_df_table.loc[_mask, "country"].unique().tolist())
            # Rows that have a specific_amount (used for the specific-duty chart tab)
            specific_chart_rows = (
                corr_df_table[corr_df_table["specific_amount"].notna()].to_dict("records")
                if "specific_amount" in corr_df_table.columns
                else []
            )

            headline_source_df = corr_df_full if not corr_df_full.empty else corr_df_table
            headline_text = _derive_headline(
                headline_source_df, countries, products, specific_only_countries=_specific_only
            )
            timestamp = datetime.now().strftime("%I:%M %p")
            risk_snapshot = build_risk_snapshot(risk_df, countries)
            non_ad_text = _format_non_ad_summary_for_text(non_ad_summary)
            duty_exclusion_message = None
            # Only warn about exclusions that are NOT specific-duty rows — those are
            # shown in the "⊞ Specific Duty" chart tab and need no separate warning.
            non_specific_exclusions = [
                item for item in duty_exclusions
                if item.get("duty_kind") not in ("specific", "mixed")
            ]
            if non_specific_exclusions:
                preview_labels = [
                    f"{item['hts_code']} ({item['general_duty_rate_text']})"
                    for item in non_specific_exclusions[:3]
                ]
                preview = ", ".join(preview_labels)
                extra = len(non_specific_exclusions) - len(preview_labels)
                if extra > 0:
                    preview = f"{preview}, +{extra} more"
                duty_exclusion_message = (
                    f"Skipped {len(non_specific_exclusions)} product-country pair(s) with non-percentage duties: {preview}."
                )
            if non_ad_summary.get("count"):
                logger.info(
                    "Analysis run has non-ad duties",
                    extra={"analysis_run": run_id, "non_ad_count": non_ad_summary.get("count")},
                )
            headline_html = (
                f"<p class='analysis-headline'><strong>{headline_text}</strong></p>" if headline_text else ""
            )

            if corr_df_full.empty:
                if has_specific_data:
                    summary_text = (
                        "This HTS code carries specific (per-unit) duties. "
                        "See the ⊞ Specific Duty tab for the rate comparison across countries."
                    )
                elif len(corr_df_table) and non_ad_summary.get("count"):
                    summary_text = (
                        "All selected products currently rely on quantity- or rule-based duty rates, so no ad valorem analysis is available."
                    )
                    if non_ad_text:
                        summary_text = f"{summary_text} {non_ad_text}"
                else:
                    summary_text = "No overlapping tariff-risk data for the current selections."
                specific_sentence = _summarize_specific_effects(corr_df_table)
                if specific_sentence:
                    summary_text = f"{summary_text}\n\n{specific_sentence}"
                if placeholder:
                    placeholder.markdown(
                        f"<div class='bubble-bot'>{headline_html}{summary_text}</div>",
                        unsafe_allow_html=True,
                    )
                display_table, display_columns = _build_display_table(corr_df_table)
                append_message(
                    {
                        "role": "assistant",
                        "content": summary_text,
                        "time": timestamp,
                        "type": "analysis",
                        "chart_data": display_table.to_dict("records"),
                        "chart_columns": display_columns,
                        "risk_snapshot": risk_snapshot,
                        "selections": {
                            "countries": countries,
                            "products": products,
                        },
                        "non_ad_summary": non_ad_summary,
                        "non_ad_summary_text": non_ad_text,
                        "headline": headline_text,
                        "ch99_summary": ch99_summary,
                        "duty_exclusions": duty_exclusions,
                        "duty_exclusion_message": duty_exclusion_message,
                        "has_specific_data": has_specific_data,
                        "specific_chart_data": specific_chart_rows,
                    }
                )
            else:
                fig = render_correlation_chart(corr_df_full)
                summary_text = stream_analysis_to_placeholder(
                    corr_df_full, deployment_id, placeholder, headline=headline_text
                )
                if non_ad_text:
                    summary_text = f"{summary_text}\n\n{non_ad_text}"
                    if placeholder:
                        placeholder.markdown(
                            f"<div class='bubble-bot'>{headline_html}{summary_text}</div>",
                            unsafe_allow_html=True,
                        )
                specific_sentence = _summarize_specific_effects(corr_df_table)
                if specific_sentence:
                    summary_text = f"{summary_text}\n\n{specific_sentence}"
                    if placeholder:
                        placeholder.markdown(
                            f"<div class='bubble-bot'>{headline_html}{summary_text}</div>",
                            unsafe_allow_html=True,
                        )
                display_table, display_columns = _build_display_table(corr_df_table)
                append_message(
                    {
                        "role": "assistant",
                        "content": summary_text,
                        "time": timestamp,
                        "type": "analysis",
                        "plotly_fig": fig.to_dict() if fig else None,
                        # raw_chart_data/raw_chart_columns: full corr_df_full for toggle re-rendering
                        "raw_chart_data": corr_df_full.to_dict("records"),
                        "raw_chart_columns": corr_df_full.columns.tolist(),
                        # chart_data/chart_columns: formatted display table
                        "chart_data": display_table.to_dict("records"),
                        "chart_columns": display_columns,
                        "risk_snapshot": risk_snapshot,
                        "selections": {
                            "countries": countries,
                            "products": products,
                        },
                        "non_ad_summary": non_ad_summary,
                        "non_ad_summary_text": non_ad_text,
                        "headline": headline_text,
                        "ch99_summary": ch99_summary,
                        "duty_exclusions": duty_exclusions,
                        "duty_exclusion_message": duty_exclusion_message,
                        "has_specific_data": has_specific_data,
                        "specific_chart_data": specific_chart_rows,
                    }
                )
            st.session_state["correlation_signature"] = signature
            success = True
            logger.info(
                "Analysis run completed",
                extra={
                    "analysis_run": run_id,
                    "pairs": len(corr_df_full),
                    "table_rows": len(corr_df_table),
                    "non_ad_rows": non_ad_summary.get("count", 0),
                    "ch99_adjusted": ch99_summary.get("n_adjusted"),
                },
            )
    except Exception as exc:  # pragma: no cover - Streamlit handles UI errors
        logger.exception("Correlation analysis failed", extra={"analysis_run": run_id})
        if placeholder:
            placeholder.markdown(
                f"<div class='bubble-bot'>Error: {exc}</div>",
                unsafe_allow_html=True,
            )
        st.error(f"Correlation analysis failed: {exc}")
    finally:
        if st.session_state.get("analysis_active_run") == run_id:
            st.session_state["analysis_inflight"] = False
            st.session_state["analysis_active_run"] = None
        if success:
            st.rerun()
