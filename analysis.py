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

SUMMARY_SAMPLE_LIMIT = 60
DISPLAY_COLUMN_MAP = [
    ("country", "Country"),
    ("risk_score", "Risk Score"),
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


def _best_ch99_rule(
    ch99_df: pd.DataFrame,
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
    if ch99_df is None or ch99_df.empty:
        return None
    mask = ch99_df["hts_code"] == hts_code
    mask &= (ch99_df["ch99_country"] == country) | (ch99_df["ch99_country"] == "Global")
    candidates = ch99_df[mask].copy()
    if candidates.empty:
        return None

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
    if "Risk Score" in display_df.columns:
        display_df["Risk Score"] = display_df["Risk Score"].apply(
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
    ch99_available = ch99_df is not None and not ch99_df.empty
    if not selected_countries or tariff_df.empty:
        empty_cols = [
            "country",
            "risk_score",
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
            "risk_score",
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
                matching_codes = special_rule.codes_for_country(country_name)
                if matching_codes:
                    applied_special = True
                    program_code_summary = ", ".join(matching_codes)
                    program_label_summary = special_rule.format_labels(matching_codes)
                    rate_source = f"Special — {program_code_summary}"
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

            rule = _best_ch99_rule(ch99_df, product_meta["hts_code"], country_name) if ch99_available else None
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
                    "risk_score": country_meta["score"],
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
            "risk_score",
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

        fig = px.scatter(
            plot_df,
            x="risk_score",
            y="specific_amount",
            color="country",
            hover_name="product_description" if "product_description" in plot_df.columns else None,
            hover_data={
                "country": True,
                "risk_score": ":.1f",
                "risk_level": True,
                "hts_code": True,
                "Base Rate": True,
                "specific_amount": ":.4f",
                "Trade Program": True,
            },
            labels={
                "risk_score": "Country Risk Score",
                "specific_amount": f"Duty Amount ({unit_label})",
            },
        )
        fig.update_layout(
            xaxis_title="Country Risk Score",
            yaxis_title=f"Specific Duty Rate ({unit_label})",
            legend_title="Country",
            template="plotly_white",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig.update_traces(marker={"size": 12, "line": {"width": 1.5, "color": "rgba(0,0,0,0.25)"}})
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

    fig = px.scatter(
        plot_df,
        x="risk_score",
        y="ad_valorem_rate",
        color="country",
        hover_name="product_description" if "product_description" in plot_df.columns else None,
        hover_data={
            "country": True,
            "risk_score": ":.1f",
            "hts_code": True,
            "Base Rate": True,
            "ad_valorem_rate": ":.2f",
            "Rate Source": True,
            "Trade Program": True,
            "Ch.99 Δ": True,
        },
    )
    fig.update_layout(
        xaxis_title="Country Risk Score",
        yaxis_title="Effective Duty Rate (% ad valorem)",
        legend_title="Country",
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(tickformat=".1f")
    fig.update_traces(marker={"size": 12, "line": {"width": 1, "color": "rgba(0,0,0,0.3)"}})
    if "ch99_applied" in plot_df.columns:
        adjusted = plot_df[plot_df["ch99_applied"]]
        if not adjusted.empty:
            fig.add_trace(
                go.Scatter(
                    x=adjusted["risk_score"],
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
        "risk_min": _round_one(df["risk_score"].min()),
        "risk_max": _round_one(df["risk_score"].max()),
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
    risk_scores = df["risk_score"]
    if duties.nunique(dropna=False) == 1 and risk_scores.nunique(dropna=False) == 1:
        duty_val = _round_display(duties.iloc[0])
        risk_val = round(float(risk_scores.iloc[0]), 1)
        return (
            f"All {len(unique_pairs)} selections share the same profile: {duty_val}% duty and risk ≈ {risk_val}."
        )

    min_duty = duties.min()
    best_by_duty = df[duties == min_duty]
    min_risk = best_by_duty["risk_score"].min()
    best_rows = best_by_duty[best_by_duty["risk_score"] == min_risk]
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
                "You are a trade intelligence assistant that combines the mindset of a "
                "pragmatic trade manager and a neutral data analyst. You work inside a "
                "tariff analysis tool.\n\n"
                "You are given structured data about country risk scores and general "
                "ad valorem duty rates for specific HTS codes.\n\n"
                "Your goals:\n"
                "- Focus on the actual numbers and patterns in the provided data.\n"
                "- Give a medium-length answer: a short paragraph plus 1–3 concise, "
                "practical suggestions.\n"
                "- Describe correlations or notable clusters, but be cautious and avoid "
                "big, sweeping claims.\n"
                "- Tie any recommendations directly to the observed data (e.g., which "
                "countries/products to scrutinize or where to investigate further).\n\n"
                "Constraints:\n"
                "- Use ONLY the supplied stats and sample rows; do not invent additional "
                "numbers or new countries/products.\n"
                "- This is not legal advice or a substitute for professional customs or "
                "trade counsel; if your answer could be interpreted that way, remind the "
                "user of this.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "Using the JSON payload below, write a medium-length summary of what the "
                "data suggests about the relationship between country risk and ad valorem "
                "duty rates. Highlight any extremes or interesting clusters, and then give "
                "1–3 actionable, business-focused suggestions. Do not make claims that are "
                "not clearly supported by the numbers.\n\n"
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
