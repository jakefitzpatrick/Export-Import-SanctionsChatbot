"""Best-country ranking for a given HTS code.

Scores all 186 countries on effective tariff rate + corruption score, then
passes the top 10 to the LLM for trade-flow reasoning before returning the
final top-3 selection.
"""
from __future__ import annotations

import json
import sqlite3

import openai
import pandas as pd
import streamlit as st

from analysis import (
    _best_ch99_rule,
    _build_ch99_lookup,
    apply_ch99_to_duty,
    fetch_ch99_for_codes_and_countries,
    fetch_tariffs_for_codes,
    queue_analysis_request,
)
from duty_parser import DutyRate
from logger import setup_logger

logger = setup_logger(__name__)

TARIFF_WEIGHT = 0.6
CORRUPTION_WEIGHT = 0.4
TOP_N_FOR_LLM = 10

EXCLUDED_COUNTRIES = {
    "iran", "north korea", "cuba", "syria", "russia", "belarus",
}

_RANKER_SYSTEM_PROMPT = (
    "You are a trade sourcing advisor. You receive a ranked list of candidate countries "
    "for a specific HTS code, already scored on tariff rate and corruption. "
    "Apply qualitative trade-flow judgment: consider whether this product is realistically "
    "sourced from each country, any active sanctions or export controls beyond the "
    "pre-filtered list, existing US trade relationships, shipping/port access, and "
    "product-category sourcing norms. "
    "Return ONLY valid JSON in exactly this format with no other text:\n"
    '{"selected": ["Country A", "Country B", "Country C"], '
    '"rationale": {"Country A": "one sentence", "Country B": "one sentence", "Country C": "one sentence"}}'
)


def rank_countries_for_hts(
    hts_code: str,
    conn: sqlite3.Connection,
    risk_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Score all countries and return (ranked_df, hts_description) sorted ascending by composite."""
    tariff_rows = fetch_tariffs_for_codes(conn, [hts_code])
    if tariff_rows.empty:
        logger.warning("No tariff data found for %s", hts_code)
        return pd.DataFrame(), hts_code

    tariff_row = tariff_rows.iloc[0]
    base_rate: DutyRate = tariff_row["parsed_duty_rate"]
    hts_description = str(tariff_row.get("description", hts_code))

    all_countries = risk_df["country"].tolist()
    ch99_df = fetch_ch99_for_codes_and_countries(conn, [hts_code], all_countries)
    ch99_lookup = _build_ch99_lookup(ch99_df)

    records = []
    for _, row in risk_df.iterrows():
        country = str(row["country"])
        if country.lower() in EXCLUDED_COUNTRIES:
            continue
        ch99 = _best_ch99_rule(ch99_lookup, hts_code, country)
        if ch99 is not None:
            effective_rate = apply_ch99_to_duty(base_rate, ch99)
        else:
            effective_rate = base_rate.ad_valorem_rate if base_rate else None
        records.append({
            "country": country,
            "corruption_score": float(row["score"]),
            "risk_level": str(row["level"]),
            "effective_rate_pct": effective_rate,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df, hts_description

    # Normalize tariff (min-max); fall back to mean for countries with no % rate
    has_rate = df["effective_rate_pct"].notna()
    if has_rate.any():
        rates = df.loc[has_rate, "effective_rate_pct"]
        r_min, r_max = rates.min(), rates.max()
        if r_max > r_min:
            df.loc[has_rate, "tariff_norm"] = (rates - r_min) / (r_max - r_min)
        else:
            df.loc[has_rate, "tariff_norm"] = 0.0
        mean_tariff_norm = float(df.loc[has_rate, "tariff_norm"].mean())
        df["tariff_norm"] = df["tariff_norm"].fillna(mean_tariff_norm)
    else:
        df["tariff_norm"] = None

    df["corruption_norm"] = df["corruption_score"] / 100.0

    if df["tariff_norm"].notna().any():
        df["composite"] = (
            TARIFF_WEIGHT * df["tariff_norm"] + CORRUPTION_WEIGHT * df["corruption_norm"]
        )
    else:
        df["composite"] = df["corruption_norm"]

    df = df.sort_values(
        ["composite", "corruption_score", "country"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df, hts_description


def select_best_countries(
    hts_code: str,
    conn: sqlite3.Connection,
    risk_df: pd.DataFrame,
    deployment_id: str,
) -> tuple[list[str], dict[str, str]]:
    """Return (top_3_names, rationale_dict) for the given HTS code."""
    ranked_df, hts_description = rank_countries_for_hts(hts_code, conn, risk_df)

    if ranked_df.empty:
        logger.warning("Ranking returned empty DataFrame for %s", hts_code)
        return [], {}

    top10 = ranked_df.head(TOP_N_FOR_LLM)
    candidates = [
        {
            "country": row["country"],
            "effective_tariff_rate_pct": row["effective_rate_pct"],
            "corruption_score": row["corruption_score"],
            "risk_level": row["risk_level"],
            "composite_score": round(float(row["composite"]), 3),
            "rank": int(row["rank"]),
        }
        for _, row in top10.iterrows()
    ]

    payload = json.dumps({
        "hts_code": hts_code,
        "hts_description": hts_description,
        "candidates": candidates,
    })

    logger.info(
        "Calling LLM for trade-flow reasoning",
        extra={"hts_code": hts_code, "candidate_count": len(candidates)},
    )
    try:
        response = openai.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": _RANKER_SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        selected = parsed.get("selected", [])
        rationale = parsed.get("rationale", {})
        candidate_names_lower = {c["country"].lower() for c in candidates}
        if (
            isinstance(selected, list)
            and len(selected) == 3
            and all(str(s).lower() in candidate_names_lower for s in selected)
        ):
            logger.info("LLM selected countries", extra={"selected": selected})
            return selected, rationale
        logger.warning("LLM returned invalid selection; falling back to quantitative top-3")
    except Exception as exc:
        logger.warning("LLM ranking call failed (%s); using quantitative top-3", exc)

    top3 = ranked_df.head(3)["country"].tolist()
    return top3, {}


def maybe_run_best_countries(
    conn: sqlite3.Connection,
    deployment_id: str,
    risk_df: pd.DataFrame,
) -> None:
    """If a best-countries request is queued, execute it and auto-queue analysis."""
    request = st.session_state.get("best_countries_request")
    if not request or st.session_state.get("best_countries_inflight"):
        return

    hts_code = request.get("hts_code")
    if not hts_code:
        st.session_state["best_countries_request"] = None
        return

    st.session_state["best_countries_request"] = None
    st.session_state["best_countries_inflight"] = True

    try:
        selected, rationale = select_best_countries(hts_code, conn, risk_df, deployment_id)
    except Exception as exc:
        logger.error("Best countries ranking failed", exc_info=exc)
        st.session_state["best_countries_inflight"] = False
        st.session_state["best_countries_error"] = str(exc)
        return

    st.session_state["best_countries_pending"] = selected
    st.session_state["best_countries_rationale"] = {
        "hts_code": hts_code,
        "selected": selected,
        "rationale": rationale,
    }
    st.session_state["best_countries_inflight"] = False

    if selected:
        queue_analysis_request(selected, [hts_code])

    st.rerun()
