"""Streamlit text-to-SQL chatbot backed by a local HTS SQLite database."""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
import itertools
from pathlib import Path

from dotenv import load_dotenv
import openai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from risk_model import get_risk_df

load_dotenv()

logger = logging.getLogger(__name__)

HTS_DB_PATH = Path(__file__).resolve().parent / "data" / "hts.db"
HTS_COLUMNS = [
    "hts_code",
    "chapter",
    "heading",
    "subheading",
    "statistical_suffix",
    "indent_level",
    "description",
    "full_description",
    "unit",
    "general_duty_rate",
    "special_duty_rate",
    "column2_duty_rate",
    "quota_quantity",
    "additional_duties",
]

MAX_PRODUCT_OPTIONS = 1000
DEFAULT_COUNTRY_SELECTION = [
    "Cameroon",
    "Russia",
]
SUMMARY_SAMPLE_LIMIT = 60
MAX_COUNTRY_SELECTION = 3
MAX_PRODUCT_SELECTION = 1

GENERAL_DUTY_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

QUESTION_PLACEHOLDER = "Ask about the HTS data"

FREE_DUTY_VALUES = {"free", "", "n/a", "none", "no", "zero"}
SPECIFIC_HINTS = [
    " per ",
    "/",
    " each",
    " doz",
    " dozen",
    " kg",
    " lb",
    " liter",
    " litres",
    " pair",
    " kg.",
    " kg)",
    "units",
    "unit",
]
CURRENCY_HINTS = ["$", "¢"]


@dataclass
class DutyRate:
    raw_text: str
    kind: str
    ad_valorem_rate: float | None = None
    specific_amount: float | None = None
    specific_unit: str | None = None
    notes: str | None = None


CH99_COLUMNS = [
    "NEWCODE",
    "HTS_MASTER_CODE",
    "COUNTRY",
    "NEWRATE",
    "NEWRATE_CLEAN",
    "RATE_MODIFIER",
    "ADDITIONAL_DUTY_PCT",
    "ADDITIONAL_VALUE",
    "ADDITIONAL_UNIT",
    "TRADEPROGRAM",
    "MATCH_PRIORITY",
]

SQL_SYSTEM_PROMPT = (
    "You are a SQL generator for a SQLite database with two tables and one view. "
    "Table `hts` has columns: "
    + ", ".join(HTS_COLUMNS)
    + ". "
    "Table `chapter_99` has columns: "
    + ", ".join(CH99_COLUMNS)
    + ". "
    "In `chapter_99`, HTS_MASTER_CODE = 'ALL' matches any product and COUNTRY = 'Global' matches any country. "
    "Use MATCH_PRIORITY (lower number = higher precedence) to resolve conflicts when multiple rows match. "
    "View `hts_with_ch99` pre-joins both tables using those wildcard and priority rules and exposes: "
    "hts_code, chapter, description, full_description, general_duty_rate, special_duty_rate, "
    "ch99_newcode, ch99_country, ch99_newrate, ch99_rate_modifier, ch99_additional_pct, "
    "ch99_tradeprogram, ch99_match_priority. "
    "Always treat HTS codes as TEXT strings — never cast them to integers. "
    "Always respond with exactly one valid SQLite SELECT statement. "
    "Do not include surrounding markdown, explanations, or additional text. "
    "Only use SELECT statements; avoid INSERT/UPDATE/DELETE/PRAGMA."
)

SELECT_PATTERN = re.compile(r"SELECT\b.*", re.IGNORECASE | re.DOTALL)

LAST_RESULT_KEY = "latest_hts_result"


def _format_css() -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root {
        --primary:#0f1f38;
        --muted:#94a3b8;
        --border:#e2e8f0;
        --card:#ffffff;
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background:#f5f7fb; }
    section.main > div.block-container {
        display:flex;
        flex-direction:column;
        min-height:100vh;
        height:100vh;
        overflow:hidden;
        gap:1rem;
        padding-bottom:0 !important;
    }
    div[data-context="true"],
    div[data-chat="true"],
    div[data-composer="true"],
    div[data-anchor="chat-end"] {
        display:none;
    }
    div[data-testid="stVerticalBlock"]:has(> div[data-context="true"]) {
        position:sticky;
        top:0;
        z-index:50;
        background:var(--card);
        border:1px solid var(--border);
        border-radius:18px;
        padding:16px 20px 6px;
        box-shadow:0 12px 24px rgba(15,31,56,0.08);
    }
    div[data-testid="stVerticalBlock"]:has(> div[data-chat="true"]) {
        flex:1;
        overflow-y:auto;
        padding-right:6px;
        padding-bottom:120px;
        scroll-behavior:smooth;
    }
    div[data-testid="stVerticalBlock"]:has(> div[data-composer="true"]) {
        position:sticky;
        bottom:0;
        z-index:40;
        background:var(--card);
        border-top:1px solid var(--border);
        border-radius:18px 18px 0 0;
        padding:12px 20px;
        box-shadow:0 -10px 25px rgba(15,31,56,0.08);
    }
    .bubble-user {
        background-color: #1a3a5c;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 4px 0;
        max-width: 70%;
        float: right;
        clear: both;
        font-size: 14px;
        line-height: 1.5;
    }
    .bubble-bot {
        background-color: #ffffff;
        color: #1a1a2e;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 4px 0;
        max-width: 70%;
        float: left;
        clear: both;
        font-size: 14px;
        line-height: 1.5;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .bubble-label {
        font-size: 10px;
        color: #94a3b8;
        margin-bottom: 2px;
        clear: both;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .bubble-label-right { text-align: right; }
    .timestamp {
        font-size: 10px;
        color: #cbd5e1;
        margin-top: 3px;
        clear: both;
    }
    .timestamp-right { text-align: right; }
    .clearfix { clear: both; }
    img { mix-blend-mode: multiply; }
    [data-testid="stSidebar"] img {
        display: block;
        margin-left: 0 !important;
        padding-left: 0 !important;
        mix-blend-mode: normal !important;
        filter: brightness(0) invert(1);
    }
    .stButton > button {
        background-color: #1a3a5c;
        color: white !important;
        border-radius: 10px;
        font-weight: 500;
        border: none;
        padding: 10px;
    }
    .stButton > button:hover { background-color: #2a5298; }
    [data-testid="stHorizontalBlock"] .stButton > button {
        background-color: transparent !important;
        color: #94a3b8 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 20px !important;
        font-size: 12px !important;
        font-weight: 400 !important;
        padding: 4px 10px !important;
        opacity: 0.8;
    }
    [data-testid="stHorizontalBlock"] .stButton > button:hover {
        background-color: #f0f4ff !important;
        color: #1a3a5c !important;
        opacity: 1;
    }
    .risk-pills {
        display:flex;
        flex-wrap:wrap;
        gap:8px;
        margin:8px 0 4px;
    }
    .risk-pill {
        border:1px solid var(--border);
        border-left:4px solid var(--border);
        padding:8px 12px;
        border-radius:12px;
        background:#f9fafc;
        font-size:12px;
        font-weight:500;
    }
    .risk-pill strong {
        display:block;
        font-size:12px;
        color:var(--primary);
    }
    .analysis-meta {
        font-size:12px;
        color:var(--muted);
        margin-bottom:6px;
    }
    .empty-chat {
        color:var(--muted);
        font-size:13px;
        text-align:center;
        padding:40px 0;
    }
    div[data-testid="stChatInputContainer"] {
        width:100%;
    }
    div[data-testid="stChatInput"] {
        position:relative;
        border:0.5px solid rgba(255,255,255,0.12);
        border-radius:24px;
        background:rgba(255,255,255,0.06);
        min-height:44px;
        padding:10px 52px 10px 16px;
        transition:border-color 0.15s ease;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color:rgba(255,255,255,0.35);
    }
    div[data-testid="stChatInput"] textarea {
        background:transparent !important;
        border:none !important;
        resize:none !important;
        min-height:24px;
        max-height:160px;
        font-size:14px;
        line-height:24px;
        color:#ffffff;
        padding:0;
    }
    div[data-testid="stChatInput"] textarea:focus {
        outline:none !important;
        box-shadow:none !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color:rgba(255,255,255,0.35);
    }
    div[data-testid="stChatInput"] button {
        position:absolute;
        right:10px;
        bottom:8px;
        width:28px;
        height:28px;
        border-radius:8px;
        border:none;
        background:#534AB7;
        color:#ffffff;
        font-size:0;
        cursor:pointer;
    }
    div[data-testid="stChatInput"] button:after {
        content:none;
    }
    </style>
    """


@st.cache_resource
def get_db_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(show_spinner=False)
def load_product_options(_conn: sqlite3.Connection) -> list[tuple[str, str]]:
    query = (
        'SELECT hts_code, description FROM hts '
        'WHERE hts_code IS NOT NULL AND hts_code <> "" '
        'ORDER BY hts_code LIMIT ?'
    )
    try:
        df = pd.read_sql_query(query, _conn, params=(MAX_PRODUCT_OPTIONS,))
    except Exception as exc:
        logger.warning("Failed to load product options: %s", exc)
        return []
    return list(df.itertuples(index=False, name=None))


def _extract_first_number(value: str) -> tuple[float | None, re.Match | None]:
    match = GENERAL_DUTY_PATTERN.search(value)
    if not match:
        return None, None
    try:
        return float(match.group(0)), match
    except ValueError:
        return None, match


def _has_specific_hint(value: str) -> bool:
    lowered = value.lower()
    return any(hint in lowered for hint in SPECIFIC_HINTS) or any(symbol in value for symbol in CURRENCY_HINTS)


def parse_general_duty(value: str | float | int | None) -> DutyRate:
    if value is None:
        return DutyRate(raw_text="", kind="text", notes="missing duty")
    if isinstance(value, (int, float)):
        return DutyRate(raw_text=str(value), kind="ad_valorem", ad_valorem_rate=float(value))

    raw_text = str(value).strip()
    normalized = raw_text.lower()
    if normalized in FREE_DUTY_VALUES:
        return DutyRate(raw_text=raw_text, kind="ad_valorem", ad_valorem_rate=0.0, notes="duty-free entry")

    has_percent = "%" in raw_text
    has_specific = _has_specific_hint(raw_text)

    number, match = _extract_first_number(raw_text)
    if has_percent and not has_specific and number is not None:
        return DutyRate(raw_text=raw_text, kind="ad_valorem", ad_valorem_rate=number)

    if has_percent and has_specific:
        return DutyRate(
            raw_text=raw_text,
            kind="text",
            notes="contains mixed ad valorem and specific components",
        )

    if has_specific and number is not None:
        unit_fragment = raw_text.replace(match.group(0), "", 1).strip() if match else raw_text
        unit_fragment = unit_fragment or None
        return DutyRate(
            raw_text=raw_text,
            kind="specific",
            specific_amount=number,
            specific_unit=unit_fragment,
        )

    if number is not None and has_percent:
        return DutyRate(raw_text=raw_text, kind="ad_valorem", ad_valorem_rate=number)

    if number is not None and not has_percent and not has_specific:
        return DutyRate(
            raw_text=raw_text,
            kind="text",
            notes="numeric value without context",
        )

    return DutyRate(raw_text=raw_text, kind="text", notes="unparsable duty text")


def compute_selection_signature(countries: list[str], products: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    if not countries or not products:
        return None
    return (tuple(countries), tuple(products))


def enforce_selection_limit(key: str, max_items: int) -> tuple[list[str], bool]:
    """Trim a list in session_state before widgets using the same key are rendered."""
    selections = st.session_state.get(key, []) or []
    if not isinstance(selections, list):
        selections = list(selections)
        st.session_state[key] = selections
    trimmed = len(selections) > max_items
    if trimmed:
        st.session_state[key] = selections[:max_items]
        selections = st.session_state[key]
    return selections, trimmed


def build_sql_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                "Return only one SELECT statement that answers the question."
            ),
        },
    ]


def extract_select_statement(text: str) -> str | None:
    cleaned = text.replace("`", "").strip()
    match = SELECT_PATTERN.search(cleaned)
    if not match:
        return None
    stmt = match.group(0)
    if ";" in stmt:
        stmt = stmt.split(";")[0]
    return stmt.strip()


def translate_question_to_sql(question: str, deployment_id: str) -> str:
    messages = build_sql_messages(question)
    response = openai.chat.completions.create(
        model=deployment_id,
        messages=messages,
        temperature=1,
    )
    raw_content = response.choices[0].message.content
    sql = extract_select_statement(raw_content)
    if not sql or not sql.strip().lower().startswith("select"):
        raise ValueError("The model did not return a valid SELECT statement.")
    return sql


def execute_sql(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    normalized = sql.strip().lower()
    if not normalized.startswith("select"):
        raise ValueError("Only SELECT statements are allowed.")
    return pd.read_sql_query(sql, conn)


def fetch_tariffs_for_codes(
    conn: sqlite3.Connection,
    selected_codes: list[str],
) -> pd.DataFrame:
    if not selected_codes:
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(selected_codes))
    query = (
        "SELECT hts_code, description, general_duty_rate "
        "FROM hts WHERE hts_code IN (" + placeholders + ")"
    )
    df = pd.read_sql_query(query, conn, params=selected_codes)
    if df.empty:
        return df
    df = df.rename(columns={"general_duty_rate": "general_duty_rate_text"})
    df["general_duty_rate_text"] = df["general_duty_rate_text"].fillna("").astype(str)
    parsed_rates = df["general_duty_rate_text"].apply(parse_general_duty)
    df["duty_kind"] = [rate.kind for rate in parsed_rates]
    df["ad_valorem_rate"] = [rate.ad_valorem_rate for rate in parsed_rates]
    df["specific_amount"] = [rate.specific_amount for rate in parsed_rates]
    df["specific_unit"] = [rate.specific_unit for rate in parsed_rates]
    df["duty_notes"] = [rate.notes for rate in parsed_rates]
    return df


def fetch_ch99_for_codes_and_countries(
    conn: sqlite3.Connection,
    selected_codes: list[str],
    selected_countries: list[str],
) -> pd.DataFrame:
    """Return the best Chapter 99 rule per (hts_code, queried_country).

    The view hts_with_ch99 already selected the best row per (hts_code,
    ch99_country bucket).  For each queried country we pull both the
    country-specific bucket AND the 'Global' bucket, then keep the one
    with the lower MATCH_PRIORITY (higher precedence).
    """
    if not selected_codes or not selected_countries:
        return pd.DataFrame()

    # Check whether the view exists (DB may have been built without ch99).
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM hts_with_ch99 LIMIT 1")
    except Exception:
        return pd.DataFrame()

    code_ph = ",".join(["?"] * len(selected_codes))
    country_ph = ",".join(["?"] * len(selected_countries))
    query = f"""
        SELECT
            hts_code,
            ch99_country,
            CAST(ch99_newrate      AS REAL) AS ch99_newrate,
            ch99_rate_modifier,
            COALESCE(CAST(ch99_additional_pct AS REAL), 0.0) AS ch99_additional_pct,
            ch99_tradeprogram,
            ch99_match_priority
        FROM hts_with_ch99
        WHERE hts_code IN ({code_ph})
          AND (ch99_country IN ({country_ph}) OR ch99_country = 'Global')
    """
    params = selected_codes + selected_countries
    df = pd.read_sql_query(query, conn, params=params)
    return df


def _best_ch99_rule(
    ch99_df: pd.DataFrame,
    hts_code: str,
    country: str,
) -> dict | None:
    """Return the highest-precedence (lowest MATCH_PRIORITY) Chapter 99 row
    for a given (hts_code, country) pair, considering 'Global' fallback."""
    if ch99_df.empty:
        return None
    mask = ch99_df["hts_code"] == hts_code
    mask &= (ch99_df["ch99_country"] == country) | (ch99_df["ch99_country"] == "Global")
    candidates = ch99_df[mask]
    if candidates.empty:
        return None
    best = candidates.loc[candidates["ch99_match_priority"].idxmin()]
    return best.to_dict()


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
      1. RATE_MODIFIER = 'Floor'  → duty = max(base_rate, NEWRATE_CLEAN)
                                    both values are in percentage-point scale; no conversion needed
      2. NEWRATE_CLEAN is a number → duty = NEWRATE_CLEAN * 100  (replaces base)
      3. Otherwise                 → duty = base_rate + ADDITIONAL_DUTY_PCT * 100
    """
    base_av  = base_rate.ad_valorem_rate  # percentage points; None for non-ad-valorem rates
    modifier = (ch99.get("ch99_rate_modifier") or "").strip()

    raw_newrate   = ch99.get("ch99_newrate")
    raw_additional = ch99.get("ch99_additional_pct")

    # Safe numeric conversions with None guards
    try:
        newrate = float(raw_newrate) if raw_newrate is not None else None
    except (TypeError, ValueError):
        newrate = None

    try:
        additional = float(raw_additional) if raw_additional is not None else 0.0
    except (TypeError, ValueError):
        additional = 0.0

    if modifier == "Floor":
        # Both base_av and newrate are in percentage-point scale here — no conversion
        if base_av is not None and newrate is not None:
            return max(base_av, newrate)
        return base_av

    if newrate is not None:
        # Non-Floor NEWRATE_CLEAN is a decimal fraction — scale to percentage points
        return newrate * 100

    if base_av is not None:
        # ADDITIONAL_DUTY_PCT is a decimal fraction — scale before adding
        return base_av + (additional * 100)

    return base_av


def build_correlation_dataframe(
    selected_countries: list[str],
    tariff_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    ch99_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    if not selected_countries or tariff_df.empty:
        return pd.DataFrame(), []

    country_subset = risk_df[risk_df["country"].isin(selected_countries)].copy()
    if country_subset.empty:
        return pd.DataFrame(), []

    # Accept both ad_valorem and specific rates; only exclude truly unparseable text rates.
    valid_products = tariff_df[
        (
            ((tariff_df["duty_kind"] == "ad_valorem") & tariff_df["ad_valorem_rate"].notna())
            | ((tariff_df["duty_kind"] == "specific") & tariff_df["specific_amount"].notna())
        )
    ].copy()
    excluded_mask = tariff_df["duty_kind"] == "text"
    excluded_records = tariff_df.loc[excluded_mask, ["hts_code", "description", "general_duty_rate_text", "duty_kind"]]
    exclusions = excluded_records.to_dict("records")

    if valid_products.empty:
        return pd.DataFrame(), exclusions

    # Build a lookup for parsed DutyRate objects keyed by hts_code.
    parsed_lookup: dict[str, DutyRate] = {}
    for _, row in valid_products.iterrows():
        parsed_lookup[row["hts_code"]] = parse_general_duty(row["general_duty_rate_text"])

    ch99_available = ch99_df is not None and not ch99_df.empty

    country_records = country_subset.to_dict("records")
    product_records = valid_products.to_dict("records")
    rows = []
    for country_meta, product_meta in itertools.product(country_records, product_records):
        country = country_meta["country"]
        hts_code = product_meta["hts_code"]
        base_rate = parsed_lookup[hts_code]
        base_av = base_rate.ad_valorem_rate

        effective_rate = base_av
        ch99_applied = False
        ch99_tradeprogram = None

        # Ch99 effective-rate adjustment applies only to ad_valorem base rates.
        # Specific rates ($/kg etc.) cannot be meaningfully adjusted by a percentage modifier.
        if ch99_available and base_rate.kind == "ad_valorem":
            rule = _best_ch99_rule(ch99_df, hts_code, country)
            if rule is not None:
                effective_rate = apply_ch99_to_duty(base_rate, rule)
                ch99_applied = True
                ch99_tradeprogram = rule.get("ch99_tradeprogram")

        rows.append(
            {
                "country": country,
                "risk_score": country_meta["score"],
                "risk_level": country_meta["level"],
                "country_color": country_meta["color"],
                "hts_code": hts_code,
                "product_description": product_meta["description"],
                "duty_kind": base_rate.kind,
                "base_duty_pct": base_av,
                "ad_valorem_rate": effective_rate,
                "specific_amount": base_rate.specific_amount,
                "specific_unit": base_rate.specific_unit,
                "general_duty_rate_text": product_meta["general_duty_rate_text"],
                "ch99_applied": ch99_applied,
                "ch99_tradeprogram": ch99_tradeprogram,
            }
        )
    return pd.DataFrame(rows), exclusions


def build_risk_snapshot(risk_df: pd.DataFrame, selected_countries: list[str]) -> list[dict]:
    if not selected_countries:
        return []
    subset = risk_df[risk_df["country"].isin(selected_countries)]
    if subset.empty:
        return []
    return subset[["country", "score", "level", "color", "year"]].to_dict("records")


def reset_app_state() -> None:
    st.session_state.messages = []
    st.session_state[LAST_RESULT_KEY] = None
    # Remove widget-controlled keys so Streamlit can recreate them cleanly.
    for widget_key in ["selected_countries", "selected_products_display"]:
        st.session_state.pop(widget_key, None)
    st.session_state["selected_product_codes"] = []
    st.session_state["correlation_signature"] = None
    st.session_state["analysis_active_run"] = None
    st.session_state["analysis_inflight"] = False
    st.session_state["analysis_request"] = None
    st.session_state["chat_scroll_token"] = 0


def append_message(message: dict) -> None:
    st.session_state.messages.append(message)
    st.session_state["chat_scroll_token"] = st.session_state.get("chat_scroll_token", 0) + 1


def render_correlation_chart(df: pd.DataFrame, mode: str = "ad_valorem") -> go.Figure | None:
    """Render the risk-vs-duty scatterplot.

    mode='ad_valorem' — Y axis = effective ad valorem rate (%, ch99-adjusted where applicable)
    mode='specific'   — Y axis = specific duty amount ($/kg, ¢/dozen, etc.)
    """
    if df.empty:
        return None

    df = df.copy()
    df["Trade Program"] = df["ch99_tradeprogram"].fillna("—")
    df["Base Rate"] = df["general_duty_rate_text"]

    if mode == "specific":
        plot_df = df[df["specific_amount"].notna()].copy()
        if plot_df.empty:
            return None

        units = plot_df["specific_unit"].dropna().unique()
        unit_label = units[0].strip() if len(units) == 1 else "unit"

        fig = px.scatter(
            plot_df,
            x="risk_score",
            y="specific_amount",
            color="country",
            hover_name="product_description",
            hover_data={
                "country": True,
                "risk_score": ":.1f",
                "risk_level": True,
                "hts_code": True,
                "Base Rate": True,
                "specific_amount": ":.4f",
                "Trade Program": True,
                "product_description": False,
                "general_duty_rate_text": False,
                "ch99_tradeprogram": False,
                "country_color": False,
                "risk_level": False,
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
    plot_df = df[df["ad_valorem_rate"].notna()].copy()
    if plot_df.empty:
        return None

    # Delta column — how much ch99 moved the rate
    def _delta_label(row: dict) -> str:
        if not row["ch99_applied"]:
            return "—"
        base = row.get("base_duty_pct")
        eff  = row.get("ad_valorem_rate")
        if base is None or eff is None:
            return "—"
        delta = eff - base
        sign  = "+" if delta >= 0 else ""
        return f"{sign}{delta:.2f}%"

    plot_df["Rate Source"] = plot_df["ch99_applied"].map({True: "Ch.99 adjusted", False: "Base rate"})
    plot_df["Ch.99 Δ"] = plot_df.apply(_delta_label, axis=1)

    fig = px.scatter(
        plot_df,
        x="risk_score",
        y="ad_valorem_rate",
        color="country",
        hover_name="product_description",
        hover_data={
            "country": True,
            "risk_score": ":.1f",
            "risk_level": True,
            "hts_code": True,
            "Base Rate": True,
            "ad_valorem_rate": ":.2f",
            "Rate Source": True,
            "Trade Program": True,
            "Ch.99 Δ": True,
            "product_description": False,
            "general_duty_rate_text": False,
            "ch99_applied": False,
            "ch99_tradeprogram": False,
            "base_duty_pct": False,
            "country_color": False,
            "risk_level": False,
        },
        labels={
            "risk_score": "Country Risk Score",
            "ad_valorem_rate": "Effective Duty Rate (%)",
        },
    )
    fig.update_layout(
        xaxis_title="Country Risk Score",
        yaxis_title="Effective Duty Rate (% ad valorem)",
        legend_title="Country",
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_traces(marker={"size": 12, "line": {"width": 1.5, "color": "rgba(0,0,0,0.25)"}})

    # Orange ring overlay on ch99-adjusted points
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
                    color="rgba(0,0,0,0)",
                    line=dict(color="rgba(255,140,0,0.85)", width=2),
                ),
                name="Ch.99 adjusted",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    return fig


def stream_analysis_to_placeholder(
    df: pd.DataFrame,
    deployment_id: str,
    placeholder: st.delta_generator.DeltaGenerator | None,
) -> str:
    subset = df.head(SUMMARY_SAMPLE_LIMIT)
    stats = {
        "count_pairs": len(df),
        "countries": sorted(df["country"].unique().tolist()),
        "hts_codes": sorted(df["hts_code"].unique().tolist()),
        "risk_min": float(df["risk_score"].min()),
        "risk_max": float(df["risk_score"].max()),
        "duty_min": float(df["ad_valorem_rate"].min()),
        "duty_max": float(df["ad_valorem_rate"].max()),
    }
    payload = {
        "stats": stats,
        "sample_rows": subset.to_dict("records"),
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a trade compliance analyst. Explain correlations between country risk scores "
                "and general duty rates in business language. Highlight extremes, clusters, and any anomalies."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write a concise paragraph (<=140 words) summarizing these tariff-risk pairs "
                "and give 1-2 actionable insights:\n"
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
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if not delta:
            continue
        full_text += delta
        if placeholder:
            placeholder.markdown(
                f"<div class='bubble-bot'>{full_text}▌</div>",
                unsafe_allow_html=True,
            )
    if placeholder:
        placeholder.markdown(
            f"<div class='bubble-bot'>{full_text}</div>",
            unsafe_allow_html=True,
        )
    return full_text.strip()


def queue_analysis_request(
    selected_countries: list[str],
    selected_products: list[str],
) -> bool:
    signature = compute_selection_signature(selected_countries, selected_products)
    if not signature:
        return False
    st.session_state["analysis_request"] = {
        "countries": list(selected_countries),
        "products": list(selected_products),
        "signature": signature,
    }
    # Reset chart mode so the new analysis always opens in ad_valorem view
    st.session_state["chart_mode"] = "ad_valorem"
    return True


def maybe_run_analysis(
    conn: sqlite3.Connection,
    deployment_id: str,
    risk_df: pd.DataFrame,
    placeholder: st.delta_generator.DeltaGenerator | None,
) -> None:
    request = st.session_state.get("analysis_request")
    if not request or st.session_state.get("analysis_inflight"):
        return

    countries = request.get("countries", [])
    products = request.get("products", [])
    signature = request.get("signature")
    if not countries or not products:
        st.session_state["analysis_request"] = None
        return

    run_id = uuid.uuid4().hex
    st.session_state["analysis_request"] = None
    st.session_state["analysis_inflight"] = True
    st.session_state["analysis_active_run"] = run_id
    success = False

    try:
        with st.spinner("Running analysis..."):
            tariff_df = fetch_tariffs_for_codes(conn, products)
            ch99_df = fetch_ch99_for_codes_and_countries(conn, products, countries)
            corr_df, duty_exclusions = build_correlation_dataframe(countries, tariff_df, risk_df, ch99_df)
            timestamp = datetime.now().strftime("%I:%M %p")
            risk_snapshot = build_risk_snapshot(risk_df, countries)
            exclusion_message = None
            if duty_exclusions:
                preview_labels = [
                    f"{item['hts_code']} ({item['general_duty_rate_text']})"
                    for item in duty_exclusions
                ]
                preview_limit = 3
                preview = ", ".join(preview_labels[:preview_limit])
                if len(preview_labels) > preview_limit:
                    preview += f", +{len(preview_labels) - preview_limit} more"
                exclusion_message = (
                    f"Skipped {len(duty_exclusions)} product(s) with non-percentage duty rates: {preview}."
                )
                st.warning(exclusion_message)
            if corr_df.empty:
                if duty_exclusions and not tariff_df.empty:
                    summary_text = (
                        "All selected products use quantity- or rule-based duty rates, so no ad valorem analysis is available."
                    )
                else:
                    summary_text = "No overlapping tariff-risk data for the current selections."
                if placeholder:
                    placeholder.markdown(
                        f"<div class='bubble-bot'>{summary_text}</div>",
                        unsafe_allow_html=True,
                    )
                append_message(
                    {
                        "role": "assistant",
                        "content": summary_text,
                        "time": timestamp,
                        "type": "analysis",
                        "chart_data": corr_df.to_dict("records") if not corr_df.empty else [],
                        "chart_columns": corr_df.columns.tolist(),
                        "risk_snapshot": risk_snapshot,
                        "selections": {
                            "countries": countries,
                            "products": products,
                        },
                        "duty_exclusions": duty_exclusions,
                        "duty_exclusion_message": exclusion_message,
                    }
                )
            else:
                fig = render_correlation_chart(corr_df)
                summary_text = stream_analysis_to_placeholder(corr_df, deployment_id, placeholder)
                n_adjusted = int(corr_df["ch99_applied"].sum()) if "ch99_applied" in corr_df.columns else 0
                n_total = len(corr_df)
                ch99_programs = (
                    corr_df.loc[corr_df["ch99_applied"], "ch99_tradeprogram"]
                    .dropna().unique().tolist()
                    if "ch99_applied" in corr_df.columns else []
                )
                append_message(
                    {
                        "role": "assistant",
                        "content": summary_text,
                        "time": timestamp,
                        "type": "analysis",
                        "plotly_fig": fig.to_dict() if fig else None,
                        "chart_data": corr_df.to_dict("records"),
                        "chart_columns": corr_df.columns.tolist(),
                        "risk_snapshot": risk_snapshot,
                        "selections": {
                            "countries": countries,
                            "products": products,
                        },
                        "duty_exclusions": duty_exclusions,
                        "duty_exclusion_message": exclusion_message,
                        "ch99_summary": {
                            "n_adjusted": n_adjusted,
                            "n_total": n_total,
                            "programs": ch99_programs,
                        },
                        "has_specific_data": (
                            "specific_amount" in corr_df.columns
                            and corr_df["specific_amount"].notna().any()
                        ),
                    }
                )
            st.session_state["correlation_signature"] = signature
            success = True
    except Exception as exc:
        logger.exception("Correlation analysis failed")
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


def main() -> None:
    st.set_page_config(page_title="ImportInsight AI", layout="wide")
    st.markdown(_format_css(), unsafe_allow_html=True)

    st.markdown(
        """
        <h1 style='color:#0f1f38; font-weight:700; letter-spacing:-0.5px;'>ImportInsight AI</h1>
        <p style='color:#64748b; font-size:15px; margin-top:-10px;'>Natural-language queries -> SQL over the HTS data.</p>
        <hr style='border: 1px solid #e2e8f0; margin-top:16px;'>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style='background-color:#fef9ec; border-left: 4px solid #f0a500; padding: 10px 16px; border-radius: 6px; margin-bottom: 20px;'>
            <span style='color:#7d5a00; font-size:13px;'>
                <b>Disclaimer:</b> This tool is for informational purposes only and does not constitute legal advice.
            </span>
        </div>
    """,
        unsafe_allow_html=True,
    )

    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        st.error("Please set AZURE_OPENAI_API_KEY before running the app.")
        return
    if not openai.api_base:
        st.error("Please set AZURE_OPENAI_ENDPOINT before running the app.")
        return
    openai.api_key = api_key

    deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")
    if not deployment_id:
        st.error("Please set AZURE_OPENAI_DEPLOYMENT_ID for the chat completion deployment.")
        return

    if not HTS_DB_PATH.exists():
        st.error(
            "The local HTS database is missing. Run `python scripts/build_hts_sqlite.py` to create data/hts.db before launching the app."
        )
        return

    conn = get_db_connection(str(HTS_DB_PATH))

    risk_df = get_risk_df()
    country_options = risk_df["country"].tolist()
    if not country_options:
        country_options = DEFAULT_COUNTRY_SELECTION.copy()
    product_options = load_product_options(conn)
    display_options = [f"{code} — {desc}" for code, desc in product_options]
    code_map = dict(zip(display_options, [code for code, _ in product_options]))
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.setdefault(LAST_RESULT_KEY, None)
    st.session_state.setdefault("selected_countries", [])
    st.session_state.setdefault("selected_products_display", [])
    st.session_state.setdefault("selected_product_codes", [])
    st.session_state.setdefault("analysis_inflight", False)
    st.session_state.setdefault("analysis_active_run", None)
    st.session_state.setdefault("analysis_request", None)
    st.session_state.setdefault("correlation_signature", None)
    st.session_state.setdefault("chat_scroll_token", 0)

    selected_countries, country_trimmed = enforce_selection_limit(
        "selected_countries",
        MAX_COUNTRY_SELECTION,
    )
    selected_products_display, product_trimmed = enforce_selection_limit(
        "selected_products_display",
        MAX_PRODUCT_SELECTION,
    )

    with st.sidebar:
        st.image("logo.png", width=150)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Settings")
        st.caption(
            "Natural language inputs are translated to SQL, executed locally against a read-only SQLite copy of the HTS data."
        )
        st.caption(
            "Responses are deterministic: the SQL output is re-run each time against the local database."
        )
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### About")
        st.caption("ImportInsight AI translates your prompt into SQL and returns the actual HTS rows.")

    context_bar = st.container()
    with context_bar:
        st.markdown('<div data-context="true"></div>', unsafe_allow_html=True)
        st.markdown("### Context")
        st.caption("Select up to three countries and products to drive the analysis below.")
        sel_cols = st.columns(2)
        with sel_cols[0]:
            st.multiselect(
                "Countries",
                options=country_options,
                key="selected_countries",
                max_selections=MAX_COUNTRY_SELECTION,
                help="Choose up to three countries to compare governance risk.",
            )
        with sel_cols[1]:
            if display_options:
                st.multiselect(
                    "HTS Products",
                    options=display_options,
                    key="selected_products_display",
                    max_selections=MAX_PRODUCT_SELECTION,
                    help="Products are loaded from the local HTS SQLite database.",
                )
            else:
                st.error("No HTS products found—rebuild the SQLite database.")
        selected_countries = st.session_state.get("selected_countries", [])
        selected_products_display = st.session_state.get("selected_products_display", [])
        st.caption(
            f"{len(selected_countries)} / {MAX_COUNTRY_SELECTION} countries · {len(selected_products_display)} / {MAX_PRODUCT_SELECTION} products"
        )
        if country_trimmed:
            st.warning(
                f"Country selection limited to {MAX_COUNTRY_SELECTION}. Extra choices were dropped."
            )
        if product_trimmed:
            st.warning(
                f"Product selection limited to {MAX_PRODUCT_SELECTION}. Extra choices were dropped."
            )
        current_product_labels = st.session_state.get("selected_products_display", [])
        valid_product_labels = [label for label in current_product_labels if label in code_map]
        if len(valid_product_labels) != len(current_product_labels):
            st.warning("Some selected products are unavailable in the current HTS list.")
        selected_products = [code_map[label] for label in valid_product_labels]
        st.session_state["selected_product_codes"] = selected_products

        action_cols = st.columns([1, 1])
        analyse_disabled = (
            st.session_state.get("analysis_inflight")
            or not selected_countries
            or not selected_products
        )
        with action_cols[0]:
            analyse_clicked = st.button(
                "Analyse",
                width="stretch",
                disabled=analyse_disabled,
            )
        with action_cols[1]:
            if st.button("Clear Chat", width="stretch", key="context_clear"):
                reset_app_state()
                st.rerun()
        if analyse_clicked:
            queued = queue_analysis_request(selected_countries, selected_products)
            if not queued:
                st.warning("Select at least one country and one product before running analysis.")
        if st.session_state.get("analysis_inflight"):
            st.caption("Running analysis…")

        # Chart mode toggle — persists across reruns so switching is instant
        st.session_state.setdefault("chart_mode", "ad_valorem")
        # Determine whether the most recent analysis has specific-rate data
        _last_analysis = next(
            (m for m in reversed(st.session_state.get("messages", []))
             if m.get("type") == "analysis"),
            None,
        )
        _has_specific = bool(_last_analysis and _last_analysis.get("has_specific_data"))
        _toggle_help = (
            "Switch the Y axis between the ad valorem percentage rate and the specific duty amount (e.g. $/kg, ¢/dozen)."
            if _has_specific
            else "This product has a percentage-based rate only. Select a product with a unit-based rate (e.g. $/kg, ¢/liter) to use this view."
        )
        if not _has_specific:
            st.session_state["chart_mode"] = "ad_valorem"
        show_specific = st.toggle(
            "Show specific duty rate ($/unit)",
            value=(st.session_state["chart_mode"] == "specific"),
            disabled=not _has_specific,
            help=_toggle_help,
        )
        if _has_specific:
            st.session_state["chart_mode"] = "specific" if show_specific else "ad_valorem"
    analysis_stream_placeholder: st.delta_generator.DeltaGenerator | None = None
    chat_feed = st.container()
    composer = st.container()

    with composer:
        st.markdown('<div data-composer="true"></div>', unsafe_allow_html=True)
        prompt = st.chat_input(QUESTION_PLACEHOLDER, key="chat_input")

    if prompt is not None:
        question = prompt.strip()
        if not question:
            st.warning("Please enter a prompt before sending.")
        else:
            timestamp = datetime.now().strftime("%I:%M %p")
            append_message({"role": "user", "content": question, "time": timestamp})
            try:
                sql = translate_question_to_sql(question, deployment_id)
                df = execute_sql(conn, sql)
                records = df.head(200).to_dict("records")
                st.session_state[LAST_RESULT_KEY] = {
                    "sql": sql,
                    "row_count": len(df),
                    "records": records,
                    "columns": list(df.columns),
                    "rows_displayed": len(records),
                }
                assistant_text = f"Executed SQL and returned {len(df)} row(s)."
            except Exception as exc:
                logger.exception("SQL execution failed")
                st.session_state[LAST_RESULT_KEY] = None
                assistant_text = f"Error: {exc}"
            append_message(
                {"role": "assistant", "content": assistant_text, "time": datetime.now().strftime("%I:%M %p")}
            )

    with chat_feed:
        st.markdown('<div data-chat="true"></div>', unsafe_allow_html=True)
        if len(st.session_state.messages) == 0:
            st.markdown(
                "<div class='empty-chat'>Select countries/products above or ask a question about the HTS data.</div>",
                unsafe_allow_html=True,
            )
        for msg in st.session_state.messages:
            timestamp = msg.get("time", "")
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='bubble-label bubble-label-right'>You</div><div class='bubble-user'>{msg['content']}</div><div class='timestamp timestamp-right'>{timestamp}</div><div class='clearfix'></div>",
                    unsafe_allow_html=True,
                )
                continue

            if msg.get("type") == "analysis":
                st.markdown(
                    "<div class='bubble-label'>Assistant</div>",
                    unsafe_allow_html=True,
                )

                # --- metadata: countries + product descriptions ---
                selections = msg.get("selections", {})
                selection_text = []
                if selections.get("countries"):
                    selection_text.append("Countries: " + ", ".join(selections["countries"]))
                if selections.get("products"):
                    # Resolve raw HTS codes to human-readable descriptions via code_map
                    descs = [
                        code_map.get(
                            next((lbl for lbl in code_map if code_map[lbl] == c), ""),
                            c,
                        )
                        for c in selections["products"]
                    ]
                    # code_map keys are "CODE — description"; extract description part
                    readable = []
                    for raw_code in selections["products"]:
                        match = next(
                            (lbl for lbl, code in code_map.items() if code == raw_code),
                            None,
                        )
                        readable.append(match.split(" — ", 1)[-1] if match else raw_code)
                    selection_text.append("Products: " + ", ".join(readable))
                if selection_text:
                    st.markdown(
                        f"<div class='analysis-meta'>{' · '.join(selection_text)}</div>",
                        unsafe_allow_html=True,
                    )

                # --- scatterplot (rebuilt live so the toggle takes effect instantly) ---
                _chart_data = msg.get("chart_data")
                _chart_cols = msg.get("chart_columns")
                if _chart_data and _chart_cols:
                    _chart_df = pd.DataFrame(_chart_data, columns=_chart_cols)
                    _mode = st.session_state.get("chart_mode", "ad_valorem")
                    fig = render_correlation_chart(_chart_df, mode=_mode)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        if _mode == "ad_valorem":
                            st.caption(
                                "Orange ring = effective rate modified by a Chapter 99 surcharge or trade program override."
                            )
                        else:
                            st.caption(
                                "Y axis shows the specific duty amount from the HTS table. "
                                "Ch.99 percentage surcharges are not applied in this view."
                            )
                    else:
                        _mode_label = "specific rate" if _mode == "specific" else "ad valorem rate"
                        st.info(
                            f"No {_mode_label} data available for this product. "
                            "Try switching the toggle above."
                        )

                # --- Fix 4: ch99 adjustment callout ---
                ch99_summary = msg.get("ch99_summary")
                if ch99_summary and ch99_summary.get("n_adjusted", 0) > 0:
                    n_adj = ch99_summary["n_adjusted"]
                    n_tot = ch99_summary["n_total"]
                    programs = ch99_summary.get("programs") or []
                    prog_str = (
                        " — " + ", ".join(sorted(set(programs))) if programs else ""
                    )
                    st.info(
                        f"Chapter 99 adjustments applied to **{n_adj} of {n_tot}** data points{prog_str}."
                    )

                # --- AI narrative bubble ---
                st.markdown(
                    f"<div class='bubble-bot'>{msg['content']}</div>"
                    f"<div class='timestamp'>{timestamp}</div>"
                    "<div class='clearfix'></div>",
                    unsafe_allow_html=True,
                )

                # --- risk pills ---
                risk_snapshot = msg.get("risk_snapshot") or []
                if risk_snapshot:
                    pills_html = "".join(
                        f"<div class='risk-pill' style='border-left-color:{snap['color']};'>"
                        f"<strong>{snap['country']}</strong> "
                        f"Score {snap['score']:.1f} · {snap['level']}"
                        "</div>"
                        for snap in risk_snapshot
                    )
                    st.markdown(f"<div class='risk-pills'>{pills_html}</div>", unsafe_allow_html=True)

                # --- Fix 1: clean summary table (replaces raw corr_df dump) ---
                chart_data = msg.get("chart_data")
                if chart_data:
                    raw_df = pd.DataFrame(chart_data, columns=msg.get("chart_columns"))
                    display_cols = {
                        "country": "Country",
                        "risk_score": "Risk Score",
                        "hts_code": "HTS Code",
                        "product_description": "Product",
                        "general_duty_rate_text": "Base Rate",
                        "ad_valorem_rate": "Effective Rate (%)",
                        "ch99_tradeprogram": "Trade Program",
                    }
                    available = [c for c in display_cols if c in raw_df.columns]
                    summary_df = raw_df[available].rename(columns=display_cols).copy()
                    if "Effective Rate (%)" in summary_df.columns:
                        summary_df["Effective Rate (%)"] = (
                            summary_df["Effective Rate (%)"]
                            .apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "—")
                        )
                    if "Trade Program" in summary_df.columns:
                        summary_df["Trade Program"] = summary_df["Trade Program"].fillna("—")
                    if "Risk Score" in summary_df.columns:
                        summary_df["Risk Score"] = (
                            summary_df["Risk Score"]
                            .apply(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
                        )
                    st.dataframe(summary_df, hide_index=True, use_container_width=True)

                # --- duty exclusion warning ---
                exclusion_text = msg.get("duty_exclusion_message")
                if exclusion_text:
                    st.warning(exclusion_text)
                elif msg.get("duty_exclusions"):
                    duty_exclusions = msg["duty_exclusions"]
                    listed = ", ".join(
                        f"{item.get('hts_code')} ({item.get('general_duty_rate_text')})"
                        for item in duty_exclusions[:3]
                    )
                    if len(duty_exclusions) > 3:
                        listed += f", +{len(duty_exclusions) - 3} more"
                    st.warning(f"Skipped non-percentage duty rates: {listed}")

                continue

            st.markdown(
                f"<div class='bubble-label'>Assistant</div><div class='bubble-bot'>{msg['content']}</div><div class='timestamp'>{timestamp}</div><div class='clearfix'></div>",
                unsafe_allow_html=True,
            )

        analysis_stream_placeholder = st.empty()
        latest_result = st.session_state.get(LAST_RESULT_KEY)
        if latest_result:
            with st.expander(
                f"Last SQL Result — {latest_result['row_count']} row(s) returned",
                expanded=False,
            ):
                st.code(latest_result["sql"], language="sql")
                st.caption(
                    f"Showing {latest_result['rows_displayed']} of {latest_result['row_count']} row(s)."
                )
                if latest_result["records"]:
                    st.dataframe(
                        pd.DataFrame(latest_result["records"], columns=latest_result["columns"]),
                        use_container_width=True,
                    )
                else:
                    st.info("The last query returned no rows.")
        st.markdown('<div data-anchor="chat-end" id="chat-end"></div>', unsafe_allow_html=True)

    if analysis_stream_placeholder is None:
        analysis_stream_placeholder = st.empty()

    maybe_run_analysis(
        conn,
        deployment_id,
        risk_df,
        analysis_stream_placeholder,
    )

    components.html(
        f"""
        <script>
        const marker = window.parent.document.querySelector('div[data-anchor="chat-end"]');
        if (marker) {{
            const chatBlock = marker.closest('div[data-testid="stVerticalBlock"]');
            if (chatBlock) {{
                chatBlock.scrollTop = chatBlock.scrollHeight;
            }}
        }}
        </script>
        """,
        height=0,
    )

if __name__ == "__main__":
    main()
