from __future__ import annotations
# ImportInsight AI
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import openai
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from analysis import maybe_run_analysis, queue_analysis_request
from ranker import maybe_run_best_countries
from chat import answer_question, append_message
from risk_model import get_risk_df
from session import enforce_selection_limit, reset_app_state
from utils import LAST_RESULT_KEY

load_dotenv()

logger = logging.getLogger(__name__)

HTS_DB_PATH = Path(__file__).resolve().parent / "data" / "hts.db"
HTS_VIEW_NAME = "hts_with_ch99"
MAX_PRODUCT_OPTIONS = 1000
ENSURE_PRODUCT_CODES = ["0406.40.44.00", "0405.90.20"]
DEFAULT_COUNTRY_SELECTION = [
    "Cameroon",
    "Russia",
]
MAX_COUNTRY_SELECTION = 3
MAX_PRODUCT_SELECTION = 1

QUESTION_PLACEHOLDER = "Ask about tariffs, compliance, or trade regulations..."

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
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background:#ffffff; }
    section.main { background: #ffffff; }
    .block-container { background: #ffffff; }
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
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .bubble-user:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 6px 20px rgba(26,58,92,0.25);
    }
    .bubble-bot-wrap:hover .bubble-bot {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    .bubble-user {
    /* country pills */
    span[style*="background:#0B2A4A"] {
        transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease !important;
        cursor: default;
        box-shadow: 0 2px 6px rgba(11,42,74,0.3);
    }
    span[style*="background:#0B2A4A"]:hover {
        transform: translateY(-2px) scale(1.05) !important;
        box-shadow: 0 6px 16px rgba(11,42,74,0.4) !important;
        background: #1a3a5c !important;
    }
    /* hts pills */
    span[style*="background:#4F6D7A"] {
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        box-shadow: 0 2px 6px rgba(79,109,122,0.3);
    }
    span[style*="background:#4F6D7A"]:hover {
        transform: translateY(-2px) scale(1.05) !important;
        box-shadow: 0 6px 16px rgba(79,109,122,0.4) !important;
    }
    /* analyse button */
    .stButton > button {
        transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease !important;
        box-shadow: 0 4px 14px rgba(26,58,92,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(26,58,92,0.4) !important;
        background-color: #2a5298 !important;
    }
    .stButton > button:active {
        transform: translateY(0px) scale(0.98) !important;
        box-shadow: 0 2px 8px rgba(26,58,92,0.2) !important;
    }
    /* risk gauge cards */
    div[style*="border-radius:10px;padding:12px"] {
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }
    div[style*="border-radius:10px;padding:12px"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1) !important;
    }
    /* sidebar multiselect tags */
    [data-testid="stSidebar"] [data-baseweb="tag"] {
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"]:hover {
        transform: translateY(-2px) scale(1.03) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    }
    /* recent session items */
    div[style*="rgba(255,255,255,0.05)"] {
        transition: background 0.15s ease, transform 0.15s ease !important;
        cursor: pointer;
    }
    div[style*="rgba(255,255,255,0.05)"]:hover {
        background: rgba(255,255,255,0.1) !important;
        transform: translateX(3px) !important;
    }

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
        background-color: #ffffff !important;
        color: #0B2A4A !important;
        border-radius: 28px !important;
        font-weight: 500 !important;
        border: 1.5px solid #D1D5DB !important;
        padding: 10px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background-color: #0B2A4A !important;
        color: white !important;
        border-color: #0B2A4A !important;
        box-shadow: 0 4px 16px rgba(11,42,74,0.2) !important;
        transform: translateY(-1px) !important;
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
    /* force chat input visibility */
    
    
    
    
    
    
    </style>
    """


@st.cache_resource
def get_db_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_resource(show_spinner=False)
def _load_full_product_catalog(db_path: str) -> pd.DataFrame:
    query = (
        f'SELECT hts_code, description FROM {HTS_VIEW_NAME} '
        'WHERE hts_code IS NOT NULL AND hts_code <> "" '
        'ORDER BY hts_code'
    )
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(query, conn)
    except Exception as exc:
        logger.warning("Failed to load HTS catalog: %s", exc)
        return pd.DataFrame(columns=["hts_code", "description"])


def load_product_options(_conn: sqlite3.Connection) -> list[tuple[str, str]]:
    df = _load_full_product_catalog(str(HTS_DB_PATH)).copy()
    if df.empty:
        return []

    missing_codes: list[str] = []
    if ENSURE_PRODUCT_CODES:
        present = set(df["hts_code"].tolist())
        missing_codes = [code for code in ENSURE_PRODUCT_CODES if code not in present]
    if missing_codes:
        placeholders = ",".join(["?"] * len(missing_codes))
        ensure_query = (
            f"SELECT hts_code, description FROM {HTS_VIEW_NAME} "
            f"WHERE hts_code IN ({placeholders})"
        )
        try:
            ensure_df = pd.read_sql_query(ensure_query, _conn, params=missing_codes)
            if not ensure_df.empty:
                df = pd.concat([df, ensure_df], ignore_index=True)
        except Exception as exc:
            logger.warning("Failed to fetch ensured HTS codes: %s", exc)

    df = df.drop_duplicates(subset=["hts_code"]).sort_values("hts_code")
    return list(df.itertuples(index=False, name=None))




def main() -> None:
    st.set_page_config(page_title="ImportInsight AI", layout="wide")
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { opacity: 1 !important; transition: none !important; }
    [data-testid="stAppViewBlockContainer"] { opacity: 1 !important; transition: none !important; }
    div[data-stale="true"] { opacity: 1 !important; transition: none !important; }
    div[data-stale="false"] { opacity: 1 !important; transition: none !important; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(_format_css(), unsafe_allow_html=True)
    st.markdown("""<style>
[data-testid="stBottom"] > div {
    background: transparent !important;
    padding: 16px 24px !important;
    border-top: none !important;
}
[data-testid="stChatInputContainer"] {
    background: #FFFFFF !important;
    border: 2px solid #CBD5E1 !important;
    border-radius: 32px !important;
    padding: 6px 10px 6px 20px !important;
    outline: 4px solid rgba(11,42,74,0.08) !important;
}
[data-testid="stChatInputContainer"]:focus-within {
    border-color: #0B2A4A !important;
    outline: 4px solid rgba(11,42,74,0.12) !important;
}
[data-testid="stChatInputContainer"] textarea {
    color: #1F2937 !important;
    font-size: 14px !important;
    background: transparent !important;
}
[data-testid="stChatInputContainer"] textarea::placeholder {
    color: #9CA3AF !important;
    opacity: 1 !important;
}
/* Tab styling */
button[data-baseweb="tab"] { font-size:13px !important; font-weight:500 !important; color:#94A3B8 !important; padding:8px 16px !important; background:transparent !important; }
button[data-baseweb="tab"]:hover { color:#0B2A4A !important; }
button[data-baseweb="tab"][aria-selected="true"] { font-weight:700 !important; color:#0B2A4A !important; border-bottom:2px solid #0B2A4A !important; }
</style>""", unsafe_allow_html=True)
    st.markdown("""
<style>
section
div
div
div
div
div
</style>
""", unsafe_allow_html=True)



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
    country_options = sorted(risk_df["country"].tolist())
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

    # Apply pending country selection from best-countries ranker before the widget renders
    _pending = st.session_state.pop("best_countries_pending", None)
    if _pending:
        st.session_state["selected_countries"] = _pending

    # do not auto-refill selections - let user control them

    selected_countries, country_trimmed = enforce_selection_limit(
        "selected_countries",
        MAX_COUNTRY_SELECTION,
    )
    selected_products_display, product_trimmed = enforce_selection_limit(
        "selected_products_display",
        MAX_PRODUCT_SELECTION,
    )

    with st.sidebar:
        st.markdown("""
        <style>
        [data-testid="stSidebar"] { background-color: #0B2A4A !important; }
        [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
        [data-testid="stSidebar"] img { filter: brightness(0) invert(1); }
        section[data-testid="stSidebar"] > div { padding-top: 0 !important; margin-top: -80px !important; }
        section[data-testid="stSidebar"] > div > div { padding-top: 0 !important; }
        section[data-testid="stSidebar"] > div > div > div { padding-top: 0 !important; }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background-color: rgba(255,255,255,0.1) !important;
            border: 0.5px solid rgba(255,255,255,0.2) !important;
            border-radius: 6px !important;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"] span { color: rgba(255,255,255,0.85) !important; font-size: 11px !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background-color: rgba(255,255,255,0.05) !important;
            border: 0.5px solid rgba(255,255,255,0.12) !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            background-color: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            color: #e2e8f0 !important;
            border-radius: 10px !important;
            width: 100%;
        }
        span[data-baseweb="tag"] { background-color: #0B2A4A !important; border-color: #4F6D7A !important; }
        span[data-baseweb="tag"] span { color: #ffffff !important; }
        </style>
        """, unsafe_allow_html=True)
        st.image("logo_clean.png", width=120)


        st.markdown("<div style='background:rgba(240,165,0,0.12);border-left:3px solid #f0a500;border-radius:6px;padding:8px 12px;font-size:11px;color:#fde68a;margin:8px 0;'><b>Disclaimer:</b> This tool is for informational purposes only and does not constitute legal advice.</div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:rgba(255,255,255,0.75);margin-bottom:8px;padding-left:10px;border-left:3px solid rgba(99,179,237,0.7);'>Recent Sessions</p>", unsafe_allow_html=True)
        if 'session_history' not in st.session_state:
            st.session_state['session_history'] = []
        if st.session_state['session_history']:
            for entry in reversed(st.session_state['session_history'][-4:]):
                st.markdown(f"<div style='padding:6px 8px;border-radius:6px;font-size:12px;color:rgba(255,255,255,0.55);margin-bottom:3px;background:rgba(255,255,255,0.05);'><span style='color:#4F8FB8;margin-right:6px;'>●</span>{entry}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:11px;color:rgba(255,255,255,0.25);padding:4px 8px;'>No sessions yet</div>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        _bc_inflight = st.session_state.get("best_countries_inflight", False)
        _bc_codes = [code_map[label] for label in st.session_state.get("selected_products_display", []) if label in code_map]
        _bc_disabled = _bc_inflight or not _bc_codes or bool(st.session_state.get("analysis_inflight"))

        if _bc_inflight:
            _btn_label = "Finding best countries…"
            _btn_bg = "rgba(255,255,255,0.06)"
            _btn_color = "rgba(255,255,255,0.4)"
            _btn_border = "rgba(255,255,255,0.15)"
        elif _bc_disabled:
            _btn_label = "Auto-select best countries"
            _btn_bg = "rgba(255,255,255,0.05)"
            _btn_color = "rgba(255,255,255,0.3)"
            _btn_border = "rgba(255,255,255,0.1)"
        else:
            _btn_label = "Auto-select best countries"
            _btn_bg = "rgba(99,179,237,0.15)"
            _btn_color = "rgba(255,255,255,0.9)"
            _btn_border = "rgba(99,179,237,0.4)"

        st.markdown(f"""<style>
        [data-testid="stSidebar"] [data-testid="stButton"] button {{
            background: {_btn_bg} !important;
            color: {_btn_color} !important;
            border: 1px solid {_btn_border} !important;
            border-radius: 8px !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            letter-spacing: 0.03em !important;
            padding: 10px 12px !important;
            width: 100% !important;
            text-align: center !important;
            box-shadow: none !important;
            cursor: {"not-allowed" if _bc_disabled else "pointer"} !important;
        }}
        </style>""", unsafe_allow_html=True)

        st.markdown("<p style='font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:rgba(255,255,255,0.75);margin:0 0 6px 0;padding-left:10px;border-left:3px solid rgba(99,179,237,0.7);'>Countries</p>", unsafe_allow_html=True)
        st.multiselect("Countries", options=country_options, key="selected_countries", max_selections=MAX_COUNTRY_SELECTION, label_visibility="hidden")

        _or_style = "display:flex;align-items:center;gap:8px;margin:8px 0;"
        _line_style = "flex:1;height:1px;background:rgba(255,255,255,0.1);"
        _or_text_style = "font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:0.08em;text-transform:uppercase;white-space:nowrap;"
        st.markdown(f"<div style='{_or_style}'><div style='{_line_style}'></div><span style='{_or_text_style}'>or</span><div style='{_line_style}'></div></div>", unsafe_allow_html=True)

        if st.button(
            _btn_label,
            key="btn_find_best_countries",
            disabled=_bc_disabled,
            help="Auto-select the 3 best countries for this HTS code based on tariff rates, corruption risk, and trade-flow reasoning.",
        ):
            st.session_state["best_countries_request"] = {"hts_code": _bc_codes[0]}
            st.rerun()
        if not _bc_codes and not _bc_inflight:
            st.markdown("<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-top:4px;margin-bottom:0;padding:0;'>Select an HTS code below to enable</div>", unsafe_allow_html=True)
        if st.session_state.get("best_countries_error"):
            st.markdown(f"<div style='font-size:10px;color:#f87171;padding:4px 0;'>{st.session_state.pop('best_countries_error')}</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:rgba(255,255,255,0.75);margin:0 0 6px 0;padding-left:10px;border-left:3px solid rgba(99,179,237,0.7);'>HTS Products</p>", unsafe_allow_html=True)
        if display_options:
            all_codes = [opt.split(" — ")[0].strip() for opt in display_options]

            # Build chapter list (first 2 digits)
            chapters = sorted(set(code[:2] for code in all_codes if len(code) >= 2))
            CHAPTER_NAMES = {
                "01": "Live animals",
                "02": "Meat and edible meat offal",
                "03": "Fish and crustaceans",
                "04": "Dairy, eggs, honey",
                "05": "Other animal products",
                "06": "Live trees and plants",
                "07": "Edible vegetables",
                "08": "Edible fruit and nuts",
                "09": "Coffee, tea, spices",
                "10": "Cereals",
                "11": "Milling industry products",
                "12": "Oil seeds and misc grains",
                "13": "Lac, gums, resins",
                "14": "Vegetable plaiting materials",
                "15": "Animal and vegetable fats",
                "16": "Prepared meat and fish",
                "17": "Sugars and confectionery",
                "18": "Cocoa and cocoa preparations",
                "19": "Preparations of cereals",
                "20": "Preparations of vegetables and fruit",
                "21": "Miscellaneous edible preparations",
                "22": "Beverages and vinegar",
                "23": "Residues from food industries",
                "24": "Tobacco",
                "25": "Salt, sulfur, earth, stone",
                "26": "Ores, slag and ash",
                "27": "Mineral fuels and oils",
                "28": "Inorganic chemicals",
                "29": "Organic chemicals",
                "30": "Pharmaceutical products",
                "31": "Fertilizers",
                "32": "Tanning and dyeing extracts",
                "33": "Essential oils and cosmetics",
                "34": "Soap and waxes",
                "35": "Albuminoidal substances",
                "36": "Explosives and pyrotechnics",
                "37": "Photographic goods",
                "38": "Miscellaneous chemicals",
                "39": "Plastics",
                "40": "Rubber",
                "41": "Raw hides and skins",
                "42": "Leather articles",
                "43": "Furskins and artificial fur",
                "44": "Wood and articles of wood",
                "45": "Cork",
                "46": "Manufactures of straw",
                "47": "Pulp of wood",
                "48": "Paper and paperboard",
                "49": "Printed books and newspapers",
                "50": "Silk",
                "51": "Wool and animal hair",
                "52": "Cotton",
                "53": "Other vegetable textile fibres",
                "54": "Man-made filaments",
                "55": "Man-made staple fibres",
                "56": "Wadding and felt",
                "57": "Carpets and floor coverings",
                "58": "Special woven fabrics",
                "59": "Impregnated textile fabrics",
                "60": "Knitted or crocheted fabrics",
                "61": "Knitted apparel",
                "62": "Woven apparel",
                "63": "Other made-up textile articles",
                "64": "Footwear",
                "65": "Headgear",
                "66": "Umbrellas and walking sticks",
                "67": "Feathers and artificial flowers",
                "68": "Stone, plaster and cement",
                "69": "Ceramic products",
                "70": "Glass and glassware",
                "71": "Precious stones and metals",
                "72": "Iron and steel",
                "73": "Articles of iron or steel",
                "74": "Copper",
                "75": "Nickel",
                "76": "Aluminium",
                "78": "Lead",
                "79": "Zinc",
                "80": "Tin",
                "81": "Other base metals",
                "82": "Tools and cutlery",
                "83": "Miscellaneous metal articles",
                "84": "Machinery and mechanical appliances",
                "85": "Electrical machinery",
                "86": "Railway locomotives",
                "87": "Vehicles",
                "88": "Aircraft and spacecraft",
                "89": "Ships and boats",
                "90": "Optical and medical instruments",
                "91": "Clocks and watches",
                "92": "Musical instruments",
                "93": "Arms and ammunition",
                "94": "Furniture and bedding",
                "95": "Toys and games",
                "96": "Miscellaneous manufactured articles",
                "97": "Works of art",
                "98": "Special classification provisions",
                "99": "Temporary legislation",
            }
            chapter_labels = {ch: f"Ch. {ch} — {CHAPTER_NAMES.get(ch.zfill(2), 'Other')}" for ch in chapters}

            st.session_state.setdefault("hts_drill_chapter", None)
            st.session_state.setdefault("hts_drill_heading", None)

            # Step 1: Chapter
            chapter_choice = st.selectbox(
                "Step 1 — Chapter",
                options=[""] + chapters,
                format_func=lambda x: "Select chapter..." if x == "" else chapter_labels.get(x, f"Ch. {x}"),
                key="hts_drill_chapter",
                label_visibility="hidden",
            )

            # Step 2: 4-digit heading
            if chapter_choice:
                headings = sorted(set(
                    code[:4] for code in all_codes
                    if code.startswith(chapter_choice) and len(code) >= 4
                ))
                heading_labels = {h: f"{h} — {next((opt.split(' — ',1)[1] for opt in display_options if opt.split(' — ')[0].strip().startswith(h)), '')}" for h in headings}
                heading_choice = st.selectbox(
                    "Step 2 — Heading",
                    options=[""] + headings,
                    format_func=lambda x: "Select heading..." if x == "" else heading_labels.get(x, x),
                    key="hts_drill_heading",
                    label_visibility="hidden",
                )
            else:
                heading_choice = None

            # Step 3: Full 8/10-digit code — add to selection
            if heading_choice:
                subheadings = [
                    opt for opt in display_options
                    if opt.split(" — ")[0].strip().startswith(heading_choice)
                ]
                if subheadings:
                    sub_choice = st.selectbox(
                        "Step 3 — Subheading",
                        options=[""] + subheadings,
                        format_func=lambda x: "Select code..." if x == "" else x,
                        key="hts_drill_sub",
                        label_visibility="hidden",
                    )
                    if sub_choice and sub_choice not in st.session_state.get("selected_products_display", []):
                        if st.button("+ Add to selection", key="hts_add_btn"):
                            current = st.session_state.get("selected_products_display", [])
                            if len(current) < MAX_PRODUCT_SELECTION:
                                current.append(sub_choice)
                                st.session_state["selected_products_display"] = current
                                for k in ["hts_drill_chapter", "hts_drill_heading", "hts_drill_sub"]:
                                    if k in st.session_state:
                                        del st.session_state[k]
                                st.rerun()

            # Show selected products
            selected_prods = st.session_state.get("selected_products_display", [])
            if selected_prods:
                for i, prod in enumerate(selected_prods):
                    st.markdown(f"<div style='font-size:11px;color:rgba(255,255,255,0.7);padding:3px 0;'>✓ {prod}</div>", unsafe_allow_html=True)


        st.markdown("<hr>", unsafe_allow_html=True)
        with st.expander("About", expanded=False):
            st.caption("ImportInsight AI translates your prompt into SQL and returns the actual HTS rows.")
        with st.expander("Settings", expanded=False):
            st.caption("Natural language inputs are translated to SQL, executed locally against a read-only SQLite copy of the HTS data.")
            st.caption("Responses are deterministic: the SQL output is re-run each time against the local database.")
        st.markdown("<hr>", unsafe_allow_html=True)


    st.markdown("""
<div style='display:flex;align-items:center;justify-content:space-between;padding:10px 0 14px;border-bottom:1px solid #E5E7EB;margin-bottom:14px;'>
  <div style='display:flex;align-items:center;gap:10px;'>
    <span style='font-family:serif;font-size:17px;font-weight:700;color:#0B2A4A;letter-spacing:-0.3px;'>HTS Analysis</span>
    <span style='font-size:11px;padding:3px 10px;border-radius:100px;background:#F0FDF4;border:0.5px solid #86EFAC;color:#15803D;display:inline-flex;align-items:center;gap:6px;'>
      <span style='position:relative;width:8px;height:8px;display:inline-block;'>
        <span style='position:absolute;inset:0;border-radius:50%;background:#22C55E;opacity:0.4;animation:ping 1.5s cubic-bezier(0,0,0.2,1) infinite;'></span>
        <span style='position:absolute;inset:1px;border-radius:50%;background:#16A34A;display:inline-block;'></span>
      </span>
      Connected
    </span>
  </div>
</div>
<style>
@keyframes ping { 0% { transform: scale(1); opacity: 0.4; } 75%, 100% { transform: scale(2); opacity: 0; } }
</style>
""", unsafe_allow_html=True)
    context_bar = st.container()
    with context_bar:
        st.markdown('<div data-context="true"></div>', unsafe_allow_html=True)
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
                "Analyze",
                width="stretch",
                disabled=analyse_disabled,
            )
        with action_cols[1]:
            if st.button("Clear Inputs", width="stretch", key="context_clear"):
                reset_app_state(extra_widget_keys=["selected_products_display", "hts_drill_chapter", "hts_drill_heading", "hts_drill_sub"])
                st.rerun()
        if analyse_clicked:
            queued = queue_analysis_request(selected_countries, selected_products)
            if not queued:
                st.warning("Select at least one country and one product before running analysis.")
        if st.session_state.get("analysis_inflight"):
            st.caption("Running analysis…")
    analysis_stream_placeholder: st.delta_generator.DeltaGenerator | None = None
    chat_feed = st.container(height=480, border=False)
    composer = st.container()

    with composer:
        st.markdown('<div data-composer="true"></div>', unsafe_allow_html=True)
        st.markdown("""
<style>




</style>
""", unsafe_allow_html=True)
        # Handle chip-triggered questions
    chip_question = st.session_state.pop("chip_question", None)

    prompt = st.chat_input(QUESTION_PLACEHOLDER, key="chat_input")

    if prompt is None and chip_question:
        prompt = chip_question.strip()

    if prompt is not None:
        question = prompt.strip()
        if not question:
            st.warning("Please enter a prompt before sending.")
        else:
            timestamp = datetime.now().strftime("%I:%M %p")
            append_message({"role": "user", "content": question, "time": timestamp})
            stream_placeholder = st.empty()
            try:
                assistant_text, metadata = answer_question(
                    question, deployment_id, conn, stream_placeholder
                )
            except Exception as exc:
                logger.exception("Chat flow failed")
                st.session_state[LAST_RESULT_KEY] = None
                assistant_text = f"Error: {exc}"
                metadata = {"mode": "error"}
            stream_placeholder.empty()
            append_message(
                {
                    "role": "assistant",
                    "content": assistant_text,
                    "time": datetime.now().strftime("%I:%M %p"),
                    "metadata": metadata,
                }
            )
        st.rerun()

    with chat_feed:
        st.markdown('<div data-chat="true"></div>', unsafe_allow_html=True)
        is_loading = st.session_state.get("analysis_inflight") or st.session_state.get("analysis_request")
        if len(st.session_state.messages) == 0 and not is_loading:
            st.markdown("""
<div style='display:flex;flex-direction:column;align-items:center;justify-content:center;height:340px;gap:14px;'>
  <div style='width:56px;height:56px;border-radius:14px;border:2px solid #D1D5DB;display:flex;align-items:center;justify-content:center;'>
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M21 15C21 15.5304 20.7893 16.0391 20.4142 16.4142C20.0391 16.7893 19.5304 17 19 17H7L3 21V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H19C19.5304 3 20.0391 3.21071 20.4142 3.58579C20.7893 3.96086 21 4.46957 21 5V15Z" stroke="#9CA3AF" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </div>
  <div style='text-align:center;'>
    <p style='font-size:15px;font-weight:600;color:#374151;margin:0 0 6px;'>Select countries and HTS products, then click <span style="color:#0B2A4A;">Analyze</span> to begin.</p>
    <p style='font-size:13px;color:#9CA3AF;margin:0;'>Or type a question in the composer below.</p>
  </div>
</div>
""", unsafe_allow_html=True)
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
                selections = msg.get("selections", {})
                selection_text = []
                if selections.get("countries"):
                    selection_text.append("Countries: " + ", ".join(selections["countries"]))
                if selections.get("products"):
                    selection_text.append("Products: " + ", ".join(selections["products"]))
                if selection_text:
                    st.markdown(
                        f"<div class='analysis-meta'>{' · '.join(selection_text)}</div>",
                        unsafe_allow_html=True,
                    )
                fig_payload = msg.get("plotly_fig")
                risk_snapshot = msg.get("risk_snapshot") or []
                chart_data = msg.get("chart_data")

                # Quick summary card
                if risk_snapshot and chart_data:
                    scores = [s['score'] for s in risk_snapshot]
                    avg_score = sum(scores) / len(scores)
                    chart_df_summary = pd.DataFrame(chart_data, columns=msg.get("chart_columns"))
                    if not chart_df_summary.empty:
                        rate_column = "Effective Rate (%)" if "Effective Rate (%)" in chart_df_summary.columns else None
                        if rate_column:
                            numeric_rates = (
                                pd.to_numeric(
                                    chart_df_summary[rate_column]
                                    .astype(str)
                                    .str.replace("%", "", regex=False)
                                    .str.strip()
                                    .replace({"": None, "—": None, "nan": None}),
                                    errors="coerce",
                                )
                            )
                            chart_df_summary["_effective_rate_numeric"] = numeric_rates
                            if numeric_rates.notna().any():
                                lowest_idx = numeric_rates.idxmin()
                                lowest_duty_row = chart_df_summary.loc[lowest_idx]
                            else:
                                lowest_duty_row = None
                        else:
                            lowest_duty_row = None
                    else:
                        lowest_duty_row = None
                    if avg_score >= 50:
                        risk_label = "⚠ High avg risk"
                        risk_color = "#C0392B"
                        risk_bg = "#FADBD8"
                    elif avg_score >= 30:
                        risk_label = "◑ Moderate avg risk"
                        risk_color = "#D68910"
                        risk_bg = "#FDEBD0"
                    else:
                        risk_label = "✓ Low avg risk"
                        risk_color = "#1E8449"
                        risk_bg = "#D5F5E3"
                    country_col = "Country" if "Country" in chart_df_summary.columns else None
                    best_country = (
                        lowest_duty_row[country_col] if (lowest_duty_row is not None and country_col) else "N/A"
                    )
                    best_rate_value = (
                        lowest_duty_row["_effective_rate_numeric"]
                        if lowest_duty_row is not None and "_effective_rate_numeric" in lowest_duty_row
                        else None
                    )
                    best_rate = f"{best_rate_value:.1f}%" if isinstance(best_rate_value, (int, float)) and pd.notna(best_rate_value) else "N/A"
                    # Build duty map directly from chart_df_summary
                    duty_by_country_map = {}
                    if chart_df_summary is not None and not chart_df_summary.empty and "_effective_rate_numeric" in chart_df_summary.columns and "Country" in chart_df_summary.columns:
                        for _, dr in chart_df_summary.iterrows():
                            cname = str(dr["Country"]).strip() if pd.notna(dr["Country"]) else None
                            rval = dr["_effective_rate_numeric"]
                            if cname and pd.notna(rval):
                                try:
                                    fval = float(rval)
                                    if cname not in duty_by_country_map or fval < duty_by_country_map[cname]:
                                        duty_by_country_map[cname] = fval
                                except (TypeError, ValueError):
                                    pass
                    # Always sort: duty first, then corruption score as tiebreaker
                    import sys
                    print("DEBUG duty_by_country_map:", duty_by_country_map, file=sys.stderr)
                    best_snap_sorted = sorted(
                        risk_snapshot,
                        key=lambda s: (round(duty_by_country_map.get(s['country'], 9999), 2), round(s['score'], 2))
                    )
                    print("DEBUG sorted order:", [(s['country'], duty_by_country_map.get(s['country'], 9999), s['score']) for s in best_snap_sorted], file=sys.stderr)
                    best_risk_snap = best_snap_sorted[0] if best_snap_sorted else (risk_snapshot[0] if risk_snapshot else None)
                    true_best_country = best_risk_snap['country'] if best_risk_snap else best_country
                    true_best_rate = f"{duty_by_country_map[true_best_country]:.1f}%" if true_best_country in duty_by_country_map else best_rate
                    best_risk_score = round(best_risk_snap['score'], 1) if best_risk_snap else "N/A"
                    best_risk_level = best_risk_snap['level'] if best_risk_snap else "N/A"
                    best_country = true_best_country
                    best_rate = true_best_rate
                    st.markdown(f"""
                    <div style='border:2px solid #16A34A;border-radius:14px;padding:18px 22px;margin-bottom:14px;background:#ffffff;position:relative;overflow:hidden;box-shadow:0 4px 20px rgba(22,163,74,0.12);'>
                        <div style='position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#0B2A4A,#378ADD);'></div>
                        <div style='display:inline-flex;align-items:center;gap:6px;padding:3px 12px;background:#F0FDF4;border:1px solid #86EFAC;border-radius:4px;font-size:11px;font-weight:600;color:#15803D;margin-bottom:10px;'>
                            <span style='color:#f59e0b;'>★</span> Best from your selection
                        </div>
                        <div style='font-size:26px;font-weight:700;color:#0B2A4A;letter-spacing:-0.03em;margin-bottom:2px;'>{true_best_country}</div>
                        <div style='font-size:12px;color:#64748B;margin-bottom:14px;'>Lowest duty + lowest corruption score among your {len(risk_snapshot)} markets</div>
                        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:10px;'>
                            <div style='background:#F8FAFC;border-radius:8px;padding:10px 14px;'>
                                <div style='font-size:10px;color:#94A3B8;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.07em;'>Effective duty</div>
                                <div style='font-size:18px;font-weight:700;color:#1E8449;'>{true_best_rate}</div>
                            </div>
                            <div style='background:#F8FAFC;border-radius:8px;padding:10px 14px;'>
                                <div style='font-size:10px;color:#94A3B8;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.07em;'>Corruption score</div>
                                <div style='font-size:18px;font-weight:700;color:#1D4ED8;'>{best_risk_score} <span style='font-size:12px;color:#94A3B8;font-weight:400;'>/ 100</span></div>
                            </div>
                            <div style='background:#F8FAFC;border-radius:8px;padding:10px 14px;'>
                                <div style='font-size:10px;color:#94A3B8;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.07em;'>Risk level</div>
                                <div style='font-size:18px;font-weight:700;color:#1E8449;'>{best_risk_level}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                has_tabs = fig_payload or risk_snapshot or chart_data
                if has_tabs:
                    tab_labels = []
                    if risk_snapshot: tab_labels.append("◎ Corruption Score")
                    tab_labels.append("∿ Graph")
                    if chart_data: tab_labels.append("≡ Data")
                    tab_labels.append("◈ Analysis")
                    tabs = st.tabs(tab_labels)
                    tab_idx = 0
                    if risk_snapshot:
                        with tabs[tab_idx]:
                            tab_idx += 1
                    def render_gauge(s):
                        score = round(s['score'], 1)
                        level = s['level']
                        country = s['country']
                        needle_pct = min(score, 99)
                        score_color = '#C0392B' if level == 'High' else '#D68910' if level in ('Medium','Moderate') else '#1E8449'
                        badge_bg = '#FADBD8' if level == 'High' else '#FDEBD0' if level in ('Medium','Moderate') else '#D5F5E3'
                        badge_color = '#922B21' if level == 'High' else '#7D6608' if level in ('Medium','Moderate') else '#1E8449'
                        bar_color = score_color
                        return (
                            f"<div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:16px;padding:20px 22px;box-shadow:0 2px 8px rgba(11,42,74,0.06);'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;'>"
                            f"<span style='font-size:14px;font-weight:700;color:#0B2A4A;letter-spacing:-0.01em;'>{country}</span>"
                            f"<span style='font-size:10px;font-weight:600;padding:3px 10px;border-radius:20px;background:{badge_bg};color:{badge_color};letter-spacing:0.04em;text-transform:uppercase;'>{level}</span>"
                            f"</div>"
                            f"<div style='height:6px;background:#F1F5F9;border-radius:100px;position:relative;margin-bottom:6px;'>"
                            f"<div style='position:absolute;left:0;top:0;height:100%;width:{needle_pct}%;background:linear-gradient(90deg,#1E8449,{bar_color});border-radius:100px;transition:width 0.4s ease;'></div>"
                            f"<div style='position:absolute;top:-5px;left:{needle_pct}%;width:2px;height:16px;background:#0B2A4A;border-radius:2px;transform:translateX(-50%);box-shadow:0 0 0 2px #fff;'></div>"
                            f"</div>"
                            f"<div style='display:flex;justify-content:space-between;font-size:9px;color:#94A3B8;letter-spacing:0.03em;margin-bottom:16px;'><span>LOW</span><span>MEDIUM</span><span>HIGH</span></div>"
                            f"<div style='display:flex;align-items:baseline;gap:4px;'>"
                            f"<span style='font-size:32px;font-weight:300;color:{score_color};letter-spacing:-0.02em;'>{score}</span>"
                            f"<span style='font-size:13px;color:#CBD5E1;font-weight:400;'>/ 100</span>"
                            f"</div></div>"
                        )
                    gauges_html = ''.join(render_gauge(s) for s in risk_snapshot)
                    gauges_div = f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:8px 0;'>{gauges_html}</div>"
                    if has_tabs:
                        with tabs[0]:
                            st.markdown(gauges_div, unsafe_allow_html=True)
                    else:
                        st.markdown(gauges_div, unsafe_allow_html=True)
                if has_tabs:
                    graph_tab_idx = 1 if risk_snapshot else 0
                    with tabs[graph_tab_idx]:
                        chart_df_for_matrix = pd.DataFrame(msg.get("chart_data", []), columns=msg.get("chart_columns", []))
                        risk_snap_for_matrix = msg.get("risk_snapshot") or []
                        if not chart_df_for_matrix.empty and "Country" in chart_df_for_matrix.columns and risk_snap_for_matrix:
                            rate_col_m = "Effective Rate (%)" if "Effective Rate (%)" in chart_df_for_matrix.columns else None
                            duty_map_m = {}
                            if rate_col_m:
                                for _, mr in chart_df_for_matrix.iterrows():
                                    cn = mr.get("Country")
                                    rv = mr.get(rate_col_m)
                                    if cn and rv is not None:
                                        try:
                                            fv = float(str(rv).replace("%","").strip())
                                            if cn not in duty_map_m or fv < duty_map_m[cn]:
                                                duty_map_m[cn] = fv
                                        except (ValueError, TypeError):
                                            pass
                            import plotly.graph_objects as go_m
                            matrix_fig = go_m.Figure()
                            colors_m = ["#1a5ccc","#0f6e56","#e74c3c","#8b5cf6","#f59e0b"]
                            max_duty_m = max(duty_map_m.values()) if duty_map_m else 20
                            max_risk_m = max(s["score"] for s in risk_snap_for_matrix)
                            mid_duty = max_duty_m / 2
                            mid_risk = max_risk_m / 2
                            def _quadrant_color(risk_score, duty_val, mid_r, mid_d):
                                low_risk = risk_score <= mid_r
                                low_duty = duty_val <= mid_d
                                if low_risk and low_duty: return "#1a5ccc"
                                if not low_risk and low_duty: return "#f59e0b"
                                if low_risk and not low_duty: return "#0f6e56"
                                return "#e74c3c"
                            for i, snap in enumerate(risk_snap_for_matrix):
                                c = snap["country"]
                                duty_v = duty_map_m.get(c)
                                if duty_v is None:
                                    continue
                                bubble_color = _quadrant_color(snap["score"], duty_v, mid_risk, mid_duty)
                                matrix_fig.add_trace(go_m.Scatter(
                                    x=[snap["score"]], y=[duty_v],
                                    mode="markers+text",
                                    name=c,
                                    text=[c],
                                    textposition="top center",
                                    textfont=dict(size=12, color=bubble_color),
                                    marker=dict(size=40, color=bubble_color, line=dict(width=2, color="white")),
                                    hovertemplate=f"<b>{c}</b><br>Risk score: {snap['score']:.1f}/100<br>Duty: {duty_v:.1f}%<extra></extra>"
                                ))
                                matrix_fig.add_annotation(x=snap["score"], y=duty_v,
                                    text=f"{duty_v:.1f}%", showarrow=False,
                                    font=dict(size=10, color="white", family="monospace"), yshift=0)
                            pad_x = max_risk_m * 0.15
                            pad_y = max_duty_m * 0.25
                            matrix_fig.update_layout(
                                template="plotly_white",
                                margin=dict(l=20, r=20, t=10, b=60),
                                height=280,
                                plot_bgcolor="white",
                                showlegend=False,
                                shapes=[
                                    dict(type="line", x0=mid_risk, x1=mid_risk, y0=0, y1=max_duty_m+pad_y,
                                         line=dict(color="rgba(0,0,0,0.15)", width=1, dash="dash")),
                                    dict(type="line", x0=0, x1=max_risk_m+pad_x, y0=mid_duty, y1=mid_duty,
                                         line=dict(color="rgba(0,0,0,0.15)", width=1, dash="dash")),
                                ],
                                xaxis=dict(title="Corruption score (lower = safer)", gridcolor="#F0F2F5", zeroline=False,
                                           range=[-2, max_risk_m+pad_x], tickfont=dict(size=12)),
                                yaxis=dict(title="Effective duty rate (%)", gridcolor="#F0F2F5", zeroline=False,
                                           ticksuffix="%", range=[-0.5, max_duty_m+pad_y], tickfont=dict(size=12)),
                            )
                            st.plotly_chart(matrix_fig, use_container_width=True)
                            st.markdown("""
                            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:4px;font-size:11px;">
                              <div style="padding:6px 10px;background:#e8f4ff;border-radius:6px;color:#1a4a8a;border:0.5px solid #b5d4f4;display:flex;align-items:center;gap:6px;"><span style="width:10px;height:10px;border-radius:50%;background:#1a5ccc;flex-shrink:0"></span>Bottom-left — low risk, low duty &rarr; ideal sourcing</div>
                              <div style="padding:6px 10px;background:#fff8e8;border-radius:6px;color:#7a5a0a;border:0.5px solid #f0d090;display:flex;align-items:center;gap:6px;"><span style="width:10px;height:10px;border-radius:50%;background:#f59e0b;flex-shrink:0"></span>Bottom-right — low duty, higher risk &rarr; cost efficient but exposed</div>
                              <div style="padding:6px 10px;background:#f0f8f0;border-radius:6px;color:#1a5a2a;border:0.5px solid #90d0a0;display:flex;align-items:center;gap:6px;"><span style="width:10px;height:10px;border-radius:50%;background:#0f6e56;flex-shrink:0"></span>Top-left — safe market, premium duty &rarr; regulatory safe haven</div>
                              <div style="padding:6px 10px;background:#fff0f0;border-radius:6px;color:#8a1a1a;border:0.5px solid #f0b0b0;display:flex;align-items:center;gap:6px;"><span style="width:10px;height:10px;border-radius:50%;background:#e74c3c;flex-shrink:0"></span>Top-right — high risk, high duty &rarr; avoid</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            if fig_payload:
                                fig = go.Figure(fig_payload)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No percentage-based duty rates available to plot for this product.")
                    data_tab_idx = graph_tab_idx + 1
                    chart_data = msg.get("chart_data")
                    if chart_data:
                        with tabs[data_tab_idx]:
                            chart_df = pd.DataFrame(chart_data, columns=msg.get("chart_columns"))
                            st.dataframe(chart_df)
                    analysis_tab_idx = data_tab_idx + (1 if chart_data else 0)
                    with tabs[analysis_tab_idx]:
                        st.markdown(
                            f"<div class='bubble-bot'>{msg['content']}</div><div class='timestamp'>{timestamp}</div><div class='clearfix'></div>",
                            unsafe_allow_html=True,
                        )
                        duty_exclusions = msg.get("duty_exclusions") or []
                        exclusion_text = msg.get("duty_exclusion_message")
                        if exclusion_text:
                            st.warning(exclusion_text)
                        elif duty_exclusions:
                            listed = ", ".join(
                                f"{item.get('hts_code')} ({item.get('general_duty_rate_text')})"
                                for item in duty_exclusions[:3]
                            )
                            if len(duty_exclusions) > 3:
                                listed += f", +{len(duty_exclusions) - 3} more"
                            st.warning(f"Skipped non-percentage duty rates: {listed}")
                        # Follow-up prompt chips
                        selections = msg.get("selections", {})
                        countries = selections.get("countries", [])
                        products = selections.get("products", [])
                        chip_country = countries[0] if countries else "this country"
                        chip_product = products[0] if products else "this product"
                        suggestions = [
                            f"Which country has the lowest duty for {chip_product}?",
                            f"What is the corruption score for {chip_country}?",
                            f"Why do duty rates differ across countries?",
                            f"What trade programs reduce tariffs for {chip_country}?",
                        ]
                        st.markdown("<div style='margin-top:14px;display:flex;flex-wrap:wrap;gap:8px;'>", unsafe_allow_html=True)
                        for suggestion in suggestions:
                            if st.button(suggestion, key=f"chip_{hash(suggestion)}_{timestamp}"):
                                st.session_state["chip_question"] = suggestion
                                st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    chart_data = msg.get("chart_data")
                    if chart_data:
                        chart_df = pd.DataFrame(chart_data, columns=msg.get("chart_columns"))
                        st.dataframe(chart_df)
                    st.markdown(
                        f"<div class='bubble-bot'>{msg['content']}</div><div class='timestamp'>{timestamp}</div><div class='clearfix'></div>",
                        unsafe_allow_html=True,
                    )
                    duty_exclusions = msg.get("duty_exclusions") or []
                    exclusion_text = msg.get("duty_exclusion_message")
                    if exclusion_text:
                        st.warning(exclusion_text)
                    elif duty_exclusions:
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
        maybe_run_best_countries(
            conn,
            deployment_id,
            risk_df,
        )
        maybe_run_analysis(
            conn,
            deployment_id,
            risk_df,
            analysis_stream_placeholder,
        )
        latest_result = st.session_state.get(LAST_RESULT_KEY)
        if latest_result:
            st.markdown("---")
            st.markdown("### Last SQL Result")
            st.code(latest_result["sql"], language="sql")
            st.caption(
                f"Returned {latest_result['row_count']} row(s); showing {latest_result['rows_displayed']} row(s) below."
            )
            if latest_result["records"]:
                st.dataframe(
                    pd.DataFrame(latest_result["records"], columns=latest_result["columns"])
                )
            else:
                st.info("The last query returned no rows.")
        st.markdown('<div data-anchor="chat-end" id="chat-end"></div>', unsafe_allow_html=True)

    # analysis runs inside chat_feed above

    components.html("""<script>
    (function() {
        function fix() {
            var doc = window.parent.document;
            var inputs = doc.querySelectorAll(');
            var tas = doc.querySelectorAll(');
            var btns = doc.querySelectorAll(');
            var bottom = doc.querySelectorAll(');
        }
        fix();
        setInterval(fix, 500);
    })();
    </script>""", height=0)

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
