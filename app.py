"""Streamlit text-to-SQL chatbot backed by a local HTS SQLite database."""
import logging
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import openai
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

COUNTRY_RISK = {
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


def get_risk_color(score: float) -> tuple[str, str]:
    if score >= 75:
        return "#c0392b", "High"
    if score >= 45:
        return "#d35400", "Medium"
    return "#1e8449", "Low"

QUESTION_PLACEHOLDER = (
    "Describe what you need from the Harmonized Tariff Schedule."
)

SQL_SYSTEM_PROMPT = (
    "You are a SQL generator for a SQLite database containing a single table named `hts`. "
    "The available text columns in `hts` are: "
    + ", ".join(HTS_COLUMNS)
    + ". Always respond with exactly one valid SQLite SELECT statement. "
    "Do not include surrounding markdown, explanations, or additional text. "
    "The SQL will be executed as-is against the HTS database, so refer only to the columns listed above and avoid modifications (INSERT/UPDATE/DELETE/PRAGMA)."
)

SELECT_PATTERN = re.compile(r"SELECT\b.*", re.IGNORECASE | re.DOTALL)

LAST_RESULT_KEY = "latest_hts_result"


def _format_css() -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] {
        background-color: #0f1f38;
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] * { color: #e8edf5 !important; }
    [data-testid="stSidebar"] h3 {
        color: #a0aec0 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
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
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        font-size: 14px;
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
    </style>
    """


@st.cache_resource
def get_db_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


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


def get_top_level_codes(conn: sqlite3.Connection) -> list[str]:
    try:
        df = pd.read_sql_query(
            'SELECT hts_code, description FROM hts WHERE indent_level = "0" ORDER BY hts_code',
            conn,
        )
    except Exception as exc:
        logger.warning("Failed to load top-level codes: %s", exc)
        return []
    return [f"{row['hts_code']} - {row['description']}" for _, row in df.iterrows()]


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

    sidebar_sources = []
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
        st.markdown("### Governance Risk")
        selected_country = st.selectbox("Select country", list(COUNTRY_RISK.keys()))
        data = COUNTRY_RISK[selected_country]
        score = data["score"]
        year = data["year"]
        color, level = get_risk_color(score)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 28, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#a0aec0", "tickfont": {"size": 9, "color": "#a0aec0"}},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 45], "color": "#1e8449"},
                    {"range": [45, 75], "color": "#d35400"},
                    {"range": [75, 100], "color": "#c0392b"},
                ],
            },
            title={"text": f"<b>{level} Risk</b><br><span style='font-size:11px;color:#a0aec0'>{selected_country} · {year}</span>", "font": {"size": 13, "color": "white"}},
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=60, b=10),
            height=200,
            font={"color": "white"},
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("Clear Chat", width="stretch"):
            st.session_state.messages = []
            st.session_state[LAST_RESULT_KEY] = None
            st.rerun()
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### About")
        st.caption("ImportInsight AI translates your prompt into SQL and returns the actual HTS rows.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if LAST_RESULT_KEY not in st.session_state:
        st.session_state[LAST_RESULT_KEY] = None

    if len(st.session_state.messages) == 0:
        starters = [
            "What is the general duty rate for HTS 0101?",
            "List entries that mention semiconductors in chapter 85.",
            "Show HTS numbers with an additional duty of 30%.",
            "Which chapters cover textiles?",
        ]
        st.markdown(
            "<p style='color:#cbd5e1; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;'>Suggested questions</p>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, q in enumerate(starters):
            with cols[i % 2]:
                if st.button(q, key=f"starter_{i}", width="stretch"):
                    st.session_state.messages.append(
                        {"role": "user", "content": q, "time": datetime.now().strftime("%I:%M %p")}
                    )
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        timestamp = msg.get("time", "")
        if msg["role"] == "user":
            st.markdown(
                f"<div class='bubble-label bubble-label-right'>You</div><div class='bubble-user'>{msg['content']}</div><div class='timestamp timestamp-right'>{timestamp}</div><div class='clearfix'></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='bubble-label'>Assistant</div><div class='bubble-bot'>{msg['content']}</div><div class='timestamp'>{timestamp}</div><div class='clearfix'></div>",
                unsafe_allow_html=True,
            )

    try:
        code_options = get_top_level_codes(conn)
    except Exception:
        code_options = []

    selected_hts = st.selectbox("Select HTS Code", [""] + code_options)
    if selected_hts:
        question = selected_hts.split(" - ", 1)[0]
    else:
        question = st.text_area(QUESTION_PLACEHOLDER, height=140, key="prompt_input")

    if st.button("Send", width="stretch"):
        if not question or not question.strip():
            st.warning("Please enter a question before sending.")
        else:
            st.session_state.messages.append(
                {"role": "user", "content": question, "time": datetime.now().strftime("%I:%M %p")}
            )
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
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_text, "time": datetime.now().strftime("%I:%M %p")}
            )
            st.session_state.prompt_input = ""
            st.rerun()

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


if __name__ == "__main__":
    main()
