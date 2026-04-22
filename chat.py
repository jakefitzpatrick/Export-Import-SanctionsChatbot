"""Chatbot utilities for ImportInsight AI.

Provides both context-first answers (summaries of Analyse results) and
text-to-SQL fallback for ad hoc HTS queries.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time

import openai
import pandas as pd
import streamlit as st

from logger import setup_logger
from risk_model import RISK_METHOD_SUMMARY
from special_rates import get_program_description
from utils import LAST_RESULT_KEY

QUESTION_PLACEHOLDER = "Ask about the HTS data"

QA_SYSTEM_PROMPT = (
    "You are a trade intelligence assistant with the mindset of a pragmatic "
    "trade manager. You help users understand tariff exposure and country risk "
    "for specific HTS codes.\n\n"
    "Your goals:\n"
    "- Give medium-length answers: a short paragraph plus 1–3 concise, "
    "actionable suggestions where appropriate.\n"
    "- Explain the tariff mechanics clearly: what the base HTS rate is, whether "
    "a Chapter 99 surcharge applies and under which trade program (e.g. "
    "Section 122, Section 232, USMCA blocker), and how those combine into the "
    "effective rate. When rates differ across countries, explain why.\n"
    "- Describe patterns and relationships (e.g., higher-risk countries with "
    "higher effective rates), but be cautious and avoid sweeping claims.\n"
    "- Keep recommendations practical and specific to the countries and HTS "
    "codes in view.\n\n"
    "Strict constraints:\n"
    "- Speak as a knowledgeable trade advisor. Never reference 'the data', "
    "'the dataset', 'sample rows', 'sample size', 'supplied data', 'the "
    "provided rows', or any other language that implies the user can see or "
    "access backend data.\n"
    "- Do NOT invent, approximate, or assume new numbers or countries beyond "
    "what you have been given.\n"
    "- Treat all rates, corruption scores, and trade program labels you are given as "
    "correct. Do not question, second-guess, or suggest that the figures may be "
    "inaccurate — accept them at face value and reason from them.\n"
    "- Do NOT generate or suggest SQL queries, and do NOT imply that you are "
    "running live database queries.\n"
    "- This is not legal advice; if an answer could be interpreted that way, "
    "remind the user to consult their customs or trade counsel.\n"
    "- If the context is too sparse to answer reliably, say so and suggest "
    "the user run the Analyse step with specific countries or HTS codes."
)

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
    "additional_duties",
]
CH99_COLUMNS = [
    "NEWCODE", "HTS_MASTER_CODE", "COUNTRY", "NEWRATE", "NEWRATE_CLEAN",
    "RATE_MODIFIER", "ADDITIONAL_DUTY_PCT", "ADDITIONAL_VALUE", "ADDITIONAL_UNIT",
    "TRADEPROGRAM", "MATCH_PRIORITY",
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
PROGRAM_CMD_PATTERN = re.compile(r"^/program\s+([A-Za-z][A-Za-z0-9\+\*]*)$", re.IGNORECASE)

logger = setup_logger(__name__)


def append_message(message: dict) -> None:
    st.session_state.messages.append(message)
    st.session_state["chat_scroll_token"] = st.session_state.get("chat_scroll_token", 0) + 1


def upsert_analysis_message(message: dict) -> None:
    """Replace the existing analysis message in chat history; append if absent."""
    msgs = st.session_state.messages
    for index, existing in enumerate(msgs):
        if existing.get("type") == "analysis":
            msgs[index] = message
            return
    msgs.append(message)


def _build_tariff_breakdown(chart_columns: list, chart_data: list) -> list[dict]:
    """Convert chart rows into structured per-country tariff breakdowns.

    chart_data is stored as to_dict("records") — rows are dicts keyed by column name.
    """
    if not chart_data:
        return []
    breakdowns = []
    for row in chart_data:
        if not isinstance(row, dict):
            continue
        entry = {
            key: row.get(field)
            for field, key in [
                ("Country", "country"),
                ("HTS Code", "hts_code"),
                ("Product", "product_description"),
                ("Base Rate", "base_rate"),
                ("Effective Rate (%)", "effective_rate_pct"),
                ("Ch.99 Δ", "ch99_surcharge"),
                ("Trade Program", "trade_program"),
                ("Rate Source", "rate_source"),
                ("Corruption Score", "corruption_score"),
            ]
        }
        if any(v is not None for v in entry.values()):
            breakdowns.append(entry)
    return breakdowns


def _build_chat_context() -> dict:
    """Summarise the latest analysis result and recent chat for the LLM."""
    messages = st.session_state.get("messages", [])

    latest_analysis = None
    for msg in reversed(messages):
        if msg.get("type") == "analysis":
            chart_columns = msg.get("chart_columns") or []
            chart_data = msg.get("chart_data") or []
            tariff_breakdown = _build_tariff_breakdown(chart_columns, chart_data)
            latest_analysis = {
                "summary": msg.get("content"),
                "selections": msg.get("selections"),
                "risk_snapshot": msg.get("risk_snapshot"),
                "tariff_breakdown": tariff_breakdown,
                "ch99_summary": msg.get("ch99_summary"),
                "non_ad_summary_text": msg.get("non_ad_summary_text"),
            }
            break

    recent_chat = [
        {"role": m.get("role"), "content": m.get("content")}
        for m in messages
        if m.get("type") != "analysis" and m.get("role") in ("user", "assistant")
    ][-6:]

    return {
        "latest_analysis": latest_analysis,
        "recent_chat": recent_chat,
        "last_sql_result": st.session_state.get(LAST_RESULT_KEY),
    }


def _format_analysis_bundle(analysis: dict | None) -> str | None:
    if not analysis:
        return None
    bundle = {
        "summary": analysis.get("summary"),
        "selections": analysis.get("selections"),
        "risk_snapshot": analysis.get("risk_snapshot"),
        "chapter99_summary": analysis.get("ch99_summary"),
        "non_ad_text": analysis.get("non_ad_summary_text"),
        "tariff_breakdown": analysis.get("tariff_breakdown"),
    }
    return json.dumps(bundle, default=str, indent=2)


def _maybe_handle_program_command(question: str) -> tuple[str, dict] | None:
    match = PROGRAM_CMD_PATTERN.match(question.strip())
    if not match:
        return None
    code = match.group(1).upper()
    description = get_program_description(code)
    if description:
        response = f"{code}: {description}"
    else:
        response = f"I don't have a description for program code '{code}'."
    metadata = {"mode": "program", "program_code": code}
    return response, metadata


def build_sql_messages(question: str, context: dict) -> list[dict]:
    messages = [{"role": "system", "content": SQL_SYSTEM_PROMPT}]
    latest = context.get("latest_analysis")
    if latest:
        summary = latest.get("summary") or ""
        selections = latest.get("selections") or {}
        context_note = f"Recent analysis summary:\n{summary}\nSelections: {json.dumps(selections)}"
        messages.append({"role": "system", "content": context_note})
    for entry in context.get("recent_chat", [])[-4:]:
        if entry.get("content"):
            messages.append({"role": entry.get("role", "user"), "content": entry["content"]})
    messages.append(
        {
            "role": "user",
            "content": f"Question: {question}\nReturn only one SELECT statement that answers the question.",
        }
    )
    return messages


def translate_question_to_sql(question: str, deployment_id: str, context: dict) -> str:
    messages = build_sql_messages(question, context)
    logger.info(
        "Translating question to SQL",
        extra={"question": question, "has_analysis_context": context.get("latest_analysis") is not None},
    )
    start = time.perf_counter()
    response = openai.chat.completions.create(
        model=deployment_id,
        messages=messages,
    )
    raw_content = response.choices[0].message.content or ""
    match = SELECT_PATTERN.search(raw_content.replace("`", "").strip())
    if not match:
        logger.error("Model response did not contain a SELECT statement", extra={"question": question})
        raise ValueError("The model did not return a valid SELECT statement.")
    sql = match.group(0).split(";")[0].strip()
    if not sql.lower().startswith("select"):
        logger.error("Model response started with a non-SELECT token", extra={"question": question, "content": raw_content})
        raise ValueError("Only SELECT statements are allowed.")
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "SQL translation complete",
        extra={"sql": sql, "duration_ms": duration_ms},
    )
    return sql


def execute_sql(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    start = time.perf_counter()
    df = pd.read_sql_query(sql, conn)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "Executed SQL against HTS database",
        extra={"sql": sql, "rows": len(df), "duration_ms": duration_ms},
    )
    return df


def decide_chat_mode(question: str, context: dict) -> tuple[str, str]:
    normalized = question.strip()
    lowered = normalized.lower()
    explicit_sql = False
    if lowered.startswith("/sql"):
        explicit_sql = True
        normalized = normalized[4:].strip()
    keywords = ("run sql", "query", "list hts", "show hts", "select", "fetch rows")
    if any(keyword in lowered for keyword in keywords):
        explicit_sql = True
    has_analysis = context.get("latest_analysis") is not None
    if explicit_sql or not has_analysis:
        return "sql", normalized
    return "context", normalized


def run_sql_chat_flow(
    question: str,
    deployment_id: str,
    conn: sqlite3.Connection,
    context: dict,
) -> tuple[str, dict]:
    sql = translate_question_to_sql(question, deployment_id, context)
    logger.info("Running SQL chat flow", extra={"sql": sql})
    df = execute_sql(conn, sql)
    row_count = len(df)
    preview_cols = ", ".join(df.columns[: min(5, len(df.columns))])
    if row_count:
        assistant_text = (
            f"Executed SQL and returned {row_count} row(s). "
            f"Preview columns: {preview_cols or 'n/a'}."
        )
    else:
        assistant_text = "Executed SQL, but the query returned no rows."

    records = df.head(200).to_dict("records")
    st.session_state[LAST_RESULT_KEY] = {
        "sql": sql,
        "row_count": row_count,
        "records": records,
        "columns": list(df.columns),
        "rows_displayed": len(records),
    }
    metadata = {"mode": "sql", "sql": sql, "row_count": row_count}
    return assistant_text, metadata


def answer_question(
    question: str,
    deployment_id: str,
    conn: sqlite3.Connection,
    stream_placeholder=None,
) -> tuple[str, dict]:
    special = _maybe_handle_program_command(question)
    if special:
        return special

    context = _build_chat_context()
    mode, normalized_question = decide_chat_mode(question, context)
    logger.info(
        "Dispatching question to chat engine",
        extra={"question": question, "mode": mode},
    )
    if mode == "context":
        return answer_question_with_context(
            normalized_question, deployment_id, context, stream_placeholder
        )
    return run_sql_chat_flow(normalized_question, deployment_id, conn, context)


def answer_question_with_context(
    question: str,
    deployment_id: str,
    context: dict,
    stream_placeholder=None,
) -> tuple[str, dict]:
    """Use the LLM to answer a question based on existing results only.

    If stream_placeholder is provided (a Streamlit empty()), tokens are written
    incrementally so the UI updates as the response arrives.
    """
    analysis = context.get("latest_analysis") or {}
    selections = analysis.get("selections") or {}
    summary = analysis.get("summary") or ""

    sections = []
    if selections:
        sections.append(f"SELECTIONS\n{json.dumps(selections, default=str)}")
    structured_bundle = _format_analysis_bundle(analysis)
    if structured_bundle:
        sections.append(f"LATEST ANALYSIS BUNDLE\n{structured_bundle}")
    elif summary:
        sections.append(f"ANALYSIS NARRATIVE\n{summary}")
    last_sql = context.get("last_sql_result")
    if last_sql:
        sql_excerpt = {
            "sql": last_sql.get("sql"),
            "row_count": last_sql.get("row_count"),
            "preview_rows": last_sql.get("records", [])[:3],
        }
        sections.append(f"LAST SQL RESULT\n{json.dumps(sql_excerpt, default=str)}")
    sections.append(f"RISK SCORE METHODOLOGY\n{RISK_METHOD_SUMMARY}")
    sections.append(f"RECENT CHAT\n{json.dumps(context.get('recent_chat', []), default=str)}")
    sections.append(f"USER QUESTION\n{question}")

    user_message = "\n\n---\n\n".join(sections)

    logger.info(
        "Answering question from cached context",
        extra={"question": question, "has_analysis": context.get("latest_analysis") is not None},
    )
    start = time.perf_counter()
    stream = openai.chat.completions.create(
        model=deployment_id,
        messages=[
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        stream=True,
    )
    full_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if not delta:
            continue
        full_text += delta
        if stream_placeholder is not None:
            stream_placeholder.markdown(
                f"<div class='bubble-label'>Assistant</div>"
                f"<div class='bubble-bot'>{full_text}▌</div>",
                unsafe_allow_html=True,
            )
    if stream_placeholder is not None:
        stream_placeholder.empty()

    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "Contextual answer generated",
        extra={"duration_ms": duration_ms, "chars": len(full_text)},
    )
    return full_text.strip(), {"mode": "context"}
