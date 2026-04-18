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
from utils import LAST_RESULT_KEY

QUESTION_PLACEHOLDER = "Ask about the HTS data"

QA_SYSTEM_PROMPT = (
    "You are a trade intelligence assistant with the mindset of a pragmatic "
    "trade manager. You help users understand tariff exposure and country risk "
    "for specific HTS codes.\n\n"
    "Your goals:\n"
    "- Give medium-length answers: a short paragraph plus 1–3 concise, "
    "actionable suggestions where appropriate.\n"
    "- Describe patterns and relationships (e.g., higher-risk countries with "
    "higher ad valorem rates), but be cautious and avoid sweeping claims.\n"
    "- Keep recommendations practical and specific to the countries and HTS "
    "codes in view.\n\n"
    "Strict constraints:\n"
    "- Speak as a knowledgeable trade advisor. Never reference 'the data', "
    "'the dataset', 'sample rows', 'sample size', 'supplied data', 'the "
    "provided rows', or any other language that implies the user can see or "
    "access backend data.\n"
    "- Do NOT invent, approximate, or assume new numbers or countries beyond "
    "what you have been given.\n"
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


def _build_chat_context() -> dict:
    """Summarise the latest analysis result and recent chat for the LLM."""
    messages = st.session_state.get("messages", [])

    latest_analysis = None
    for msg in reversed(messages):
        if msg.get("type") == "analysis":
            latest_analysis = {
                "summary": msg.get("content"),
                "selections": msg.get("selections"),
                "risk_snapshot": msg.get("risk_snapshot"),
                "chart_columns": msg.get("chart_columns"),
                "chart_data_sample": (msg.get("chart_data") or [])[:30],
                "non_ad_summary": msg.get("non_ad_summary"),
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
    }


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


def answer_question(question: str, deployment_id: str, conn: sqlite3.Connection) -> tuple[str, dict]:
    context = _build_chat_context()
    mode, normalized_question = decide_chat_mode(question, context)
    logger.info(
        "Dispatching question to chat engine",
        extra={"question": question, "mode": mode},
    )
    if mode == "context":
        return answer_question_with_context(normalized_question, deployment_id, context)
    return run_sql_chat_flow(normalized_question, deployment_id, conn, context)


def answer_question_with_context(question: str, deployment_id: str, context: dict) -> tuple[str, dict]:
    """Use the LLM to answer a question based on existing results only."""
    user_payload = {
        "context": context,
        "question": question,
    }
    user_message = (
        "You are given JSON containing the current analytic context and the user's question.\n"
        "Use only this data and the chat history to answer. Do not guess new numbers.\n\n"
        f"{json.dumps(user_payload, default=str)[:6000]}"
        "\n\nRisk score methodology reference:\n"
        f"{RISK_METHOD_SUMMARY}"
    )

    logger.info(
        "Answering question from cached context",
        extra={"question": question, "has_analysis": context.get("latest_analysis") is not None},
    )
    start = time.perf_counter()
    response = openai.chat.completions.create(
        model=deployment_id,
        messages=[
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    content = response.choices[0].message.content or ""
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "Contextual answer generated",
        extra={"duration_ms": duration_ms, "chars": len(content)},
    )
    return content.strip(), {"mode": "context"}
