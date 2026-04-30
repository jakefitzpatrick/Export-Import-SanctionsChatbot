"""
Text-to-SQL evaluation script for ImportInsight AI.

Runs a fixed set of natural-language questions through the Azure OpenAI
SQL generation pipeline, executes each query against data/hts.db, and
checks whether the result satisfies a correctness predicate.

Usage:
    python3 scripts/eval_text_to_sql.py

Requires AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and
AZURE_OPENAI_DEPLOYMENT_ID to be set in the environment.
"""
from __future__ import annotations

import os
import sys
import time
import re
import sqlite3

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Azure OpenAI client setup ────────────────────────────────────────────────
endpoint   = os.environ["AZURE_OPENAI_ENDPOINT"]
api_key    = os.environ["AZURE_OPENAI_API_KEY"]
api_ver    = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
deploy_id  = os.environ["AZURE_OPENAI_DEPLOYMENT_ID"]

openai.api_type    = "azure"
openai.api_base    = endpoint
openai.api_version = api_ver
openai.api_key     = api_key

client = openai.AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_ver,
)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hts.db")

# ── SQL generation (mirrors chat.py logic) ───────────────────────────────────
SELECT_PATTERN = re.compile(r"SELECT\b.*", re.IGNORECASE | re.DOTALL)

SQL_SYSTEM_PROMPT = (
    "You are a SQL generator for a SQLite database with two tables and one view. "
    "Table `hts` has columns: hts_code, chapter, heading, subheading, statistical_suffix, "
    "indent_level, description, full_description, unit, general_duty_rate, special_duty_rate. "
    "View `hts_with_ch99` pre-joins hts with chapter_99 using wildcard and priority rules. "
    "IMPORTANT — the `hts` table does NOT have columns named column2_duty_rate, additional_duties, "
    "col2_duty_rate, or any variant. Do not reference those names in any query. "
    "The `chapter` column stores values as unpadded strings ('1', '2', ... '97'), never zero-padded. "
    "Always use the unpadded form when filtering by chapter (e.g., WHERE chapter = '1', NOT '01'). "
    "Always treat HTS codes as TEXT strings. "
    "Always respond with exactly one valid SQLite SELECT statement. "
    "Do not include surrounding markdown, explanations, or additional text. "
    "Only use SELECT statements."
)


def translate_to_sql(question: str) -> str:
    messages = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\nReturn only one SELECT statement."},
    ]
    resp = client.chat.completions.create(model=deploy_id, messages=messages)
    raw = resp.choices[0].message.content or ""
    match = SELECT_PATTERN.search(raw.replace("`", "").strip())
    if not match:
        raise ValueError(f"No SELECT in response: {raw!r}")
    return match.group(0).split(";")[0].strip()


def run_sql(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


# ── Test cases ────────────────────────────────────────────────────────────────
# Each entry: (question, check_fn, description_of_what_pass_means)
def _contains(df: pd.DataFrame, value: str) -> bool:
    """True if `value` appears in any cell of the dataframe (case-insensitive)."""
    return df.apply(lambda col: col.astype(str).str.contains(value, case=False, na=False)).any().any()


def _row_count_gte(n: int):
    return lambda df: len(df) >= n


TEST_CASES: list[tuple[str, object, str]] = [
    (
        "What is the general duty rate for HTS code 0406.40.44.00?",
        lambda df: _contains(df, "12.80"),
        "result contains '12.80%'",
    ),
    (
        "What is the general duty rate for HTS code 0209.10.00.00?",
        lambda df: _contains(df, "3.20"),
        "result contains '3.20%'",
    ),
    (
        "What is the description for HTS code 0104.20.00.00?",
        lambda df: _contains(df, "goat"),
        "result contains 'goat'",
    ),
    (
        "What is the general duty rate for HTS code 8471.30.01.00?",
        lambda df: len(df) > 0,
        "query returns at least one row",
    ),
    (
        "What is the general duty rate for 2204.21.20.00?",
        lambda df: _contains(df, "19.8"),
        "result contains '19.8'",
    ),
    (
        "What is the general duty rate for 8401.10.00.00?",
        lambda df: _contains(df, "3.30"),
        "result contains '3.30%'",
    ),
    (
        "List HTS codes and descriptions for products related to cheese.",
        lambda df: len(df) >= 3 and df.apply(
            lambda col: col.astype(str).str.contains("0406|cheese", case=False, na=False)
        ).any().any(),
        "returns ≥3 rows with cheese/0406 references",
    ),
    (
        "How many HTS codes are in the database?",
        lambda df: any(
            abs(int(str(v).replace(",", "")) - 23065) < 500
            for cell in df.values.flatten()
            for v in [str(cell).replace(",", "")]
            if v.isdigit()
        ),
        "count is within 500 of 23065",
    ),
    (
        "What HTS codes are in chapter 84?",
        _row_count_gte(50),
        "returns ≥50 rows",
    ),
    (
        "What is the special duty rate for 0405.90.20?",
        lambda df: len(df) > 0,
        "query returns at least one row",
    ),
    (
        "List HTS codes with a specific (non-percentage) duty rate.",
        _row_count_gte(5),
        "returns ≥5 rows",
    ),
    (
        "What is the column 2 duty rate for HTS code 0406.40.44.00?",
        lambda df: len(df) > 0,
        "query returns at least one row",
    ),
    (
        "Show me all HTS codes in chapter 01.",
        lambda df: len(df) > 0 and df.apply(
            lambda col: col.astype(str).str.startswith("01")
        ).any().any(),
        "returns rows with HTS codes starting with '01'",
    ),
    (
        "What is the special duty rate for HTS code 0406.40.44.00?",
        lambda df: _contains(df, "CO") or _contains(df, "KR") or _contains(df, "free"),
        "result contains known free-duty program codes (CO, KR) or 'free'",
    ),
    (
        "Which HTS codes in chapter 22 have a duty rate involving cents per liter?",
        _row_count_gte(1),
        "returns at least 1 row",
    ),
]


# ── Runner ────────────────────────────────────────────────────────────────────
def run_eval():
    conn = sqlite3.connect(DB_PATH)
    results = []

    print(f"\n{'='*70}")
    print(f"  ImportInsight AI — Text-to-SQL Evaluation ({len(TEST_CASES)} test cases)")
    print(f"  Model: {deploy_id}  |  DB: {DB_PATH}")
    print(f"{'='*70}\n")

    for i, (question, check_fn, expectation) in enumerate(TEST_CASES, 1):
        print(f"[{i:02d}/{len(TEST_CASES)}] {question}")
        sql = None
        try:
            t0 = time.perf_counter()
            sql = translate_to_sql(question)
            df  = run_sql(conn, sql)
            elapsed = round((time.perf_counter() - t0) * 1000)
            passed  = check_fn(df)
            status  = "PASS" if passed else "FAIL"
            print(f"       SQL: {sql[:120]}{'...' if len(sql)>120 else ''}")
            print(f"       Rows returned: {len(df)}  |  {elapsed}ms  |  [{status}] expect: {expectation}")
        except Exception as exc:
            passed  = False
            status  = "ERROR"
            print(f"       SQL: {sql or '(not generated)'}")
            print(f"       [{status}] {exc}")
        results.append(passed)
        print()

    conn.close()

    passed_count = sum(results)
    total        = len(results)
    accuracy     = passed_count / total * 100

    print(f"{'='*70}")
    print(f"  Results: {passed_count}/{total} passed")
    print(f"  Execution Accuracy: {accuracy:.1f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_eval()
