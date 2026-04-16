#!/usr/bin/env python3
"""Inspect HTS duty rate patterns for documentation/debugging."""
from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "hts.db"


def _fetch_all(conn: sqlite3.Connection, query: str) -> list[str]:
    cur = conn.execute(query)
    return [row[0] for row in cur.fetchall() if row[0]]


def summarize_rates(conn: sqlite3.Connection) -> None:
    total_rows = conn.execute("SELECT COUNT(*) FROM hts").fetchone()[0]
    general_rates = _fetch_all(
        conn, "SELECT general_duty_rate FROM hts WHERE general_duty_rate IS NOT NULL AND general_duty_rate <> ''"
    )
    counter = Counter()
    samples: dict[str, list[str]] = {
        "percent": [],
        "currency": [],
        "per_unit": [],
        "mixed": [],
    }
    for rate in general_rates:
        text = str(rate)
        lowered = text.lower()
        has_percent = "%" in text
        has_currency = "$" in text or "¢" in text or "cents" in lowered
        has_per = "/" in text or " per " in lowered
        has_plus = "+" in text or " plus " in lowered

        if has_percent:
            counter["percent"] += 1
            if len(samples["percent"]) < 5:
                samples["percent"].append(text)
        if has_currency:
            counter["currency"] += 1
            if len(samples["currency"]) < 5:
                samples["currency"].append(text)
        if has_per:
            counter["per_unit"] += 1
            if len(samples["per_unit"]) < 5:
                samples["per_unit"].append(text)
        if has_plus:
            counter["mixed"] += 1
            if len(samples["mixed"]) < 5:
                samples["mixed"].append(text)

    print("HTS duty-rate snapshot")
    print(f"Total HTS rows: {total_rows:,}")
    print(f"Rows with a duty string: {len(general_rates):,}")
    print()
    for key, label in (
        ("percent", "Contains % (ad valorem)"),
        ("currency", "Contains currency symbol (specific)"),
        ("per_unit", "Contains '/' or 'per'"),
        ("mixed", "Contains '+' or 'plus'"),
    ):
        count = counter.get(key, 0)
        pct = (count / len(general_rates) * 100) if general_rates else 0
        print(f"{label:<35}: {count:>6} ({pct:5.1f}%)")
        for sample in samples[key]:
            print(f"    • {sample}")
        if samples[key]:
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect duty rate patterns in the HTS SQLite DB.")
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help="Path to data/hts.db (defaults to repository copy).",
    )
    args = parser.parse_args()
    if not args.db.exists():
        raise SystemExit(f"Database not found: {args.db}")
    conn = sqlite3.connect(args.db)
    try:
        summarize_rates(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
