"""Create or update the HTS SQLite database.

Two modes:
  Full rebuild  – provide --csv to load/replace the `hts` table and add ch99.
  Ch99-only     – omit --csv (or point to a missing file) to only add/replace
                  the `chapter_99` table and `hts_with_ch99` view in an
                  existing database.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
import sys

MAX_DISPLAY_ROWS = 1000

# View that joins Master_HTS (Chapters 1-97) with Chapter 99 surcharges.
#
# Join logic:
#   - ch99.HTS_MASTER_CODE = master.hts_code  OR  ch99.HTS_MASTER_CODE = 'ALL'
#   - COUNTRY = 'Global' is preserved as-is; Python filters both the specific
#     country and 'Global' at query time and picks the lower MATCH_PRIORITY.
#   - ROW_NUMBER partitioned by (hts_code, ch99.COUNTRY) keeps only the
#     highest-precedence row per (product, country-bucket).
CREATE_VIEW_SQL = """
CREATE VIEW hts_with_ch99 AS
WITH matched AS (
    SELECT
        m.hts_code,
        m.chapter,
        m.description,
        m.full_description,
        m.general_duty_rate,
        m.special_duty_rate,
        c.NEWCODE                                           AS ch99_newcode,
        c.COUNTRY                                           AS ch99_country,
        c.NEWRATE_CLEAN                                     AS ch99_newrate,
        c.RATE_MODIFIER                                     AS ch99_rate_modifier,
        COALESCE(CAST(c.ADDITIONAL_DUTY_PCT AS REAL), 0.0) AS ch99_additional_pct,
        c.TRADEPROGRAM                                      AS ch99_tradeprogram,
        CAST(c.MATCH_PRIORITY AS INTEGER)                  AS ch99_match_priority,
        ROW_NUMBER() OVER (
            PARTITION BY m.hts_code, c.COUNTRY
            ORDER BY
                CAST(c.MATCH_PRIORITY AS INTEGER) ASC,
                -- Within same priority, prefer rows with actual rate content:
                -- Floor rows first (they express the clearest intent),
                -- then replacement rates, then additive surcharges, then blanks last.
                CASE c.RATE_MODIFIER WHEN 'Floor' THEN 0 ELSE 1 END ASC,
                CASE WHEN c.NEWRATE_CLEAN != '' AND c.NEWRATE_CLEAN IS NOT NULL THEN 0 ELSE 1 END ASC,
                CASE WHEN c.ADDITIONAL_DUTY_PCT != '' AND c.ADDITIONAL_DUTY_PCT IS NOT NULL THEN 0 ELSE 1 END ASC
        ) AS _rn
    FROM hts AS m
    JOIN chapter_99 AS c
        ON (
            -- Exact match (13-char Ch99 codes, e.g. "2208.50.00.30")
            c.HTS_MASTER_CODE = m.hts_code
            -- 8-digit Ch99 code (10 chars, e.g. "0402.21.90") vs 10-digit HTS ("0402.21.90.00")
            OR (LENGTH(c.HTS_MASTER_CODE) = 10
                AND c.HTS_MASTER_CODE = SUBSTR(m.hts_code, 1, 10))
            -- 10-digit Ch99 code missing the final period (12 chars, e.g. "8541.42.0010")
            -- → re-insert the missing dot: "8541.42.00" + "." + "10" = "8541.42.00.10"
            OR (LENGTH(c.HTS_MASTER_CODE) = 12
                AND SUBSTR(c.HTS_MASTER_CODE, 1, 10) || '.' || SUBSTR(c.HTS_MASTER_CODE, 11) = m.hts_code)
            -- ALL wildcard matches every product
            OR c.HTS_MASTER_CODE = 'ALL'
        )
)
SELECT
    hts_code,
    chapter,
    description,
    full_description,
    general_duty_rate,
    special_duty_rate,
    ch99_newcode,
    ch99_country,
    ch99_newrate,
    ch99_rate_modifier,
    ch99_additional_pct,
    ch99_tradeprogram,
    ch99_match_priority
FROM matched
WHERE _rn = 1;
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("hts_cleaned_final.csv"),
        help="Path to the cleaned HTS CSV (Chapters 1-97). If absent, the "
             "existing `hts` table in --db is left untouched.",
    )
    parser.add_argument(
        "--ch99",
        type=Path,
        default=Path("Final_Wildcard_Optimized_Chapter99.csv"),
        help="Path to the Chapter 99 CSV.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/hts.db"),
        help="SQLite file path.",
    )
    return parser.parse_args()


def load_csv_into_table(
    cur: sqlite3.Cursor,
    csv_path: Path,
    table_name: str,
    extra_indexes: list[str] | None = None,
) -> int:
    """Load a CSV into a SQLite table (all TEXT columns). Returns row count."""
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        try:
            headers = next(reader)
        except StopIteration:
            sys.exit(f"CSV file is empty: {csv_path}")

        column_defs = ", ".join(f'"{col}" TEXT' for col in headers)
        quoted_cols = ", ".join(f'"{col}"' for col in headers)
        placeholders = ", ".join("?" for _ in headers)

        cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        cur.execute(f'CREATE TABLE "{table_name}" ({column_defs})')
        insert_sql = f'INSERT INTO "{table_name}" ({quoted_cols}) VALUES ({placeholders})'

        batch: list[list[str]] = []
        row_count = 0
        for idx, row in enumerate(reader, start=1):
            if len(row) != len(headers):
                raise ValueError(
                    f"Row {idx} in {csv_path} has {len(row)} columns "
                    f"but header has {len(headers)}."
                )
            batch.append(row)
            row_count += 1
            if len(batch) >= MAX_DISPLAY_ROWS:
                cur.executemany(insert_sql, batch)
                batch.clear()
        if batch:
            cur.executemany(insert_sql, batch)

    for col in (extra_indexes or []):
        idx_name = f"idx_{table_name}_{col}"
        cur.execute(
            f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}"("{col}")'
        )

    return row_count


def build_database(csv_path: Path, ch99_path: Path, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    full_rebuild = csv_path.exists()

    if full_rebuild:
        if db_path.exists():
            db_path.unlink()
        print(f"Full rebuild: loading `hts` from {csv_path}.")
    else:
        if not db_path.exists():
            sys.exit(
                f"Database not found at {db_path} and no --csv provided for a full build."
            )
        print(f"Updating existing database: {db_path}")

    if not ch99_path.exists():
        sys.exit(f"Chapter 99 CSV not found: {ch99_path}")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        if full_rebuild:
            n_hts = load_csv_into_table(
                cur, csv_path, "hts",
                extra_indexes=["hts_code", "chapter"],
            )
            print(f"Loaded {n_hts:,} rows into `hts`.")

        # Always (re)load chapter_99 and recreate the view.
        n_ch99 = load_csv_into_table(
            cur, ch99_path, "chapter_99",
            extra_indexes=["HTS_MASTER_CODE", "COUNTRY", "MATCH_PRIORITY"],
        )
        print(f"Loaded {n_ch99:,} rows into `chapter_99`.")

        cur.execute("DROP VIEW IF EXISTS hts_with_ch99")
        cur.execute(CREATE_VIEW_SQL)
        print("Created view `hts_with_ch99`.")

        conn.commit()
    finally:
        conn.close()

    print(f"Done: {db_path}")


def main() -> None:
    args = parse_args()
    build_database(args.csv, args.ch99, args.db)


if __name__ == "__main__":
    main()
