"""Create a SQLite database from the cleaned HTS CSV."""
import argparse
import csv
import sqlite3
from pathlib import Path
import sys

MAX_DISPLAY_ROWS = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the HTS SQLite database from CSV data."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("hts_cleaned_final.csv"),
        help="Path to the cleaned HTS CSV file.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/hts.db"),
        help="Output SQLite file path.",
    )
    return parser.parse_args()


def build_database(csv_path: Path, db_path: Path) -> None:
    if not csv_path.exists():
        sys.exit(f"CSV file not found: {csv_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()

    with csv_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        try:
            headers = next(reader)
        except StopIteration:
            sys.exit(f"CSV file is empty: {csv_path}")

        column_defs = ", ".join(f'"{col}" TEXT' for col in headers)
        quoted_cols = ", ".join(f'"{col}"' for col in headers)
        placeholders = ", ".join("?" for _ in headers)

        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(f"DROP TABLE IF EXISTS hts")
            cur.execute(f"CREATE TABLE hts ({column_defs})")
            insert_sql = f"INSERT INTO hts ({quoted_cols}) VALUES ({placeholders})"
            batch = []
            for idx, row in enumerate(reader, start=1):
                if len(row) != len(headers):
                    raise ValueError(
                        f"Row {idx} in {csv_path} has {len(row)} columns but header has {len(headers)}."
                    )
                batch.append(row)
                if len(batch) >= MAX_DISPLAY_ROWS:
                    cur.executemany(insert_sql, batch)
                    batch.clear()
            if batch:
                cur.executemany(insert_sql, batch)
            cur.execute('CREATE INDEX idx_hts_code ON hts("hts_code")')
            cur.execute('CREATE INDEX idx_hts_chapter ON hts("chapter")')
            conn.commit()
        finally:
            conn.close()

    print(f"Built SQLite database {db_path} from {csv_path} ({len(headers)} columns).")


def main() -> None:
    args = parse_args()
    build_database(args.csv, args.db)


if __name__ == "__main__":
    main()
