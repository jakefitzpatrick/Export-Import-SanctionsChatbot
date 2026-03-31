#!/usr/bin/env python3
"""
Clean HTS dataset for Streamlit tariff lookup app.

Assumptions:
  - Input CSV has a header row matching expected column names.
  - 'Indent' indicates hierarchy level and is preserved as-is for dropdown grouping.
  - Blank tariff fields are allowed to simplify initial testing.

Outputs cleaned CSV to data/hts_cleaned.csv and prints summary info.
"""

import pandas as pd


def main():
    # Load raw HTS dataset as strings to preserve formatting
    input_path = 'data/hts_revision.csv'
    df = pd.read_csv(input_path, dtype=str)

    # Keep only the columns needed for tariff lookups
    cols_to_keep = [
        'HTS Number', 'Indent', 'Description',
        'General Rate of Duty', 'Special Rate of Duty',
        'Column 2 Rate of Duty', 'Additional Duties'
    ]
    df = df[cols_to_keep]

    # Rename columns to simpler identifiers
    df = df.rename(columns={
        'HTS Number': 'hts_code',
        'Indent': 'indent',
        'Description': 'product_description',
        'General Rate of Duty': 'general_rate',
        'Special Rate of Duty': 'special_rate',
        'Column 2 Rate of Duty': 'column2_rate',
        'Additional Duties': 'additional_duties'
    })

    # Drop rows where both key fields are missing
    df = df.dropna(subset=['hts_code', 'product_description'], how='all')

    # Preserve indent hierarchy; allow blank tariff fields for initial dropdowns/testing

    # Write cleaned data to CSV
    output_path = 'data/hts_cleaned.csv'
    df.to_csv(output_path, index=False)

    # Print summary information
    print(f'Columns after cleaning: {list(df.columns)}')
    print(f'Number of rows: {len(df)}')
    print('First 10 rows:')
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
