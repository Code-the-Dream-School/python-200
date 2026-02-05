"""mini_etl.py

A small ETL module. 
"""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


def clean_amount_series(s: pd.Series) -> pd.Series:
    """Clean a currency amount column.

    - strip whitespace
    - remove "$" and "," characters
    - convert to float
    - invalid values become NaN
    """
    s = s.astype("string").str.strip()
    s = s.str.replace("$", "", regex=False)
    
    return pd.to_numeric(s, errors="coerce")


def parse_timestamp_series(s: pd.Series) -> pd.Series:
    """Parse a timestamp column into pandas datetimes.

    - pd.to_datetime with errors="coerce"
    - invalid values become NaT
    """
    return pd.to_datetime(s, errors="ignore")


def summarize_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize data by day.

    Expected behavior 
    - input df has columns "timestamp" (datetime) and "amount" (float)
    - create "date" from timestamp (date only)
    - group by date
    - output columns: date, total_amount, num_rows
      - total_amount: sum of amount (NaNs ignored)
      - num_rows: number of rows in that date group (including rows with NaN amounts)
    """
    out = df.copy()
    out["date"] = out["timestamp"].dt.date

    grouped = out.groupby("date", as_index=False).agg(
        total_amount=("amount", "sum"),
        num_rows=("amount", "count"),  # BUG: counts non-NaN only
    )

    return grouped.sort_values("date").reset_index(drop=True)


def run_pipeline(csv_path: str | Path) -> pd.DataFrame:
    """
    Run all of our functions (mini pipeline) on a CSV:
    First, read csv given by `csv_path`, then run our functions on the data:

        parse_time_stamp_series -> clean_amount_series -> summarize_daily
    
    """
    df = pd.read_csv(csv_path)

    # Normalize column names a tiny bit
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["timestamp"] = parse_timestamp_series(df["timestamp"])
    df["amount"] = clean_amount_series(df["amount"])

    # Drop rows where timestamp is missing (keeps the summary simple)
    df = df.dropna(subset=["timestamp"])

    return summarize_daily(df)


if __name__ == "__main__":
    # Simple playground: edit this path if you want to try a different file.
    csv_path = "data_sample.csv"
    summary = run_pipeline(csv_path)
    print(summary.to_string(index=False))  # don't print index column
