import math
from pathlib import Path

import pandas as pd
import pytest

import mini_etl


def test_clean_amount_series_strips_dollar_commas_whitespace():
    # Input: [" $1,200.50 ", "9", "bad"]
    # Output should become: [1200.50, 9.0, NaN]
    s = pd.Series([" $1,200.50 ", "9", "bad"])
    out = mini_etl.clean_amount_series(s)

    assert out.iloc[0] == pytest.approx(1200.50)
    assert out.iloc[1] == pytest.approx(9.0)
    assert math.isnan(out.iloc[2])


def test_parse_timestamp_series_coerces_invalid_to_nat():
    """
    parse_timestamp_series should turn timestamp strings into real datetimes.

    Two things should be true:
    1) The result should be a datetime-typed Series (not a Series of strings).
    2) Bad timestamps should not crash the function. They should become "missing".

       Pandas represents a missing datetime value as NaT ("Not a Time").
       pd.isna(...) returns True for NaT.
    """
    s = pd.Series(["2024-01-01 10:00:00", "not-a-date"])
    out = mini_etl.parse_timestamp_series(s)

    # 1) The whole Series should be datetime dtype (so we can use .dt later)
    assert str(out.dtype).startswith("datetime64"), f"Expected datetime dtype, got {out.dtype}"

    # 2) The invalid timestamp should become missing (NaT)
    assert pd.isna(out.iloc[1])


def test_summarize_daily_counts_rows_including_nan_amounts():
    # Input rows:
    #   2024-01-01 10:00:00 amount=10.0
    #   2024-01-01 12:00:00 amount=NaN
    #   2024-01-02 09:00:00 amount=5.0
    #
    # Expected daily summary:
    #   2024-01-01: total_amount=10.0, num_rows=2  (NaN ignored in sum, but row still counted)
    #   2024-01-02: total_amount=5.0,  num_rows=1
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 10:00:00", "2024-01-01 12:00:00", "2024-01-02 09:00:00"]
            ),
            "amount": [10.0, float("nan"), 5.0],
        }
    )

    out = mini_etl.summarize_daily(df)

    row_0101 = out[out["date"] == pd.to_datetime("2024-01-01").date()].iloc[0]
    assert row_0101["total_amount"] == pytest.approx(10.0)
    assert row_0101["num_rows"] == 2  # counts rows, even if amount is NaN



def test_run_pipeline_smoke(tmp_path):
    # The pipeline should run end-to-end and return columns:
    # ["date", "total_amount", "num_rows"]
    here = Path(__file__).resolve().parent.parent
    sample = here / "data_sample.csv"
    dst = tmp_path / "data_sample.csv"
    dst.write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")

    out = mini_etl.run_pipeline(dst)

    # Expected columns
    assert list(out.columns) == ["date", "total_amount", "num_rows"]

    # Expected dates (sorted)
    assert list(out["date"]) == [
        pd.to_datetime("2024-01-01").date(),
        pd.to_datetime("2024-01-02").date(),
    ]

    # Expected number of summary rows
    assert len(out) == 2


def test_run_pipeline_expected_totals(tmp_path):
    # Expected daily summary for the included sample CSV:
    #
    # 2024-01-01: 1200.50 + 9.00 - 5.00 = 1204.50 (3 rows)
    # 2024-01-02: 3.00 + 2.00 = 5.00 (2 rows)
    #
    # Note: the row with "not-a-date" should be dropped by the pipeline.
    here = Path(__file__).resolve().parent.parent
    sample = here / "data_sample.csv"
    dst = tmp_path / "data_sample.csv"
    dst.write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")

    out = mini_etl.run_pipeline(dst)

    expected = pd.DataFrame(
        {
            "date": [
                pd.to_datetime("2024-01-01").date(),
                pd.to_datetime("2024-01-02").date(),
            ],
            "total_amount": [1204.50, 5.00],
            "num_rows": [3, 2],
        }
    )

    pd.testing.assert_frame_equal(out, expected)
