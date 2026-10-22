"""Small synthetic OHLC inputs only; these are not real gold history."""

import numpy as np
import pandas as pd
import pytest

from anttrader.data import load_gold_data


def write_csv(tmp_path, **changes):
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-05T20:00:00Z", "2024-01-05T21:00:00Z"],
            "open": [100.0, 101.0],
            "high": [103.0, 104.0],
            "low": [99.0, 100.0],
            "close": [102.0, 103.0],
        }
    )
    for column, values in changes.items():
        frame[column] = values
    path = tmp_path / "synthetic.csv"
    frame.to_csv(path, index=False)
    return path


def test_valid_csv_and_optional_volume(tmp_path):
    path = write_csv(tmp_path, volume=[0.5, 0.8])
    data = load_gold_data(path, as_of="2024-01-06T00:00:00Z")
    assert list(data) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert str(data.timestamp.dt.tz) == "UTC"
    assert data.close.tolist() == [102, 103]
    assert data.volume.tolist() == [0.5, 0.8]


def test_missing_column(tmp_path):
    path = write_csv(tmp_path)
    pd.read_csv(path).drop(columns="close").to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns: close"):
        load_gold_data(path)


@pytest.mark.parametrize(
    "column,values",
    [
        ("high", [101, 104]),
        ("low", [101, 100]),
        ("high", [98, 104]),
    ],
)
def test_invalid_ohlc(tmp_path, column, values):
    with pytest.raises(ValueError, match="OHLC"):
        load_gold_data(write_csv(tmp_path, **{column: values}))


@pytest.mark.parametrize("value", [-1, 0, np.nan, np.inf, -np.inf, "invalid"])
@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
def test_invalid_prices(tmp_path, column, value):
    with pytest.raises(ValueError, match=f"{column} prices"):
        load_gold_data(write_csv(tmp_path, **{column: [value, 101]}))


def test_duplicate_timestamps_after_utc_conversion(tmp_path):
    path = write_csv(
        tmp_path,
        timestamp=[
            "2024-01-05T20:00:00Z",
            "2024-01-05T21:00:00+01:00",
        ],
    )
    with pytest.raises(ValueError, match="Duplicate candle timestamps"):
        load_gold_data(path)


def test_unsorted_input_is_sorted_with_prices(tmp_path):
    path = write_csv(
        tmp_path,
        timestamp=[
            "2024-01-05T22:00:00+01:00",
            "2024-01-05T21:00:00+01:00",
        ],
    )
    data = load_gold_data(path)
    assert data.timestamp.is_monotonic_increasing
    assert data.close.tolist() == [103, 102]
    assert data.timestamp.iloc[0] == pd.Timestamp("2024-01-05T20:00:00Z")


def test_naive_timestamps_require_source_timezone(tmp_path):
    path = write_csv(tmp_path, timestamp=["2024-01-05 20:00", "2024-01-05 21:00"])
    with pytest.raises(ValueError, match="explicit source_timezone"):
        load_gold_data(path)
    assert load_gold_data(path, source_timezone="UTC").timestamp.iloc[0] == (
        pd.Timestamp("2024-01-05T20:00:00Z")
    )


@pytest.mark.parametrize("timestamp", ["2024-10-27 02:00", "2024-03-31 02:00"])
def test_dst_uncertainty_is_rejected(tmp_path, timestamp):
    path = write_csv(tmp_path, timestamp=[timestamp, "2024-01-05 20:00"])
    with pytest.raises(ValueError, match="Invalid timestamp"):
        load_gold_data(path, source_timezone="Europe/Berlin")


@pytest.mark.parametrize("timestamp", ["2024-01-05T20:30:00Z", "2024-01-05T20:00:01Z"])
def test_non_hourly_boundary(tmp_path, timestamp):
    with pytest.raises(ValueError, match="UTC hourly boundaries"):
        load_gold_data(write_csv(tmp_path, timestamp=[timestamp, "2024-01-05T21:00Z"]))


def test_weekend_gap_is_reported_without_filling(tmp_path):
    data = load_gold_data(
        write_csv(
            tmp_path,
            timestamp=[
                "2024-01-05T21:00:00Z",
                "2024-01-07T23:00:00Z",
            ],
        )
    )
    assert len(data) == 2
    report = data.attrs["validation_report"]
    assert report["gap_count"] == 1
    assert report["gap_examples"][0]["missing_hours"] == 49


def test_completion_boundary_is_inclusive(tmp_path):
    path = write_csv(tmp_path)
    before = load_gold_data(path, as_of="2024-01-05T20:59:59Z")
    at = load_gold_data(path, as_of="2024-01-05T21:00:00Z")
    assert before.empty
    assert len(at) == 1
    assert at.attrs["validation_report"]["excluded_incomplete_candles"] == 1
    with pytest.raises(ValueError, match="explicit timezone"):
        load_gold_data(path, as_of="2024-01-05 21:00")


def test_dukascopy_export_header_and_explicit_date_format(tmp_path):
    path = tmp_path / "synthetic_export.csv"
    path.write_text(
        "Gmt time,Open,High,Low,Close,Volume\n"
        "05.01.2024 20:00:00.000,100,103,99,102,0.5\n",
        encoding="utf-8",
    )
    data = load_gold_data(
        path, source_timezone="UTC", timestamp_format="%d.%m.%Y %H:%M:%S.%f"
    )
    assert data.timestamp.iloc[0] == pd.Timestamp("2024-01-05T20:00Z")


@pytest.mark.parametrize("content", ["", "timestamp,open,high,low,close\n"])
def test_empty_csv(tmp_path, content):
    path = tmp_path / "empty.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_gold_data(path)


@pytest.mark.parametrize("timestamp", [None, "invalid", "1704484800000"])
def test_invalid_timestamp(tmp_path, timestamp):
    with pytest.raises(ValueError, match="Invalid timestamp"):
        load_gold_data(write_csv(tmp_path, timestamp=[timestamp, "2024-01-05T21:00Z"]))
