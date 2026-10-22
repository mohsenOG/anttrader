"""Load and validate local XAU/USD H1 candles without filling market gaps."""

from pathlib import Path

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]
HOUR = pd.Timedelta(hours=1)


def utc_timestamp(value: object) -> pd.Timestamp:
    """Require an explicit timezone for a decision time, then convert to UTC."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("Decision/as_of timestamp must include an explicit timezone.")
    return timestamp.tz_convert("UTC")


def validate_candles(
    frame: pd.DataFrame,
    *,
    source_timezone: str | None = None,
    as_of: object | None = None,
    timestamp_format: str = "ISO8601",
) -> pd.DataFrame:
    """Return sorted, validated candles completed at as_of (default: now in UTC).

    Input timestamps are candle opening times. Strings use ISO 8601 unless an
    explicit strptime format is supplied. Naive times require source_timezone;
    ambiguous/nonexistent local times raise errors. No candle is inserted or
    repaired. The returned frame's attrs['validation_report'] records excluded
    incomplete candles and gaps over one hour (all gaps need inspection; no
    trading-calendar classification is attempted). Optional volume is preserved.
    """
    required = ["timestamp", *OHLC]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The candle dataset is empty.")
    candles = frame[required + (["volume"] if "volume" in frame else [])].copy()
    timestamps = []
    for row, value in enumerate(candles["timestamp"], start=2):
        try:
            if isinstance(value, str):
                timestamp = pd.to_datetime(value, format=timestamp_format)
            elif isinstance(value, (int, float, np.number)):
                raise TypeError("Numeric timestamps require conversion to ISO 8601.")
            else:
                timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                raise ValueError("Missing timestamp.")
            if timestamp.tzinfo is None:
                if source_timezone is None:
                    raise ValueError(
                        "Naive timestamps require an explicit source_timezone."
                    )
                timestamp = timestamp.tz_localize(source_timezone)
            timestamps.append(timestamp.tz_convert("UTC"))
        except Exception as exc:
            raise ValueError(f"Invalid timestamp at CSV row {row}: {exc}") from exc
    candles["timestamp"] = pd.DatetimeIndex(timestamps)
    if candles["timestamp"].duplicated().any():
        raise ValueError("Duplicate candle timestamps after UTC normalization.")
    if (candles["timestamp"] != candles["timestamp"].dt.floor("h")).any():
        raise ValueError("Candle timestamps must lie on UTC hourly boundaries.")
    for column in OHLC:
        candles[column] = pd.to_numeric(candles[column], errors="coerce")
        values = candles[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError(f"{column} prices must be numeric, finite, and positive.")
    invalid = (
        (candles["high"] < candles[["open", "close"]].max(axis=1))
        | (candles["low"] > candles[["open", "close"]].min(axis=1))
        | (candles["high"] < candles["low"])
    )
    if invalid.any():
        raise ValueError(
            "Invalid OHLC relationship: high/low do not contain open/close."
        )
    candles = candles.sort_values("timestamp").reset_index(drop=True)
    cutoff = utc_timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    completed = candles["timestamp"] + HOUR <= cutoff
    excluded = int((~completed).sum())
    candles = candles.loc[completed].reset_index(drop=True)
    intervals = candles["timestamp"].diff()
    gaps = intervals > HOUR
    examples = [
        {
            "previous_open": candles.loc[i - 1, "timestamp"].isoformat(),
            "next_open": candles.loc[i, "timestamp"].isoformat(),
            "missing_hours": int(intervals.loc[i] / HOUR) - 1,
        }
        for i in candles.index[gaps][:5]
    ]
    candles.attrs["validation_report"] = {
        "valid_candles": len(candles),
        "excluded_incomplete_candles": excluded,
        "gap_count": int(gaps.sum()),
        "gap_examples": examples,
        "as_of": cutoff.isoformat(),
    }
    return candles


def load_gold_data(
    path: str | Path,
    *,
    source_timezone: str | None = None,
    as_of: object | None = None,
    timestamp_format: str = "ISO8601",
) -> pd.DataFrame:
    """Load a comma-delimited CSV; see validate_candles for the input contract.

    Headers are case-insensitive. Dukascopy 'Gmt time' and 'Local time' headers
    are accepted as timestamp aliases, but still require explicit timezone
    information for naive values. For DD.MM.YYYY exports pass
    timestamp_format='%d.%m.%Y %H:%M:%S.%f'.
    """
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The candle CSV is empty.") from exc
    frame.columns = frame.columns.str.strip().str.lower()
    frame = frame.rename(columns={"gmt time": "timestamp", "local time": "timestamp"})
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate CSV column names after header normalization.")
    return validate_candles(
        frame,
        source_timezone=source_timezone,
        as_of=as_of,
        timestamp_format=timestamp_format,
    )
