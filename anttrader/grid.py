"""Deterministic binary log-return encoding and chronological rolling grids."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import prod

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from anttrader.data import HOUR, validate_candles


@dataclass(frozen=True)
class HistoricalGrid:
    """A completed window, identified by its first/last candle opening times."""

    grid: NDArray[np.uint8]
    decision_timestamp: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    returns: NDArray[np.float64]


def _grid_shape(grid_size: Sequence[int]) -> tuple[int, int]:
    if len(grid_size) != 2 or any(
        isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1
        for n in grid_size
    ):
        raise ValueError("grid_size must contain two positive integers.")
    return int(grid_size[0]), int(grid_size[1])


def log_returns(closes: Sequence[float]) -> NDArray[np.float64]:
    """Calculate ln(C[i] / C[i-1]), oldest first, from positive finite closes."""
    values = np.asarray(closes, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("At least two closing prices in a 1D sequence are required.")
    if not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("Closing prices must be finite and positive.")
    with np.errstate(over="raise", under="raise", divide="raise", invalid="raise"):
        try:
            return np.log(values[1:] / values[:-1])
        except FloatingPointError as exc:
            raise ValueError(
                "Closing price ratio exceeds the numerical range."
            ) from exc


def binary_states(returns: Sequence[float]) -> NDArray[np.uint8]:
    """Positive returns become 1; negative and exactly zero returns become 0."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Returns must be a finite 1D sequence.")
    return (values > 0).astype(np.uint8)


def serpentine_grid(
    values: Sequence[float], grid_size: Sequence[int] = (16, 16)
) -> np.ndarray:
    """Place chronological values in alternating row directions; preserve dtype."""
    shape = _grid_shape(grid_size)
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != prod(shape):
        raise ValueError(f"Exactly {prod(shape)} values are required for grid {shape}.")
    grid = values.reshape(shape).copy()
    grid[1::2] = grid[1::2, ::-1]
    return grid


def closes_to_grid(
    closes: Sequence[float], grid_size: Sequence[int] = (16, 16)
) -> NDArray[np.uint8]:
    """Encode one exact window: 257 closes for the default 16 x 16 binary grid."""
    shape = _grid_shape(grid_size)
    if len(closes) != prod(shape) + 1:
        raise ValueError(f"Exactly {prod(shape) + 1} closing prices are required.")
    return serpentine_grid(binary_states(log_returns(closes)), shape)


def iter_historical_grids(
    candles: pd.DataFrame,
    *,
    grid_size: Sequence[int] = (16, 16),
    as_of: object | None = None,
) -> Iterator[HistoricalGrid]:
    """Yield each eligible completed window without storing all grids.

    Input uses normalized OHLC columns and timezone-aware opening timestamps.
    Revalidate at this public boundary. as_of defaults to now in UTC. The last
    included candle completes exactly at decision_timestamp, even across gaps.
    """
    shape = _grid_shape(grid_size)
    count = prod(shape)
    if candles.empty:
        return
    completed = validate_candles(candles, as_of=as_of)
    if len(completed) <= count:
        return
    returns = log_returns(completed["close"].to_numpy())
    states = binary_states(returns)
    timestamps = completed["timestamp"]
    for end in range(count, len(completed)):
        start = end - count
        yield HistoricalGrid(
            grid=serpentine_grid(states[start:end], shape),
            decision_timestamp=timestamps.iloc[end] + HOUR,
            window_start=timestamps.iloc[start],
            window_end=timestamps.iloc[end],
            returns=returns[start:end].copy(),
        )
