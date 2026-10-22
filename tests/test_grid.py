"""Deterministic checks using synthetic candles, never real gold history."""

import numpy as np
import pandas as pd
import pytest

from anttrader.grid import (
    binary_states,
    closes_to_grid,
    iter_historical_grids,
    log_returns,
    serpentine_grid,
)


def synthetic_candles(count=260):
    closes = 100 + np.arange(count) % 7
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=count, freq="h", tz="UTC"),
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
        }
    )


def test_log_return_formula_and_binary_signs():
    np.testing.assert_allclose(log_returns([100, 110, 99, 99]), np.log([1.1, 0.9, 1]))
    np.testing.assert_array_equal(
        binary_states(log_returns([100, 110, 99, 99])), [1, 0, 0]
    )
    assert len(log_returns(np.arange(1, 258))) == 256


@pytest.mark.parametrize("count", [0, 1, 256, 258])
def test_default_window_requires_exactly_257_closes(count):
    with pytest.raises(ValueError, match="Exactly 257 closing prices"):
        closes_to_grid(np.arange(1, count + 1))


def test_default_grid_shape_binary_values_and_determinism():
    closes = synthetic_candles(257).close.to_numpy()
    first = closes_to_grid(closes)
    assert first.shape == (16, 16)
    assert first.dtype == np.uint8
    assert set(np.unique(first)) == {0, 1}
    np.testing.assert_array_equal(first, closes_to_grid(closes))
    states = binary_states(log_returns(closes))
    assert first[0, 0] == states[0]
    assert first[15, 0] == states[-1]
    assert closes_to_grid(np.ones(257)).sum() == 0


def test_serpentine_index_example_and_default_endpoints():
    np.testing.assert_array_equal(
        serpentine_grid(np.arange(16), (4, 4)),
        [
            [0, 1, 2, 3],
            [7, 6, 5, 4],
            [8, 9, 10, 11],
            [15, 14, 13, 12],
        ],
    )
    grid = serpentine_grid(np.arange(256))
    assert grid[0, 0] == 0
    assert grid[15, 0] == 255


@pytest.mark.parametrize("shape", [(0, 16), (16,), (16, 16.0), (True, 256)])
def test_invalid_grid_shape(shape):
    with pytest.raises(ValueError, match="positive integers"):
        serpentine_grid(np.arange(256), shape)


def test_wrong_number_of_states():
    with pytest.raises(ValueError, match="Exactly 256 values"):
        serpentine_grid([0, 1])


@pytest.mark.parametrize(
    "closes", [[0, 1], [-1, 1], [np.inf, 1], [np.nan, 1], [[1, 2]]]
)
def test_invalid_close_inputs(closes):
    with pytest.raises(ValueError):
        log_returns(closes)


def test_nonfinite_returns_are_rejected():
    with pytest.raises(ValueError, match="finite 1D"):
        binary_states([np.nan])


def test_rolling_windows_match_independent_windows_and_times():
    candles = synthetic_candles()
    # A closure changes elapsed wall time, but not the number of input closes.
    candles.loc[200:, "timestamp"] += pd.Timedelta(days=2)
    grids = list(iter_historical_grids(candles))
    assert len(grids) == 4
    for start, result in enumerate(grids):
        window = candles.iloc[start : start + 257]
        np.testing.assert_array_equal(
            result.grid, closes_to_grid(window.close.to_numpy())
        )
        np.testing.assert_array_equal(
            result.returns, log_returns(window.close.to_numpy())
        )
        assert result.window_start == window.timestamp.iloc[0]
        assert result.window_end == window.timestamp.iloc[-1]
        assert result.decision_timestamp == window.timestamp.iloc[-1] + pd.Timedelta(
            hours=1
        )
        assert (
            window.timestamp + pd.Timedelta(hours=1) <= result.decision_timestamp
        ).all()


def test_insufficient_completed_history_and_exact_completion_time():
    candles = synthetic_candles()
    decision = candles.timestamp.iloc[256] + pd.Timedelta(hours=1)
    assert list(iter_historical_grids(candles.iloc[:0])) == []
    assert list(iter_historical_grids(candles.iloc[:256])) == []
    assert (
        list(
            iter_historical_grids(candles, as_of=decision - pd.Timedelta(nanoseconds=1))
        )
        == []
    )
    grids = list(iter_historical_grids(candles, as_of=decision))
    assert len(grids) == 1
    assert grids[0].decision_timestamp == decision
    assert grids[0].window_end == candles.timestamp.iloc[256]


def test_no_look_ahead_when_future_candles_change():
    original = synthetic_candles()
    changed = original.copy()
    changed.loc[257:, ["open", "high", "low", "close"]] *= 3
    decision = original.timestamp.iloc[256] + pd.Timedelta(hours=1)
    first = next(iter_historical_grids(original, as_of=decision))
    second = next(iter_historical_grids(changed, as_of=decision))
    truncated = next(iter_historical_grids(original.iloc[:257]))
    future_included = next(iter_historical_grids(changed))
    for other in [second, truncated, future_included]:
        assert first.decision_timestamp == other.decision_timestamp == decision
        np.testing.assert_array_equal(first.grid, other.grid)
        np.testing.assert_array_equal(first.returns, other.returns)
