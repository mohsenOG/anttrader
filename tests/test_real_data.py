"""Optional integration check against the configured local Dukascopy dataset."""

import numpy as np
import pandas as pd
import pytest
import yaml

from anttrader.data import load_gold_data
from anttrader.grid import iter_historical_grids
from anttrader.pipeline import DEFAULT_CONFIG


def test_local_dukascopy_history_and_grid_alignment():
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    path = DEFAULT_CONFIG.parent / config["historical_csv"]
    if not path.is_file():
        pytest.skip(
            "Local Dukascopy CSV is not present; datasets are excluded from Git."
        )
    cutoff = pd.Timestamp("2026-10-01T00:00:00Z")
    candles = load_gold_data(
        path,
        source_timezone=config["timestamp_timezone"],
        timestamp_format=config["timestamp_format"],
        drop_zero_volume=config["drop_zero_volume"],
        as_of=cutoff,
    )
    assert len(candles) >= 257
    if "volume" in candles and config["drop_zero_volume"]:
        assert candles.volume.gt(0).all()
    count = 0
    latest = None
    previous_decision = None
    for count, latest in enumerate(iter_historical_grids(candles, as_of=cutoff), 1):
        assert latest.grid.shape == (16, 16)
        assert np.isin(latest.grid, [0, 1]).all()
        assert latest.decision_timestamp == candles.timestamp.iloc[
            count + 255
        ] + pd.Timedelta(hours=1)
        if previous_decision is not None:
            assert latest.decision_timestamp > previous_decision
        previous_decision = latest.decision_timestamp
    assert count == len(candles) - 256
    assert latest is not None
    assert latest.window_start == candles.timestamp.iloc[-257]
    assert latest.window_end == candles.timestamp.iloc[-1]
    closes = candles.close.iloc[-257:].to_numpy()
    expected_returns = np.log(closes[1:] / closes[:-1])
    np.testing.assert_array_equal(latest.returns, expected_returns)
    chronological = latest.grid.copy()
    chronological[1::2] = chronological[1::2, ::-1]
    np.testing.assert_array_equal(
        chronological.ravel(), (expected_returns > 0).astype(np.uint8)
    )
    # A real-data counterpart to the small synthetic no-look-ahead test.
    first = next(iter_historical_grids(candles, as_of=cutoff))
    changed = candles.copy()
    changed.loc[257:, ["open", "high", "low", "close"]] *= 1.1
    same_time = next(iter_historical_grids(changed, as_of=first.decision_timestamp))
    np.testing.assert_array_equal(first.grid, same_time.grid)
    np.testing.assert_array_equal(first.returns, same_time.returns)
