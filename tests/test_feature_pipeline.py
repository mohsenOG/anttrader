"""Historical timing, independent windows, CSV output, and failure handling."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from anttrader.features import feature_columns, iter_feature_rows
from anttrader.grid import iter_historical_grids
from anttrader.pipeline import DEFAULT_CONFIG, main
from anttrader.simulation import SimulationConfig


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


def configured_history(tmp_path):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["historical_csv"] = "synthetic.csv"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    candles = synthetic_candles()
    candles.to_csv(tmp_path / "synthetic.csv", index=False)
    return path, config, candles


def test_one_row_per_grid_with_no_future_leakage():
    candles = synthetic_candles()
    windows = list(iter_historical_grids(candles))
    originals = [window.grid.copy() for window in windows]
    rows = list(iter_feature_rows(windows))
    assert len(rows) == 4
    for index, (window, row) in enumerate(zip(windows, rows, strict=True)):
        assert row["decision_timestamp"] == window.decision_timestamp.isoformat()
        assert window.decision_timestamp == candles.timestamp.iloc[
            256 + index
        ] + pd.Timedelta(hours=1)
        assert list(row) == ["decision_timestamp", *feature_columns(SimulationConfig())]
        np.testing.assert_array_equal(window.grid, originals[index])
        assert row == next(iter_feature_rows([window]))  # Independent resets.
    changed = candles.copy()
    changed.loc[257:, ["open", "high", "low", "close"]] *= 3
    assert next(iter_feature_rows(iter_historical_grids(changed))) == rows[0]
    cutoff = windows[0].decision_timestamp
    assert (
        list(iter_feature_rows(iter_historical_grids(changed, as_of=cutoff)))
        == rows[:1]
    )
    assert (
        list(iter_feature_rows(iter_historical_grids(candles.iloc[:257]))) == rows[:1]
    )
    assert list(iter_feature_rows(iter_historical_grids(candles))) == rows


def test_csv_export_metadata_limit_and_reproducibility(tmp_path, capsys):
    path, config, candles = configured_history(tmp_path)
    main(path, stage="features")
    output = tmp_path / config["feature_output"]
    first = output.read_bytes()
    frame = pd.read_csv(output)
    expected_rows = list(iter_feature_rows(iter_historical_grids(candles)))
    assert list(frame) == list(expected_rows[0])
    assert frame.shape == (4, 61)
    pd.testing.assert_frame_equal(frame, pd.DataFrame(expected_rows), check_exact=False)
    timestamps = pd.to_datetime(frame.decision_timestamp, utc=True)
    assert timestamps.is_unique and timestamps.is_monotonic_increasing
    assert np.isfinite(frame.iloc[:, 1:]).all().all()
    metadata = json.loads(
        output.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["simulations"] == metadata["eligible_grids"] == 4
    assert metadata["transitions"] == 8192
    assert metadata["feature_count"] == 60
    assert len(metadata["source_sha256"]) == len(metadata["config_sha256"]) == 64
    main(path, stage="features")
    assert output.read_bytes() == first
    main(path, stage="features", limit=2)
    pd.testing.assert_frame_equal(pd.read_csv(output), frame.iloc[:2])
    assert "failures: 0" in capsys.readouterr().out
    assert not list(output.parent.glob("*.tmp"))


def test_unexpected_grid_failure_preserves_output(tmp_path, monkeypatch):
    path, config, candles = configured_history(tmp_path)
    output = tmp_path / config["feature_output"]
    output.parent.mkdir()
    output.write_text("previous verified output\n", encoding="utf-8")
    window = next(iter_historical_grids(candles))
    invalid = replace(
        window,
        grid=np.full((16, 16), 2),
        decision_timestamp=window.decision_timestamp + pd.Timedelta(hours=1),
    )
    monkeypatch.setattr(
        "anttrader.pipeline.iter_historical_grids",
        lambda *a, **k: iter([window, invalid]),
    )
    with pytest.raises(SystemExit, match="Feature simulation failed") as failure:
        main(path, stage="features")
    assert str(invalid.decision_timestamp) in str(failure.value)
    assert output.read_text(encoding="utf-8") == "previous verified output\n"
    assert not list(output.parent.glob("*.tmp"))


def test_duplicate_or_naive_decisions_fail_explicitly():
    window = next(iter_historical_grids(synthetic_candles()))
    with pytest.raises(ValueError, match="increasing"):
        list(iter_feature_rows([window, window]))
    with pytest.raises(ValueError, match="timezone-aware"):
        list(
            iter_feature_rows(
                [
                    replace(
                        window,
                        decision_timestamp=window.decision_timestamp.tz_localize(None),
                    )
                ]
            )
        )


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit(tmp_path, limit):
    path, _, _ = configured_history(tmp_path)
    with pytest.raises(SystemExit, match="positive integer"):
        main(path, stage="features", limit=limit)


@pytest.mark.parametrize("target", ["synthetic.csv", "config.yaml"])
def test_protect_source_and_config(tmp_path, target):
    path, config, _ = configured_history(tmp_path)
    config["feature_output"] = target
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    original = (tmp_path / target).read_bytes()
    with pytest.raises(SystemExit, match="must not overwrite"):
        main(path, stage="features")
    assert (tmp_path / target).read_bytes() == original


def test_pipeline_rejects_wrong_grid_dimensions(tmp_path):
    path, config, _ = configured_history(tmp_path)
    config["grid_size"] = [8, 32]
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SystemExit, match="requires grid_size"):
        main(path, stage="features")
