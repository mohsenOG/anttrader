"""Target timing, leakage protection, fixed inputs, and deterministic decoders."""

import numpy as np
import pandas as pd
import pytest
import yaml

from anttrader.data import HOUR
from anttrader.decoder import (
    RAW_COLUMNS,
    chronological_partitions,
    decode_signals,
    fit_decoder,
    future_targets,
    historical_inputs,
    predict_up,
    validate_research_config,
)
from anttrader.features import iter_feature_rows
from anttrader.grid import binary_states, iter_historical_grids, log_returns
from anttrader.pipeline import DEFAULT_CONFIG


def history(count=500):
    i = np.arange(count)
    close = 200 + np.sin(i * 0.31) * 8 + np.cos(i * 0.17) * 3
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=count, freq="h", tz="UTC"),
            "open": close + np.sin(i) * 0.2,
            "high": close + 2,
            "low": close - 2,
            "close": close,
        }
    )


def config():
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_targets_entry_exit_zero_negative_and_tail():
    candles = history(12)
    candles["open"] = [100, 100, 100, 100, 100, 110, 100, 90, 100, 100, 100, 100]
    decisions = candles.timestamp.iloc[:-1] + HOUR
    targets = future_targets(candles, decisions)
    first = targets.iloc[0]
    assert (
        first.decision_timestamp == first.entry_timestamp == candles.timestamp.iloc[1]
    )
    assert first.exit_timestamp == candles.timestamp.iloc[5]
    assert first.future_return == pytest.approx(np.log(110 / 100))
    assert targets.target.iloc[:3].tolist() == [1, 0, 0]
    assert targets.target_valid.sum() == 7
    assert targets.target.iloc[-4:].isna().all()
    assert targets.target_exclusion.iloc[-4:].eq("unavailable_tail").all()


@pytest.mark.parametrize("missing", [1, 2, 3, 4, 5])
def test_gap_in_any_execution_hour_excludes_label(missing):
    candles = history(12)
    decision = candles.timestamp.iloc[1]
    targets = future_targets(candles.drop(index=missing), [decision])
    assert not targets.target_valid.iloc[0]
    assert pd.isna(targets.target.iloc[0])
    assert targets.target_exclusion.iloc[0] == "market_gap"


def test_split_before_filter_and_exclusive_boundary_purge():
    candles = history(50)
    decisions = candles.timestamp.iloc[:40] + HOUR
    splits, boundaries = chronological_partitions(decisions, 0.6, 0.2)
    targets = future_targets(candles, decisions)
    allowed = splits.outcome_cutoff.isna() | targets.exit_timestamp.lt(
        splits.outcome_cutoff
    )
    assert (
        splits.partition.tolist() == ["train"] * 24 + ["validation"] * 8 + ["test"] * 8
    )
    assert boundaries["validation"]["first_decision"] == decisions.iloc[24].isoformat()
    for name in ("train", "validation"):
        selected = allowed & splits.partition.eq(name)
        assert (
            targets.loc[selected, "exit_timestamp"]
            < splits.loc[selected, "outcome_cutoff"]
        ).all()
        assert (splits.partition.eq(name) & ~allowed).sum() == 4
    # A missing label cannot move the precomputed chronological boundaries.
    changed = future_targets(candles.drop(index=10), decisions)
    assert not changed.target_valid.equals(targets.target_valid)
    pd.testing.assert_frame_equal(
        splits, chronological_partitions(decisions, 0.6, 0.2)[0]
    )


def test_thresholds_and_probability_validation():
    assert decode_signals([0, 0.45, 0.45001, 0.54999, 0.55, 1]).tolist() == [
        "SELL",
        "SELL",
        "HOLD",
        "HOLD",
        "BUY",
        "BUY",
    ]
    for values in ([np.nan], [np.inf], [-0.1], [1.1]):
        with pytest.raises(ValueError, match="Probabilities"):
            decode_signals(values)
    with pytest.raises(ValueError, match="Thresholds"):
        decode_signals([0.5], buy=0.45, sell=0.55)


def test_raw_states_and_momentum_use_only_completed_candles():
    candles = history()
    decisions = candles.timestamp.iloc[256:270] + HOUR
    raw, momentum, valid = historical_inputs(candles, decisions)
    assert list(raw) == RAW_COLUMNS and raw.shape == (14, 256)
    np.testing.assert_array_equal(
        raw.iloc[0], binary_states(log_returns(candles.close.iloc[:257]))
    )
    assert np.isin(raw, [0, 1]).all() and valid.all()
    expected = np.sign(
        np.log(
            candles.close.iloc[256:270].to_numpy()
            / candles.close.iloc[252:266].to_numpy()
        )
    )
    assert (
        momentum.tolist()
        == np.where(
            expected > 0, "BUY", np.where(expected < 0, "SELL", "HOLD")
        ).tolist()
    )
    changed = candles.copy()
    changed.loc[257:, ["open", "high", "low", "close"]] *= 2
    changed_raw, changed_momentum, _ = historical_inputs(changed, decisions)
    pd.testing.assert_series_equal(raw.iloc[0], changed_raw.iloc[0])
    assert momentum.iloc[0] == changed_momentum.iloc[0]
    gapped = candles.copy()
    gapped.loc[:254, "timestamp"] -= HOUR
    assert not historical_inputs(gapped, decisions)[2][0]
    candles["close"] = 100
    assert historical_inputs(candles, decisions)[1].eq("HOLD").all()


def test_train_only_scaler_estimator_frozen_deterministic_and_stable_columns():
    candles = history()
    decisions = candles.timestamp.iloc[256:] + HOUR
    inputs, _, _ = historical_inputs(candles, decisions)
    labels = future_targets(candles, decisions)
    splits, _ = chronological_partitions(decisions, 0.6, 0.2)
    eligible = labels.target_valid & (
        splits.outcome_cutoff.isna() | labels.exit_timestamp.lt(splits.outcome_cutoff)
    )
    train = eligible & splits.partition.eq("train")
    later = eligible & splits.partition.ne("train")
    model = fit_decoder(inputs.loc[train], labels.loc[train, "target"], config())
    assert model.named_steps["scaler"].n_samples_seen_ == train.sum()
    np.testing.assert_allclose(
        model.named_steps["scaler"].mean_, inputs.loc[train].mean()
    )
    separate = fit_decoder(inputs.loc[train], labels.loc[train, "target"], config())
    np.testing.assert_array_equal(
        model.named_steps["logistic"].coef_, separate.named_steps["logistic"].coef_
    )
    before = model.named_steps["logistic"].coef_.copy()
    scaler_before = model.named_steps["scaler"].mean_.copy()
    predictions = predict_up(model, inputs.loc[later])
    np.testing.assert_array_equal(predictions, predict_up(separate, inputs.loc[later]))
    assert (
        np.isfinite(predictions).all()
        and ((predictions >= 0) & (predictions <= 1)).all()
    )
    np.testing.assert_array_equal(before, model.named_steps["logistic"].coef_)
    np.testing.assert_array_equal(scaler_before, model.named_steps["scaler"].mean_)
    with pytest.raises(ValueError, match="feature names"):
        predict_up(model, inputs.loc[later, list(reversed(RAW_COLUMNS))])
    changed = candles.copy()
    row = np.flatnonzero(later)[0]
    cutoff = decisions.iloc[row]
    changed.loc[changed.timestamp >= cutoff, ["open", "high", "low", "close"]] *= 3
    changed_inputs, _, _ = historical_inputs(changed, decisions)
    np.testing.assert_array_equal(
        predict_up(model, inputs.iloc[[row]]),
        predict_up(model, changed_inputs.iloc[[row]]),
    )
    # Ant features also use only completed candles, with the same frozen-model invariant.
    window = next(
        w for w in iter_historical_grids(candles) if w.decision_timestamp == cutoff
    )
    changed_window = next(
        w for w in iter_historical_grids(changed) if w.decision_timestamp == cutoff
    )
    assert list(iter_feature_rows([window])) == list(
        iter_feature_rows([changed_window])
    )
    with pytest.raises(ValueError, match="both target classes"):
        fit_decoder(
            inputs.loc[train], pd.Series(1, index=inputs.index[train]), config()
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("holding_period_bars", 3),
        ("maximum_concurrent_positions", 2),
        ("position_quantity_ounces", 0),
        ("train_fraction", 0.8),
        ("standard_scaler", False),
    ],
)
def test_research_configuration_validation(key, value):
    settings = config()
    settings[key] = value
    with pytest.raises(ValueError):
        validate_research_config(settings)
