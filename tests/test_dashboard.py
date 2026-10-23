"""Dashboard contracts, recorded replay, chart timestamps, and Streamlit controls."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from test_experiment import exported_history as exported_history  # noqa: PLC0414 -- pytest fixture registration

from anttrader.backtest import predictive_metrics, trading_metrics
from anttrader.dashboard import (
    STRATEGY_NAMES,
    comparison_table,
    equity_chart,
    format_metric,
    input_signature,
    load_research,
    price_chart,
    replay_state,
    selected_simulation,
)
from anttrader.decoder import sha256
from anttrader.experiment import OUTPUT_KEYS, STRATEGIES
from anttrader.features import checkpoint_features
from anttrader.grid import iter_historical_grids
from anttrader.pipeline import main
from anttrader.simulation import SimulationConfig, simulate


@pytest.fixture
def bundle(exported_history):
    path, settings, _, _ = exported_history
    main(path, stage="backtest")
    return path, settings


def save_result(path, settings, key, value):
    output = path.parent / settings[key]
    if isinstance(value, pd.DataFrame):
        value.to_csv(output, index=False)
    else:
        output.write_text(json.dumps(value), encoding="utf-8")
    metadata_path = path.parent / settings["experiment_metadata_output"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output_sha256"][key] = sha256(output)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_bundle_loads_utc_separate_partitions_and_metric_views(bundle):
    path, settings = bundle
    research = load_research(path)
    for key in OUTPUT_KEYS[:-1]:
        assert research.metadata["output_sha256"][key] == sha256(
            path.parent / settings[key]
        )
    assert str(research.signals.decision_timestamp.dt.tz) == "UTC"
    assert research.features.decision_timestamp.is_monotonic_increasing
    assert set(research.signals.partition) == {"validation", "test"}
    assert (
        research.equity.timestamp.duplicated().any()
    )  # Different strategies are valid.
    for partition in ("validation", "test"):
        table = comparison_table(research.metrics, partition)
        chart = equity_chart(research.equity, partition)
        assert list(table) == list(STRATEGY_NAMES.values())
        for strategy, trace in zip(STRATEGIES, chart.data, strict=True):
            stored = research.metrics["partitions"][partition][strategy]["trading"]
            assert trace.name == STRATEGY_NAMES[strategy]
            assert trace.y[0] == 0
            assert trace.y[-1] == pytest.approx(stored["total_gross_pnl_usd"])
            assert table.loc["Completed trades", trace.name] == format_metric(
                stored["completed_trades"]
            )
            selected = research.equity.loc[
                research.equity.partition.eq(partition)
                & research.equity.strategy.eq(strategy)
            ]
            assert list(trace.x) == list(selected.timestamp)
        assert partition.title() in chart.layout.title.text
    assert format_metric(None) == "Undefined"
    assert format_metric("infinity") == "Infinity"


@pytest.mark.parametrize(
    "key",
    [
        "historical_csv",
        "feature_output",
        "feature_metadata",
        *OUTPUT_KEYS,
    ],
)
def test_missing_files_name_path_and_recovery_command(bundle, key):
    path, settings = bundle
    missing = path.parent / settings.get(key, settings["feature_output"])
    if key == "feature_metadata":
        missing = missing.with_suffix(".metadata.json")
    before = input_signature(path)
    missing.unlink()
    assert input_signature(path) != before
    with pytest.raises(ValueError, match="Required file missing") as failure:
        load_research(path)
    assert str(missing) in str(failure.value)
    assert "--stage features" in str(failure.value)
    assert "--stage backtest" in str(failure.value)


@pytest.mark.parametrize(
    "change",
    [
        "legacy",
        "manifest_type",
        "config",
        "source",
        "feature",
        "feature_metadata",
        "history_count",
        "partition_count",
        "output_hash",
    ],
)
def test_incompatible_or_stale_metadata_is_rejected(bundle, change):
    path, settings = bundle
    output = path.parent / settings["experiment_metadata_output"]
    metadata = json.loads(output.read_text(encoding="utf-8"))
    if change == "legacy":
        del metadata["output_sha256"]
    elif change == "manifest_type":
        metadata["output_sha256"] = []
    elif change == "config":
        metadata["config"]["buy_threshold"] = 0.6
    elif change in ("source", "feature", "feature_metadata"):
        metadata[f"{change}_sha256"] = "0" * 64
    elif change == "history_count":
        metadata["candles"] -= 1
    elif change == "partition_count":
        metadata["partitions"]["test"]["eligible_rows"] -= 1
    else:
        metadata["output_sha256"]["signal_output"] = "0" * 64
    output.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot load research"):
        load_research(path)


@pytest.mark.parametrize(
    "change,key",
    [
        ("column", "signal_output"),
        ("duplicate", "signal_output"),
        ("naive", "signal_output"),
        ("unordered", "signal_output"),
        ("nonfinite", "signal_output"),
        ("probability", "signal_output"),
        ("signal", "signal_output"),
        ("target", "signal_output"),
        ("partial", "signal_output"),
        ("duplicate", "trade_output"),
        ("strategy", "trade_output"),
        ("pnl", "trade_output"),
        ("exit", "trade_output"),
        ("equity", "equity_output"),
        ("duplicate", "equity_output"),
    ],
)
def test_malformed_and_partial_result_contracts_are_rejected(bundle, change, key):
    path, settings = bundle
    output = path.parent / settings[key]
    frame = pd.read_csv(output)
    if change == "column":
        frame = frame.drop(columns="ant_p_up")
    elif change == "duplicate":
        frame = pd.concat([frame.iloc[:1], frame], ignore_index=True)
    elif change == "naive":
        frame.loc[0, "decision_timestamp"] = "2024-01-01 00:00:00"
    elif change == "unordered":
        frame = frame.iloc[::-1]
    elif change == "nonfinite":
        frame.loc[0, "ant_p_up"] = np.inf
    elif change == "probability":
        frame.loc[0, "ant_p_up"] = 1.1
    elif change == "signal":
        frame.loc[0, "ant_signal"] = "INVALID"
    elif change == "target":
        frame.loc[0, "target"] = 1 - frame.loc[0, "target"]
    elif change == "partial":
        frame = frame.iloc[:-1]
    elif change == "strategy":
        frame.loc[0, "strategy"] = "unknown"
    elif change == "pnl":
        frame.loc[0, "gross_pnl_usd"] += 10
    elif change == "exit":
        frame.loc[0, "exit_timestamp"] = frame.loc[0, "entry_timestamp"]
    else:
        frame.loc[len(frame) - 1, "cumulative_gross_pnl_usd"] += 10
    # Re-sign intentionally to exercise semantic checks, not only file identity.
    save_result(path, settings, key, frame)
    with pytest.raises(ValueError, match="Cannot load research"):
        load_research(path)


def test_mixed_outputs_and_incorrect_stored_metrics_are_rejected(bundle):
    path, settings = bundle
    output = path.parent / settings["signal_output"]
    output.write_bytes(output.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="output SHA-256"):
        load_research(path)
    metadata_path = path.parent / settings["experiment_metadata_output"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output_sha256"]["signal_output"] = sha256(output)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    metrics = json.loads(
        (path.parent / settings["metrics_output"]).read_text(encoding="utf-8")
    )
    metrics["partitions"]["test"]["ant"]["trading"]["completed_trades"] += 1
    save_result(path, settings, "metrics_output", metrics)
    with pytest.raises(ValueError, match="completed_trades"):
        load_research(path)


@pytest.mark.parametrize("empty_strategies", [("ant",), STRATEGIES])
def test_no_trade_bundle_keeps_zero_baseline_and_undefined_metrics(
    bundle, empty_strategies
):
    path, settings = bundle
    research = load_research(path)
    signals = research.signals.copy()
    trades = research.trades.loc[~research.trades.strategy.isin(empty_strategies)]
    curves = [research.equity.loc[~research.equity.strategy.isin(empty_strategies)]]
    for strategy in empty_strategies:
        signals[f"{strategy}_signal"] = "HOLD"
        if strategy != "momentum":
            signals[f"{strategy}_p_up"] = 0.5
        for partition in ("validation", "test"):
            selected = signals.loc[signals.partition.eq(partition)]
            curve = pd.DataFrame(
                {
                    "timestamp": [selected.decision_timestamp.iloc[0]],
                    "cumulative_gross_pnl_usd": [0.0],
                    "partition": [partition],
                    "strategy": [strategy],
                }
            )
            curves.append(curve)
            research.metrics["partitions"][partition][strategy] = {
                "predictive": predictive_metrics(
                    selected.target,
                    selected[f"{strategy}_signal"],
                    None if strategy == "momentum" else selected[f"{strategy}_p_up"],
                ),
                "trading": trading_metrics(
                    trades.loc[trades.strategy.eq(strategy)], curve
                ),
            }
    equity = pd.concat(curves).sort_values(["timestamp", "strategy"], kind="stable")
    for key, value in (
        ("signal_output", signals),
        ("trade_output", trades),
        ("equity_output", equity),
        ("metrics_output", research.metrics),
    ):
        save_result(path, settings, key, value)
    loaded = load_research(path)
    table = comparison_table(loaded.metrics, "test")
    assert table.loc["Win rate", "AntTrader"] == "Undefined"
    assert equity_chart(loaded.equity, "test").data[0].y == (0.0,)


def test_selected_grid_replay_checkpoints_and_future_independence(bundle):
    path, _ = bundle
    research = load_research(path)
    original = research.candles.copy(deep=True)
    decision = research.features.decision_timestamp.iloc[3]
    window, result = selected_simulation(research, decision)
    expected = list(iter_historical_grids(research.candles, as_of=decision))[-1]
    np.testing.assert_array_equal(window.grid, expected.grid)
    initial = replay_state(result, 0)
    np.testing.assert_array_equal(initial.grid, window.grid)
    assert (initial.row, initial.col, initial.heading) == (15, 0, "north")
    assert initial.visit_counts.sum() == initial.revisits == 0
    for step in (1, 8, 137, 256, 512, 1024, 2048):
        state = replay_state(result, step)
        expected_state = simulate(
            window.grid, replace(result.config, steps=step, checkpoints=(step,))
        ).checkpoints[0]
        for name in state.__dataclass_fields__:
            np.testing.assert_array_equal(
                getattr(state, name), getattr(expected_state, name)
            )
        assert (state.row, state.col, state.direction) == result.trajectory[step]
        assert state.visit_counts.sum() == step
        if step in result.config.checkpoints:
            saved = research.features.loc[
                research.features.decision_timestamp.eq(decision)
            ].iloc[0]
            for name, value in checkpoint_features(state).items():
                assert saved[f"s{step:04d}_{name}"] == pytest.approx(value)
    pd.testing.assert_frame_equal(research.candles, original)
    research.candles.loc[
        research.candles.timestamp.ge(decision), ["open", "high", "low", "close"]
    ] *= 2
    _, unchanged = selected_simulation(research, decision)
    assert unchanged.trajectory == result.trajectory
    with pytest.raises(ValueError, match="saved historical"):
        selected_simulation(research, "2000-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="outside"):
        replay_state(result, 2049)
    with pytest.raises(ValueError, match="recorded trajectory"):
        replay_state(simulate(window.grid), 1)
    research.features.loc[3, "s0256_final_row"] += 1
    with pytest.raises(ValueError, match="checkpoint features"):
        selected_simulation(research, decision)


def test_recorded_replay_reconstructs_wrapped_edges():
    result = simulate(
        np.zeros((16, 16)),
        SimulationConfig(steps=8, checkpoints=(8,)),
        record_trajectory=True,
    )
    # Includes east, south, west and north movement, with wrapped arrivals.
    for step, expected in ((1, (15, 1)), (2, (0, 1)), (5, (15, 15)), (7, (14, 0))):
        state = replay_state(result, step)
        assert (state.row, state.col) == expected
    np.testing.assert_array_equal(
        replay_state(result, 8).grid, result.checkpoints[-1].grid
    )


def test_price_markers_use_saved_decision_entry_exit_times(bundle):
    path, _ = bundle
    research = load_research(path)
    signals = research.signals.loc[research.signals.partition.eq("test")]
    trades = research.trades.loc[
        research.trades.partition.eq("test") & research.trades.strategy.eq("ant")
    ]
    chart = price_chart(research.candles, signals, trades, "ant", "test")
    assert chart.data[0].type == "candlestick"
    for signal, pred_index, entry_index in (("BUY", 1, 2), ("SELL", 3, 4)):
        predictions = signals.loc[signals.ant_signal.eq(signal)]
        entries = trades.loc[trades.signal.eq(signal)]
        assert list(chart.data[pred_index].x) == list(predictions.decision_timestamp)
        assert list(chart.data[entry_index].x) == list(entries.entry_timestamp)
        np.testing.assert_array_equal(chart.data[pred_index].y, predictions.entry_price)
        np.testing.assert_array_equal(chart.data[entry_index].y, entries.entry_price)
        assert chart.data[pred_index].marker.size < chart.data[entry_index].marker.size
    assert list(chart.data[-1].x) == list(trades.exit_timestamp)
    np.testing.assert_array_equal(chart.data[-1].y, trades.exit_price)
    assert all("HOLD" not in trace.name for trace in chart.data)
    assert sum(len(chart.data[i].x) for i in (2, 4)) == len(trades)


def test_streamlit_controls_and_missing_files(bundle):
    path, settings = bundle
    app = AppTest.from_string(
        f"from pathlib import Path\nfrom app import main\nmain(Path({str(path)!r}))",
        default_timeout=30,
    ).run()
    assert not app.exception and not app.error
    assert [tab.label for tab in app.tabs] == [
        "Ant Simulation",
        "Gold & Signals",
        "Backtest & Comparison",
    ]
    assert app.selectbox(key="performance_partition").value == "Test"
    assert any(item.value == "Latest Historical Signal" for item in app.subheader)
    assert app.slider(key="replay_step").value == 0
    app.slider(key="replay_step").set_value(137).run()
    assert not app.exception and app.slider(key="replay_step").value == 137
    app.selectbox(key="checkpoint_step").set_value(2048).run()
    assert not app.exception and app.slider(key="replay_step").value == 2048
    app.selectbox(key="performance_partition").select("Validation").run()
    assert (
        not app.exception
        and app.selectbox(key="performance_partition").value == "Validation"
    )
    app.selectbox(key="price_partition").select("Validation").run()
    assert not app.exception
    end = app.date_input[0].value[-1]
    app.date_input[0].set_value((end, end)).run()
    assert not app.exception
    app.date_input[0].set_value((end,)).run()
    assert not app.exception
    assert any("Select both the start and end date" in item.value for item in app.info)
    (path.parent / settings["signal_output"]).unlink()
    app.run()
    assert not app.exception
    assert "signals.csv" in app.error[0].value
    assert "--stage backtest" in app.info[0].value
    assert len(app.metric) == 0


def test_malformed_feature_metadata_has_readable_error(bundle):
    path, settings = bundle
    feature_metadata = (path.parent / settings["feature_output"]).with_suffix(
        ".metadata.json"
    )
    metadata = json.loads(feature_metadata.read_text(encoding="utf-8"))
    metadata["config"] = []
    feature_metadata.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="config and simulation must be objects"):
        load_research(path)
