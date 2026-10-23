"""Dashboard contracts, recorded replay, chart timestamps, and Streamlit controls."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from test_experiment import exported_history as exported_history  # noqa: PLC0414 -- pytest fixture registration

from anttrader.dashboard import (
    STRATEGY_NAMES,
    comparison_table,
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
    main(path, stage="signals")
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


def test_bundle_loads_utc_separate_partitions_and_metric_views(bundle, monkeypatch):
    def forbidden_fit(*args, **kwargs):
        pytest.fail("The dashboard must not train a model.")

    monkeypatch.setattr("anttrader.experiment.fit_decoder", forbidden_fit)
    path, settings = bundle
    research = load_research(path)
    for key in OUTPUT_KEYS[:-1]:
        assert research.metadata["output_sha256"][key] == sha256(
            path.parent / settings[key]
        )
    assert str(research.signals.decision_timestamp.dt.tz) == "UTC"
    assert research.features.decision_timestamp.is_monotonic_increasing
    assert set(research.signals.partition) == {"validation", "test"}
    for partition in ("validation", "test"):
        table = comparison_table(research.metrics, partition)
        selected = research.signals.loc[research.signals.partition.eq(partition)]
        assert list(table) == list(STRATEGY_NAMES.values())
        for strategy in STRATEGIES:
            stored = research.metrics["partitions"][partition][strategy]["predictive"]
            assert table.loc[
                "Total signals", STRATEGY_NAMES[strategy]
            ] == format_metric(len(selected))
            assert sum(stored["signal_counts"].values()) == len(selected)
    assert format_metric(None) == "Undefined"
    assert not hasattr(research, "trades") and not hasattr(research, "equity")
    assert not (path.parent / "results/trades.csv").exists()
    assert not (path.parent / "results/equity.csv").exists()
    before = input_signature(path)
    # Unused legacy files must not enter the loader or its cache signature.
    (path.parent / "results/trades.csv").write_text("obsolete")
    (path.parent / "results/equity.csv").write_text("obsolete")
    assert input_signature(path) == before
    load_research(path)


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
    assert "--stage signals" in str(failure.value)


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
        ("reference", "signal_output"),
        ("partition", "signal_output"),
        ("momentum", "signal_output"),
        ("label_end", "signal_output"),
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
    elif change == "reference":
        frame.loc[0, "reference_price"] += 1
    elif change == "partition":
        frame.loc[0, "partition"] = "test"
    elif change == "momentum":
        frame.loc[0, "momentum_signal"] = "INVALID"
    elif change == "label_end":
        frame.loc[0, "label_end_timestamp"] = frame.loc[0, "decision_timestamp"]
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
    metrics["partitions"]["test"]["ant"]["predictive"]["signal_counts"]["BUY"] += 1
    save_result(path, settings, "metrics_output", metrics)
    with pytest.raises(ValueError, match="BUY"):
        load_research(path)


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


def test_price_markers_use_decision_times_and_only_completed_candles(bundle):
    path, _ = bundle
    research = load_research(path)
    signals = research.signals.loc[research.signals.partition.eq("test")].copy()
    # Exercise both sides even if a small trained dataset contains mostly HOLD.
    signals["ant_signal"] = np.where(np.arange(len(signals)) % 2, "BUY", "SELL")
    chart = price_chart(research.candles, signals, "ant", "test")
    assert chart.data[0].type == "candlestick"
    assert chart.layout.height == 800
    for signal, index in (("BUY", 1), ("SELL", 2)):
        predictions = signals.loc[signals.ant_signal.eq(signal)]
        trace = chart.data[index]
        assert list(trace.x) == list(predictions.decision_timestamp)
        reference = research.candles.set_index("timestamp").reindex(
            predictions.decision_timestamp - pd.Timedelta(hours=1)
        )
        assert (
            (np.asarray(trace.y) < reference.low.to_numpy()).all()
            if signal == "BUY"
            else (np.asarray(trace.y) > reference.high.to_numpy()).all()
        )
        assert trace.marker.size == 16
        assert "UTC" in trace.hovertemplate and "P(up)" in trace.hovertemplate
        assert (
            "Partition" in trace.hovertemplate
            and "Last completed close" in trace.hovertemplate
        )
        assert [row[1] for row in trace.customdata] == list(predictions.reference_price)
        assert [row[2] for row in trace.customdata] == ["test"] * len(predictions)
    assert all("HOLD" not in trace.name for trace in chart.data)
    one = signals.iloc[[0]]
    cutoff = one.decision_timestamp.iloc[0]
    before = price_chart(research.candles, one, "ant", "test")
    changed = research.candles.copy()
    changed.loc[changed.timestamp.ge(cutoff), ["open", "high", "low", "close"]] *= 10
    after = price_chart(changed, one, "ant", "test")
    for index in (1, 2):
        np.testing.assert_array_equal(before.data[index].y, after.data[index].y)
        assert before.data[index].x == after.data[index].x
        assert before.data[index].customdata == after.data[index].customdata
    filtered = price_chart(
        research.candles, research.signals, "ant", "validation", visible_sides=("SELL",)
    )
    assert len(filtered.data) == 2 and filtered.data[1].name == "SELL prediction"
    assert {row[2] for row in filtered.data[1].customdata} <= {"validation"}
    # The first visible decision still uses the preceding completed candle.
    boundary = price_chart(
        research.candles,
        one,
        "ant",
        "test",
        start=cutoff,
        stop=cutoff + pd.Timedelta(hours=1),
    )
    assert boundary.data[1].x == before.data[1].x
    assert boundary.data[2].x == before.data[2].x


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
        "Experiment Lab",
    ]
    assert app.selectbox(key="price_partition").value == "Test"
    assert any(item.value == "Latest Historical Signal" for item in app.subheader)
    assert app.slider(key="replay_step").value == 0
    app.slider(key="replay_step").set_value(137).run()
    assert not app.exception and app.slider(key="replay_step").value == 137
    app.selectbox(key="checkpoint_step").set_value(2048).run()
    assert not app.exception and app.slider(key="replay_step").value == 2048
    app.selectbox(key="price_partition").select("Validation").run()
    assert not app.exception
    app.multiselect(key="visible_sides").set_value(["BUY"]).run()
    assert not app.exception
    app.multiselect(key="visible_sides").set_value([]).run()
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
    assert any("--stage signals" in item.value for item in app.info)
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
