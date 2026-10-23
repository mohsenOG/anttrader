"""Small lab checks: deterministic grids, common samples, persistence, and UI."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest
from test_decoder import config
from test_decoder import history as synthetic_history

from anttrader.dashboard import replay_state, selected_simulation
from anttrader.decoder import historical_inputs, signal_metrics
from anttrader.experiments import (
    ExperimentConfig,
    as_research,
    batch_configs,
    checkpoints,
    comparison_rows,
    compatible,
    delete_result,
    digest,
    experiment_root,
    history,
    load_result,
    mark_candidate,
    prepare_plan,
    rerun_result,
    run_plan,
)
from anttrader.features import extract_features, feature_columns
from anttrader.grid import closes_to_grid, serpentine_grid
from anttrader.lab_dashboard import heatmap, overlay_chart
from anttrader.simulation import SimulationConfig, simulate


@pytest.fixture
def lab_config(tmp_path):
    settings = config()
    settings["historical_csv"] = "sample.csv"
    settings["experiment_lab"] = {"sample_candles": 430}
    synthetic_history(460).drop(index=[180, 370]).to_csv(
        tmp_path / "sample.csv", index=False
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return path


def test_all_supported_checkpoints():
    for steps in range(4, 16385):
        points = checkpoints(steps)
        assert (
            len(points) == 4
            and 0 < points[0] < points[1] < points[2] < points[3] == steps
        )
    assert checkpoints(2048) == (256, 512, 1024, 2048)
    assert checkpoints(5) == (1, 2, 3, 5)


@pytest.mark.parametrize(
    "rows,columns", [(8, 8), (16, 6), (16, 16), (20, 20), (3, 7), (2, 2)]
)
def test_rectangular_mapping_simulation_and_replay(rows, columns):
    settings = ExperimentConfig(rows, columns, 33, "west")
    indices = serpentine_grid(np.arange(settings.cells), (rows, columns))
    pose = settings.simulation
    assert indices[pose.start_row, pose.start_col] == settings.cells - 1
    center = replace(settings, start="center").simulation
    assert (center.start_row, center.start_col) == (rows // 2, columns // 2)
    grid = closes_to_grid(
        np.exp(np.sin(np.arange(settings.cells + 1)) / 100), (rows, columns)
    )
    before = grid.copy()
    result = simulate(grid, pose, record_trajectory=True)
    np.testing.assert_array_equal(grid, before)
    assert len(feature_columns(pose)) == 60
    assert list(extract_features(result)) == feature_columns(pose)
    for state in result.checkpoints:
        assert state.row == (pose.start_row + state.vertical_displacement) % rows
        assert state.col == (pose.start_col + state.horizontal_displacement) % columns
        assert state.visit_counts.sum() == state.steps
        replay = replay_state(result, state.steps)
        np.testing.assert_array_equal(replay.grid, state.grid)
        assert replay.row == state.row and replay.col == state.col
    values = extract_features(result)
    for key, value in values.items():
        if key.endswith(("fraction", "entropy", "concentration")):
            assert -1e-12 <= value <= 1 + 1e-12
    assert values == extract_features(simulate(grid, pose))
    with pytest.raises(ValueError, match="outside"):
        simulate(grid, replace(pose, start_col=columns))


@pytest.mark.parametrize(
    "changes",
    [
        {"rows": 1},
        {"columns": 33},
        {"rows": True},
        {"steps": 3},
        {"steps": 16385},
        {"steps": 8.5},
        {"start": "unknown"},
        {"direction": "up"},
    ],
)
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        ExperimentConfig(**changes)


def test_matched_history_and_future_independence():
    candles = synthetic_history(250)
    decisions = candles.timestamp.iloc[100:110] + pd.Timedelta(hours=1)
    for lookback in (64, 96):
        raw, _, _ = historical_inputs(candles, decisions, lookback=lookback)
        assert raw.shape == (10, lookback)
        changed = candles.copy()
        changed.loc[111:, "close"] *= 2
        pd.testing.assert_frame_equal(
            raw, historical_inputs(changed, decisions, lookback=lookback)[0]
        )


def test_metrics_side_accuracy_brier_and_undefined():
    m = signal_metrics(
        [1, 0, 1, 0], ["BUY", "SELL", "HOLD", "BUY"], [0.8, 0.2, 0.5, 0.8]
    )
    assert m["signal_coverage"] == 0.75
    assert m["buy_accuracy"] == 0.5 and m["sell_accuracy"] == 1
    assert m["actionable_accuracy"] == pytest.approx(2 / 3)
    assert m["brier_score"] == pytest.approx((0.04 + 0.04 + 0.25 + 0.64) / 4)
    empty = signal_metrics([1], ["HOLD"], [0.5])
    assert empty["roc_auc"] is None and empty["actionable_accuracy"] is None
    assert empty["buy_accuracy"] is None and empty["sell_accuracy"] is None


def test_batch_common_training_samples_independent_decoders_cache_rerun_and_delete(
    lab_config, monkeypatch
):
    import anttrader.experiment as evaluation

    original = evaluation.fit_decoder
    fits = []

    def checked(inputs, targets, settings):
        model = original(inputs, targets, settings)
        fits.append((inputs.copy(), targets.copy(), model))
        np.testing.assert_allclose(model.named_steps["scaler"].mean_, inputs.mean())
        return model

    monkeypatch.setattr(evaluation, "fit_decoder", checked)
    configs = batch_configs([(8, 8), (16, 6)], [16, 32])
    plan = prepare_plan(lab_config, configs)
    source = lab_config.parent / "sample.csv"
    unchanged = source.read_bytes()
    statuses = []
    results = run_plan(
        plan, progress=lambda i, n, c, stage: statuses.append((i, n, stage))
    )
    assert all("error" not in r for r in results), results
    assert len(results) == 4 and len(fits) == 8
    assert compatible(results)
    for inputs, targets, model in fits:
        pd.testing.assert_index_equal(inputs.index, fits[0][0].index)
        pd.testing.assert_series_equal(targets, fits[0][1])
        assert model.named_steps["logistic"].random_state == 42
    assert [len(fits[i][0].columns) for i in (1, 3, 5, 7)] == [64, 64, 96, 96]
    assert len({id(v[2]) for v in fits}) == 8
    assert any(stage == "Feature generation" for _, _, stage in statuses)
    reference = results[0]
    experiment_id = reference["metadata"]["id"]
    root = plan["root"]
    for partition in ("train", "validation"):
        boundary = reference["metadata"]["partitions"][partition]
        assert pd.Timestamp(boundary["latest_eligible_label_end"]) < pd.Timestamp(
            boundary["exclusive_outcome_cutoff"]
        )
    for result in results:
        pd.testing.assert_series_equal(
            reference["signals"].decision_timestamp,
            result["signals"].decision_timestamp,
        )
        assert result["metadata"]["partitions"] == reference["metadata"]["partitions"]
    saved = (root / experiment_id / "predictions.csv").read_bytes()
    for result in run_plan(plan):
        assert result["cached"]
    assert len(fits) == 8
    mark_candidate(root, experiment_id, True)
    rerun = rerun_result(root, experiment_id)
    assert "error" not in rerun, rerun
    assert len(fits) == 10
    assert (root / experiment_id / "predictions.csv").read_bytes() == saved
    assert (root / experiment_id / "candidate.json").is_file()
    view = as_research(load_result(root, experiment_id))
    window, replay = selected_simulation(
        view, view.features.decision_timestamp.iloc[-1]
    )
    assert window.grid.shape == (8, 8) and replay.config.steps == 16
    table = comparison_rows(results)
    assert table["ROC AUC"].dropna().is_monotonic_decreasing
    assert len(heatmap(table, "ROC AUC").data[0].z) == 2
    chart = overlay_chart(
        view.candles,
        results[:2],
        "validation",
        reference["signals"].decision_timestamp.iloc[0],
        reference["signals"].decision_timestamp.iloc[-1],
    )
    assert len(chart.data) >= 3
    with pytest.raises(ValueError, match="confirmation"):
        delete_result(root, experiment_id)
    with pytest.raises(ValueError, match="ID"):
        delete_result(root, "../sample.csv", confirmed=True)
    foreign = root / experiment_id / "user.txt"
    foreign.write_text("keep")
    with pytest.raises(ValueError, match="unrecognized"):
        delete_result(root, experiment_id, confirmed=True)
    assert (
        foreign.read_text() == "keep"
        and (root / experiment_id / "predictions.csv").exists()
    )
    foreign.unlink()
    delete_result(root, experiment_id, confirmed=True)
    assert not (root / experiment_id).exists()
    assert len(history(root)) == 3 and source.read_bytes() == unchanged


def test_incompatible_cohorts_failures_and_safe_cache(lab_config, monkeypatch):
    first_plan = prepare_plan(lab_config, [ExperimentConfig(8, 8, 8)])
    first = run_plan(first_plan)[0]
    second = run_plan(prepare_plan(lab_config, [ExperimentConfig(16, 6, 8)]))[0]
    assert not compatible([first, second])
    with pytest.raises(ValueError, match="not directly comparable"):
        comparison_rows([first, second])

    def failed(*args, **kwargs):
        raise RuntimeError("deliberate small test failure")

    monkeypatch.setattr("anttrader.experiments.evaluate_features", failed)
    failed_plan = prepare_plan(lab_config, [ExperimentConfig(8, 8, 9)])
    failure = run_plan(failed_plan)[0]
    assert "deliberate" in failure["error"]
    directory = failed_plan["root"] / failure["id"]
    assert json.loads((directory / "metadata.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="incomplete"):
        load_result(failed_plan["root"], failure["id"])
    old = (
        first_plan["root"] / first["metadata"]["id"] / "predictions.csv"
    ).read_bytes()
    assert "error" in run_plan(first_plan, rerun=True)[0]
    assert (
        first_plan["root"] / first["metadata"]["id"] / "predictions.csv"
    ).read_bytes() == old
    directory = first_plan["root"] / first["metadata"]["id"]
    (directory / "predictions.csv").write_text("corrupt")
    assert "explicit rerun" in run_plan(first_plan)[0]["error"]
    with pytest.raises(ValueError, match="artifact"):
        load_result(first_plan["root"], first["metadata"]["id"])


def test_plan_limits_identity_and_no_source_overwrite(lab_config):
    assert (
        prepare_plan(lab_config, [ExperimentConfig(8, 8, 8)])["identities"]
        == prepare_plan(lab_config, [ExperimentConfig(8, 8, 8)])["identities"]
    )
    with pytest.raises(ValueError, match="30"):
        prepare_plan(lab_config, [ExperimentConfig(8, 8, n) for n in range(4, 35)])
    with pytest.raises(ValueError, match="sample_candles"):
        prepare_plan(lab_config, sample_candles=0)
    settings = yaml.safe_load(lab_config.read_text())
    settings["experiment_lab"]["output_dir"] = "data"
    with pytest.raises(ValueError, match="within"):
        experiment_root(lab_config, settings)
    plan = prepare_plan(lab_config, [ExperimentConfig(8, 8, 8)])
    assert digest(plan["identities"][0]) != digest(
        prepare_plan(lab_config, [ExperimentConfig(8, 8, 9)])["identities"][0]
    )


def test_lab_controls_single_batch_reset_open_rerun_delete(lab_config):
    app = AppTest.from_string(
        f"from pathlib import Path\nfrom app import main\nmain(Path({str(lab_config)!r}))",
        default_timeout=30,
    ).run()
    assert not app.exception
    app.selectbox(key="lab_grid").set_value("8 x 8")
    app.selectbox(key="lab_step_preset").set_value("Custom")
    app.number_input(key="lab_steps").set_value(16)
    app.checkbox(key="lab_limited").set_value(True)
    app.number_input(key="lab_sample").set_value(430)
    app.button(key="lab_run_single").click().run()
    assert not app.exception and not any(
        "Experiment failed" in e.value for e in app.error
    )
    assert any("Completed:" in s.value for s in app.success)
    assert any("Estimated workload" in s.value for s in app.info)
    first_id = app.selectbox(key="lab_saved").value
    app.button(key="lab_open").click().run()
    assert not app.exception and not app.error
    assert app.slider(key="replay_step").max == 16
    app.selectbox(key="checkpoint_step").set_value(16).run()
    assert app.slider(key="replay_step").value == 16 and not app.exception
    app.button(key="lab_rerun").click().run()
    assert not app.exception and not app.error
    app.button(key="lab_reset").click().run()
    assert not app.exception and app.selectbox(key="lab_grid").value == "16 x 16"
    assert app.selectbox(key="lab_step_preset").value == 2048
    app.multiselect(key="lab_grids").set_value(["8 x 8", "16 x 6"])
    app.multiselect(key="lab_batch_steps").set_value([256])
    app.button(key="lab_run_batch").click().run()
    assert not app.exception and not app.error
    assert len(app.selectbox(key="lab_saved").options) == 3
    app.selectbox(key="lab_saved").set_value(first_id).run()
    app.checkbox(key=f"lab_confirm_delete_{first_id}").set_value(True).run()
    app.button(key="lab_delete").click().run()
    assert not app.exception
    assert len(app.selectbox(key="lab_saved").options) == 2


def test_original_reference_simulator_regression():
    # Golden digests from the committed v1 simulator on this fixed binary grid.
    import hashlib

    grid = (np.arange(256).reshape(16, 16) % 3 == 0).astype(np.uint8)
    result = simulate(grid)
    values = np.round(list(extract_features(result).values()), 12).astype("<f8")
    assert (
        hashlib.sha256(values.tobytes()).hexdigest()
        == "a37e9d5a80b53291db258eece787e9fe85f9ae7dbb0a371775df0d98017cb144"
    )
    states = b"".join(
        s.grid.tobytes() + s.visit_counts.astype("<i8").tobytes()
        for s in result.checkpoints
    )
    assert (
        hashlib.sha256(states).hexdigest()
        == "9e4211eca52bccd9287cc2fefa47b41fe1e1ce9b65a16c8c6913b23355b565ea"
    )


def test_future_test_prices_do_not_change_validation_predictions(lab_config):
    config = ExperimentConfig(8, 8, 16)
    before = run_plan(prepare_plan(lab_config, [config]))[0]
    source = lab_config.parent / "sample.csv"
    candles = pd.read_csv(source)
    # Change only the test tail, preserving consistent OHLC values.
    candles.loc[410:, ["open", "high", "low", "close"]] *= 1.05
    candles.to_csv(source, index=False)
    after = run_plan(prepare_plan(lab_config, [config]))[0]
    for method in ("ant", "raw"):
        np.testing.assert_array_equal(
            before["signals"].loc[
                lambda f: f.partition.eq("validation"), f"{method}_p_up"
            ],
            after["signals"].loc[
                lambda f: f.partition.eq("validation"), f"{method}_p_up"
            ],
        )


def test_failed_run_can_be_rerun_and_metadata_tampering_is_rejected(
    lab_config, monkeypatch
):
    from anttrader.experiments import evaluate_features

    plan = prepare_plan(lab_config, [ExperimentConfig(8, 8, 16)])

    def fail(*args, **kwargs):
        raise ValueError("failed fixture")

    monkeypatch.setattr("anttrader.experiments.evaluate_features", fail)
    failed = run_plan(plan)[0]
    monkeypatch.setattr("anttrader.experiments.evaluate_features", evaluate_features)
    result = rerun_result(plan["root"], failed["id"])
    assert "error" not in result
    path = plan["root"] / result["metadata"]["id"] / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["partitions"]["train"]["eligible_rows"] += 1
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        load_result(plan["root"], failed["id"])


@pytest.mark.parametrize(
    "row,col,direction,expected",
    [
        (0, 0, "west", (15, 0)),
        (0, 5, "north", (0, 0)),
        (15, 0, "east", (0, 0)),
        (0, 0, "south", (0, 5)),
    ],
)
def test_rectangular_wrapped_edges(row, col, direction, expected):
    state = simulate(
        np.zeros((16, 6)),
        SimulationConfig(
            steps=1,
            checkpoints=(1,),
            start_row=row,
            start_col=col,
            initial_direction=direction,
        ),
    ).checkpoints[0]
    assert (state.row, state.col) == expected


def test_single_cell_entropy_edge():
    result = simulate(
        np.zeros((1, 1)), SimulationConfig(steps=4, checkpoints=(4,), start_row=0)
    )
    assert extract_features(result)["s0004_visit_entropy"] == 0


def test_corrupt_history_remains_visible_without_breaking_dashboard(lab_config):
    plan = prepare_plan(lab_config, [ExperimentConfig(8, 8, 16)])
    result = run_plan(plan)[0]
    directory = plan["root"] / result["metadata"]["id"]
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    del metadata["cohort_id"]
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert history(plan["root"])[0]["status"] == "invalid"
    app = AppTest.from_string(
        f"from pathlib import Path\nfrom app import main\nmain(Path({str(lab_config)!r}))",
        default_timeout=30,
    ).run()
    assert not app.exception
