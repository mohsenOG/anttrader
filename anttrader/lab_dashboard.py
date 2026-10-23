"""Streamlit controls and comparison views for the shared local experiment engine."""

from itertools import combinations
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from anttrader.dashboard import price_chart
from anttrader.experiments import (
    GRID_PRESETS,
    STEP_PRESETS,
    ExperimentConfig,
    as_research,
    batch_configs,
    comparison_rows,
    compatible,
    delete_result,
    digest,
    disagreement,
    experiment_root,
    history,
    load_result,
    mark_candidate,
    prepare_plan,
    rerun_result,
    run_plan,
)

DEFAULT_CONTROLS = {
    "lab_grid": "16 x 16",
    "lab_rows": 16,
    "lab_columns": 16,
    "lab_step_preset": 2048,
    "lab_steps": 2048,
    "lab_direction": "north",
    "lab_start": "newest_return",
    "lab_grids": ["16 x 16"],
    "lab_batch_steps": [2048],
    "lab_custom_step": False,
}


def reset_controls():
    st.session_state.update(DEFAULT_CONTROLS)


def _clear_replay():
    for key in ("replay_step", "checkpoint_step"):
        st.session_state.pop(key, None)


def _signature(root, experiment_id):
    directory = Path(root) / experiment_id
    files = [
        directory / name
        for name in ("metadata.json", "features.csv", "predictions.csv", "metrics.json")
    ]
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_result(root, experiment_id, signature):
    return load_result(root, experiment_id)


@st.cache_data(show_spinner=False, max_entries=2)
def cached_view(_result, signature):
    return as_research(_result)


def _load(root, experiment_id):
    return cached_result(str(root), experiment_id, _signature(root, experiment_id))


def heatmap(table, metric):
    if table.duplicated(["Grid", "Steps"]).any():
        raise ValueError(
            "Filter direction and starting position before displaying the heatmap."
        )
    matrix = table.pivot(index="Grid", columns="Steps", values=metric).sort_index(
        axis=1
    )
    return go.Figure(
        go.Heatmap(
            x=[str(v) for v in matrix.columns],
            y=matrix.index,
            z=matrix.to_numpy(),
            colorbar_title=metric,
            colorscale="Viridis",
            hoverongaps=False,
        )
    ).update_layout(xaxis_title="Simulation steps", yaxis_title="Grid (rows x columns)")


def overlay_chart(candles, results, partition, start, stop):
    if not 1 <= len(results) <= 3 or not compatible(results):
        raise ValueError("Select one to three compatible experiments for the overlay.")
    figure = None
    colors = ("#16865d", "#9056d9", "#1574c4")
    for result, color in zip(results, colors, strict=False):
        signals = result["signals"].loc[
            lambda f: (
                f.partition.eq(partition)
                & f.decision_timestamp.ge(start)
                & f.decision_timestamp.lt(stop)
            )
        ]
        chart = price_chart(candles, signals, "ant", partition, start=start, stop=stop)
        if figure is None:
            figure = go.Figure(chart.data[:1], layout=chart.layout)
        for trace in chart.data[1:]:
            trace.name = f"{result['metadata']['id'][:8]} {trace.name}"
            trace.marker.color = color
            figure.add_trace(trace)
    return figure


def _report_results(results):
    for result in results:
        if "error" in result:
            st.error(f"{result['id']}: {result['error']}")
        elif result.get("cached"):
            st.info(f"Cached result loaded: {result['metadata']['id']}")
        else:
            st.success(f"Completed: {result['metadata']['id']}")


def _comparisons(root, metadata):
    completed = [m for m in metadata if m.get("status") == "completed"]
    if not completed:
        return
    st.subheader("Compare saved experiments")
    st.caption(
        f"{len(completed)} completed experiments. Testing many configurations increases "
        "the chance of a high score by luck. Rank candidates using validation only. "
        "Predictive improvement requires fresh forward data or an untouched period."
    )
    cohorts = list(dict.fromkeys(m["cohort_id"] for m in completed))
    cohort = st.selectbox(
        "Comparison cohort", cohorts, format_func=lambda v: v[:16], key="lab_cohort"
    )
    if len(cohorts) > 1:
        st.info(
            "Different cohorts are not directly comparable. Rankings include the selected cohort only."
        )
    cohort_members = [m for m in completed if m["cohort_id"] == cohort]
    detail = cohort_members[0]
    st.caption(
        f"Common lookback: {detail['cohort']['lookback']} returns. "
        f"Completed candles: {detail['dataset']['candles']:,}. "
        f"Dataset cutoff (UTC): {detail['dataset']['cutoff']}."
    )
    versions = list(
        dict.fromkeys(digest(m["identity"]["software"]) for m in cohort_members)
    )
    version = versions[0]
    if len(versions) > 1:
        st.info(
            "Software versions differ. Comparisons include the selected version only."
        )
        version = st.selectbox(
            "Comparison software version",
            versions,
            format_func=lambda v: v[:16],
            key=f"lab_version_{cohort}",
        )
    results = []
    for meta in cohort_members:
        if digest(meta["identity"]["software"]) == version:
            try:
                results.append(_load(root, meta["id"]))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                st.error(f"{meta['id']}: {exc}")
    if not results:
        return
    if not compatible(results):
        st.warning(
            "Software versions differ. These results cannot share a ranking. Run a common batch."
        )
        return
    validation = comparison_rows(results)
    st.write("Validation leaderboard — click a column to sort")
    st.dataframe(validation, hide_index=True, width="stretch")
    st.caption(
        "Accuracy must be read with BUY/SELL counts and coverage. Fewer than 30 "
        "actionable observations is a small sample."
    )
    for result in results:
        if (
            result["metrics"]["partitions"]["validation"]["ant"]["predictive"][
                "actionable_observations"
            ]
            < 30
        ):
            st.warning(
                f"{result['metadata']['id'][:8]}: fewer than 30 validation BUY/SELL signals."
            )
    directions = list(dict.fromkeys(validation.Direction))
    starts = list(dict.fromkeys(validation.Start))
    left, right = st.columns(2)
    direction = left.selectbox("Heatmap direction", directions)
    start_mode = right.selectbox("Heatmap starting position", starts)
    metric = st.selectbox(
        "Heatmap metric", ["ROC AUC", "BUY/SELL accuracy", "Coverage"]
    )
    filtered = validation.loc[
        validation.Direction.eq(direction) & validation.Start.eq(start_mode)
    ]
    st.plotly_chart(heatmap(filtered, metric), width="stretch", key="lab_heatmap")
    st.plotly_chart(
        px.scatter(
            validation,
            x="Coverage",
            y="BUY/SELL accuracy",
            hover_name="ID",
            hover_data=["Grid", "Steps", "BUY", "SELL", "HOLD", "Observations"],
        ),
        width="stretch",
        key="lab_accuracy_coverage",
    )
    ids = [r["metadata"]["id"] for r in results]
    selected = st.multiselect(
        "Experiments for side-by-side comparison and overlay (maximum 3)",
        ids,
        default=ids[: min(3, len(ids))],
        max_selections=3,
        key=f"lab_compare_{cohort}",
    )
    chosen = [r for r in results if r["metadata"]["id"] in selected]
    if not chosen:
        return
    partition_label = st.selectbox(
        "Comparison results",
        ["Validation", "Test — previously inspected diagnostics"],
    )
    partition = "validation" if partition_label == "Validation" else "test"
    if partition == "test":
        st.warning(
            "Previously inspected exploratory test data. Do not use it to choose configuration winners."
        )
    rows = []
    for result in chosen:
        for name in ("ant", "raw", "momentum"):
            m = result["metrics"]["partitions"][partition][name]["predictive"]
            rows.append(
                dict(
                    ID=result["metadata"]["id"],
                    Method=name,
                    **{k: v for k, v in m.items() if not isinstance(v, dict)},
                )
            )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.dataframe(
        pd.DataFrame(
            [
                {"ID": r["metadata"]["id"], "Partition": p, **values}
                for r in chosen
                for p, values in r["metadata"]["partitions"].items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    for a, b in combinations(chosen, 2):
        value = disagreement(a, b, partition)
        st.write(
            f"Signal disagreement {a['metadata']['id'][:8]} / {b['metadata']['id'][:8]}: "
            + (
                f"{value:.2%}"
                if value is not None
                else "undefined (no common timestamps)"
            )
        )
    try:
        view = as_research(chosen[0])
        times = chosen[0]["signals"].loc[
            lambda f: f.partition.eq(partition), "decision_timestamp"
        ]
        end = times.iloc[-1].date()
        begin = max(times.iloc[0].date(), end - pd.Timedelta(days=10))
        dates = st.date_input(
            "Overlay date range (UTC)",
            (begin, end),
            min_value=times.iloc[0].date(),
            max_value=end,
            key=f"lab_overlay_{cohort}_{partition}",
        )
        if len(dates) == 2:
            st.plotly_chart(
                overlay_chart(
                    view.candles,
                    chosen,
                    partition,
                    pd.Timestamp(dates[0], tz="UTC"),
                    pd.Timestamp(dates[1], tz="UTC") + pd.Timedelta(days=1),
                ),
                width="stretch",
                key="lab_overlay",
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Cannot load signal overlay: {exc}")


def render_lab(config_path):
    """Render controls; return the selected saved research view, if any."""
    st.subheader("Experiment Lab")
    try:
        settings = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise TypeError("Configuration must be a YAML mapping.")
        root = experiment_root(config_path, settings)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        st.error(f"Cannot load Experiment Lab settings: {exc}")
        return None
    lab = settings.get("experiment_lab", {})
    for key, value in DEFAULT_CONTROLS.items():
        st.session_state.setdefault(key, value)
    st.caption(
        "Original reference: 16 x 16, 2,048 steps, North, newest return. "
        "Rows x columns determine the exact historical lookback. "
        "Settings changes do not run experiments."
    )
    with st.form("experiment_configuration"):
        left, right = st.columns(2)
        grid = left.selectbox(
            "Grid preset",
            [f"{r} x {c}" for r, c in GRID_PRESETS] + ["Custom"],
            key="lab_grid",
        )
        step_preset = right.selectbox(
            "Simulation steps", [*STEP_PRESETS, "Custom"], key="lab_step_preset"
        )
        rows = left.number_input("Custom rows (2–32)", 2, 32, key="lab_rows")
        columns = right.number_input("Custom columns (2–32)", 2, 32, key="lab_columns")
        steps = left.number_input("Custom steps (4–16,384)", 4, 16384, key="lab_steps")
        direction = right.selectbox(
            "Starting direction",
            ["north", "east", "south", "west"],
            format_func=str.title,
            key="lab_direction",
        )
        start = left.selectbox(
            "Starting position",
            ["newest_return", "center"],
            format_func=lambda v: v.replace("_", " ").title(),
            key="lab_start",
        )
        st.caption(
            "Custom values apply when Custom is selected. Center uses (rows // 2, columns // 2)."
        )
        grids = st.multiselect(
            "Batch grids",
            [f"{r} x {c}" for r, c in GRID_PRESETS] + ["Custom"],
            key="lab_grids",
        )
        batch_steps = st.multiselect(
            "Batch step counts", STEP_PRESETS, key="lab_batch_steps"
        )
        custom_step = st.checkbox(
            "Include custom steps in batch", key="lab_custom_step"
        )
        limited = st.checkbox(
            "Use limited historical sample",
            value=lab.get("sample_candles") is not None,
            key="lab_limited",
        )
        sample = st.number_input(
            "Sample completed candles (first N, used only when enabled)",
            1,
            value=lab.get("sample_candles") or 1000,
            key="lab_sample",
        )
        expected_count = len(grids) * len(
            {*batch_steps, *([steps] if custom_step else [])}
        )
        st.caption(
            f"Expected batch count from submitted settings: {expected_count}. "
            "Maximum 30. Exact windows and transition workload are shown before execution."
        )
        run_single = st.form_submit_button("Run Experiment", key="lab_run_single")
        run_batch = st.form_submit_button("Run Batch", key="lab_run_batch")
    st.button("Reset Settings", on_click=reset_controls, key="lab_reset")
    status = st.empty()
    bar = st.progress(0.0)
    completed_status = st.container()

    def progress(i, n, config, stage):
        status.info(f"Experiment {i}/{n} · {config.label} · {stage}")
        if stage in ("Completed", "Cached result loaded"):
            completed_status.success(f"{i}/{n}: {config.label} — {stage}")
        elif stage.startswith("Error"):
            completed_status.error(f"{i}/{n}: {config.label} — {stage}")
        bar.progress(
            i / n
            if stage in ("Completed", "Cached result loaded")
            or stage.startswith("Error")
            else (i - 1) / n
        )

    if run_single or run_batch:
        try:
            shape = (
                (rows, columns)
                if grid == "Custom"
                else tuple(map(int, grid.split(" x ")))
            )
            configs = [
                ExperimentConfig(
                    *shape,
                    steps if step_preset == "Custom" else step_preset,
                    direction,
                    start,
                )
            ]
            if run_batch:
                shapes = [
                    (rows, columns)
                    if v == "Custom"
                    else tuple(map(int, v.split(" x ")))
                    for v in grids
                ]
                configs = batch_configs(
                    shapes,
                    [*batch_steps, *([steps] if custom_step else [])],
                    direction=direction,
                    start=start,
                )
            plan = prepare_plan(
                config_path,
                configs,
                sample_candles=sample if limited else None,
                base_config={
                    **settings,
                    "experiment_lab": {
                        **lab,
                        "sample_candles": sample if limited else None,
                    },
                },
            )
            st.info(
                f"Estimated workload: {len(configs)} experiments × {len(plan['decisions']):,} windows; "
                f"{plan['transitions']:,} ant transitions plus two decoder fits per uncached experiment."
            )
            _report_results(run_plan(plan, progress=progress))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.error(f"Experiment failed: {exc}")
    metadata = history(root)
    st.subheader("Experiment history")
    if not metadata:
        st.info("No saved experiments. Run a configuration to create one.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    dict(
                        ID=m["id"],
                        Status=m["status"],
                        **m.get("configuration", {}),
                        Error=m.get("error", ""),
                    )
                    for m in metadata
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        ids = [m["id"] for m in metadata]
        selected = st.selectbox("Saved experiment", ids, key="lab_saved")
        col1, col2, col3 = st.columns(3)
        if col1.button("Open experiment", key="lab_open"):
            try:
                _load(root, selected)
                st.session_state.lab_active = selected
                _clear_replay()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                st.error(str(exc))
        if col2.button("Rerun experiment", key="lab_rerun"):
            try:
                _report_results([rerun_result(root, selected, progress)])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                st.error(f"Cannot rerun experiment: {exc}")
        if col3.button("Mark research candidate", key="lab_candidate"):
            try:
                mark_candidate(root, selected, True)
                st.success("Marked as a research candidate.")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                st.error(str(exc))
        confirmed = st.checkbox(
            "Confirm deletion of the selected experiment",
            key=f"lab_confirm_delete_{selected}",
        )
        if st.button(
            "Delete selected experiment", disabled=not confirmed, key="lab_delete"
        ):
            try:
                delete_result(root, selected, confirmed=confirmed)
                if st.session_state.get("lab_active") == selected:
                    st.session_state.pop("lab_active", None)
                    _clear_replay()
                st.rerun()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                st.error(str(exc))
        _comparisons(root, metadata)
    if st.button("View original v1 reference", key="lab_original"):
        st.session_state.pop("lab_active", None)
        _clear_replay()
    active = st.session_state.get("lab_active")
    if active:
        try:
            result = _load(root, active)
            source = Path(result["metadata"]["dataset"]["source"])
            signature = (
                _signature(root, active),
                source.stat().st_mtime_ns,
                source.stat().st_size,
            )
            view = cached_view(result, signature)
            st.info(
                f"Selected research view: {active}. Inspect Ant Simulation and Gold & Signals."
            )
            st.json(result["metadata"]["configuration"])
            return view
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.error(f"Cannot open experiment: {exc}")
    return None
