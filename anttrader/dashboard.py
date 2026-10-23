"""Read-only research loading, trajectory replay, and Plotly views."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

from anttrader.data import HOUR, load_gold_data, utc_timestamp
from anttrader.decoder import (
    chronological_partitions,
    decode_signals,
    future_targets,
    historical_inputs,
    load_feature_dataset,
    predictive_metrics,
    sha256,
    validate_research_config,
)
from anttrader.experiment import OUTPUT_KEYS, STRATEGIES
from anttrader.features import extract_features
from anttrader.grid import HistoricalGrid, iter_historical_grids
from anttrader.pipeline import DEFAULT_CONFIG
from anttrader.simulation import (
    DELTAS,
    Checkpoint,
    SimulationConfig,
    SimulationResult,
    simulate,
)

STRATEGY_NAMES = {
    "ant": "AntTrader",
    "momentum": "Momentum",
    "raw": "Raw-input logistic regression",
}
SIGNAL_COLORS = {"BUY": "#16865d", "SELL": "#d64550"}
PARTITIONS = ("test", "validation")
SIGNAL_COLUMNS = [
    "decision_timestamp",
    "label_start_timestamp",
    "label_start_price",
    "label_end_timestamp",
    "label_end_price",
    "future_return",
    "target",
    "target_valid",
    "target_exclusion",
    "reference_price",
    "partition",
    "momentum_signal",
    "ant_p_up",
    "ant_signal",
    "raw_p_up",
    "raw_signal",
]
DISCLAIMER = (
    "AntTrader is a signal research tool, not a trade simulator. "
    "BUY and SELL are directional predictions. Historical accuracy does not establish profitability."
)
RECOVERY = (
    "Restore the configured Dukascopy CSV using README.md: Market data if needed. "
    "Generate complete features with python -m anttrader.pipeline --stage features "
    "(without --limit), then run python -m anttrader.pipeline --stage signals."
)


@dataclass
class ResearchData:
    config: dict
    candles: pd.DataFrame
    features: pd.DataFrame
    signals: pd.DataFrame
    metrics: dict
    metadata: dict


def _json(path: Path) -> dict:
    def invalid(value):
        raise ValueError(f"Invalid JSON number: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object.")
    return value


def input_signature(config_path: Path = DEFAULT_CONFIG) -> tuple:
    """Small file-stat cache key; include missing files so restoration invalidates it."""
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"{config_path}: configuration must be a YAML mapping.")
    paths = [config_path]
    for key in ("historical_csv", "feature_output", *OUTPUT_KEYS):
        paths.append((config_path.parent / config[key]).resolve())
    paths.append(paths[2].with_suffix(".metadata.json"))
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        if path.is_file()
        else (str(path), None, None)
        for path in paths
    )


def _csv(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(columns) - set(frame)
    if list(frame) != columns:
        raise ValueError(
            f"{path}: invalid signal columns; missing: {', '.join(sorted(missing))}."
        )
    return frame


def _times(frame: pd.DataFrame, column: str, keys: list[str]) -> None:
    times = pd.DatetimeIndex([utc_timestamp(value) for value in frame[column]])
    if times.empty:
        frame[column] = pd.to_datetime(frame[column], utc=True)
        return
    frame[column] = times
    if (
        not times.is_monotonic_increasing
        or not (times == times.floor("h")).all()
        or frame.duplicated([*keys, column]).any()
    ):
        raise ValueError(
            f"{column}: timestamps must be ordered UTC hours with unique keys."
        )


def _metrics_match(stored: dict, expected: dict) -> None:
    """Reconcile with the same functions used by the exporter."""
    if not isinstance(stored, dict) or stored.keys() != expected.keys():
        raise ValueError("Stored metric schema does not match the signal experiment.")
    for key, value in expected.items():
        actual = stored[key]
        if isinstance(value, dict):
            _metrics_match(actual, value)
        elif isinstance(value, float):
            if (
                type(actual) not in (int, float)
                or not np.isfinite(actual)
                or not np.isclose(actual, value, rtol=1e-9, atol=1e-8)
            ):
                raise ValueError(f"Stored metric does not reconcile: {key}.")
        elif actual != value:
            raise ValueError(f"Stored metric does not reconcile: {key}.")


def load_research(config_path: Path = DEFAULT_CONFIG) -> ResearchData:
    """Load one complete, identified bundle. Never fit or predict a model."""
    try:
        signature = input_signature(config_path)
        config_path = config_path.resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        validate_research_config(config)
        paths = {
            key: (config_path.parent / config[key]).resolve()
            for key in ("historical_csv", "feature_output", *OUTPUT_KEYS)
        }
        feature_meta_path = paths["feature_output"].with_suffix(".metadata.json")
        for path in [*paths.values(), feature_meta_path]:
            if not path.is_file():
                raise ValueError(f"Required file missing: {path}.")
        metadata = _json(paths["experiment_metadata_output"])
        if metadata.get("schema") != "signals-only-v1":
            raise ValueError(
                "Signal-only metadata required; regenerate with --stage signals."
            )
        feature_metadata = _json(feature_meta_path)
        if not all(
            isinstance(feature_metadata.get(key), dict)
            for key in ("config", "simulation")
        ):
            raise TypeError(
                f"{feature_meta_path}: config and simulation must be objects."
            )
        identities = {
            "config_sha256": config_path,
            "source_sha256": paths["historical_csv"],
            "feature_sha256": paths["feature_output"],
            "feature_metadata_sha256": feature_meta_path,
        }
        for key, path in identities.items():
            if metadata.get(key) != sha256(path) and not (
                key == "config_sha256"
                and metadata["config"].get("experiment_lab")
                != config.get("experiment_lab")
                and {
                    k: v for k, v in metadata["config"].items() if k != "experiment_lab"
                }
                == {k: v for k, v in config.items() if k != "experiment_lab"}
            ):
                raise ValueError(
                    f"{path}: experiment {key} mismatch; stale or mixed inputs."
                )
        if {k: v for k, v in metadata["config"].items() if k != "experiment_lab"} != {
            k: v for k, v in config.items() if k != "experiment_lab"
        } or metadata["simulation"] != config["simulation"]:
            raise ValueError("Experiment configuration metadata mismatch.")
        manifest = metadata.get("output_sha256", {})
        if not isinstance(manifest, dict):
            raise TypeError("Experiment output SHA-256 manifest must be an object.")
        for key in OUTPUT_KEYS[:-1]:
            path = paths[key]
            if manifest.get(key) != sha256(path):
                raise ValueError(
                    f"{path}: missing or mismatched experiment output SHA-256. "
                    "Regenerate the result bundle with "
                    "python -m anttrader.pipeline --stage signals."
                )
        candles = load_gold_data(
            paths["historical_csv"],
            as_of=feature_metadata["as_of"],
            source_timezone=config["timestamp_timezone"],
            timestamp_format=config["timestamp_format"],
            drop_zero_volume=config["drop_zero_volume"],
        )
        features, _ = load_feature_dataset(
            paths["feature_output"], candles, paths["historical_csv"], config
        )
        if (
            metadata["candles"] != len(candles)
            or metadata["historical_observations"] != len(features)
            or metadata["feature_source_cutoff"] != feature_metadata["as_of"]
        ):
            raise ValueError("Experiment history count/cutoff mismatch.")
        signals = _csv(paths["signal_output"], SIGNAL_COLUMNS)
        for column in (
            "decision_timestamp",
            "label_start_timestamp",
            "label_end_timestamp",
        ):
            _times(signals, column, [])
        if signals.empty or set(signals.partition) != set(PARTITIONS):
            raise ValueError("Complete validation and test signals are required.")
        numeric = [
            "label_start_price",
            "label_end_price",
            "future_return",
            "target",
            "reference_price",
            "ant_p_up",
            "raw_p_up",
        ]
        signals[numeric] = signals[numeric].astype(float)
        if not np.isfinite(signals[numeric].to_numpy()).all():
            raise ValueError("Missing or nonfinite signal values.")
        splits, boundaries = chronological_partitions(
            features.decision_timestamp,
            config["train_fraction"],
            config["validation_fraction"],
        )
        targets = future_targets(
            candles, features.decision_timestamp, config["prediction_horizon_bars"]
        )
        _, momentum, momentum_valid = historical_inputs(
            candles,
            features.decision_timestamp,
            config["momentum_lookback_bars"],
            config["lookback_returns"],
        )
        boundary_valid = splits.outcome_cutoff.isna() | targets.label_end_timestamp.lt(
            splits.outcome_cutoff
        )
        eligible = targets.target_valid & momentum_valid & boundary_valid
        evaluation = eligible & splits.partition.ne("train")
        expected_targets = targets.loc[evaluation].reset_index(drop=True)
        if not pd.DatetimeIndex(signals.decision_timestamp).equals(
            pd.DatetimeIndex(expected_targets.decision_timestamp)
        ):
            raise ValueError("Incomplete or incompatible signal timestamps.")
        for column in ("label_start_timestamp", "label_end_timestamp", "target_valid"):
            if not signals[column].eq(expected_targets[column]).all():
                raise ValueError(f"Signal {column} mismatch with historical candles.")
        for column in (
            "label_start_price",
            "label_end_price",
            "future_return",
            "target",
        ):
            if not np.allclose(
                signals[column],
                expected_targets[column].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-10,
            ):
                raise ValueError(f"Signal {column} mismatch with historical candles.")
        reference = (
            candles.set_index("timestamp")
            .close.reindex(pd.DatetimeIndex(signals.decision_timestamp) - HOUR)
            .to_numpy()
        )
        if not np.allclose(signals.reference_price, reference, rtol=1e-10, atol=1e-10):
            raise ValueError(
                "Signal reference price differs from the last completed candle."
            )
        if (
            not signals.target_exclusion.astype("string").fillna("").eq("").all()
            or not signals.partition.eq(
                splits.loc[evaluation, "partition"].to_numpy()
            ).all()
            or not signals.momentum_signal.eq(momentum.loc[evaluation].to_numpy()).all()
        ):
            raise ValueError("Signal partition, momentum, or label exclusion mismatch.")
        for strategy in STRATEGIES:
            if not signals[f"{strategy}_signal"].isin(["BUY", "SELL", "HOLD"]).all():
                raise ValueError(f"Invalid {strategy} signal.")
            if strategy != "momentum":
                expected = decode_signals(
                    signals[f"{strategy}_p_up"],
                    buy=config["buy_threshold"],
                    sell=config["sell_threshold"],
                )
                if not signals[f"{strategy}_signal"].eq(expected).all():
                    raise ValueError(
                        f"{strategy} signals do not match saved probabilities."
                    )
        for partition, boundary in boundaries.items():
            selected = eligible & splits.partition.eq(partition)
            times = features.loc[selected, "decision_timestamp"]
            boundary.update(
                eligible_rows=int(selected.sum()),
                first_eligible_decision=times.iloc[0].isoformat(),
                last_eligible_decision=times.iloc[-1].isoformat(),
                latest_eligible_label_end=targets.loc[selected, "label_end_timestamp"]
                .iloc[-1]
                .isoformat(),
                boundary_purged_rows=int(
                    (
                        splits.partition.eq(partition)
                        & targets.target_valid
                        & momentum_valid
                        & ~boundary_valid
                    ).sum()
                ),
            )
        counts = {
            "valid_target_labels": int(targets.target_valid.sum()),
            "excluded_market_gap": int(targets.target_exclusion.eq("market_gap").sum()),
            "excluded_unavailable_tail": int(
                targets.target_exclusion.eq("unavailable_tail").sum()
            ),
            "excluded_momentum_gap_after_labeling": int(
                (targets.target_valid & ~momentum_valid).sum()
            ),
            "excluded_boundary_after_intersection": int(
                (targets.target_valid & momentum_valid & ~boundary_valid).sum()
            ),
            "shared_eligible_observations": int(eligible.sum()),
        }
        if metadata["partitions"] != boundaries or any(
            metadata[key] != value for key, value in counts.items()
        ):
            raise ValueError(
                "Experiment partition boundaries/counts or exclusions mismatch."
            )
        metrics = _json(paths["metrics_output"])
        expected_metrics = {"partitions": {}}
        for partition in PARTITIONS:
            selected = signals.loc[signals.partition.eq(partition)]
            expected_metrics["partitions"][partition] = {
                strategy: {
                    "predictive": predictive_metrics(
                        selected.target,
                        selected[f"{strategy}_signal"],
                        None
                        if strategy == "momentum"
                        else selected[f"{strategy}_p_up"],
                    )
                }
                for strategy in STRATEGIES
            }
        _metrics_match(metrics, expected_metrics)
        if input_signature(config_path) != signature:
            raise ValueError(
                "Research files changed while loading. Reload after the CLI finishes."
            )
        return ResearchData(config, candles, features, signals, metrics, metadata)
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        IndexError,
        yaml.YAMLError,
    ) as exc:
        raise ValueError(f"Cannot load research: {exc}\n\n{RECOVERY}") from exc


def selected_simulation(
    research: ResearchData,
    decision: object,
) -> tuple[HistoricalGrid, SimulationResult]:
    timestamp = utc_timestamp(decision)
    saved = research.features.loc[research.features.decision_timestamp.eq(timestamp)]
    if len(saved) != 1:
        raise ValueError(
            "Select a timestamp from the saved historical feature dataset."
        )
    end = pd.Index(research.candles.timestamp).get_indexer([timestamp - HOUR])[0]
    lookback = int(np.prod(research.config["grid_size"]))
    if end < lookback:
        raise ValueError("Selected decision has no complete historical input window.")
    window = next(
        iter_historical_grids(
            research.candles.iloc[end - lookback : end + 1],
            grid_size=research.config["grid_size"],
            as_of=timestamp,
        )
    )
    if window.decision_timestamp != timestamp:
        raise ValueError("Reconstructed grid decision timestamp mismatch.")
    result = simulate(
        window.grid,
        SimulationConfig(**research.config["simulation"]),
        record_trajectory=True,
    )
    values = extract_features(result)
    if not np.allclose(
        list(values.values()),
        saved[list(values)].iloc[0].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Reconstructed simulation does not match saved checkpoint features."
        )
    return window, result


def replay_state(result: SimulationResult, step: int) -> Checkpoint:
    """Count recorded departures and flips; use recorded headings for displacement."""
    if type(step) is not int or not 0 <= step <= result.config.steps:
        raise ValueError("Replay step is outside the recorded simulation.")
    if result.trajectory is None:
        raise ValueError("Replay requires a recorded trajectory.")
    poses = np.asarray(result.trajectory[: step + 1], dtype=int)
    visits = np.zeros_like(result.initial_grid, dtype=np.int64)
    np.add.at(visits, (poses[:-1, 0], poses[:-1, 1]), 1)
    grid = result.initial_grid ^ (visits % 2).astype(np.uint8)
    dy, dx = np.asarray(DELTAS)[poses[1:, 2]].sum(axis=0)
    row, col, direction = poses[-1]
    return Checkpoint(
        steps=step,
        row=int(row),
        col=int(col),
        direction=int(direction),
        horizontal_displacement=int(dx),
        vertical_displacement=int(dy),
        right_turns=int(np.count_nonzero((np.diff(poses[:, 2]) % 4) == 1)),
        unique_cells_visited=int(np.count_nonzero(visits)),
        black_cells=int(grid.sum()),
        changed_cells=int(np.count_nonzero(grid != result.initial_grid)),
        grid=grid,
        visit_counts=visits,
    )


def grid_chart(grid: np.ndarray, state: Checkpoint | None, title: str) -> go.Figure:
    figure = go.Figure(
        go.Heatmap(
            z=grid,
            zmin=0,
            zmax=1,
            colorscale=[[0, "#f1f4f8"], [1, "#263445"]],
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate="Row %{y}, column %{x}<br>Cell %{z}<extra></extra>",
        )
    )
    if state is not None:
        figure.add_trace(
            go.Scatter(
                x=[state.col],
                y=[state.row],
                mode="markers",
                marker={
                    "symbol": (
                        "triangle-up",
                        "triangle-right",
                        "triangle-down",
                        "triangle-left",
                    )[state.direction],
                    "color": "#f0b429",
                    "size": 19,
                    "line": {"color": "#694b0a", "width": 1},
                },
                name=f"Ant: {state.heading.title()}",
                hovertemplate=f"Row {state.row}, column {state.col}<br>{state.heading}<extra></extra>",
            )
        )
    figure.update_layout(
        title=title,
        height=480,
        margin={"l": 35, "r": 15, "t": 55, "b": 35},
        showlegend=False,
        xaxis={
            "title": "Column",
            "dtick": 1,
            "range": [-0.5, grid.shape[1] - 0.5],
            "constrain": "domain",
        },
        yaxis={
            "title": "Row",
            "dtick": 1,
            "range": [grid.shape[0] - 0.5, -0.5],
            "scaleanchor": "x",
        },
    )
    return figure


def price_chart(
    candles: pd.DataFrame,
    signals: pd.DataFrame,
    strategy: str,
    partition: str,
    *,
    visible_sides: tuple[str, ...] = ("BUY", "SELL"),
    start: pd.Timestamp | None = None,
    stop: pd.Timestamp | None = None,
) -> go.Figure:
    # Resolve references before display filtering, including the preceding boundary candle.
    signals = signals.loc[signals.partition.eq(partition)]
    if start is not None:
        signals = signals.loc[signals.decision_timestamp.ge(start)]
    if stop is not None:
        signals = signals.loc[signals.decision_timestamp.lt(stop)]
    observed = candles.set_index("timestamp").reindex(
        pd.DatetimeIndex(signals.decision_timestamp) - HOUR
    )
    if observed[["high", "low", "close"]].isna().any().any():
        raise ValueError("Signal markers require the last completed candle.")
    displayed = candles
    if start is not None:
        displayed = displayed.loc[displayed.timestamp.ge(start)]
    if stop is not None:
        displayed = displayed.loc[displayed.timestamp.lt(stop)]
    figure = go.Figure(
        go.Candlestick(
            x=displayed.timestamp.tolist(),
            open=displayed.open,
            high=displayed.high,
            low=displayed.low,
            close=displayed.close,
            name="XAU/USD bid OHLC",
            increasing_line_color="#16865d",
            decreasing_line_color="#d64550",
        )
    )
    offset = np.maximum(
        (observed.high - observed.low).to_numpy() * 0.15,
        observed.close.to_numpy() * 0.0008,
    )
    for signal, symbol in (("BUY", "triangle-up"), ("SELL", "triangle-down")):
        if signal not in visible_sides:
            continue
        mask = signals[f"{strategy}_signal"].eq(signal).to_numpy()
        predictions = signals.loc[mask]
        levels = (
            observed.low.to_numpy() - offset
            if signal == "BUY"
            else observed.high.to_numpy() + offset
        )
        probability = (
            ["Unavailable (momentum)"] * len(predictions)
            if strategy == "momentum"
            else [f"{value:.4f}" for value in predictions[f"{strategy}_p_up"]]
        )
        figure.add_trace(
            go.Scatter(
                x=predictions.decision_timestamp.tolist(),
                y=levels[mask],
                mode="markers",
                name=f"{signal} prediction",
                marker={
                    "symbol": symbol,
                    "size": 16,
                    "color": SIGNAL_COLORS[signal],
                    "line": {"color": "#152536", "width": 1.5},
                },
                customdata=list(
                    zip(
                        probability,
                        predictions.reference_price,
                        predictions.partition,
                        strict=True,
                    )
                ),
                hovertemplate="Decision: %{x|%Y-%m-%d %H:%M} UTC<br>Signal: "
                + signal
                + "<br>P(up): %{customdata[0]}<br>Last completed close: $%{customdata[1]:.3f}"
                "<br>Partition: %{customdata[2]}<extra></extra>",
            )
        )
    figure.update_layout(
        title=f"XAU/USD 1H · {STRATEGY_NAMES[strategy]} · {partition.title()}",
        template="plotly_white",
        height=800,
        font={"size": 14},
        xaxis_title="Candle opening / decision time (UTC)",
        yaxis_title="Gold bid price (USD / ounce)",
        xaxis_rangeslider_visible=False,
        dragmode="pan",
        hovermode="closest",
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 65, "r": 25, "t": 110, "b": 60},
    )
    if start is not None and stop is not None:
        figure.update_xaxes(range=[start, stop])
    return figure


def format_metric(value, *, percent: bool = False) -> str:
    if value is None:
        return "Undefined"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:.2%}" if percent else f"{value:,.3f}"


def comparison_table(metrics: dict, partition: str) -> pd.DataFrame:
    rows = [
        ("Total signals", "valid_observations", False),
        ("BUY signals", "BUY", False),
        ("SELL signals", "SELL", False),
        ("HOLD signals", "HOLD", False),
        ("BUY fraction", "BUY", True),
        ("SELL fraction", "SELL", True),
        ("HOLD fraction", "HOLD", True),
        ("Overall classification accuracy", "directional_accuracy", True),
        ("BUY/SELL directional accuracy", "actionable_accuracy", True),
        ("ROC AUC", "roc_auc", False),
    ]
    columns = {}
    for strategy in STRATEGIES:
        result = metrics["partitions"][partition][strategy]["predictive"]
        columns[STRATEGY_NAMES[strategy]] = [
            format_metric(
                result["signal_fractions" if percent else "signal_counts"][key]
                if key in ("BUY", "SELL", "HOLD")
                else None
                if strategy == "momentum" and key == "directional_accuracy"
                else result[key],
                percent=percent,
            )
            for _, key, percent in rows
        ]
    return pd.DataFrame(columns, index=[row[0] for row in rows])
