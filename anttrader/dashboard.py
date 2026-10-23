"""Read-only research loading, trajectory replay, and Plotly views."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

from anttrader.backtest import (
    EQUITY_COLUMNS,
    TRADE_COLUMNS,
    predictive_metrics,
    trading_metrics,
)
from anttrader.data import HOUR, load_gold_data, utc_timestamp
from anttrader.decoder import (
    decode_signals,
    future_targets,
    load_feature_dataset,
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
COLORS = {"ant": "#d39b28", "momentum": "#3182ce", "raw": "#9867c5"}
SIGNAL_COLORS = {"BUY": "#16865d", "SELL": "#d64550"}
PARTITIONS = ("test", "validation")
SIGNAL_COLUMNS = [
    "decision_timestamp",
    "entry_timestamp",
    "entry_price",
    "exit_timestamp",
    "exit_price",
    "future_return",
    "target",
    "target_valid",
    "target_exclusion",
    "partition",
    "momentum_signal",
    "ant_p_up",
    "ant_signal",
    "raw_p_up",
    "raw_signal",
]
DISCLAIMER = (
    "All backtest P&L represents theoretical price-only performance. Spread, "
    "slippage, commissions, financing, and execution latency are not modeled. "
    "Historical results do not establish a profitable live trading strategy."
)
RECOVERY = (
    "Restore the configured Dukascopy CSV using README.md: Market data if needed. "
    "Generate complete features with python -m anttrader.pipeline --stage features "
    "(without --limit), then run python -m anttrader.pipeline --stage backtest."
)


@dataclass
class ResearchData:
    config: dict
    candles: pd.DataFrame
    features: pd.DataFrame
    signals: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame
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
    if missing:
        raise ValueError(
            f"{path}: missing required columns: {', '.join(sorted(missing))}."
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
        raise ValueError("Stored metric schema does not match the backtester.")
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
            if metadata.get(key) != sha256(path):
                raise ValueError(
                    f"{path}: experiment {key} mismatch; stale or mixed inputs."
                )
        if (
            metadata["config"] != config
            or metadata["simulation"] != config["simulation"]
        ):
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
                    "python -m anttrader.pipeline --stage backtest."
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
        trades = _csv(paths["trade_output"], [*TRADE_COLUMNS, "partition", "strategy"])
        equity = _csv(
            paths["equity_output"], [*EQUITY_COLUMNS, "partition", "strategy"]
        )
        for column in ("decision_timestamp", "entry_timestamp", "exit_timestamp"):
            _times(signals, column, [])
            _times(trades, column, ["partition", "strategy"])
        _times(equity, "timestamp", ["partition", "strategy"])
        if signals.empty or set(signals.partition) != set(PARTITIONS):
            raise ValueError("Complete validation and test signals are required.")
        for frame in (trades, equity):
            if (
                not frame.partition.isin(PARTITIONS).all()
                or not frame.strategy.isin(STRATEGIES).all()
            ):
                raise ValueError("Unknown result strategy or partition.")
        for frame, columns in (
            (
                signals,
                [
                    "entry_price",
                    "exit_price",
                    "future_return",
                    "target",
                    "ant_p_up",
                    "raw_p_up",
                ],
            ),
            (
                trades,
                [
                    "entry_price",
                    "exit_price",
                    "direction",
                    "quantity_ounces",
                    "holding_period_bars",
                    "gross_pnl_usd",
                ],
            ),
            (equity, ["cumulative_gross_pnl_usd"]),
        ):
            frame[columns] = frame[columns].astype(float)
            if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
                raise ValueError(f"Missing or nonfinite result values in {columns}.")
        if not signals.decision_timestamp.isin(features.decision_timestamp).all():
            raise ValueError("Signal timestamps do not match saved feature windows.")
        expected_targets = future_targets(
            candles, signals.decision_timestamp, config["prediction_horizon_bars"]
        )
        for column in ("entry_timestamp", "exit_timestamp", "target_valid"):
            if not signals[column].eq(expected_targets[column]).all():
                raise ValueError(f"Signal {column} mismatch with historical candles.")
        for column in ("entry_price", "exit_price", "future_return", "target"):
            if not np.allclose(
                signals[column].to_numpy(dtype=float),
                expected_targets[column].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-10,
                equal_nan=False,
            ):
                raise ValueError(f"Signal {column} mismatch with historical candles.")
        if not signals.target_valid.eq(True).all():
            raise ValueError("Signals contain unavailable future outcomes.")
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
        metrics = _json(paths["metrics_output"])
        if metrics["pnl_basis"] != metadata["pnl_basis"]:
            raise ValueError("Metric P&L basis differs from experiment metadata.")
        for partition in PARTITIONS:
            selected = signals.loc[signals.partition.eq(partition)]
            info = metadata["partitions"][partition]
            if (
                len(selected) != info["eligible_rows"]
                or selected.decision_timestamp.iloc[0]
                != utc_timestamp(info["first_eligible_decision"])
                or selected.decision_timestamp.iloc[-1]
                != utc_timestamp(info["last_eligible_decision"])
                or selected.exit_timestamp.iloc[-1]
                != utc_timestamp(info["latest_eligible_exit"])
                or selected.decision_timestamp.min()
                < utc_timestamp(info["first_decision"])
                or selected.decision_timestamp.max()
                > utc_timestamp(info["last_decision"])
            ):
                raise ValueError(f"Incomplete or incompatible {partition} signals.")
            cutoff = info["exclusive_outcome_cutoff"]
            if cutoff and not selected.exit_timestamp.lt(utc_timestamp(cutoff)).all():
                raise ValueError(f"{partition} outcomes cross the partition boundary.")
            for strategy in STRATEGIES:
                completed = trades.loc[
                    trades.partition.eq(partition) & trades.strategy.eq(strategy)
                ]
                curve = equity.loc[
                    equity.partition.eq(partition) & equity.strategy.eq(strategy)
                ]
                if not completed.decision_timestamp.isin(
                    selected.decision_timestamp
                ).all():
                    raise ValueError("Trade decision absent from saved predictions.")
                matched = selected.set_index("decision_timestamp").reindex(
                    completed.decision_timestamp
                )
                for column in ("entry_timestamp", "exit_timestamp"):
                    if not np.array_equal(completed[column], matched[column]):
                        raise ValueError(f"Trade {column} differs from saved signals.")
                for column in ("entry_price", "exit_price"):
                    if not np.allclose(completed[column], matched[column]):
                        raise ValueError(f"Trade {column} differs from saved signals.")
                if (
                    not completed.signal.isin(["BUY", "SELL"]).all()
                    or not np.array_equal(
                        completed.signal, matched[f"{strategy}_signal"]
                    )
                    or not completed.direction.eq(
                        np.where(completed.signal.eq("BUY"), 1, -1)
                    ).all()
                    or not completed.quantity_ounces.eq(
                        config["position_quantity_ounces"]
                    ).all()
                    or not completed.holding_period_bars.eq(
                        config["holding_period_bars"]
                    ).all()
                    or not np.allclose(
                        completed.gross_pnl_usd,
                        completed.direction
                        * (completed.exit_price - completed.entry_price)
                        * completed.quantity_ounces,
                    )
                ):
                    raise ValueError(
                        "Trade direction, quantity, holding period, or P&L mismatch."
                    )
                if strategy == "momentum":
                    if completed.p_up.notna().any():
                        raise ValueError(
                            "Momentum trades cannot have model probabilities."
                        )
                elif not np.allclose(
                    completed.p_up.to_numpy(dtype=float), matched[f"{strategy}_p_up"]
                ):
                    raise ValueError(
                        "Trade probability differs from saved predictions."
                    )
                if (
                    len(completed) > 1
                    and (
                        completed.entry_timestamp.iloc[1:].to_numpy()
                        < completed.exit_timestamp.iloc[:-1].to_numpy()
                    ).any()
                ):
                    raise ValueError("Overlapping positions in saved trades.")
                expected_times = pd.DatetimeIndex(
                    [selected.decision_timestamp.iloc[0], *completed.exit_timestamp]
                )
                if not pd.DatetimeIndex(curve.timestamp).equals(expected_times):
                    raise ValueError(
                        "Equity timestamps/baseline do not match completed trades."
                    )
                stored = metrics["partitions"][partition][strategy]
                _metrics_match(stored["trading"], trading_metrics(completed, curve))
                _metrics_match(
                    stored["predictive"],
                    predictive_metrics(
                        selected.target,
                        selected[f"{strategy}_signal"],
                        None
                        if strategy == "momentum"
                        else selected[f"{strategy}_p_up"],
                    ),
                )
        if input_signature(config_path) != signature:
            raise ValueError(
                "Research files changed while loading. Reload after the CLI finishes."
            )
        return ResearchData(
            config, candles, features, signals, trades, equity, metrics, metadata
        )
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
    if end < 256:
        raise ValueError("Selected decision has no complete 257-candle input window.")
    window = next(
        iter_historical_grids(
            research.candles.iloc[end - 256 : end + 1],
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
    visits = np.zeros((16, 16), dtype=np.int64)
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
            "range": [-0.5, 15.5],
            "constrain": "domain",
        },
        yaxis={"title": "Row", "dtick": 1, "range": [15.5, -0.5], "scaleanchor": "x"},
    )
    return figure


def price_chart(
    candles: pd.DataFrame,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    strategy: str,
    partition: str,
) -> go.Figure:
    figure = go.Figure(
        go.Candlestick(
            x=candles.timestamp.tolist(),
            open=candles.open,
            high=candles.high,
            low=candles.low,
            close=candles.close,
            name="XAU/USD bid OHLC",
            increasing_line_color="#16865d",
            decreasing_line_color="#d64550",
        )
    )
    for signal, symbol in (("BUY", "triangle-up"), ("SELL", "triangle-down")):
        predictions = signals.loc[
            signals[f"{strategy}_signal"].eq(signal)
            & signals.decision_timestamp.isin(candles.timestamp)
        ]
        figure.add_trace(
            go.Scatter(
                x=predictions.decision_timestamp.tolist(),
                y=predictions.entry_price,
                mode="markers",
                name=f"{signal} prediction",
                marker={"symbol": symbol, "size": 7, "color": SIGNAL_COLORS[signal]},
                hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>Decision reference: $%{y:.3f}"
                "<extra>" + signal + " prediction</extra>",
            )
        )
        entries = trades.loc[
            trades.signal.eq(signal) & trades.entry_timestamp.isin(candles.timestamp)
        ]
        figure.add_trace(
            go.Scatter(
                x=entries.entry_timestamp.tolist(),
                y=entries.entry_price,
                mode="markers",
                name=f"{signal} executed entry",
                marker={
                    "symbol": "diamond-open",
                    "size": 14,
                    "color": SIGNAL_COLORS[signal],
                    "line": {"width": 2},
                },
                hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>Entry: $%{y:.3f}"
                "<extra>" + signal + " entry</extra>",
            )
        )
    exits = trades.loc[trades.exit_timestamp.isin(candles.timestamp)]
    figure.add_trace(
        go.Scatter(
            x=exits.exit_timestamp.tolist(),
            y=exits.exit_price,
            mode="markers",
            name="Completed exit",
            marker={"symbol": "x", "size": 11, "color": "#64748b"},
            customdata=exits[["gross_pnl_usd"]].to_numpy(),
            hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>Exit: $%{y:.3f}"
            "<br>Gross P&L: $%{customdata[0]:.3f}<extra>Exit</extra>",
        )
    )
    figure.update_layout(
        title=f"XAU/USD 1H · {STRATEGY_NAMES[strategy]} · {partition.title()}",
        height=570,
        xaxis_title="Candle opening / decision time (UTC)",
        yaxis_title="Bid reference price (USD / ounce)",
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 40, "r": 20, "t": 100, "b": 40},
    )
    return figure


def equity_chart(equity: pd.DataFrame, partition: str) -> go.Figure:
    figure = go.Figure()
    for strategy in STRATEGIES:
        curve = equity.loc[
            equity.partition.eq(partition) & equity.strategy.eq(strategy)
        ]
        figure.add_trace(
            go.Scatter(
                x=curve.timestamp.tolist(),
                y=curve.cumulative_gross_pnl_usd,
                mode="lines",
                name=STRATEGY_NAMES[strategy],
                line={"color": COLORS[strategy], "shape": "hv", "width": 2},
                hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>$%{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )
    figure.add_hline(y=0, line_dash="dot", line_color="#64748b")
    figure.update_layout(
        title=f"{partition.title()} · Cumulative realized gross P&L",
        xaxis_title="Completed-trade timestamp (UTC)",
        yaxis_title="Gross P&L (USD)",
        height=480,
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 40, "r": 20, "t": 90, "b": 40},
    )
    return figure


def format_metric(value, *, percent: bool = False) -> str:
    if value is None:
        return "Undefined"
    if value == "infinity":
        return "Infinity"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:.2%}" if percent else f"{value:,.3f}"


def comparison_table(metrics: dict, partition: str) -> pd.DataFrame:
    rows = [
        ("Gross total P&L (USD)", "trading", "total_gross_pnl_usd", False),
        ("Completed trades", "trading", "completed_trades", False),
        ("Win rate", "trading", "win_rate", True),
        ("Average P&L / trade (USD)", "trading", "average_gross_pnl_usd", False),
        ("Profit factor", "trading", "profit_factor", False),
        (
            "Maximum realized drawdown (USD)",
            "trading",
            "maximum_realized_drawdown_usd",
            False,
        ),
        ("Directional accuracy", "predictive", "directional_accuracy", True),
        ("ROC AUC", "predictive", "roc_auc", False),
        ("BUY signals", "predictive", "BUY", False),
        ("SELL signals", "predictive", "SELL", False),
        ("HOLD signals", "predictive", "HOLD", False),
    ]
    columns = {}
    for strategy in STRATEGIES:
        result = metrics["partitions"][partition][strategy]
        columns[STRATEGY_NAMES[strategy]] = [
            format_metric(
                result[group]["signal_counts"][key]
                if key in ("BUY", "SELL", "HOLD")
                else result[group][key],
                percent=percent,
            )
            for _, group, key, percent in rows
        ]
    return pd.DataFrame(columns, index=[row[0] for row in rows])
