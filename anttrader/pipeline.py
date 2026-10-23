"""Run configured historical XAU/USD grids, ant features, or the fixed backtest."""

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict
from itertools import islice
from math import prod
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from anttrader.data import load_gold_data
from anttrader.features import feature_columns, iter_feature_rows
from anttrader.grid import _grid_shape, iter_historical_grids
from anttrader.simulation import SimulationConfig

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def main(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    stage: str = "grids",
    limit: int | None = None,
) -> None:
    """Run the grid demonstration, feature export, or historical backtest."""
    config_path = Path(config_path)
    started = perf_counter()
    as_of = pd.Timestamp.now(tz="UTC")
    try:
        if stage not in ("grids", "features", "backtest"):
            raise ValueError("stage must be grids, features, or backtest.")
        if limit is not None and (
            type(limit) is not int or limit < 1 or stage != "features"
        ):
            raise ValueError("limit must be a positive integer for stage features.")
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        expected = {
            "symbol": "XAU/USD",
            "timeframe": "1H",
            "data_source": "Dukascopy",
            "candle_timestamp": "open",
            "quote_side": "bid",
            "encoding": "binary_log_return_sign",
            "layout": "serpentine",
        }
        if not isinstance(config, dict):
            raise TypeError("Configuration must be a YAML mapping.")
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"Topic 2 requires {key}: {value}.")
        shape = _grid_shape(config["grid_size"])
        if config["lookback_returns"] != prod(shape):
            raise ValueError("lookback_returns must equal the number of grid cells.")
        simulation = None
        if stage == "features":
            if shape != (16, 16):
                raise ValueError("Topic 3 requires grid_size: [16, 16].")
            simulation = SimulationConfig(**config["simulation"])
        path = config_path.resolve().parent / config["historical_csv"]
        if not path.is_file():
            if stage in ("features", "backtest"):
                raise FileNotFoundError(
                    f"Historical CSV missing: {path}. See README.md."
                )
            print(f"AntTrader: historical CSV missing: {path}")
            print(
                "Export Dukascopy XAU/USD H1 bid OHLC with UTC opening timestamps. "
                "Place it at this path. See README.md: Market data."
            )
            return
        candles = load_gold_data(
            path,
            as_of=as_of,
            source_timezone=config["timestamp_timezone"],
            timestamp_format=config["timestamp_format"],
            drop_zero_volume=config["drop_zero_volume"],
        )
        print(f"AntTrader: {len(candles):,} validated completed candles.")
        if not candles.empty:
            print(
                f"UTC opening range: {candles.timestamp.iloc[0]} to "
                f"{candles.timestamp.iloc[-1]}"
            )
        print(f"Validation report: {candles.attrs['validation_report']}")
        if stage == "features":
            export_features(
                candles,
                simulation,
                config=config,
                config_path=config_path,
                source=path,
                as_of=as_of,
                limit=limit,
                started=started,
            )
            return
        if stage == "backtest":
            from anttrader.experiment import run_experiment

            run_experiment(candles, config, config_path, path)
            return
        latest = None
        count = 0
        for count, latest in enumerate(
            iter_historical_grids(candles, grid_size=shape, as_of=as_of), 1
        ):
            pass
        print(f"Historical grids generated: {count:,}")
        if latest is None:
            print(f"Need {prod(shape) + 1} completed closing prices for a grid.")
            return
        print(f"Latest decision timestamp: {latest.decision_timestamp}")
        print(f"Window opening range: {latest.window_start} to {latest.window_end}")
        print(
            f"Grid shape: {latest.grid.shape}; values: {np.unique(latest.grid).tolist()}"
        )
        print(latest.grid)
    except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as exc:
        label = stage if stage != "grids" else "data-to-grid"
        raise SystemExit(f"AntTrader {label} error: {exc}") from exc


def export_features(
    candles: pd.DataFrame,
    simulation: SimulationConfig,
    *,
    config: dict,
    config_path: Path,
    source: Path,
    as_of: pd.Timestamp,
    limit: int | None,
    started: float,
) -> None:
    """Stream to a temporary CSV; preserve previous output if a window fails."""
    output = (config_path.resolve().parent / config["feature_output"]).resolve()
    metadata_path = output.with_suffix(".metadata.json")
    protected = {source.resolve(), config_path.resolve()}
    if output in protected or metadata_path in protected or output == metadata_path:
        raise ValueError("Feature output/metadata must not overwrite source or config.")
    if len(candles) < 257:
        raise ValueError("Need 257 completed closing prices for historical features.")
    output.parent.mkdir(parents=True, exist_ok=True)
    grids = iter_historical_grids(candles, as_of=as_of)
    if limit is not None:
        grids = islice(grids, limit)
    columns = ["decision_timestamp", *feature_columns(simulation)]
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            count = 0
            for row in iter_feature_rows(grids, simulation):
                writer.writerow(row)
                count += 1
        expected = len(candles) - 256
        if limit is not None:
            expected = min(expected, limit)
        if count != expected:
            raise ValueError(f"Expected {expected} feature rows, generated {count}.")
        with source.open("rb") as stream:
            source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata = {
            "source": str(source.resolve()),
            "source_sha256": source_hash,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "config": config,
            "simulation": asdict(simulation),
            "as_of": as_of.isoformat(),
            "limit": limit,
            "candles": len(candles),
            "eligible_grids": len(candles) - 256,
            "simulations": count,
            "transitions": count * simulation.steps,
            "feature_count": len(columns) - 1,
            "columns": columns,
            "elapsed_seconds": perf_counter() - started,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "failures": 0,
        }
        # Prepare metadata only after every window succeeds. Both files stay local.
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            metadata_temporary = Path(stream.name)
            try:
                json.dump(metadata, stream, indent=2)
                stream.write("\n")
            except Exception:
                stream.close()
                metadata_temporary.unlink(missing_ok=True)
                raise
        try:
            metadata_temporary.replace(metadata_path)
            temporary.replace(output)
        finally:
            metadata_temporary.unlink(missing_ok=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(
        f"Ant simulations completed: {count:,}; transitions: {count * simulation.steps:,}"
    )
    print(
        f"Simulation steps: {simulation.steps}; checkpoints: {list(simulation.checkpoints)}"
    )
    print(f"Feature columns: {len(columns) - 1}; failures: 0")
    print(f"Output CSV: {output}")
    print(f"Metadata: {metadata_path}")
    print(f"Execution time: {perf_counter() - started:.3f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage", choices=("grids", "features", "backtest"), default="grids"
    )
    parser.add_argument("--limit", type=int, help="First N windows (features only).")
    args = parser.parse_args()
    main(args.config, stage=args.stage, limit=args.limit)
