"""Run the configured historical XAU/USD data-to-grid demonstration."""

import argparse
from math import prod
from pathlib import Path

import numpy as np
import yaml

from anttrader.data import load_gold_data
from anttrader.grid import iter_historical_grids

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def main(config_path: str | Path = DEFAULT_CONFIG) -> None:
    """Validate configured history and print the latest completed binary grid."""
    config_path = Path(config_path)
    try:
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
        shape = config["grid_size"]
        if config["lookback_returns"] != prod(shape):
            raise ValueError("lookback_returns must equal the number of grid cells.")
        path = config_path.resolve().parent / config["historical_csv"]
        if not path.is_file():
            print(f"AntTrader: historical CSV missing: {path}")
            print(
                "Export Dukascopy XAU/USD H1 bid OHLC with UTC opening timestamps. "
                "Place it at this path. See README.md: Market data."
            )
            return
        candles = load_gold_data(
            path,
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
        latest = None
        count = 0
        for count, latest in enumerate(
            iter_historical_grids(candles, grid_size=shape), 1
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
        raise SystemExit(f"AntTrader data-to-grid error: {exc}") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
