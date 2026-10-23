"""Run one fixed historical experiment and export comparable research results."""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from anttrader.backtest import predictive_metrics, simulate_trades, trading_metrics
from anttrader.decoder import (
    RAW_COLUMNS,
    chronological_partitions,
    decode_signals,
    fit_decoder,
    future_targets,
    historical_inputs,
    load_feature_dataset,
    predict_up,
    sha256,
    validate_research_config,
)

STRATEGIES = ("ant", "momentum", "raw")
OUTPUT_KEYS = (
    "signal_output",
    "trade_output",
    "equity_output",
    "metrics_output",
    "experiment_metadata_output",
)


def run_experiment(
    candles: pd.DataFrame, config: dict, config_path: Path, source: Path
) -> dict:
    validate_research_config(config)
    base = config_path.resolve().parent
    feature_path = (base / config["feature_output"]).resolve()
    outputs = [(base / config[key]).resolve() for key in OUTPUT_KEYS]
    protected = {
        source.resolve(),
        config_path.resolve(),
        feature_path,
        feature_path.with_suffix(".metadata.json"),
    }
    if len(set(outputs)) != len(outputs) or any(path in protected for path in outputs):
        raise ValueError(
            "Experiment outputs must be distinct and must not overwrite inputs or config."
        )
    features, feature_metadata = load_feature_dataset(
        feature_path, candles, source, config
    )
    decisions = features.decision_timestamp
    splits, boundaries = chronological_partitions(
        decisions, config["train_fraction"], config["validation_fraction"]
    )
    targets = future_targets(candles, decisions, config["prediction_horizon_bars"])
    raw, momentum, momentum_valid = historical_inputs(
        candles, decisions, config["momentum_lookback_bars"]
    )
    boundary_valid = splits.outcome_cutoff.isna() | targets.exit_timestamp.lt(
        splits.outcome_cutoff
    )
    eligible = targets.target_valid & momentum_valid & boundary_valid
    for partition in ("train", "validation", "test"):
        if not (eligible & splits.partition.eq(partition)).any():
            raise ValueError(
                "No shared eligible observations in one or more partitions."
            )
    train = eligible & splits.partition.eq("train")
    evaluation = eligible & splits.partition.ne("train")
    signals = targets.loc[evaluation].copy()
    signals["partition"] = splits.loc[evaluation, "partition"]
    signals["momentum_signal"] = momentum.loc[evaluation]
    signals = signals.reset_index(drop=True)
    for name, inputs in (("ant", features.iloc[:, 1:]), ("raw", raw)):
        model = fit_decoder(inputs.loc[train], targets.loc[train, "target"], config)
        # Both frozen models predict the same shared later rows.
        probabilities = predict_up(model, inputs.loc[evaluation])
        signals[f"{name}_p_up"] = probabilities
        signals[f"{name}_signal"] = decode_signals(
            probabilities, buy=config["buy_threshold"], sell=config["sell_threshold"]
        )
    metrics = {
        "pnl_basis": "gross price-only simulated P&L in USD; all costs zero",
        "partitions": {},
    }
    all_trades, all_equity = [], []
    for partition in ("validation", "test"):
        frame = signals.loc[signals.partition.eq(partition)].reset_index(drop=True)
        metrics["partitions"][partition] = {}
        for strategy in STRATEGIES:
            probability = None if strategy == "momentum" else frame[f"{strategy}_p_up"]
            orders = pd.DataFrame(
                {
                    "decision_timestamp": frame.decision_timestamp,
                    "signal": frame[f"{strategy}_signal"],
                }
            )
            if probability is not None:
                orders["p_up"] = probability
            trades, equity = simulate_trades(
                candles,
                orders,
                quantity=config["position_quantity_ounces"],
                holding_bars=config["holding_period_bars"],
            )
            metrics["partitions"][partition][strategy] = {
                "predictive": predictive_metrics(
                    frame.target, orders.signal, probability
                ),
                "trading": trading_metrics(trades, equity),
            }
            all_trades.append(trades.assign(partition=partition, strategy=strategy))
            all_equity.append(equity.assign(partition=partition, strategy=strategy))
    trades = (
        pd.concat(all_trades, ignore_index=True)
        .sort_values(["decision_timestamp", "strategy"], kind="stable")
        .reset_index(drop=True)
    )
    equity = (
        pd.concat(all_equity, ignore_index=True)
        .sort_values(["timestamp", "strategy"], kind="stable")
        .reset_index(drop=True)
    )
    for partition in ("train", "validation", "test"):
        selected = eligible & splits.partition.eq(partition)
        times = decisions.loc[selected]
        boundaries[partition].update(
            eligible_rows=int(selected.sum()),
            first_eligible_decision=times.iloc[0].isoformat(),
            last_eligible_decision=times.iloc[-1].isoformat(),
            latest_eligible_exit=targets.loc[selected, "exit_timestamp"]
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
    metadata = {
        "scope": "complete available feature history; validation and test reported separately",
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "feature_dataset": str(feature_path),
        "feature_sha256": sha256(feature_path),
        "feature_metadata_sha256": sha256(feature_path.with_suffix(".metadata.json")),
        "feature_source_cutoff": feature_metadata["as_of"],
        "config": config,
        "config_sha256": sha256(config_path),
        "simulation": feature_metadata["simulation"],
        "candles": len(candles),
        "historical_observations": len(features),
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
        "partitions": boundaries,
        "ant_input_columns": list(features.columns[1:]),
        "raw_input_columns": RAW_COLUMNS,
        "model": {
            "scaler": "StandardScaler",
            "estimator": "LogisticRegression",
            "random_state": config["random_state"],
            **config["logistic_regression"],
        },
        "execution": "decision D after candle D-1h completes; bid open entry D; exit D+horizon; idealized zero latency",
        "pnl_basis": metrics["pnl_basis"],
        "costs": "zero spread, slippage, commission, swap, financing",
        "equity_basis": "realized trades with initial zero; no intratrade drawdown",
        "boundary_rule": "exit strictly before next partition first decision; test exits must exist in source",
        "model_fit": "training only; no refit, calibration, tuning, or threshold search",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    # Serialize before touching outputs. Strict JSON rejects accidental nonfinite metrics.
    metrics_json = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    metadata_json = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(outputs[0], index=False, lineterminator="\n")
    trades.to_csv(outputs[1], index=False, lineterminator="\n")
    equity.to_csv(outputs[2], index=False, lineterminator="\n")
    outputs[3].write_text(metrics_json, encoding="utf-8")
    metadata["output_sha256"] = {
        key: sha256(path) for key, path in zip(OUTPUT_KEYS[:-1], outputs[:-1], strict=True)
    }
    metadata_json = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    outputs[4].write_text(metadata_json, encoding="utf-8")
    print(
        f"Complete history: {len(features):,} feature rows; {int(eligible.sum()):,} shared eligible observations."
    )
    print(metrics["pnl_basis"])
    for partition, values in metrics["partitions"].items():
        print(f"{partition}: {boundaries[partition]['eligible_rows']:,} observations")
        for strategy, result in values.items():
            trading = result["trading"]
            print(
                f"  {strategy}: {trading['completed_trades']} trades; gross P&L USD {trading['total_gross_pnl_usd']:.3f}"
            )
    print("Outputs: " + ", ".join(str(path) for path in outputs))
    return metrics
