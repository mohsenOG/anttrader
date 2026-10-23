"""Run one fixed historical experiment and export comparable research results."""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from anttrader.decoder import (
    chronological_partitions,
    decode_signals,
    fit_decoder,
    future_targets,
    historical_inputs,
    load_feature_dataset,
    predict_up,
    predictive_metrics,
    sha256,
    validate_research_config,
)

STRATEGIES = ("ant", "momentum", "raw")
OUTPUT_KEYS = (
    "signal_output",
    "metrics_output",
    "experiment_metadata_output",
)


def evaluation_samples(candles, decisions, config, lookback=256) -> dict:
    """Construct one common label/partition eligibility set before model fitting."""
    splits, boundaries = chronological_partitions(
        decisions, config["train_fraction"], config["validation_fraction"]
    )
    targets = future_targets(candles, decisions, config["prediction_horizon_bars"])
    _, momentum, momentum_valid = historical_inputs(
        candles, decisions, config["momentum_lookback_bars"], lookback
    )
    boundary_valid = splits.outcome_cutoff.isna() | targets.label_end_timestamp.lt(
        splits.outcome_cutoff
    )
    eligible = targets.target_valid & momentum_valid & boundary_valid
    for partition in ("train", "validation", "test"):
        if not (eligible & splits.partition.eq(partition)).any():
            raise ValueError(
                "No shared eligible observations in one or more partitions."
            )
    for partition in ("train", "validation", "test"):
        selected = eligible & splits.partition.eq(partition)
        times = decisions.loc[selected]
        boundaries[partition].update(
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
    return {
        "splits": splits,
        "boundaries": boundaries,
        "targets": targets,
        "momentum": momentum,
        "momentum_valid": momentum_valid,
        "boundary_valid": boundary_valid,
        "eligible": eligible,
    }


def evaluate_features(
    candles: pd.DataFrame,
    features: pd.DataFrame,
    config: dict,
    lookback: int = 256,
    prepared: dict | None = None,
) -> tuple:
    """Fit independent decoders on shared, purged chronological observations."""
    decisions = features.decision_timestamp
    sample = (
        prepared
        if prepared is not None
        else evaluation_samples(candles, decisions, config, lookback)
    )
    splits, boundaries, targets = (
        sample[k] for k in ("splits", "boundaries", "targets")
    )
    momentum, momentum_valid, boundary_valid, eligible = (
        sample[k] for k in ("momentum", "momentum_valid", "boundary_valid", "eligible")
    )
    raw, _, _ = historical_inputs(
        candles, decisions, config["momentum_lookback_bars"], lookback
    )
    train = eligible & splits.partition.eq("train")
    evaluation = eligible & splits.partition.ne("train")
    signals = targets.loc[evaluation].copy()
    signals["reference_price"] = (
        candles.set_index("timestamp")
        .close.reindex(
            pd.DatetimeIndex(signals.decision_timestamp) - pd.Timedelta(hours=1)
        )
        .to_numpy()
    )
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
    metrics = {"partitions": {}}
    for partition in ("validation", "test"):
        frame = signals.loc[signals.partition.eq(partition)]
        metrics["partitions"][partition] = {
            strategy: {
                "predictive": predictive_metrics(
                    frame.target,
                    frame[f"{strategy}_signal"],
                    None if strategy == "momentum" else frame[f"{strategy}_p_up"],
                )
            }
            for strategy in STRATEGIES
        }
    return (
        signals,
        metrics,
        boundaries,
        targets,
        eligible,
        momentum_valid,
        boundary_valid,
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
    (
        signals,
        metrics,
        boundaries,
        targets,
        eligible,
        momentum_valid,
        boundary_valid,
    ) = evaluate_features(candles, features, config, config["lookback_returns"])
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
        "raw_input_columns": [
            f"return_state_{i:03d}" for i in range(config["lookback_returns"])
        ],
        "model": {
            "scaler": "StandardScaler",
            "estimator": "LogisticRegression",
            "random_state": config["random_state"],
            **config["logistic_regression"],
        },
        "schema": "signals-only-v1",
        "label_rule": f"ln(open at D+{config['prediction_horizon_bars']}h / open at D) > 0; retrospective labels only",
        "reference_rule": "last completed candle close at D-1h; display reference only",
        "boundary_rule": "label end strictly before next partition first decision",
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
    outputs[1].write_text(metrics_json, encoding="utf-8")
    metadata["output_sha256"] = {
        key: sha256(path)
        for key, path in zip(OUTPUT_KEYS[:-1], outputs[:-1], strict=True)
    }
    metadata_json = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    outputs[2].write_text(metadata_json, encoding="utf-8")
    print(
        f"Complete history: {len(features):,} feature rows; {int(eligible.sum()):,} shared eligible observations."
    )
    for partition, values in metrics["partitions"].items():
        print(f"{partition}: {boundaries[partition]['eligible_rows']:,} observations")
        for strategy, result in values.items():
            counts = result["predictive"]["signal_counts"]
            print(
                f"  {strategy}: "
                + ", ".join(f"{name} {count:,}" for name, count in counts.items())
            )
    print("Outputs: " + ", ".join(str(path) for path in outputs))
    return metrics
