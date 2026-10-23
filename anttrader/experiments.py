"""Small synchronous, reproducible historical experiment lab (JSON and CSV)."""

import argparse
import hashlib
import json
import platform
import re
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn
import yaml

from anttrader.data import HOUR, load_gold_data
from anttrader.decoder import (
    decode_signals,
    sha256,
    signal_metrics,
    validate_research_config,
)
from anttrader.experiment import STRATEGIES, evaluate_features, evaluation_samples
from anttrader.features import feature_columns, iter_feature_rows
from anttrader.grid import iter_historical_grids
from anttrader.pipeline import DEFAULT_CONFIG
from anttrader.simulation import DIRECTIONS, SimulationConfig

SCHEMA = "anttrader-lab-v1.1"
GRID_PRESETS = ((8, 8), (12, 12), (16, 16), (16, 6), (20, 20))
STEP_PRESETS = (256, 512, 1024, 2048, 4096)
ARTIFACTS = ("features.csv", "predictions.csv", "metrics.json")
ID_PATTERN = re.compile(r"[0-9a-f]{24}")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def checkpoints(steps: int) -> tuple[int, ...]:
    """Half-up integer rounding; reserve one step for each later checkpoint."""
    if type(steps) is not int or not 4 <= steps <= 16384:
        raise ValueError("Experiment steps must be an integer in 4..16384.")
    points = []
    for index, divisor in enumerate((8, 4, 2, 1)):
        rounded = (steps + divisor // 2) // divisor
        points.append(
            max(points[-1] + 1 if points else 1, min(rounded, steps - (3 - index)))
        )
    return tuple(points)


@dataclass(frozen=True)
class ExperimentConfig:
    rows: int = 16
    columns: int = 16
    steps: int = 2048
    direction: str = "north"
    start: str = "newest_return"

    def __post_init__(self):
        if any(
            type(x) is not int or not 2 <= x <= 32 for x in (self.rows, self.columns)
        ):
            raise ValueError("Experiment rows and columns must be integers in 2..32.")
        checkpoints(self.steps)
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}.")
        if self.start not in ("newest_return", "center"):
            raise ValueError("start must be newest_return or center.")

    @property
    def cells(self):
        return self.rows * self.columns

    @property
    def simulation(self):
        row, col = (
            (self.rows // 2, self.columns // 2)
            if self.start == "center"
            else (self.rows - 1, 0 if self.rows % 2 == 0 else self.columns - 1)
        )
        return SimulationConfig(
            steps=self.steps,
            start_row=row,
            start_col=col,
            initial_direction=self.direction,
            checkpoints=checkpoints(self.steps),
        )

    @property
    def label(self):
        return (
            f"{self.rows} x {self.columns}, {self.steps:,} steps, "
            f"{self.direction}, {self.start}"
        )


def batch_configs(grids, steps, *, direction="north", start="newest_return"):
    return list(
        dict.fromkeys(
            ExperimentConfig(r, c, s, direction, start)
            for (r, c), s in product(grids, steps)
        )
    )


def software_identity():
    names = (
        "data.py",
        "grid.py",
        "simulation.py",
        "features.py",
        "decoder.py",
        "experiment.py",
        "experiments.py",
    )
    return {
        "schema": SCHEMA,
        "code": {n: sha256(Path(__file__).with_name(n)) for n in names},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def experiment_root(config_path, settings):
    allowed = Path(config_path).resolve().parent / "results" / "experiments"
    root = Path(config_path).resolve().parent / settings.get("experiment_lab", {}).get(
        "output_dir", "results/experiments"
    )
    # Resolve before writes; custom roots may only be descendants of the lab folder.
    if (
        root.is_symlink()
        or root.is_junction()
        or not root.resolve().is_relative_to(allowed)
    ):
        raise ValueError("Experiment output_dir must stay within results/experiments.")
    return root.resolve()


def _directory(root, experiment_id):
    root = Path(root).resolve()
    if not isinstance(experiment_id, str) or not ID_PATTERN.fullmatch(experiment_id):
        raise ValueError("Invalid experiment ID.")
    directory = root / experiment_id
    if (
        directory.is_symlink()
        or directory.is_junction()
        or directory.resolve().parent != root
    ):
        raise ValueError("Experiment path must be a direct, regular child of its root.")
    return directory


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}.")
    return value


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _metrics(signals):
    return {
        "partitions": {
            partition: {
                name: {
                    "predictive": signal_metrics(
                        frame.target,
                        frame[f"{name}_signal"],
                        None if name == "momentum" else frame[f"{name}_p_up"],
                    )
                }
                for name in STRATEGIES
            }
            for partition in ("validation", "test")
            for frame in [signals.loc[signals.partition.eq(partition)]]
        }
    }


def prepare_plan(
    config_path=DEFAULT_CONFIG,
    configurations=None,
    *,
    sample_candles=None,
    cohort_lookback=None,
    base_config=None,
    as_of=None,
):
    """Load once and prepare common labels. No simulations or model fitting here."""
    path = Path(config_path).resolve()
    settings = (
        base_config
        if base_config is not None
        else yaml.safe_load(path.read_text(encoding="utf-8"))
    )
    validate_research_config(settings)
    fixed = {
        "symbol": "XAU/USD",
        "timeframe": "1H",
        "data_source": "Dukascopy",
        "candle_timestamp": "open",
        "quote_side": "bid",
        "encoding": "binary_log_return_sign",
        "layout": "serpentine",
        "prediction_horizon_bars": 4,
        "momentum_lookback_bars": 4,
        "buy_threshold": 0.55,
        "sell_threshold": 0.45,
        "random_state": 42,
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
    }
    if any(settings.get(k) != v for k, v in fixed.items()) or settings[
        "logistic_regression"
    ] != {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}:
        raise ValueError(
            "The lab requires the approved v1 market, label, split and model settings."
        )
    lab = settings.get("experiment_lab", {})
    configs = list(
        dict.fromkeys(
            configurations
            if configurations is not None
            else [ExperimentConfig(**v) for v in lab.get("configurations", [{}])]
        )
    )
    maximum = lab.get("max_batch", 30)
    if type(maximum) is not int or not 1 <= maximum <= 30:
        raise ValueError("experiment_lab.max_batch must be an integer in 1..30.")
    if (
        not configs
        or len(configs) > maximum
        or any(not isinstance(v, ExperimentConfig) for v in configs)
    ):
        raise ValueError(f"Select between 1 and {maximum} experiment configurations.")
    if sample_candles is None:
        sample_candles = lab.get("sample_candles")
    if sample_candles is not None and (
        type(sample_candles) is not int or sample_candles < 1
    ):
        raise ValueError("sample_candles must be a positive integer or null.")
    source = (path.parent / settings["historical_csv"]).resolve()
    root = experiment_root(path, settings)
    if source.is_relative_to(root) or path.is_relative_to(root):
        raise ValueError(
            "Source data and configuration must stay outside the experiment output directory."
        )
    source_hash = sha256(source)
    candles = load_gold_data(
        source,
        source_timezone=settings["timestamp_timezone"],
        timestamp_format=settings["timestamp_format"],
        drop_zero_volume=settings["drop_zero_volume"],
        as_of=as_of,
    )
    if sample_candles is not None:
        candles = candles.iloc[:sample_candles].copy()
    lookback = (
        max(v.cells for v in configs) if cohort_lookback is None else cohort_lookback
    )
    if (
        type(lookback) is not int
        or not max(v.cells for v in configs) <= lookback <= 1024
    ):
        raise ValueError("Invalid common cohort lookback.")
    if len(candles) <= lookback:
        raise ValueError(
            f"Need more than {lookback} completed candles for this cohort."
        )
    decisions = (candles.timestamp.iloc[lookback:] + HOUR).reset_index(drop=True)
    shared = evaluation_samples(candles, decisions, settings, lookback)
    if sha256(source) != source_hash:
        raise ValueError("Source changed while loading; retry after the data update.")
    dataset = {
        "source": str(source),
        "source_sha256": source_hash,
        "sample_candles": sample_candles,
        "candles": len(candles),
        "cutoff": (candles.timestamp.iloc[-1] + HOUR).isoformat(),
        "completed_sha256": hashlib.sha256(
            candles.to_csv(index=False).encode()
        ).hexdigest(),
    }
    cohort = {
        "lookback": lookback,
        "decisions": [v.isoformat() for v in decisions],
        "eligible": shared["eligible"].tolist(),
        "partitions": shared["splits"].partition.tolist(),
        "boundaries": shared["boundaries"],
        "evaluation": fixed,
        "dataset": dataset,
    }
    cohort_id = digest(cohort)
    software = software_identity()
    identities = [
        {
            "configuration": asdict(v),
            "dataset": dataset,
            "cohort_id": cohort_id,
            "software": software,
            "research": {
                k: settings[k]
                for k in (
                    "timestamp_timezone",
                    "timestamp_format",
                    "drop_zero_volume",
                    "decoder",
                    "standard_scaler",
                    "logistic_regression",
                )
            },
        }
        for v in configs
    ]
    return {
        "config_path": path,
        "settings": settings,
        "configs": configs,
        "candles": candles,
        "decisions": decisions,
        "shared": shared,
        "dataset": dataset,
        "cohort": cohort,
        "cohort_id": cohort_id,
        "software": software,
        "identities": identities,
        "root": experiment_root(path, settings),
        "transitions": sum(len(decisions) * v.steps for v in configs),
    }


def load_result(root, experiment_id):
    """Verify generated artifacts without fitting models or running simulations."""
    return _load_directory(_directory(root, experiment_id), experiment_id)


def _validate_metadata(metadata, experiment_id):
    if (
        metadata.get("status") != "completed"
        or metadata.get("schema") != SCHEMA
        or metadata.get("id") != experiment_id
        or digest(metadata["identity"])[:24] != experiment_id
        or metadata["configuration"] != metadata["identity"]["configuration"]
        or digest(metadata["cohort"]) != metadata["identity"]["cohort_id"]
        or metadata["cohort_id"] != metadata["identity"]["cohort_id"]
        or metadata["dataset"] != metadata["identity"]["dataset"]
        or metadata["partitions"] != metadata["cohort"]["boundaries"]
        or any(
            metadata["base_config"].get(k) != v
            for k, v in metadata["identity"]["research"].items()
        )
        or any(
            metadata["base_config"].get(k) != v
            for k, v in metadata["cohort"]["evaluation"].items()
        )
    ):
        raise ValueError("Experiment metadata is incomplete or inconsistent.")


def _load_directory(directory, experiment_id):
    metadata = _json(directory / "metadata.json")
    _validate_metadata(metadata, experiment_id)
    for name in ARTIFACTS:
        target = directory / name
        if (
            target.is_symlink()
            or target.is_junction()
            or sha256(target) != metadata["artifacts"].get(name)
        ):
            raise ValueError(f"Experiment artifact is missing or changed: {name}.")
    features = pd.read_csv(directory / "features.csv")
    signals = pd.read_csv(directory / "predictions.csv")
    for frame in (features, signals):
        frame["decision_timestamp"] = pd.to_datetime(frame.decision_timestamp, utc=True)
    for column in ("label_start_timestamp", "label_end_timestamp"):
        signals[column] = pd.to_datetime(signals[column], utc=True)
    config = ExperimentConfig(**metadata["configuration"])
    if list(features) != ["decision_timestamp", *feature_columns(config.simulation)]:
        raise ValueError("Experiment feature columns do not match its configuration.")
    if (
        features.decision_timestamp.astype(str).tolist()
        != pd.to_datetime(metadata["cohort"]["decisions"], utc=True)
        .astype(str)
        .tolist()
        or not np.isfinite(features.iloc[:, 1:].to_numpy()).all()
    ):
        raise ValueError("Experiment feature samples are inconsistent.")
    eligible = np.asarray(metadata["cohort"]["eligible"], dtype=bool)
    partitions = np.asarray(metadata["cohort"]["partitions"])
    expected = features.loc[eligible & (partitions != "train"), "decision_timestamp"]
    if not pd.DatetimeIndex(signals.decision_timestamp).equals(
        pd.DatetimeIndex(expected)
    ) or not np.array_equal(
        signals.partition, partitions[eligible & (partitions != "train")]
    ):
        raise ValueError("Experiment prediction samples are inconsistent.")
    numeric = [
        "target",
        "reference_price",
        "ant_p_up",
        "raw_p_up",
        "future_return",
        "label_start_price",
        "label_end_price",
    ]
    if (
        not signals.target.isin([0, 1]).all()
        or not np.isfinite(signals[numeric].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Experiment predictions contain invalid numerical values.")
    for method in ("ant", "raw"):
        expected_signals = decode_signals(signals[f"{method}_p_up"])
        if not signals[f"{method}_signal"].eq(expected_signals).all():
            raise ValueError("Saved signals differ from threshold classifications.")
    if not signals.momentum_signal.isin(["BUY", "SELL", "HOLD"]).all():
        raise ValueError("Invalid saved momentum signal.")
    metrics = _json(directory / "metrics.json")
    from anttrader.dashboard import _metrics_match

    _metrics_match(metrics, _metrics(signals))
    return {
        "metadata": metadata,
        "features": features,
        "signals": signals,
        "metrics": metrics,
        "cached": False,
    }


def history(root):
    """List completed and failed runs; corrupted results remain visible as errors."""
    root = Path(root)
    rows = []
    if root.exists():
        for directory in sorted(root.iterdir()):
            if not ID_PATTERN.fullmatch(directory.name):
                continue
            try:
                meta = _json(_directory(root, directory.name) / "metadata.json")
                if meta.get("id") != directory.name or meta.get("schema") != SCHEMA:
                    raise ValueError("Unrecognized experiment metadata.")
                if meta.get("status") not in ("completed", "failed"):
                    raise ValueError("Incomplete experiment metadata.")
                if meta["status"] == "completed":
                    _validate_metadata(meta, directory.name)
                meta["research_candidate"] = (
                    directory / "candidate.json"
                ).is_file() and _json(directory / "candidate.json").get(
                    "research_candidate", False
                )
                rows.append(meta)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                rows.append(
                    {"id": directory.name, "status": "invalid", "error": str(exc)}
                )
    return rows


def run_plan(plan, *, rerun=False, progress=None):
    """Run a prepared batch synchronously. Each configuration gets its own decoder."""
    results = []
    root = plan["root"]
    root.mkdir(parents=True, exist_ok=True)
    for number, (config, identity) in enumerate(
        zip(plan["configs"], plan["identities"], strict=True), 1
    ):
        experiment_id = digest(identity)[:24]
        directory = _directory(root, experiment_id)

        def report(stage, number=number, config=config):
            if progress:
                progress(number, len(plan["configs"]), config, stage)

        if (directory / "metadata.json").exists() and not rerun:
            try:
                result = load_result(root, experiment_id)
                result["cached"] = True
                results.append(result)
                report("Cached result loaded")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                results.append(
                    {"error": f"{exc} Use explicit rerun.", "id": experiment_id}
                )
                report(f"Error: {exc}")
            continue
        started = perf_counter()
        staging = root / f".{experiment_id}-{uuid4().hex}"
        backup = root / f".{experiment_id}-previous-{uuid4().hex}"
        staging.mkdir()
        try:
            report("Feature generation")
            grids = (
                w
                for w in iter_historical_grids(
                    plan["candles"],
                    grid_size=(config.rows, config.columns),
                    as_of=plan["dataset"]["cutoff"],
                )
                if w.decision_timestamp >= plan["decisions"].iloc[0]
            )
            features = pd.DataFrame(iter_feature_rows(grids, config.simulation))
            features["decision_timestamp"] = pd.to_datetime(
                features.decision_timestamp, utc=True
            )
            if not pd.DatetimeIndex(features.decision_timestamp).equals(
                pd.DatetimeIndex(plan["decisions"])
            ):
                raise ValueError("Feature generation differs from the common cohort.")
            report("Model evaluation")
            signals, _, boundaries, *_ = evaluate_features(
                plan["candles"],
                features,
                plan["settings"],
                config.cells,
                plan["shared"],
            )
            metrics = _metrics(signals)
            report("Saving results")
            features.to_csv(staging / "features.csv", index=False, lineterminator="\n")
            signals.to_csv(
                staging / "predictions.csv", index=False, lineterminator="\n"
            )
            _write_json(staging / "metrics.json", metrics)
            warnings = [
                "Test data was previously inspected; use validation for comparisons."
            ]
            for partition in ("validation", "test"):
                if (
                    metrics["partitions"][partition]["ant"]["predictive"][
                        "actionable_observations"
                    ]
                    < 30
                ):
                    warnings.append(
                        f"{partition}: fewer than 30 BUY/SELL observations."
                    )
            meta = {
                "id": experiment_id,
                "schema": SCHEMA,
                "status": "completed",
                "identity": identity,
                "configuration": asdict(config),
                "dataset": plan["dataset"],
                "cohort": plan["cohort"],
                "cohort_id": plan["cohort_id"],
                "base_config": plan["settings"],
                "config_path": str(plan["config_path"]),
                "partitions": boundaries,
                "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "elapsed_seconds": perf_counter() - started,
                "warnings": warnings,
                "artifacts": {name: sha256(staging / name) for name in ARTIFACTS},
            }
            _write_json(staging / "metadata.json", meta)
            # Validate before publication. Reject foreign files before replacing a run.
            _load_directory(staging, experiment_id)
            if directory.exists():
                _owned_files(directory)
                if (directory / "candidate.json").exists():
                    (staging / "candidate.json").write_bytes(
                        (directory / "candidate.json").read_bytes()
                    )
                directory.rename(backup)
            try:
                staging.rename(directory)
            except BaseException:
                if backup.exists():
                    backup.rename(directory)
                raise
            if backup.exists():
                # Backup contains only this experiment's prior generated files.
                _remove_generated(backup)
            result = load_result(root, experiment_id)
            results.append(result)
            report("Completed")
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
            ArithmeticError,
        ) as exc:
            if not directory.exists():
                directory.mkdir()
                _write_json(
                    directory / "metadata.json",
                    {
                        "id": experiment_id,
                        "schema": SCHEMA,
                        "status": "failed",
                        "configuration": asdict(config),
                        "error": str(exc),
                        "identity": identity,
                        "dataset": plan["dataset"],
                        "cohort": plan["cohort"],
                        "cohort_id": plan["cohort_id"],
                        "base_config": plan["settings"],
                        "config_path": str(plan["config_path"]),
                    },
                )
            results.append({"error": str(exc), "id": experiment_id})
            report(f"Error: {exc}")
        finally:
            if staging.exists():
                _remove_generated(staging)
    return results


def _owned_files(directory):
    """No recursive deletion: reject links, subdirectories and unowned files."""
    allowed = {*ARTIFACTS, "metadata.json", "candidate.json"}
    if directory.is_symlink() or directory.is_junction():
        raise ValueError("Refusing to delete a linked experiment directory.")
    files = list(directory.iterdir())
    if any(
        p.name not in allowed or not p.is_file() or p.is_symlink() or p.is_junction()
        for p in files
    ):
        raise ValueError("Experiment contains unrecognized files; nothing was deleted.")
    return files


def _remove_generated(directory):
    for path in _owned_files(directory):
        path.unlink()
    directory.rmdir()


def delete_result(root, experiment_id, *, confirmed=False):
    if confirmed is not True:
        raise ValueError("Explicit confirmation is required to delete an experiment.")
    directory = _directory(root, experiment_id)
    meta = _json(directory / "metadata.json")
    if meta.get("id") != experiment_id or meta.get("schema") != SCHEMA:
        raise ValueError("Refusing to delete an unrecognized experiment.")
    _remove_generated(directory)


def mark_candidate(root, experiment_id, candidate):
    load_result(root, experiment_id)
    _write_json(
        _directory(root, experiment_id) / "candidate.json",
        {"research_candidate": bool(candidate)},
    )


def rerun_result(root, experiment_id, progress=None):
    meta = _json(_directory(root, experiment_id) / "metadata.json")
    plan = prepare_plan(
        meta["config_path"],
        [ExperimentConfig(**meta["configuration"])],
        sample_candles=meta["dataset"]["sample_candles"],
        cohort_lookback=meta["cohort"]["lookback"],
        base_config=meta["base_config"],
        as_of=meta["dataset"]["cutoff"],
    )
    if (
        digest(plan["identities"][0])[:24] != experiment_id
        or plan["root"] != Path(root).resolve()
    ):
        raise ValueError("Inputs or software changed. Run a new experiment instead.")
    return run_plan(plan, rerun=True, progress=progress)[0]


def compatible(results):
    return (
        bool(results)
        and len(
            {
                (
                    r["metadata"]["cohort_id"],
                    digest(r["metadata"]["identity"]["software"]),
                )
                for r in results
            }
        )
        == 1
    )


def comparison_rows(results, partition="validation"):
    if partition not in ("validation", "test"):
        raise ValueError("Choose validation or test.")
    if not compatible(results):
        raise ValueError(
            "Experiments are not directly comparable. Select one cohort or rerun a common batch."
        )
    rows = []
    for result in results:
        meta = result["metadata"]
        config = meta["configuration"]
        m = result["metrics"]["partitions"][partition]["ant"]["predictive"]
        rows.append(
            {
                "ID": meta["id"],
                "Grid": f"{config['rows']} x {config['columns']}",
                "Steps": config["steps"],
                "Start": config["start"],
                "Direction": config["direction"],
                "ROC AUC": m["roc_auc"],
                "BUY/SELL accuracy": m["actionable_accuracy"],
                "Coverage": m["signal_coverage"],
                **m["signal_counts"],
                "Observations": m["valid_observations"],
                "Seconds": meta["elapsed_seconds"],
                "Status": meta["status"],
            }
        )
    frame = pd.DataFrame(rows)
    return (
        frame.sort_values("ROC AUC", ascending=False, na_position="last")
        if partition == "validation"
        else frame
    )


def disagreement(first, second, partition="validation"):
    a = first["signals"].loc[lambda f: f.partition.eq(partition)]
    b = second["signals"].loc[lambda f: f.partition.eq(partition)]
    shared = a.merge(b, on="decision_timestamp", suffixes=("_a", "_b"))
    return (
        float(shared.ant_signal_a.ne(shared.ant_signal_b).mean())
        if len(shared)
        else None
    )


def as_research(result):
    """Load candles for a selected saved view; no model fitting on tab changes."""
    from anttrader.dashboard import ResearchData

    meta = result["metadata"]
    settings = dict(meta["base_config"])
    config = ExperimentConfig(**meta["configuration"])
    source = Path(meta["dataset"]["source"])
    if sha256(source) != meta["dataset"]["source_sha256"]:
        raise ValueError(
            "Saved experiment source changed; restore its dataset to inspect it."
        )
    candles = load_gold_data(
        source,
        source_timezone=settings["timestamp_timezone"],
        timestamp_format=settings["timestamp_format"],
        drop_zero_volume=settings["drop_zero_volume"],
        as_of=meta["dataset"]["cutoff"],
    )
    if (
        hashlib.sha256(candles.to_csv(index=False).encode()).hexdigest()
        != meta["dataset"]["completed_sha256"]
    ):
        raise ValueError("Saved experiment completed dataset mismatch.")
    settings.update(
        grid_size=[config.rows, config.columns],
        lookback_returns=config.cells,
        simulation=asdict(config.simulation),
    )
    view_meta = dict(
        meta,
        source_sha256=meta["dataset"]["source_sha256"],
        feature_source_cutoff=meta["dataset"]["cutoff"],
        config_sha256=digest(settings),
    )
    return ResearchData(
        settings,
        candles,
        result["features"],
        result["signals"],
        result["metrics"],
        view_meta,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--sample-candles",
        type=int,
        help="First N completed candles; omitted uses YAML.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Recompute only the specified configuration results.",
    )
    args = parser.parse_args()
    try:
        plan = prepare_plan(args.config, sample_candles=args.sample_candles)
        print(
            f"{len(plan['configs'])} experiments; {len(plan['decisions']):,} windows each; "
            f"{plan['transitions']:,} ant transitions plus decoder fitting."
        )
        results = run_plan(
            plan,
            rerun=args.rerun,
            progress=lambda i, n, c, s: print(f"[{i}/{n}] {c.label}: {s}"),
        )
        for result in results:
            print(result.get("error") or result["metadata"]["id"])
        if any("error" in r for r in results):
            raise SystemExit(1)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
