"""Exact historical alignment, purged chronological labels, and frozen decoders."""

import hashlib
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from anttrader.data import HOUR, utc_timestamp
from anttrader.features import feature_columns
from anttrader.grid import binary_states, log_returns
from anttrader.simulation import SimulationConfig

RAW_COLUMNS = [f"return_state_{i:03d}" for i in range(256)]


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decision_times(values) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex([utc_timestamp(value) for value in values])
    if (
        times.empty
        or not times.is_unique
        or not times.is_monotonic_increasing
        or not (times == times.floor("h")).all()
    ):
        raise ValueError("Decision timestamps must be unique, ordered UTC hours.")
    return times


def validate_research_config(config: dict) -> None:
    if (
        config["decoder"] != "logistic_regression"
        or config["standard_scaler"] is not True
    ):
        raise ValueError("Topic 4 requires logistic_regression and StandardScaler.")
    horizon = config["prediction_horizon_bars"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("prediction_horizon_bars must be a positive integer.")
    if (
        type(config["momentum_lookback_bars"]) is not int
        or config["momentum_lookback_bars"] < 1
    ):
        raise ValueError("momentum_lookback_bars must be a positive integer.")
    fractions = [
        config[key]
        for key in ("train_fraction", "validation_fraction", "test_fraction")
    ]
    if any(
        isinstance(x, bool) or not np.isfinite(x) or not 0 < x < 1 for x in fractions
    ) or not np.isclose(sum(fractions), 1):
        raise ValueError("Split fractions must be positive and sum to 1.")
    decode_signals([0.5], buy=config["buy_threshold"], sell=config["sell_threshold"])


def load_feature_dataset(
    path: Path, candles: pd.DataFrame, source: Path, config: dict
) -> tuple[pd.DataFrame, dict]:
    """Require the complete export; compare only feature-producing settings."""
    if not path.is_file() or not path.with_suffix(".metadata.json").is_file():
        raise ValueError(
            "Feature CSV/metadata missing. Run python -m anttrader.pipeline --stage features without --limit."
        )
    metadata = json.loads(
        path.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    simulation = SimulationConfig(**config["simulation"])
    columns = ["decision_timestamp", *feature_columns(simulation)]
    if len(columns) != 61:
        raise ValueError("Topic 4 requires exactly 60 configured ant features.")
    stored_simulation = SimulationConfig(**metadata["simulation"])
    if asdict(stored_simulation) != asdict(simulation):
        raise ValueError("Feature simulation metadata mismatch.")
    keys = (
        "symbol",
        "timeframe",
        "data_source",
        "timestamp_timezone",
        "timestamp_format",
        "candle_timestamp",
        "quote_side",
        "drop_zero_volume",
        "lookback_returns",
        "grid_size",
        "encoding",
        "layout",
        "simulation",
    )
    if any(metadata["config"].get(key) != config[key] for key in keys):
        raise ValueError("Feature market/encoding/configuration metadata mismatch.")
    if metadata["source_sha256"] != sha256(source):
        raise ValueError("Feature source SHA-256 mismatch; regenerate features.")
    expected = len(candles) - 256
    if (
        metadata["limit"] is not None
        or metadata["simulations"] != expected
        or metadata["eligible_grids"] != expected
        or metadata["candles"] != len(candles)
        or metadata["failures"] != 0
        or metadata["feature_count"] != 60
        or metadata["columns"] != columns
        or metadata["transitions"] != expected * simulation.steps
    ):
        raise ValueError(
            "Complete historical feature dataset required; regenerate without --limit."
        )
    frame = pd.read_csv(path)
    if list(frame) != columns or len(frame) != expected:
        raise ValueError(
            "Feature CSV columns/count mismatch; only the 60 ant features are permitted."
        )
    times = decision_times(frame["decision_timestamp"])
    expected_times = pd.DatetimeIndex(candles.timestamp.iloc[256:] + HOUR)
    if not times.equals(expected_times):
        raise ValueError(
            "Feature timestamps do not exactly match completed market windows."
        )
    values = frame.iloc[:, 1:].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "Ant features must be finite numerical values without missing data."
        )
    frame["decision_timestamp"] = times
    return frame, metadata


def future_targets(candles: pd.DataFrame, decisions, horizon: int = 4) -> pd.DataFrame:
    """Build retrospective labels; require consecutive hours through the horizon."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    times = decision_times(decisions)
    prices = candles.set_index("timestamp")["open"]
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Market timestamps must be unique and chronological.")
    if not (np.isfinite(prices) & prices.gt(0)).all():
        raise ValueError("Opening prices must be finite and positive.")
    valid = np.ones(len(times), dtype=bool)
    for offset in range(-1, horizon + 1):
        valid &= prices.index.get_indexer(times + offset * HOUR) >= 0
    start_price = prices.reindex(times).to_numpy()
    ends = times + horizon * HOUR
    label_end_price = prices.reindex(ends).to_numpy()
    returns = np.full(len(times), np.nan)
    returns[valid] = log_returns_pairs(start_price[valid], label_end_price[valid])
    result = pd.DataFrame(
        {
            "decision_timestamp": times,
            "label_start_timestamp": times,
            "label_start_price": start_price,
            "label_end_timestamp": ends,
            "label_end_price": label_end_price,
            "future_return": returns,
            "target": pd.array(np.where(valid, returns > 0, None), dtype="Int64"),
            "target_valid": valid,
        }
    )
    # Missing hours inside observed history are gaps; the final horizon is unavailable.
    result["target_exclusion"] = np.where(
        valid, "", np.where(ends > prices.index[-1], "unavailable_tail", "market_gap")
    )
    return result


def log_returns_pairs(start_price: np.ndarray, end_price: np.ndarray) -> np.ndarray:
    with np.errstate(over="raise", under="raise", divide="raise", invalid="raise"):
        try:
            return np.log(end_price / start_price)
        except FloatingPointError as exc:
            raise ValueError("Price ratio exceeds the numerical range.") from exc


def chronological_partitions(
    decisions, train: float, validation: float
) -> tuple[pd.DataFrame, dict]:
    times = decision_times(decisions)
    if not 0 < train < 1 or not 0 < validation < 1 or train + validation >= 1:
        raise ValueError("Invalid chronological split fractions.")
    a, b = int(len(times) * train), int(len(times) * (train + validation))
    if not 0 < a < b < len(times):
        raise ValueError("Need observations in all three chronological partitions.")
    partitions = np.full(len(times), "test", dtype=object)
    partitions[:a], partitions[a:b] = "train", "validation"
    cutoffs = pd.Series([pd.NaT] * len(times), dtype="datetime64[ns, UTC]")
    cutoffs.iloc[:a], cutoffs.iloc[a:b] = times[a], times[b]
    boundaries = {}
    for name, start, stop in (
        ("train", 0, a),
        ("validation", a, b),
        ("test", b, len(times)),
    ):
        boundaries[name] = {
            "original_rows": stop - start,
            "first_decision": times[start].isoformat(),
            "last_decision": times[stop - 1].isoformat(),
            "exclusive_outcome_cutoff": times[stop].isoformat()
            if stop < len(times)
            else None,
        }
    return pd.DataFrame(
        {"partition": partitions, "outcome_cutoff": cutoffs}
    ), boundaries


def historical_inputs(
    candles: pd.DataFrame, decisions, momentum_bars: int = 4
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Original chronological Topic 2 states, and past-only consecutive momentum."""
    times = decision_times(decisions)
    end = pd.Index(candles.timestamp).get_indexer(times - HOUR)
    if (end < 256).any():
        raise ValueError("Each decision requires 257 completed historical closes.")
    states = binary_states(log_returns(candles.close.to_numpy()))
    windows = np.lib.stride_tricks.sliding_window_view(states, 256)
    raw = pd.DataFrame(windows[end - 256], columns=RAW_COLUMNS)
    if momentum_bars > 256:
        raise ValueError("Momentum lookback cannot exceed the available input window.")
    stamps = pd.DatetimeIndex(candles.timestamp)
    momentum_valid = np.ones(len(end), dtype=bool)
    for offset in range(1, momentum_bars + 1):
        momentum_valid &= stamps[end - offset] == times - (offset + 1) * HOUR
    closes = candles.close.to_numpy()
    momentum = log_returns_pairs(closes[end - momentum_bars], closes[end])
    signals = pd.Series(
        np.where(momentum > 0, "BUY", np.where(momentum < 0, "SELL", "HOLD"))
    )
    return raw, signals, momentum_valid


def fit_decoder(inputs: pd.DataFrame, targets: pd.Series, config: dict) -> Pipeline:
    if targets.nunique() != 2 or targets.isna().any():
        raise ValueError(
            "Training requires both target classes without missing labels."
        )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    random_state=config["random_state"], **config["logistic_regression"]
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(inputs, targets.astype(int))
    return model


def predict_up(model: Pipeline, inputs: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(inputs)[:, list(model.classes_).index(1)]
    if (
        not np.isfinite(probabilities).all()
        or not ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise ValueError("Predicted probabilities must be finite and in [0, 1].")
    return probabilities


def decode_signals(
    probabilities, *, buy: float = 0.55, sell: float = 0.45
) -> np.ndarray:
    if not 0 <= sell < buy <= 1:
        raise ValueError("Thresholds must satisfy 0 <= sell < buy <= 1.")
    values = np.asarray(probabilities, dtype=float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or not ((values >= 0) & (values <= 1)).all()
    ):
        raise ValueError("Probabilities must be finite and in [0, 1].")
    return np.where(values >= buy, "BUY", np.where(values <= sell, "SELL", "HOLD"))


def predictive_metrics(targets, signals, probabilities=None) -> dict:
    labels = np.asarray(targets, dtype=int)
    signals = np.asarray(signals)
    count = len(labels)
    result = {
        "valid_observations": count,
        "signal_counts": {
            name: int((signals == name).sum()) for name in ("BUY", "SELL", "HOLD")
        },
        "signal_fractions": {
            name: float((signals == name).mean()) if count else None
            for name in ("BUY", "SELL", "HOLD")
        },
    }
    if probabilities is None:
        active = signals != "HOLD"
        accuracy = (
            float(accuracy_score(labels[active], signals[active] == "BUY"))
            if active.any()
            else None
        )
        auc, note = None, "momentum has no probability estimate"
        scope = "non-HOLD momentum observations"
    else:
        values = np.asarray(probabilities, dtype=float)
        accuracy = float(accuracy_score(labels, values >= 0.5)) if count else None
        defined = len(np.unique(labels)) == 2
        auc = float(roc_auc_score(labels, values)) if defined else None
        note = None if defined else "ROC AUC requires both target classes"
        scope = "all eligible observations; probability >= 0.5 predicts up"
    actionable = signals != "HOLD"
    result.update(
        actionable_observations=int(actionable.sum()),
        actionable_accuracy=float(
            accuracy_score(labels[actionable], signals[actionable] == "BUY")
        )
        if actionable.any()
        else None,
        actionable_accuracy_scope="BUY/SELL observations only; BUY predicts up, SELL predicts non-up",
        actionable_accuracy_note=None
        if actionable.any()
        else "no BUY/SELL observations",
        directional_accuracy=accuracy,
        accuracy_scope=scope,
        roc_auc=auc,
        roc_auc_note=note,
    )
    return result
