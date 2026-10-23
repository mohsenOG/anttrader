"""Full Topic 3/4 synthetic integration, strict metadata, and reproducible exports."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from test_decoder import config, history

from anttrader.decoder import RAW_COLUMNS, fit_decoder, load_feature_dataset
from anttrader.experiment import OUTPUT_KEYS
from anttrader.pipeline import main


@pytest.fixture
def exported_history(tmp_path):
    settings = config()
    settings["historical_csv"] = "synthetic.csv"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    candles = history()
    # Exercise both forward label gaps and past momentum gaps in the shared dates.
    candles = candles.drop(index=[310, 402]).reset_index(drop=True)
    source = tmp_path / "synthetic.csv"
    candles.to_csv(source, index=False)
    main(path, stage="features")
    return path, settings, candles, source


def test_pipeline_outputs_shared_dates_rules_counts_and_reproducibility(
    exported_history,
    monkeypatch,
):
    fitted = []

    def checked_fit(inputs, labels, settings):
        model = fit_decoder(inputs, labels, settings)
        np.testing.assert_allclose(model.named_steps["scaler"].mean_, inputs.mean())
        fitted.append(
            (
                inputs.copy(),
                labels.copy(),
                model,
                model.named_steps["logistic"].coef_.copy(),
            )
        )
        return model

    monkeypatch.setattr("anttrader.experiment.fit_decoder", checked_fit)
    path, settings, candles, _source = exported_history
    main(path, stage="backtest")
    outputs = [path.parent / settings[key] for key in OUTPUT_KEYS]
    first = [output.read_bytes() for output in outputs]
    signals, trades, equity = [pd.read_csv(output) for output in outputs[:3]]
    metrics, metadata = [
        json.loads(output.read_text(encoding="utf-8")) for output in outputs[3:]
    ]
    assert len(fitted) == 2
    pd.testing.assert_index_equal(fitted[0][0].index, fitted[1][0].index)
    pd.testing.assert_series_equal(fitted[0][1], fitted[1][1])
    assert list(fitted[0][0]) == metadata["ant_input_columns"]
    assert list(fitted[1][0]) == RAW_COLUMNS
    for inputs, labels, model, coefficients in fitted:
        assert len(inputs) == metadata["partitions"]["train"]["eligible_rows"]
        assert inputs.index.max() < metadata["partitions"]["train"]["original_rows"]
        assert model.named_steps["scaler"].n_samples_seen_ == len(inputs)
        np.testing.assert_array_equal(model.named_steps["logistic"].coef_, coefficients)
    assert len(signals) and set(signals.partition) == {"validation", "test"}
    assert (
        signals.decision_timestamp.is_monotonic_increasing
        and signals.decision_timestamp.is_unique
    )
    assert signals[["ant_p_up", "raw_p_up"]].notna().all().all()
    assert signals[["ant_p_up", "raw_p_up"]].ge(0).all().all()
    assert metadata["historical_observations"] == len(candles) - 256
    assert metadata["excluded_market_gap"] > 0
    assert metadata["excluded_momentum_gap_after_labeling"] > 0
    assert (
        sum(x["eligible_rows"] for x in metadata["partitions"].values())
        == metadata["shared_eligible_observations"]
    )
    for partition in ("train", "validation"):
        info = metadata["partitions"][partition]
        assert pd.Timestamp(info["latest_eligible_exit"]) < pd.Timestamp(
            info["exclusive_outcome_cutoff"]
        )
    for partition in ("validation", "test"):
        expected_dates = signals.loc[
            signals.partition.eq(partition), "decision_timestamp"
        ]
        for strategy in ("ant", "raw", "momentum"):
            result = metrics["partitions"][partition][strategy]
            assert result["predictive"]["valid_observations"] == len(expected_dates)
            assert sum(result["predictive"]["signal_counts"].values()) == len(
                expected_dates
            )
            completed = trades.loc[
                trades.partition.eq(partition) & trades.strategy.eq(strategy)
            ]
            curve = equity.loc[
                equity.partition.eq(partition) & equity.strategy.eq(strategy)
            ]
            assert completed.decision_timestamp.is_unique
            assert completed.decision_timestamp.isin(expected_dates).all()
            assert len(completed) == result["trading"]["completed_trades"]
            assert curve.cumulative_gross_pnl_usd.iloc[0] == 0
            assert curve.cumulative_gross_pnl_usd.iloc[-1] == pytest.approx(
                completed.gross_pnl_usd.sum()
            )
            assert completed.gross_pnl_usd.sum() == pytest.approx(
                result["trading"]["total_gross_pnl_usd"]
            )
            if len(completed):
                duration = pd.to_datetime(completed.exit_timestamp) - pd.to_datetime(
                    completed.entry_timestamp
                )
                assert duration.eq(pd.Timedelta(hours=4)).all()
                assert completed.quantity_ounces.eq(1).all()
                np.testing.assert_allclose(
                    completed.gross_pnl_usd,
                    completed.direction
                    * (completed.exit_price - completed.entry_price),
                )
    main(path, stage="backtest")
    assert [output.read_bytes() for output in outputs] == first
    with pytest.raises(SystemExit, match="features"):
        main(path, stage="backtest", limit=10)


@pytest.mark.parametrize(
    "change",
    [
        "simulation",
        "source",
        "partial",
        "columns",
        "nonfinite",
        "timestamps",
        "rows",
        "market",
    ],
)
def test_bad_feature_metadata_or_data_is_rejected(exported_history, change):
    path, settings, candles, source = exported_history
    feature = path.parent / settings["feature_output"]
    metadata_path = feature.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(feature)
    if change == "simulation":
        metadata["simulation"]["initial_direction"] = "east"
    elif change == "source":
        metadata["source_sha256"] = "0" * 64
    elif change == "partial":
        metadata["limit"] = 10
    elif change == "columns":
        frame["target"] = 1
    elif change == "nonfinite":
        frame[frame.columns[1]] = frame.iloc[:, 1].astype(float)
        frame.iloc[0, 1] = np.inf
    elif change == "timestamps":
        frame.iloc[1, 0] = frame.iloc[0, 0]
    elif change == "rows":
        frame = frame.iloc[:-1]
    elif change == "market":
        metadata["config"]["quote_side"] = "ask"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    frame.to_csv(feature, index=False)
    with pytest.raises(ValueError):
        load_feature_dataset(feature, candles, source, settings)


def test_output_paths_cannot_overwrite_inputs_or_each_other(exported_history):
    path, settings, _, source = exported_history
    for output in ("historical_csv", "feature_output", "trade_output"):
        altered = {**settings, "signal_output": settings[output]}
        path.write_text(yaml.safe_dump(altered), encoding="utf-8")
        before = source.read_bytes()
        with pytest.raises(SystemExit, match="must be distinct"):
            main(path, stage="backtest")
        assert source.read_bytes() == before
