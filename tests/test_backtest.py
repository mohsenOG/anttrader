"""Execution, position state, exact gross P&L, and undefined metrics."""

import numpy as np
import pandas as pd
import pytest
from test_decoder import history

from anttrader.backtest import predictive_metrics, simulate_trades, trading_metrics
from anttrader.data import HOUR


def orders(candles, signals):
    return pd.DataFrame(
        {
            "decision_timestamp": candles.timestamp.iloc[
                1 : len(signals) + 1
            ].to_numpy(),
            "signal": signals,
            "p_up": 0.6,
        }
    )


@pytest.mark.parametrize("signal,direction", [("BUY", 1), ("SELL", -1)])
def test_side_pnl_quantity_zero_cost_and_four_hour_timing(signal, direction):
    candles = history(12)
    trades, equity = simulate_trades(candles, orders(candles, [signal]), quantity=2)
    row = trades.iloc[0]
    assert row.direction == direction and row.quantity_ounces == 2
    assert row.decision_timestamp == row.entry_timestamp == candles.timestamp.iloc[1]
    assert (
        row.exit_timestamp
        == row.entry_timestamp + 4 * HOUR
        == candles.timestamp.iloc[5]
    )
    assert row.holding_period_bars == 4
    expected = direction * (candles.open.iloc[5] - candles.open.iloc[1]) * 2
    assert row.gross_pnl_usd == pytest.approx(expected)
    assert equity.cumulative_gross_pnl_usd.tolist() == [0, expected]
    metrics = trading_metrics(trades, equity)
    assert (
        metrics["completed_trades"]
        == metrics["buy_trades"] + metrics["sell_trades"]
        == 1
    )
    assert metrics["total_gross_pnl_usd"] == pytest.approx(expected)


def test_hold_overlaps_exit_reentry_and_no_incomplete_trade():
    candles = history(16)
    decisions = orders(candles, ["HOLD"] + ["BUY"] * 14)
    trades, equity = simulate_trades(candles, decisions)
    assert (
        trades.entry_timestamp.tolist() == candles.timestamp.iloc[[2, 6, 10]].tolist()
    )
    assert (
        trades.exit_timestamp.tolist() == candles.timestamp.iloc[[6, 10, 14]].tolist()
    )
    assert trades.decision_timestamp.is_unique
    assert equity.cumulative_gross_pnl_usd.iloc[-1] == pytest.approx(
        trades.gross_pnl_usd.sum()
    )
    assert (
        trades.entry_timestamp.iloc[1:].reset_index(drop=True)
        >= trades.exit_timestamp.iloc[:-1].reset_index(drop=True)
    ).all()
    trades, equity = simulate_trades(candles, orders(candles, ["HOLD"] * 14))
    metrics = trading_metrics(trades, equity)
    assert metrics["completed_trades"] == 0
    assert (
        metrics["win_rate"]
        is metrics["average_gross_pnl_usd"]
        is metrics["profit_factor"]
        is None
    )
    assert metrics["maximum_realized_drawdown_usd"] == 0


def test_gap_and_end_without_exit_do_not_open_trades():
    candles = history(12)
    decisions = orders(candles, ["BUY"])
    for missing in range(1, 6):
        assert simulate_trades(candles.drop(index=missing), decisions)[0].empty
    assert simulate_trades(candles.iloc[:5], decisions)[0].empty
    with pytest.raises(ValueError, match="unique"):
        simulate_trades(candles, pd.concat([decisions, decisions]))
    with pytest.raises(ValueError, match="Unknown"):
        simulate_trades(candles, decisions.assign(signal="WAIT"))


def test_realized_drawdown_includes_initial_zero_and_undefined_factor():
    candles = history(24)
    decisions = orders(candles, ["BUY"] * 16)
    trades, equity = simulate_trades(candles, decisions)
    trades["gross_pnl_usd"] = [-5, 3, 10, -8]
    equity["cumulative_gross_pnl_usd"] = [0, -5, -2, 8, 0]
    metrics = trading_metrics(trades, equity)
    assert metrics["maximum_realized_drawdown_usd"] == 8
    assert metrics["gross_profit_usd"] == metrics["gross_loss_usd"] == 13
    assert metrics["profit_factor"] == 1
    assert metrics["win_rate"] == 0.5
    trades["gross_pnl_usd"] = [-5, 0, 0, 0]
    equity["cumulative_gross_pnl_usd"] = [0, -5, -5, -5, -5]
    assert trading_metrics(trades, equity)["maximum_realized_drawdown_usd"] == 5
    trades["gross_pnl_usd"] = [1, 0, 0, 0]
    equity["cumulative_gross_pnl_usd"] = [0, 1, 1, 1, 1]
    assert trading_metrics(trades, equity)["profit_factor"] == "infinity"
    with pytest.raises(ValueError, match="reconcile"):
        trading_metrics(trades, equity.assign(cumulative_gross_pnl_usd=0))


def test_predictive_metrics_counts_and_undefined_auc():
    metrics = predictive_metrics([1, 1, 1], ["BUY", "SELL", "HOLD"], [0.55, 0.45, 0.5])
    assert metrics["roc_auc"] is None and metrics["roc_auc_note"]
    assert metrics["directional_accuracy"] == pytest.approx(2 / 3)
    assert sum(metrics["signal_counts"].values()) == metrics["valid_observations"] == 3
    assert sum(metrics["signal_fractions"].values()) == 1
    assert predictive_metrics([1, 0], ["BUY", "SELL"], [0.6, 0.4])["roc_auc"] == 1
    momentum = predictive_metrics([1, 0, 1], ["BUY", "SELL", "HOLD"])
    assert momentum["directional_accuracy"] == 1 and momentum["roc_auc"] is None
    assert not any(
        isinstance(v, float) and not np.isfinite(v) for v in metrics.values()
    )
