"""One-position historical simulation and gross price-only research metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from anttrader.decoder import future_targets

TRADE_COLUMNS = [
    "decision_timestamp",
    "signal",
    "p_up",
    "entry_timestamp",
    "entry_price",
    "exit_timestamp",
    "exit_price",
    "direction",
    "quantity_ounces",
    "holding_period_bars",
    "gross_pnl_usd",
]
EQUITY_COLUMNS = ["timestamp", "cumulative_gross_pnl_usd"]


def simulate_trades(
    candles: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    quantity: float = 1.0,
    holding_bars: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recheck continuous execution hours; signals at an exit can open a new trade.

    Input columns: decision_timestamp, signal, and optional p_up. Each call
    starts flat and at zero realized P&L. No costs or intratrade accounting.
    """
    if isinstance(quantity, bool) or not np.isfinite(quantity) or quantity <= 0:
        raise ValueError("Quantity must be finite and positive.")
    if type(holding_bars) is not int or holding_bars < 1:
        raise ValueError("holding_bars must be a positive integer.")
    if not signals.signal.isin(["BUY", "SELL", "HOLD"]).all():
        raise ValueError("Unknown trading signal.")
    if "p_up" in signals:
        probabilities = signals.p_up.dropna()
        if not (np.isfinite(probabilities) & probabilities.between(0, 1)).all():
            raise ValueError("Trade probabilities must be in [0, 1].")
    if signals.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS), pd.DataFrame(columns=EQUITY_COLUMNS)
    targets = future_targets(candles, signals.decision_timestamp, holding_bars)
    records = []
    busy_until = None
    for i, row in enumerate(targets.itertuples(index=False)):
        signal = signals.signal.iloc[i]
        if (
            not row.target_valid
            or signal == "HOLD"
            or (busy_until is not None and row.decision_timestamp < busy_until)
        ):
            continue
        direction = 1 if signal == "BUY" else -1
        records.append(
            {
                "decision_timestamp": row.decision_timestamp,
                "signal": signal,
                "p_up": signals.p_up.iloc[i] if "p_up" in signals else None,
                "entry_timestamp": row.entry_timestamp,
                "entry_price": row.entry_price,
                "exit_timestamp": row.exit_timestamp,
                "exit_price": row.exit_price,
                "direction": direction,
                "quantity_ounces": quantity,
                "holding_period_bars": holding_bars,
                "gross_pnl_usd": direction
                * (row.exit_price - row.entry_price)
                * quantity,
            }
        )
        busy_until = row.exit_timestamp
    trades = pd.DataFrame(records, columns=TRADE_COLUMNS)
    equity = pd.DataFrame(
        {
            "timestamp": [targets.decision_timestamp.iloc[0], *trades.exit_timestamp],
            "cumulative_gross_pnl_usd": [0.0, *trades.gross_pnl_usd.cumsum()],
        }
    )
    return trades, equity


def trading_metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> dict:
    pnl = trades.gross_pnl_usd.astype(float)
    profit = float(pnl[pnl > 0].sum())
    loss = float(-pnl[pnl < 0].sum())
    cumulative = equity.cumulative_gross_pnl_usd.astype(float)
    if not np.isfinite(pnl).all() or not np.isfinite(cumulative).all():
        raise ValueError("P&L and realized equity must be finite.")
    if len(trades) and len(cumulative) != len(trades) + 1:
        raise ValueError("Missing realized-equity baseline or trade observations.")
    if len(cumulative) and (
        len(cumulative) != len(trades) + 1
        or cumulative.iloc[0] != 0
        or not np.allclose(cumulative.to_numpy(), np.r_[0.0, pnl.cumsum()])
    ):
        raise ValueError(
            "Realized equity must reconcile with trades, including zero baseline."
        )
    factor = profit / loss if loss > 0 else ("infinity" if profit > 0 else None)
    return {
        "total_gross_pnl_usd": float(pnl.sum()),
        "completed_trades": len(trades),
        "buy_trades": int(trades.signal.eq("BUY").sum()),
        "sell_trades": int(trades.signal.eq("SELL").sum()),
        "win_rate": float(pnl.gt(0).mean()) if len(pnl) else None,
        "average_gross_pnl_usd": float(pnl.mean()) if len(pnl) else None,
        "gross_profit_usd": profit,
        "gross_loss_usd": loss,
        "profit_factor": factor,
        "profit_factor_note": "no gross losses" if loss == 0 else None,
        "maximum_realized_drawdown_usd": float((cumulative.cummax() - cumulative).max())
        if len(cumulative)
        else 0.0,
        "long_gross_pnl_usd": float(pnl[trades.direction.eq(1)].sum()),
        "short_gross_pnl_usd": float(pnl[trades.direction.eq(-1)].sum()),
        "undefined_note": "no completed trades" if not len(trades) else None,
    }


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
    result.update(
        directional_accuracy=accuracy,
        accuracy_scope=scope,
        roc_auc=auc,
        roc_auc_note=note,
    )
    return result
