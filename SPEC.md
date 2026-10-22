# AntTrader specification

## Purpose and scope

Investigate whether Langton's Ant can provide predictive information for spot
gold (XAU/USD) on the 1-hour timeframe. This is historical research only.
Profitability is unknown. Topic 1 provides the foundation. Topic 2 implements
historical market data and grid encoding. Simulation, features, signals,
backtesting, and a full dashboard remain pending.

## Technology and structure

Use Python 3.13 and the local .venv. Existing dependencies are NumPy, pandas,
scikit-learn, Streamlit, Plotly, and PyYAML; local checks use pytest and Ruff.
Packaging uses pyproject.toml. No new dependency is required for Topic 2.
Use one Codex agent. The existing public GitHub repository is mohsenOG/anttrader.
No license, CI/CD, Docker, database, broker connection, or live trading is added.

Historical candles -> data -> grid -> simulation -> features -> decoder -> backtest

`anttrader.pipeline` demonstrates data-to-grid integration. `app.py` remains a
Streamlit scaffold. Configuration is file-based in `config.yaml`. Local history
belongs in `data/` and generated outputs in `results/`; both are ignored by Git.

## Topic 2: approved market data

- Market: XAU/USD spot gold. Timeframe: H1 (one hour).
- Provider: Dukascopy historical market data. No provider substitution.
- Initial target: approximately five years; record the actual acquired period.
- Ingestion: local CSV, with required timestamp, open, high, low, close columns.
  Volume is optional and is preserved when supplied.
- Quote basis: bid OHLC for v1, consistently across the dataset.
- Internal timestamps: UTC. Candle timestamps identify interval opening times.
- A candle opening at `s` completes at `s + 1 hour`. Use it only at or after that
  completion time. Current or future candles cannot enter a historical grid.
- Naive input timestamps require an explicitly supplied source timezone.
  Never infer a timezone or silently resolve ambiguous daylight-saving times.
- Sort chronologically, reject duplicate timestamps, and require UTC hourly
  boundaries. Prices must be numeric, finite, and strictly positive.
- Require `high >= max(open, close)`, `low <= min(open, close)`, and `high >= low`.
  Reject corrupt prices; do not repair them.
- Normal session and weekend closures are permitted. Never invent, forward-fill,
  or interpolate candles. Report gaps over one hour for inspection without
  claiming that a gap is necessarily a data error. No trading calendar in v1.

### Source conventions and acquisition

Use the official Dukascopy historical export where available:
https://www.dukascopy.com/api/data/get/historical-data-export

Verify the source's quote side, timestamp format, timezone, opening-time
convention, and price scale before conversion. Record the download procedure,
period, and any source-specific filtering in README.md. Direct binary downloads
may use only verified source conventions. Do not infer a price scale from price
plausibility. If access or conventions cannot be verified, identify the precise
manual export step; do not fabricate history. Full datasets stay outside Git.
Volume is a source field, not an assumed exchange-wide trading volume.

## Topic 2: approved mathematical encoding

At decision position `t`, select the last 257 completed closes:
`C[t-256], ..., C[t]`. Here `t` indexes ordered completed candles, not necessarily
continuous wall-clock hours. Consecutive returns can span market closures.

Calculate 256 natural logarithmic returns in chronological order:

`r[i] = ln(C[i] / C[i-1])`

Encode each return with no threshold or volatility filter:

`state[i] = 1 if r[i] > 0 else 0`

Positive returns map to 1; negative and exactly zero returns map to 0. The initial
grid is deterministic and binary, with 16 rows and 16 columns (256 cells).
Exactly 257 closes are required for a single default grid.

Use NumPy indexing `grid[row, column]` and chronological serpentine placement:
even rows run left to right, odd rows right to left. The oldest return is at
`(0, 0)` and the newest at `(15, 0)`. For a 4 x 4 example, chronological indices are:

```text
 0   1   2   3
 7   6   5   4
 8   9  10  11
15  14  13  12
```

Historical rolling generation yields one result per eligible completed candle.
Each result includes the binary grid, chronological returns, decision timestamp,
and the first and last candle opening timestamps that identify the exact window.
The decision timestamp equals the final included candle's opening plus one hour.
Use only data observable at that decision time. Changing later candles must not
change an earlier grid. With N completed candles there are max(0, N - 256) default
grids. Use an iterator to avoid storing every grid in memory.

## Configuration and future topics

Confirmed Topic 2 settings are XAU/USD, 1H, Dukascopy, local CSV, UTC opening
timestamps, bid OHLC, 256 returns, a 16 x 16 grid, binary log-return sign encoding,
and serpentine layout. They drive the data-to-grid pipeline.

Preserve provisional future defaults: one ant, 2,048 simulation steps, toroidal
boundaries, logistic regression, and a four-bar prediction horizon. These values
are experimental starting points; Topic 2 does not implement or finalize them.

Open decisions remain:

- Topic 3: ant initial state, cell rules, simulation conventions, and features.
- Topic 4: labels, chronological splits, training, BUY/SELL/HOLD rules, execution
  timing, positions, spread, slippage, and other costs.
- Topic 5: dashboard controls, charts, and final integration workflows.

No missing trading rule or scientific assumption is implied by a placeholder.
