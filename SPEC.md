# AntTrader specification

## Purpose and scope

Investigate whether Langton's Ant can provide predictive information for spot
gold (XAU/USD) on the 1-hour timeframe. This is historical research only.
Profitability is unknown. Topic 1 provides the foundation. Topic 2 implements
historical market data and grid encoding. Topic 3 adds ant simulation and
behavioral features. Topic 4 adds historical signals and backtesting. A full dashboard remains pending.

## Technology and structure

Use Python 3.13 and the local .venv. Existing dependencies are NumPy, pandas,
scikit-learn, Streamlit, Plotly, and PyYAML; local checks use pytest and Ruff.
Packaging uses pyproject.toml. No new dependency is required for Topics 2 and 3.
Use one Codex agent. The existing public GitHub repository is mohsenOG/anttrader.
No license, CI/CD, Docker, database, broker connection, or live trading is added.

Historical candles -> data -> grid -> simulation -> features -> decoder -> backtest

`anttrader.pipeline` supports data-to-grid and historical feature generation.
`app.py` remains a Streamlit scaffold. Configuration is file-based in `config.yaml`. Local history
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
- The supplied Dukascopy archive includes flat, zero-volume rows during closures.
  With `drop_zero_volume: true`, exclude these rows before selecting windows and
  report their count. Validate prices first; a non-flat zero-volume row is an
  error, not a row to discard silently. Retain flat candles with positive volume.
  If volume is absent, do not infer closures from unchanged prices. Keep the
  source CSV unchanged. The loader API defaults this source-specific option off;
  config.yaml enables it for the supplied dataset.

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

Topics 3 and 4 finalize the simulation and historical research rules below.

Open decisions remain:

- Topic 5: dashboard controls, charts, and final integration workflows.

No missing trading rule or scientific assumption is implied by a placeholder.

## Topic 3: approved ant simulation

Each historical binary 16 x 16 grid runs one independent ant. Copy the input;
reset grid, ant, and counters for each window. Start at row 15, column 0, heading
North. Directions use clockwise integer codes: North=0, East=1, South=2, West=3.
North decreases row; East increases column; South increases row; West decreases
column. There is no randomness or market-dependent movement rule.

Each transition is READ -> TURN -> FLIP -> MOVE. A white cell (0) turns right
90 degrees and becomes 1; a black cell (1) turns left 90 degrees and becomes 0.
Move one cell along the new heading. Wrap with row/column modulo 16 (toroidal).
Maintain unwrapped displacement separately: x increases east, y increases south;
wrapping never adds a discontinuity to these displacements.

Run 2,048 transitions continuously. Save independent checkpoint snapshots after
256, 512, 1,024, and 2,048 completed transitions. A cell is visited when READ at
the start of a transition, not on arrival. At step S, visit counts sum to S.
The endpoint has not yet been read unless visited on an earlier transition.
Keep the initial grid, checkpoint grids and visit counts, turn counts, wrapped
pose, and unwrapped displacement. Do not restart at intermediate checkpoints.

Configuration uses simulation.steps, start_row, start_col, initial_direction,
boundary, checkpoints, ants, and reset_each_window. Reject nonpositive/noninteger
lengths, invalid headings or starts, unordered/duplicate/out-of-range checkpoints,
and a final checkpoint different from steps. Only 16 x 16, one ant, toroidal,
and reset_each_window=true are supported. The reusable simulator permits valid
shorter runs and other starting poses for testing/replay; config.yaml records
the approved historical experiment. No settings are tuned against returns.

## Topic 3: deterministic behavioral features

At checkpoint S, let V_i be cell read counts, U the number of nonzero V_i,
R the right-turn count, B the number of black cells, D the number of cells
which differ from the initial grid, and p_i=V_i/S. There are M=256 cells.
Export these 15 numeric columns per checkpoint, in this order:

| Suffix | Definition |
| --- | --- |
| final_row | Wrapped row, 0..15 |
| final_col | Wrapped column, 0..15 |
| heading_north | 1 if heading is North, otherwise 0 |
| heading_east | 1 if heading is East, otherwise 0 |
| heading_south | 1 if heading is South, otherwise 0 |
| heading_west | 1 if heading is West, otherwise 0 |
| unique_cells_visited | U |
| revisit_fraction | (S-U)/S |
| horizontal_displacement | Net unwrapped movement east (+) / west (-) |
| vertical_displacement | Net unwrapped movement south (+) / north (-) |
| right_turn_fraction | R/S |
| final_black_fraction | B/M |
| changed_cell_fraction | D/M; repeated flips can restore a cell |
| visit_entropy | -sum(p_i*ln(p_i))/ln(M), ignoring V_i=0 |
| max_visit_concentration | max(V_i)/S |

Prefix each suffix with s{S:04d}_, e.g. s0256_final_row. Human-readable headings
remain available in checkpoint results. All numeric features must be finite;
fractions and normalized entropy lie in [0,1] within floating-point tolerance.
Do not add redundant left-turn fraction, weights, scaling, feature selection,
future returns, labels, predictions, trading signals, or P&L in Topic 3.

## Topic 3: historical export and replay

Use the existing historical grid iterator. One grid produces exactly one CSV
row at its UTC decision timestamp, strictly chronological and without duplicates.
Only candles completed at that timestamp may affect that row. Never carry state
between windows, mutate source grids/history, or skip a failed simulation.

python -m anttrader.pipeline --stage features streams rows to the configured
feature_output, default results/ant_features.csv. Columns are decision_timestamp
then the 15 features for each ascending checkpoint (60 numeric columns total).
--limit N optionally selects the first N eligible windows; default is all.
Publish output only after the batch succeeds, so failures preserve an existing
CSV. Reject output paths that overwrite the source CSV or configuration file.
Write a small adjacent .metadata.json with source/config SHA-256, UTC as_of,
configuration, counts, environment, and elapsed time for reproducibility.
Generated outputs remain ignored by Git. The default data-to-grid command stays
available. Dashboard animation remains pending.

Batch mode stores no trajectory. Optional replay records (row, column,
direction code) initially and after every move, plus the initial grid and
configuration. For transition k, flip trajectory[k]'s cell, then move to
trajectory[k+1]. This reconstructs every flip and the final grid without a full
grid copy per step. Recording must not change checkpoints or features.

## Topic 4: approved signal decoder and historical backtest

- XAU/USD, one-hour Dukascopy bid candles; use the existing 60 ant features.
- At decision D, the last observed candle opened at D-1h and has completed.
  Entry is the next candle opening at D. Exit is the opening at D+4h.
  Require every hourly opening from D-1h through D+4h. Exclude gaps and
  unavailable exits; never interpolate prices. This is idealized zero-latency
  execution at the hourly open, not a claim of executable prices.
- Label ln(exit_open / entry_open) > 0 as 1, otherwise 0. No return filter.
- StandardScaler then scikit-learn regularized LogisticRegression (L2,
  C=1, lbfgs, max_iter=1000, random_state=42). Fit each scaler and model only
  on training data. Freeze both models; no refitting, calibration, threshold
  optimization, feature selection, or test-set tuning.
- BUY if p_up >= 0.55; SELL if p_up <= 0.45; HOLD strictly between.
  Probabilities are uncalibrated model estimates, not expected profit.
- Split all feature rows chronologically before filtering: floor(0.60*N)
  training rows, then through floor(0.80*N) validation rows, then test rows.
  Purge training/validation rows with exit >= the next partition's first
  decision. This conservative exclusive cutoff prevents boundary contamination.
  Report original boundaries, retained dates/counts, and exclusion counts.
- Require the complete Topic 3 export and metadata, exact UTC timestamps,
  finite ordered feature columns, source SHA-256, matching market/encoding/
  simulation settings, and every expected feature timestamp. Partial exports
  are rejected. No approximate timestamp joins.
- Compare AntTrader, four-interval close momentum, and raw-input logistic
  regression on the same eligible observations and partitions. Momentum uses
  five completed closes, with four consecutive hourly close-to-close returns;
  positive = BUY, negative = SELL, zero = HOLD. Exclude momentum windows with
  gaps for all strategies. Raw inputs are the original 256 chronological
  binary return states from Topic 2, without running the ant again. Use the
  same labels, scaling/model settings, thresholds, and execution rules.
- One fixed troy ounce, maximum one open position per strategy and partition.
  Ignore signals while open; retain all predictions. Close after four hours;
  a new position may enter at the prior exit opening. No stop-loss,
  take-profit, compounding, leverage, broker integration, or live trading.
- Gross P&L = direction * (exit_open - entry_open) * quantity, with long +1
  and short -1. Both directions use the same bid opening reference price.
  All costs are zero: no spread, synthetic spread, slippage, commission,
  swap, or financing. Report **gross price-only simulated P&L**, never net
  broker profits. Realistic transaction costs remain future work.
- Equity is cumulative realized trade P&L, initially zero. Maximum drawdown
  includes that initial zero and measures completed-trade equity only.
  Trading metrics: trade/side counts, total/average P&L, win rate, gross
  profit, gross loss (positive loss magnitude), profit factor, realized
  drawdown, and side P&L. No trades gives undefined win rate/average/profit
  factor. Positive profit without losses gives infinite profit factor.
- Predictive metrics: accuracy (p_up >= 0.5), ROC AUC when both classes
  exist, observation counts, and BUY/SELL/HOLD counts and fractions.
  Momentum has no probability or ROC AUC; its directional accuracy uses
  non-HOLD signals. Undefined metrics are JSON null with a reason; infinite
  profit factor is the JSON string "infinity". No nonstandard JSON NaN values.
- Report validation and test separately. Export signals.csv, trades.csv,
  equity.csv, metrics.json, and experiment.metadata.json under results/.
  Include strategy and partition in trades/equity, an initial equity baseline
  per partition/strategy, identities and hashes, versions, model settings,
  boundaries, exclusions, timing and zero-cost assumptions in metadata.
  Generated data stays excluded from Git. Run with
  python -m anttrader.pipeline --stage backtest.
- One fixed chronological experiment cannot establish sensitivity to other
  periods, proven predictability, or future profitability. Report results
  without selecting parameters to improve performance.
