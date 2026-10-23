# AntTrader specification

## Purpose and scope

Investigate whether Langton's Ant can provide predictive information for spot
gold (XAU/USD) on the 1-hour timeframe. This is historical research only.
AntTrader is a signal research tool, not a trade simulator. Topic 1 provides the foundation. Topic 2 implements
historical market data and grid encoding. Topic 3 adds ant simulation and
behavioral features. Topic 4 adds historical signal research. Topic 5
adds the local research dashboard.

## Technology and structure

Use Python 3.13 and the local .venv. Existing dependencies are NumPy, pandas,
scikit-learn, Streamlit, Plotly, and PyYAML; local checks use pytest and Ruff.
Packaging uses pyproject.toml. No new dependency is required for Topics 2 and 3.
Use one Codex agent. The existing public GitHub repository is mohsenOG/anttrader.
No license, CI/CD, Docker, database, broker connection, or live trading is added.

Historical candles -> data -> grid -> simulation -> features -> decoder -> historical signals

`anttrader.pipeline` supports data-to-grid and historical feature generation.
`app.py` is the Streamlit entry point. Configuration is file-based in
`config.yaml`. Local history
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

Topic 5 adds display controls only. It introduces no additional research rules.

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
available. Topic 5 reuses this trajectory for interactive replay.

Batch mode stores no trajectory. Optional replay records (row, column,
direction code) initially and after every move, plus the initial grid and
configuration. For transition k, flip trajectory[k]'s cell, then move to
trajectory[k+1]. This reconstructs every flip and the final grid without a full
grid copy per step. Recording must not change checkpoints or features.

## Topic 4: signal decoder and historical prediction evaluation

- Preserve the 60 ant features and the original 256 binary states used by the
  raw-input comparison model. Preserve four-interval momentum (five completed
  consecutive hourly closes). Compare all three methods on identical eligible rows.
- At decision D, only candles opening at or before D-1h have completed. Inputs
  use those candles only. Preserve the four-hour forecast label:
  `future_return = ln(open[D+4h] / open[D])`; target is 1 when positive, else 0.
  Require each hourly opening from D-1h through D+4h. Exclude gaps and unavailable
  outcomes. Future prices are retrospective label data only, never model inputs.
- Split all feature rows before exclusions: floor(0.60*N) training, through
  floor(0.80*N) validation, then test. Purge training and validation labels whose
  end is at or after the next partition's first decision. Preserve the existing
  common eligibility intersection with valid momentum history.
- Fit StandardScaler and LogisticRegression on training rows only: L2, C=1,
  lbfgs, max_iter=1000, random_state=42. Freeze both models. No model changes,
  refitting, calibration, feature selection, threshold search, or optimization.
- BUY when p_up >= 0.55; SELL when p_up <= 0.45; HOLD between these values.
  BUY and SELL are directional predictions. They do not open orders or positions.
  The four-hour horizon defines labels only. There is no trade simulation,
  position management, forced closing, quantity, P&L, equity, or drawdown logic.
- Keep full feature-export checks: metadata, source hash, market/encoding and
  simulation settings, 60 finite ordered features, exact expected UTC timestamps,
  and complete historical coverage. Reject partial feature exports.
- Run `python -m anttrader.pipeline --stage signals`. Save validation and test
  rows to results/signals.csv, predictive metrics to results/metrics.json, and
  identities/configuration/partitions/exclusions/versions to
  results/experiment.metadata.json. Print signal counts by method and partition.
- Signal rows include the decision, partition, last completed close as a display
  reference, model probabilities and classifications, momentum, and explicitly
  named retrospective label fields. They contain no execution or position fields.
- Keep overall logistic classification accuracy at p_up >= 0.5 across every
  eligible row, including HOLD. Also report accuracy restricted to BUY/SELL rows:
  BUY predicts target 1, SELL predicts target 0. Momentum accuracy uses non-HOLD
  rows. Save total observations, BUY/SELL/HOLD counts and fractions, actionable
  count/accuracy, and ROC AUC where defined. Undefined values use JSON null with
  a reason. Momentum has no probability or ROC AUC. Never combine partitions.
- Save hashes for signals and metrics. Validate the complete signal bundle and
  reconcile exact eligible timestamps, labels, reference prices, momentum,
  boundaries, exclusions, thresholds, and stored predictive metrics. Legacy
  result metadata requires the signals command once. Old local trade/equity
  files can remain unused. Never delete source data or ant features.
- Identical inputs, feature history, and model settings must preserve historical
  predictions. Historical accuracy does not establish profitable trading.

## Topic 5: local signal research dashboard

Use app.py, Streamlit, and Plotly. Provide exactly two primary tabs:
Ant Simulation and Gold & Signals. Keep config.yaml read-only in the UI.
Do not train models or run the pipeline when Streamlit opens.

### Ant Simulation

Preserve the 16 x 16 grid and recorded trajectory replay. Select a historical
feature timestamp; reconstruct and verify its checkpoint features. Keep the
0–2,048 step slider, checkpoint jumps, original/replay grids, current pose and
heading, behavioral features, and exact associated saved signal when available.
Selecting a window must not create a prediction or change the movement algorithm.

### Gold & Signals

- Use a full-width 800-pixel hourly candlestick chart with high contrast,
  readable UTC axes and legend, and Plotly zoom and pan. Default to the latest
  ten calendar days within the selected partition (or all available if shorter).
- Keep display-only date range, validation/test (Test default), comparison
  method, and BUY/SELL visibility controls. No model or strategy controls.
- Plot 16-pixel green upward BUY and red downward SELL triangles with dark
  outlines. Keep HOLD in the signal table without chart markers.
- Marker x equals the actual saved decision D. Derive marker y from the candle
  opening at D-1h, which completed at D. Place BUY below its low and SELL above
  its high, with offset max(0.15 * that candle's range, 0.0008 * its close).
  The offset is visual only. Future candles cannot affect marker placement.
  Resolve the preceding candle before applying the visible date range.
- Hover shows decision UTC, direction, P(up), last completed close, and partition.
  Momentum probability is unavailable. Do not show execution or trading values.
- Show Latest Historical Signal with its saved timestamp, probability, and
  partition. Keep signal history, including HOLD and clearly identified future
  label outcomes. Keep compact full-partition prediction statistics below the
  chart. Explain overall versus BUY/SELL-only accuracy and undefined metrics.
- Require only candles, features/metadata, signals, predictive metrics, and
  experiment metadata. Missing/stale outputs must show the signals CLI recovery
  command. Never require or display obsolete trade/equity files.
- Preserve bounded caches and changed-file invalidation. Validate the signal
  bundle without fitting models. Show source/cutoff and partition metadata.

### Verification

Run full pytest and Ruff. Preserve simulation, grid, model, threshold, chronology,
and leakage checks. Verify unchanged historical predictions, completed-candle
marker coordinates and future-price independence, partition separation, signal
statistics, missing/stale/partial results, replay controls, and loading without
trade/equity files. Launch Streamlit and inspect the chart and controls in a
browser. Record actual results and limits in TASKS.md and README.md.


## AntTrader v1.1: Experiment Lab

This extension supersedes only the fixed dimensions, simulation controls, and
number of dashboard tabs above. The original 16 x 16, 2,048-step, North,
newest-return experiment remains the default reference. Research stays
signals-only. The ant rules, encoding, model family, thresholds, four-hour
labels, market-gap rules, and chronological partition purging stay unchanged.

Support 2..32 rows and columns, 4..16,384 steps, and at most 30 configurations
per batch. Dimensions mean rows x columns. Each grid uses rows*columns returns
and rows*columns+1 completed closes. Newest return starts at (rows-1, 0) for
even rows, or (rows-1, columns-1) for odd rows. Center is (rows//2, columns//2)
in zero-based coordinates. Four checkpoints use integer half-up rounding of
steps/8, steps/4, steps/2, steps, clamped in order to leave room for four unique,
positive increasing checkpoints. Features divide cell counts by actual cell
count and entropy by ln(cell count); a one-cell simulator has zero entropy.

The lab uses one shared set of decision timestamps based on the largest batch
lookback. Split this set 60/20/20 before exclusions. Apply the same labels,
momentum eligibility, and exclusive outcome cutoffs to each configuration.
Fit independent training-only scaled logistic models for ant and matched raw
sign inputs. Rank validation results only. Test results are previously inspected
exploratory diagnostics. Incompatible cohorts must never share a ranking.

Store each result under results/experiments/<stable-id> using JSON and CSV.
Identity includes normalized parameters, source and completed-data identities,
cohort observations and evaluation rules, code and library versions. Publish
completion metadata last; reject partial or altered results. Reuse completed
results unless rerun is explicit. Delete only a selected validated experiment
directory after explicit confirmation. Preserve source data and original outputs.

CLI and Streamlit share the same synchronous backend. Forms do not train on
change or tab selection. Show workload, stage, progress, cache hits, and errors.
Provide history, reset, rerun, compatible comparisons, validation leaderboard,
heatmap, accuracy/coverage chart, disagreement, and past-only signal overlays.
Selected saved experiments can be replayed and viewed in the original tabs.
Show counts, coverage, small-sample and multiple-comparison warnings. Undefined
metrics remain null. Research candidates are labels, never deployed models.

Development verification uses pytest, Ruff, and small historical samples only.
The user runs full historical experiments and optimization separately.
