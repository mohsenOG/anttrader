# AntTrader

Experimental Langton's Ant research for spot gold (XAU/USD) on the **1-hour**
timeframe. Profitability is unknown. This project uses historical data only.

Implemented: project setup, validated CSV ingestion, binary log-return encoding,
rolling serpentine grids, deterministic ant simulation, behavioral feature CSV
export, optional replay recording, frozen logistic decoders, comparable historical
backtests, and a three-tab Streamlit dashboard.

The ant is a deterministic transformation of a binary market grid. The research
compares its behavioral features with momentum and the original binary inputs;
complex ant behavior alone does not imply useful market predictions.

Market data -> Grid encoding -> Ant simulation -> Feature extraction ->
Signal decoder -> Backtester -> Streamlit dashboard.

## Installation

Use Python 3.13. In PowerShell:

```powershell
git clone https://github.com/mohsenOG/anttrader.git
cd anttrader
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\Activate.ps1
```

Activation is optional. Use `.venv\Scripts\python.exe` instead of `python` and
`.venv\Scripts\ruff.exe` instead of `ruff` if PowerShell blocks activation.
No global execution-policy change is required. Configuration is in `config.yaml`.
No additional dependencies or services are needed for the dashboard.

## Run the local application

Prepare the Dukascopy CSV as described below, then run these commands from the
repository root with the virtual environment active:

```powershell
python -m anttrader.pipeline
python -m anttrader.pipeline --stage features
python -m anttrader.pipeline --stage backtest
streamlit run app.py
```

The first command checks candles and grid encoding. Feature generation must
cover the full dataset: do not use --limit for a research run. Open the local
URL printed by Streamlit. Configuration and output paths come from config.yaml;
paths are resolved relative to that file. Datasets and generated research
results remain local and are excluded from Git.

## Dashboard

- **Ant Simulation:** select a saved feature timestamp. Compare its original
  market grid with the recorded ant replay from step 0 through 2,048. Jump to a
  checkpoint and inspect current pose, read counts, turn/cell fractions, entropy,
  and unwrapped displacement. Saved checkpoint features must match the selected
  simulation. An associated prediction appears only if it exists in signals.csv.
- **Gold & Signals:** inspect historical bid candlesticks, BUY/SELL predictions,
  executed entries, and completed exits. Select a partition, displayed strategy,
  and UTC date range. The initial range covers the recent month of that
  partition. Sort the trade and signal tables by clicking their headers.
  HOLD observations remain in the table. Future targets are retrospective
  outcomes, never model inputs. Latest Historical Signal refers to the last
  saved AntTrader prediction.
- **Backtest & Comparison:** select Test (default) or Validation. Compare saved
  metrics and realized gross-P&L curves for AntTrader, Momentum, and raw-input
  logistic regression. Undefined metrics include explanatory notes; profit
  factor can be infinite. Partitions are never combined.

Research settings are read-only. The dashboard never downloads data, generates
features, trains models, or reruns trading simulations. It caches validated
inputs and up to eight selected replay runs, using file identities to invalidate
changed results. It reconstructs only the selected ant window.

If a file is missing or incompatible, the app shows the path or validation error
and the recovery commands. Restore the configured market CSV when needed, run
the complete feature stage, then run the backtest stage. Never substitute dummy
files or edit exported results to bypass validation. For older metadata without
result hashes, rerun the backtest stage. Stop any running pipeline before
loading a result bundle.

## Limitations

Historical research only. There is no live market feed, broker connection, or
order execution. P&L uses idealized bid reference prices with zero spread,
slippage, commission, financing, and execution latency. Results depend on the
provided dataset and fixed chronological split. Positive historical gross P&L
does not establish a market edge or guarantee future profitability.

## Market data

The approved source is **Dukascopy**, using **XAU/USD H1 bid OHLC**. Aim for five
years. Full datasets are local and excluded from Git. Do not substitute a feed or
use synthetic test prices as market history.

### Download or export

1. Open [Dukascopy Historical Data Export](https://www.dukascopy.com/swiss/english/marketwatch/historical/).
2. Select XAU/USD (spot gold), period **1 hour**, price side **BID**, timezone
   **UTC/GMT**, and CSV output. Timestamps must identify candle opening times.
3. For the initial five-year sample, use **2021-10-01 00:00 UTC through
   2026-09-30 23:00 UTC**, including the final completed candle. For tools with an
   exclusive end, use **2026-10-01 00:00 UTC**. Export smaller periods and concatenate
   them with one header if the tool limits the range; do not overlap timestamps.
4. Disable creation of flat/no-tick or forward-filled candles when the exporter
   allows it. If the source includes flat zero-volume closure rows, the configured
   `drop_zero_volume: true` excludes them in memory and reports the count. Keep
   normal closures as gaps. Do not use bid/ask averages or ask prices.
5. Save the CSV as `data/xauusd_h1_bid.csv`.
6. Check the actual header and timestamp format before running the pipeline.

If the web widget is unavailable, use **JForex -> Tools -> Historical Data Manager**
with the same settings. Confirm the opening-time convention and timezone in the
export settings. Do not infer them from a filename or your computer timezone.

The official [export page](https://www.dukascopy.com/api/data/get/historical-data-export)
describes bid/ask data and volume. Direct requests were blocked on 2026-10-22.
The user supplied the Dukascopy CSV and 60 cached H1 BID archives on 2026-10-23;
real-data integration is now verified. The dataset and cache remain outside Git.
A fresh clone still requires its own local dataset.

### Verified local dataset (2026-10-23)

- Opening times: **2021-10-01 00:00 UTC through 2026-09-30 23:00 UTC**.
- Source CSV: **43,824 rows** with `time_utc,open,high,low,close,volume`.
- **14,257** flat zero-volume closure rows excluded; **29,567** completed candles
  retained. The source file is unchanged. All excluded rows also match the prior
  close; flat candles with positive volume are retained by the loader.
- All CSV timestamps and prices match the 60 cached monthly archives. Archives
  use zero-based month paths, LZMA compression, and big-endian `>IIIIIf` records:
  seconds from UTC month start, open, close, low, high, volume. Prices are divided
  by 1,000; source volume is rounded to six decimals in this CSV.
- Validation found no duplicates, invalid OHLC, missing prices, or incomplete
  candles. Filtering exposes **1,290 gaps** over one hour. The largest interval
  between candle openings is **74 hours**. All missing slots correspond to excluded
  zero-volume rows; no extra hours are missing within the supplied archives.
  This is a source comparison, not an independent trading-calendar audit.
- Generated **29,311** historical grids. The latest decision is **2026-10-01
  00:00 UTC**, using openings from **2026-09-15 19:00 UTC** through **2026-09-30
  23:00 UTC**. Last close: **4155.675**. The 16 x 16 grid has **130 zeros and
  126 ones**. Its formula, layout, and timing were checked independently.
- Source CSV SHA-256:
  `ec75b1339f50ebb9fa05ba8dece72da184709c3acd098c4e80ccb63d2d107bed`.

The local `results/topic2_validation.json` records the full grid, gap distribution,
and archive hashes. Run `python -m anttrader.pipeline` to reproduce data-to-grid
output, or `python -m pytest -q tests/test_real_data.py` to repeat real-data checks.
The integration test skips when the ignored local CSV is absent.

### CSV contract

Use comma-separated columns:

```text
timestamp,open,high,low,close,volume
```

`volume` is optional. Use ISO 8601 timestamps with `Z` or a UTC offset, for example
`2024-01-05T20:00:00Z`. Columns are case-insensitive. `Gmt time`, `Local time`,
and `time_utc` are supported timestamp header aliases. A naive timestamp requires
an explicit `timestamp_timezone` in config; its default `UTC` describes the required export
settings, not an inferred timezone. For a `DD.MM.YYYY HH:MM:SS.sss` export, set:

```yaml
timestamp_timezone: UTC
timestamp_format: '%d.%m.%Y %H:%M:%S.%f'
```

The default `timestamp_format: ISO8601` is for normalized CSV files. Use an exact
format matching your export. Do not guess the day/month order. Other export
formats must be explicitly converted to this contract first.

Prices must be numeric, finite, positive, and consistent OHLC values. The loader
sorts rows and rejects duplicate times, invalid prices, missing prices, and
non-hourly UTC boundaries. It does not repair data. Optional volume is preserved
as supplied unless closure filtering requires numeric volume. It is not assumed
to represent exchange-wide gold trading volume. With `drop_zero_volume: true`,
volume must be finite and nonnegative; zero-volume rows must have flat OHLC or
validation raises an error. The loader API defaults this source-specific filter
off; `config.yaml` enables it for the supplied dataset. If volume is absent,
filtering does nothing: unchanged prices alone never identify a closure.

All returned timestamps are UTC candle **opening times**. The close of a candle
opening at 20:00 becomes usable at 21:00. Incomplete candles are excluded. Session
and weekend gaps are allowed without filling or interpolation. The validation
report counts gaps over one hour and shows up to five examples for inspection.
It does not classify gaps with a trading calendar or certify completeness of the
provider's history. Data-source and quote-side identity must be verified at export;
OHLC values alone cannot establish these facts.

### Python interface

```python
from anttrader.data import load_gold_data
from anttrader.grid import iter_historical_grids

candles = load_gold_data(
    "data/xauusd_h1_bid.csv",
    source_timezone="UTC",  # Only for an export explicitly made in UTC.
    drop_zero_volume=True,  # Supplied Dukascopy flat/no-tick convention.
    as_of="2026-10-01T00:00:00Z",
)
print(candles.attrs["validation_report"])
for result in iter_historical_grids(candles, as_of="2026-10-01T00:00:00Z"):
    # Consume one grid at a time. result.returns is chronological.
    grid = result.grid
    decision_time = result.decision_timestamp
```

Omitting `as_of` uses the current UTC time. Explicit historical cutoffs make
experiments reproducible. Decision cutoffs must include a timezone. Results also
contain `window_start` and `window_end`, both candle opening times.

## Grid encoding

Each 16 x 16 grid requires exactly **257 completed closes**, which yield **256**
natural log returns `r[i] = ln(C[i] / C[i-1])`. A positive return becomes **1**;
negative and exactly zero returns become **0**. There are no thresholds or random
initial states. Lookback counts candles, including across closures, not continuous
wall-clock hours.

Returns run oldest to newest in alternating row directions: row 0 left to right,
row 1 right to left, and so on. A 4 x 4 example of chronological indices is:

```text
 0   1   2   3
 7   6   5   4
 8   9  10  11
15  14  13  12
```

For the default grid, the oldest return is at `[0, 0]` and the newest at `[15, 0]`.
NumPy indexing is `[row, column]`. `closes_to_grid(closes)` encodes one exact window.
`iter_historical_grids(candles)` yields `max(0, N - 256)` grids from N completed
candles. Each decision time is the last included candle's opening plus one hour.
Later candles do not enter that window. Tests explicitly change future candles
and confirm that the earlier grid remains identical.

## Run and verify

With the virtual environment active:

```powershell
python -m anttrader.pipeline
python -m anttrader.pipeline --config config.yaml
python -m anttrader.pipeline --stage features
python -m anttrader.pipeline --stage features --limit 100
streamlit run app.py
python -m pytest -q
ruff check .
```

The pipeline resolves the CSV path relative to the configuration file. It prints
the candle count, date range, validation report, number of historical grids, and
latest grid with its decision time and window. If the CSV is absent, it prints
setup instructions. It never supplies fake history. The Streamlit application
loads saved research results. Generate features and backtests separately;
missing inputs produce file-specific errors and recovery commands.

Unit tests use small **synthetic** inputs in temporary directories. They validate OHLC,
timezones, completion times, gaps, sign encoding, layout, rolling alignment, and
no look-ahead. A separate optional integration test uses the local CSV to check
all rolling decision timestamps, grid count, the latest grid, and no look-ahead
on genuine source data. It uses the explicit 2026-10-01 UTC cutoff for this sample.

Read [SPEC.md](SPEC.md) for approved rules and open decisions, [TASKS.md](TASKS.md)
for verified progress, and [AGENTS.md](AGENTS.md) for agent instructions.
`config.yaml` contains the approved simulation, decoder, and backtest settings.

## Ant simulation and behavioral features

Each historical market grid seeds one independent Langton's Ant. The simulator
copies the 16 x 16 binary grid and starts at **row 15, column 0, heading North**.
Each step reads the current cell, turns, flips that cell, then moves one cell:

- White (0): turn right 90 degrees; flip to 1.
- Black (1): turn left 90 degrees; flip to 0.

North decreases row. East increases column. The grid wraps on all four edges
(toroidal boundaries). Unwrapped displacement measures east/west and south/north
movement without jumps at wrapped edges. The ant does not use prices or future
returns. Every window resets all state. The source dataset and input grid stay
unchanged.

One continuous **2,048-step** run captures checkpoints after **256, 512, 1,024,
and 2,048** completed transitions. A visit means reading a cell before moving;
the final arrival alone is not a visit. At each checkpoint, visit counts sum to
the completed steps.

Each checkpoint has 15 numeric features, for **60 feature columns** in total:

- `final_row`, `final_col`.
- `heading_north`, `heading_east`, `heading_south`, `heading_west` (one-hot).
- `unique_cells_visited`, `revisit_fraction`.
- `horizontal_displacement`, `vertical_displacement` (east and south positive).
- `right_turn_fraction`, `final_black_fraction`, `changed_cell_fraction`.
- `visit_entropy`, `max_visit_concentration`.

Names use checkpoint prefixes such as `s0256_final_row` and
`s2048_visit_entropy`. Revisit fraction is `(steps - unique)/steps`. Black and
changed-cell fractions divide counts by 256. Entropy is
`-sum(p*ln(p))/ln(256)` over nonzero visit probabilities `p=visits/steps`.
Concentration is `max(visits)/steps`. Changed cells compare the final state with
the initial grid; repeated flips can cancel. Full definitions are in SPEC.md.

### Historical feature CSV

```powershell
python -m anttrader.pipeline --stage features
```

This validates configured history, streams the existing chronological grid
iterator, simulates each grid independently, and exports one row per decision
timestamp. `results/ant_features.csv` starts with UTC `decision_timestamp`, then
ascending checkpoint features in the order above. All numeric features are
finite. No labels, future returns, predictions, signals, or P&L are included.

The command reports candle/grid counts, transitions, checkpoints, feature count,
output path, elapsed time, and failures. A failed window raises an error with
its decision timestamp; no window is silently skipped. A temporary CSV preserves
the previous output if simulation fails. The adjacent
`results/ant_features.metadata.json` records the input/configuration SHA-256,
configuration, UTC cutoff, counts, runtime, and Python/library versions.
Both generated files stay outside Git.

`--limit 100` processes only the **first** 100 eligible windows for quick checks.
It writes the same configured output path and replaces an existing result on
success. Run without `--limit` to regenerate the complete dataset. To retain a
sample separately, use a separate config file with a different `feature_output`.
CSV and output paths are relative to that config file. The default command
`python -m anttrader.pipeline` still prints the latest Topic 2 grid.

### Replay interface

```python
from anttrader.simulation import simulate
from anttrader.features import extract_features

# window comes from iter_historical_grids(candles).
result = simulate(window.grid, record_trajectory=True)
features = extract_features(result)
print(result.checkpoints[-1].heading)
reconstructed = result.initial_grid.copy()
for row, col, direction in result.trajectory[:-1]:
    reconstructed[row, col] ^= 1
```

The trajectory contains the initial pose, then the pose after each move. Direction
codes are North=0, East=1, South=2, West=3. Flip each departing pose's cell to
reconstruct the final grid; consecutive poses describe each movement. Replay
retains no per-step grid copies. Batch mode leaves `trajectory=None` and keeps
only four checkpoint grids and visit-count matrices per active simulation.
There is no interactive animation in Topic 3.

Unit tests include a hand-calculated eight-step trajectory, all four wrapped
edges, exact checkpoint timing, feature mathematics, replay reconstruction,
independent windows, CSV ordering, failure preservation, and changes to future
candles. Full historical processing is a separate verification run, not part of
each ordinary pytest invocation.

### Verified local batch (2026-10-23)

The full configured dataset retained **29,567** candles and produced **29,311**
feature rows, covering **60,028,928** transitions. Each row has one UTC timestamp
and 60 numeric features. Every timestamp matches its completed-candle decision
time, from **2021-10-18 04:00 UTC** through **2026-10-01 00:00 UTC**.
There are **zero** missing values, nonfinite features, duplicate timestamps,
or failed simulations. Source SHA-256 remains unchanged.

Full end-to-end processing took **23.378 seconds**, with **101.574 MiB** peak
process working set, measured through Windows GetProcessMemoryInfo. Environment:
Windows 11 build 26200, local Python 3.13.11, NumPy 2.5.4, pandas 3.1.0. A separate
100-grid simulation/feature sample took **0.0654 seconds** after loading, and
reproduced the corresponding CSV rows exactly. The earlier 100-window CLI run,
including data loading and validation, took **3.093 seconds**. No extra dependency
or parallel processing was needed.

All **131 pytest tests** and Ruff pass. The Topic 2 CLI remains functional;
Streamlit passes AppTest and starts with HTTP 200 / health=ok. Local
`results/topic3_validation.json` records full-data checks and measurements.
This Topic 3 verification did not evaluate predictive value or trading performance.
The separate Topic 4 experiment below implements the approved trading rules.

## Signal decoder and historical backtest

```powershell
python -m anttrader.pipeline --stage backtest
```

This command requires the complete configured market CSV and Topic 3 export.
If features are missing, regenerate them with
`python -m anttrader.pipeline --stage features` **without `--limit`**. A partial
export is rejected, even when its metadata declares a limit equal to the full
row count. There is no partial-history backtest mode. Restore the source CSV
using the Market data steps above if it is missing; never substitute fake prices.

The loader checks source SHA-256, market/encoding and simulation metadata,
expected row count, exact UTC decision timestamps, stable feature columns,
and finite numeric values. Only feature-producing configuration is compared
with Topic 3 metadata: adding Topic 4 configuration does not invalidate features.
Changing history or the simulation requires a new complete feature export.

AntTrader uses all **60 ant features** listed above, without feature selection.
A scikit-learn pipeline fits **StandardScaler + LogisticRegression** on training
rows only. Logistic regression uses ordinary L2 regularization, `C=1`, `lbfgs`,
`max_iter=1000`, and `random_state=42`. The second logistic model uses only the
**256 original chronological binary return states**, with the same training
rows and settings. It does not run the ant. Both models remain frozen during
validation and test. No tuning, calibration, or validation refit is performed.
Estimated probabilities are not established calibrated real-world probabilities
and do not directly estimate expected profit.

### Labels, partitions, and signals

At decision **D**, the candle opening at **D-1h** has just completed. The
idealized entry is the next candle's **bid opening price at D**. Exit uses the
bid opening price at **D+4h**. Every hourly timestamp from D-1h through D+4h
must exist. Missing entry, intermediate, or exit hours exclude the observation;
weekends and session gaps cannot become continuous four-hour trades. No prices
are interpolated. Entry assumes zero processing and execution latency at the
hourly boundary. This is a theoretical reference-price convention.

`future_return = ln(exit_open / entry_open)`. The label is **1** when positive,
otherwise **0**. Future prices and labels never enter the input features.

Split all feature rows chronologically before exclusions: first 60% training,
next 20% validation, final 20% test. Boundaries use floor(0.60*N) and
floor(0.80*N). Training and validation outcomes must end **strictly before**
the next partition's first decision. Purge crossing labels. Test exits must be
available in the historical source. Validation and test are reported separately.

- **BUY:** p_up >= 0.55.
- **SELL:** p_up <= 0.45.
- **HOLD:** 0.45 < p_up < 0.55.

Momentum uses four completed consecutive hourly close-to-close intervals
(**five completed closes**). Positive momentum means BUY, negative means SELL,
and exactly zero means HOLD. A missing historical hour makes momentum invalid.
All three approaches use the intersection of valid labels, valid momentum
windows, and protected partition dates. They use identical execution rules.

### Positions, P&L, and metrics

Each strategy starts flat in each evaluation partition. A BUY opens a long;
a SELL opens a short; HOLD opens nothing. Hold exactly four consecutive hours,
with at most one position open. Ignore new signals while open, but retain
predictions in signal history. Reentry is allowed at the preceding exit opening.
Do not enter without an available full exit. Quantity is fixed at one troy ounce;
there is no compounding, stop-loss, take-profit, leverage, or live execution.

`gross_pnl_usd = direction * (exit_open - entry_open) * quantity_ounces`, where
long direction is +1 and short is -1. Both sides use bid opening prices as the
same theoretical reference. **All reported P&L is gross price-only simulated
P&L.** There is no spread, synthetic spread, slippage, commission, swap, or
financing. These results are theoretical price-only backtests, not executable
net returns.

Trading metrics include completed/BUY/SELL trade counts, total and average
gross P&L, win rate, gross profit, gross loss (positive magnitude), profit
factor, maximum realized-equity drawdown, and long/short P&L. Equity starts at
zero and accumulates only completed trades. Drawdown includes the initial zero
baseline; it does not measure intratrade losses. No-trade win rate, average
P&L, and profit factor are undefined. Profit factor without losses is
`"infinity"` when profit exists, otherwise null; associated notes explain why.

Predictive metrics include observation counts, BUY/SELL/HOLD counts/fractions,
directional accuracy, and ROC AUC. Logistic accuracy uses p_up >= 0.5 across
all eligible observations, including HOLD predictions. Momentum accuracy uses
non-HOLD observations. Momentum has no probability/AUC. A single-class target
has undefined ROC AUC. Undefined JSON metrics use null with a reason, never NaN.

### Outputs

Paths are configured in `config.yaml` and resolved relative to that file:

| File | Content |
| --- | --- |
| results/signals.csv | Validation/test decision, partition, target, execution prices, both model probabilities/signals, momentum signal |
| results/trades.csv | Strategy/partition, decision, signal/probability, entry/exit times and prices, direction, quantity, duration, gross P&L |
| results/equity.csv | Strategy/partition, initial zero, completed-trade timestamp and cumulative gross P&L |
| results/metrics.json | Separate validation/test predictive and trading metrics for all three strategies |
| results/experiment.metadata.json | Source/feature/config identities and hashes, simulation, model settings, columns, dates/counts, exclusions, assumptions, library versions |

Momentum trade probability is empty because this strategy has no probability
estimate. Signals include all eligible predictions, including those ignored
while a position is open. Generated outputs remain local and excluded from Git.
Identical source files, configuration, feature metadata, and library versions
produce byte-identical exports. Re-running replaces these result files.
Output paths must be distinct and must not overwrite source, features, their
metadata, or configuration. The dashboard reads these saved outputs without
training models or generating
predictions. Experiment metadata also includes SHA-256 hashes for all four
result files. After upgrading from Topic 4, run the backtest command once to
generate this manifest. Mixed, modified, or stale result files are rejected.

### Verified full historical experiment (2026-10-23)

The source has **29,567 retained candles** and **29,311 feature observations**.
There are **22,914 valid four-hour labels**. Exclusions are **6,393** forward
market-gap observations, **4** unavailable final outcomes, **5,112** additional
past-momentum gap observations, and **4** boundary rows after intersection.
The shared dataset contains **17,798 observations**. Prediction horizon and
holding period are both **four consecutive hourly bars**; quantity is one ounce.

All dates below are UTC. Original partition boundaries are fixed before filtering:

| Partition | Original rows | Original first decision | Original last decision |
| --- | ---: | --- | --- |
| Training | 17,586 | 2021-10-18 04:00 | 2024-10-07 07:00 |
| Validation | 5,862 | 2024-10-07 08:00 | 2025-10-03 02:00 |
| Test | 5,863 | 2025-10-03 03:00 | 2026-10-01 00:00 |

| Partition | Eligible rows | First eligible decision | Last eligible decision | Last exit |
| --- | ---: | --- | --- | --- |
| Training | 10,679 | 2021-10-18 04:00 | 2024-10-07 03:00 | 2024-10-07 07:00 |
| Validation | 3,558 | 2024-10-07 08:00 | 2025-10-02 16:00 | 2025-10-02 20:00 |
| Test | 3,561 | 2025-10-03 03:00 | 2026-09-30 16:00 | 2026-09-30 20:00 |

| Partition / strategy | BUY | SELL | HOLD | Directional accuracy | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation / AntTrader | 469 | 171 | 2,918 | 50.70% | 0.49833 |
| Validation / Momentum | 1,909 | 1,645 | 4 | 50.56% | Undefined |
| Validation / Raw input | 1,108 | 907 | 1,543 | 52.25% | 0.52220 |
| Test / AntTrader | 507 | 297 | 2,757 | 51.81% | 0.52596 |
| Test / Momentum | 1,828 | 1,733 | 0 | 48.72% | Undefined |
| Test / Raw input | 1,078 | 1,000 | 1,483 | 50.94% | 0.50495 |

Test-period results, **gross price-only simulated P&L**, with all costs zero:

| Metric | AntTrader | Momentum | Raw-input model |
| --- | ---: | ---: | ---: |
| Completed trades | 465 | 1,015 | 809 |
| Win rate | 54.84% | 49.75% | 50.68% |
| Total gross P&L (USD) | 161.969 | 1,475.739 | -1,082.725 |
| Average gross P&L/trade (USD) | 0.348 | 1.454 | -1.338 |
| Profit factor | 1.02981 | 1.14538 | 0.88637 |
| Maximum realized drawdown (USD) | 718.317 | 734.926 | 1,512.654 |

Validation gross P&L: AntTrader **-211.655 USD**, momentum **-387.864 USD**,
raw input **372.207 USD**. AntTrader's test accuracy/AUC exceed raw input,
but its validation accuracy/AUC are lower. AntTrader's test gross P&L is
positive and below momentum. HOLD occurs in **82.01%** of validation and
**77.42%** of test observations. These findings do not establish a reliable
market edge. Performance differs between validation and test; sensitivity to
other split periods has not been tested. No alternative periods or parameters
were selected to improve results. Zero costs, idealized latency, uncalibrated
probabilities, overlapping predictive labels, one historical source, and one
fixed split prevent claims of real trading profitability.

Verification: **162 pytest tests pass** and **Ruff passes** in local Python
3.13.11. Two complete reruns reproduce all five exports byte for byte.
Independent execution/label/equity reconciliation passes; 60 sampled ant
windows reproduce the saved features across the full period. Streamlit passes
AppTest and starts with HTTP 200 / health=ok. Source and feature files remain
unchanged. Local `results/topic4_validation.json` records the verification and
output hashes. No manual input or external-data blocker remains.

## Topic 5 integration verification (2026-10-23)

Local Python 3.13.11 passes **203 pytest tests**, Ruff, and git diff --check.
A fresh virtual environment installs .[dev], passes pip check, all 203 tests,
and Ruff. Automated checks cover malformed/missing/stale inputs, duplicate and
UTC timestamps, replay parity, prediction/trade markers, empty trade files,
metric reconciliation, partition filters, and Streamlit controls.

Full-data verification loads all saved research outputs, reproduces 60
historical replay windows and 100 feature-CLI sample rows, and reproduces two
complete backtest reruns byte for byte. The source, features, and all four
Topic 4 result files are unchanged. Only metadata adds result hashes.

Headless Chromium exercises the three tabs, timestamp selection, replay slider
and checkpoints, strategy/partition controls, and charts. Screenshots were
inspected. Test and validation equity values reconcile with saved metrics;
no browser JavaScript errors occurred. Streamlit startup returns HTTP 200 and
health=ok. Local evidence remains in results/topic5_validation.json and
results/topic5_browser_validation.json. No verification blocker remains.

## Disclaimer

This is experimental research, not financial advice. Historical gross price-only
results do not establish future profitability. No license is granted yet.
