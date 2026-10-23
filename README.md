# AntTrader

AntTrader is a signal research tool, not a trade simulator. It uses historical
Dukascopy XAU/USD hourly candles, Langton's Ant, and a frozen logistic decoder
to study BUY / SELL / HOLD predictions. BUY and SELL are directional predictions.

Market data -> Grid -> Ant simulation -> Features -> Decoder -> Saved signals.

## Installation

Use Python 3.13. In PowerShell:

```powershell
git clone https://github.com/mohsenOG/anttrader.git
cd anttrader
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\Activate.ps1
```

Activation is optional. Use `.venv\Scripts\python.exe` and
`.venv\Scripts\ruff.exe` directly if activation is blocked.
Configuration is in config.yaml. No additional dependency or service is needed.

## Run

Prepare the local Dukascopy CSV as described below. From the repository root:

```powershell
python -m anttrader.pipeline
python -m anttrader.pipeline --stage features
python -m anttrader.pipeline --stage signals
streamlit run app.py
```

The first command checks candles and grids. Generate the complete feature export
without `--limit`. Existing complete features can be reused when source and
simulation settings match. Run the signals command once after upgrading to
replace legacy signals/metrics/metadata. Old local trade/equity files are unused
and can remain. Paths resolve relative to config.yaml. Generated data stays out
of Git. The dashboard never generates features, trains models, or runs experiments.

## Dashboard

- **Ant Simulation:** select a feature timestamp, inspect the original 16 x 16
  grid, replay steps 0–2,048, jump to checkpoints, and inspect pose, direction,
  behavioral features, and an exact associated saved signal when available.
- **Gold & Signals:** inspect a full-width 800-pixel candlestick chart. Large
  green upward BUY and red downward SELL triangles mark actual decision times.
  Select Test or Validation, a comparison method, dates, and visible marker sides.
  The default range is the latest ten calendar days of the selected partition.
  Use the Plotly toolbar, drag to pan, and scroll to zoom. HOLD stays in the table.
  Latest Historical Signal shows the latest saved AntTrader row, not a live signal.

Markers use only the latest completed candle's high/low/close, with a visual
spacing offset. Hover shows the UTC decision, direction, P(up), last completed
close, and evaluation partition. Marker coordinates do not represent execution
prices. Date and marker controls change display only.

Prediction statistics below the chart cover the full selected partition,
independent of chart filters. Overall model classification accuracy uses
`p_up >= 0.5` across all eligible observations, including HOLD. BUY/SELL accuracy
excludes HOLD and compares the predicted direction with the four-hour label.
Momentum accuracy uses non-HOLD rows. ROC AUC is undefined for momentum or a
single target class. No actionable signals gives undefined BUY/SELL accuracy.

Missing or incompatible files show their path and recovery commands. Restore
the source CSV if needed, generate complete features, then run `--stage signals`.
Do not modify exports to bypass validation. Stop the pipeline before loading
its saved bundle. Bounded caches invalidate when input files change.

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
loads saved research results. Generate features and signals separately;
missing inputs produce file-specific errors and recovery commands.

Unit tests use small **synthetic** inputs in temporary directories. They validate OHLC,
timezones, completion times, gaps, sign encoding, layout, rolling alignment, and
no look-ahead. A separate optional integration test uses the local CSV to check
all rolling decision timestamps, grid count, the latest grid, and no look-ahead
on genuine source data. It uses the explicit 2026-10-01 UTC cutoff for this sample.

Read [SPEC.md](SPEC.md) for approved rules and open decisions, [TASKS.md](TASKS.md)
for verified progress, and [AGENTS.md](AGENTS.md) for agent instructions.
`config.yaml` contains the approved simulation, decoder, and signal research settings.

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
The signal experiment below retains the approved prediction methodology.

## Signal decoder and historical evaluation

The decoder keeps all 60 ant features. A raw-input comparison model uses the
original 256 chronological binary states. Both models use StandardScaler and
LogisticRegression with L2, C=1, lbfgs, max_iter=1000, random_state=42. They fit
training rows only and remain frozen. No thresholds or model settings are tuned.
Momentum uses four consecutive completed close-to-close intervals.

At decision D, the candle opening at D-1h has completed. The four-hour horizon
is retained **only for prediction labeling**:
`future_return = ln(open[D+4h] / open[D])`; target is 1 when positive, else 0.
Require every hourly opening from D-1h through D+4h. Exclude gaps and unavailable
outcomes. Future price fields are retrospective evaluation data, never inputs.

Split all feature rows before exclusions: 60% training, 20% validation, 20% test,
using floor boundaries. Purge training and validation labels ending at or after
the next partition's first decision. Keep the existing intersection with valid
momentum windows. Report validation and test separately.

- BUY: p_up >= 0.55.
- SELL: p_up <= 0.45.
- HOLD: 0.45 < p_up < 0.55.

There are no orders, positions, quantities, forced exits, trade simulations,
P&L, equity curves, or drawdown calculations. Probabilities are uncalibrated
model estimates. Historical prediction accuracy does not prove profitable trading.

### Saved outputs

| File | Content |
| --- | --- |
| results/signals.csv | Decision, partition, last completed close, model probabilities/signals, momentum, retrospective label fields |
| results/metrics.json | Separate validation/test predictive counts, fractions, overall/actionable accuracy and ROC AUC |
| results/experiment.metadata.json | Input/output hashes, configuration, model settings, boundaries, counts, exclusions and versions |

Labels use `label_start_timestamp`, `label_start_price`, `label_end_timestamp`,
`label_end_price`, `future_return`, `target`, `target_valid`, and `target_exclusion`.
`reference_price` is the last completed candle close. There are no entry/exit
fields. Complete source/feature identities, exact eligible timestamps, labels,
partitions, momentum, threshold classifications, and predictive metrics are
validated on loading. Partial, mixed, stale, or altered results are rejected.
Same inputs and library versions produce reproducible outputs.

### Verification

Use the local virtual environment:

```powershell
python -m pytest -q
ruff check .
```

The tests cover unchanged ant/grid behavior, frozen training-only predictions,
thresholds, future-data independence, label timestamps, boundary purging,
signal statistics, past-only marker placement, replay, missing/stale bundles,
and Streamlit controls without trade/equity files.

Signals-only verification (2026-10-23): local Python 3.13.11 passes all 192
applicable tests, Ruff, and git diff --check. All 7,119 saved historical
predictions match the previous implementation exactly, including both model
probabilities (maximum difference 0.0). Source data, ant features, and grid/ant
computation are unchanged. Repeated signal runs produce byte-identical outputs.
Browser checks verify both tabs, partitions, marker filtering, final ant replay,
and BUY/SELL hover data. Streamlit health is ok; no browser errors were reported.
Local ignored chart screenshots and detailed evidence are listed in TASKS.md.

## Limitations

Historical research only. There is no live feed, broker connection, trade
execution, or trade simulation. One dataset and one chronological split do not
establish a reliable market edge. This is experimental research, not financial
advice. No license is granted yet.


## AntTrader v1.1 Experiment Lab

Open **Experiment Lab** with `streamlit run app.py`. The original reference is
16 x 16, 2,048 steps, North, newest return. The original pipeline and saved signal
outputs remain available. No experiment runs on startup or tab selection.

### Controls and workload

Choose a preset (8 x 8, 12 x 12, 16 x 16, 16 x 6, 20 x 20) or Custom.
Dimensions mean **rows x columns**; each dimension supports 2..32 cells.
The lookback is exactly rows*columns returns and rows*columns+1 completed closes.
Custom steps support 4..16,384. Four checkpoints use half-up integer rounding at
12.5%, 25%, 50%, and 100%, adjusted to stay positive, unique, and increasing.
The reference checkpoints remain 256, 512, 1,024, and 2,048.

Choose North, East, South, or West. Newest return uses (rows-1, 0) for even rows
and (rows-1, columns-1) for odd rows. Center uses (rows//2, columns//2), with
zero-based coordinates. Only the original one-ant, binary, toroidal rules apply.

Submit **Run Experiment** for one configuration. For **Run Batch**, select grids
and step counts; the backend runs their Cartesian product with shared direction
and start settings. Select Custom in the batch grids to use the custom dimensions.
Include custom steps with the checkbox. A batch supports at most 30 unique
configurations. The application shows experiment count, common window count,
and total ant transitions before execution. Decoder fitting adds work. Results
load from cache when available. There is no additional large-run approval step.

Enable **Use limited historical sample** to use the first N completed candles.
This is for development checks, not a representative research conclusion.
Otherwise the lab uses the available completed history. **Reset Settings**
restores the original ant controls and does not remove saved experiments.

### Compare and inspect

History includes completed and failed runs. Open a completed experiment to use
its configuration and predictions in **Ant Simulation** and **Gold & Signals**.
Use **View original v1 reference** to return to the original research view.
Opening a result does not train a model. Marking a research candidate records a
local label only.

Each batch starts with the largest grid lookback. Every model uses the same
decision timestamps, chronological 60/20/20 boundaries, gap exclusions, and
purged labels. Each configuration fits its own StandardScaler and logistic
regression. Its raw-input baseline uses exactly the same return signs as its ant.
Momentum retains the original four completed intervals.

Choose a comparison cohort to see its **validation** leaderboard, sortable table,
heatmap, and accuracy-versus-coverage chart. Filter direction and starting
position for the heatmap. Compare up to three compatible configurations, inspect
all model and baseline metrics, signal disagreement on common timestamps, and
historical signal overlays. Incompatible cohorts or software versions cannot
share a ranking. To compare them, run their configurations together in a new
batch. Test results are separate, previously inspected exploratory diagnostics.

Metrics include ROC AUC, overall accuracy at probability >= 0.5, BUY/SELL-only
accuracy, coverage, side-specific accuracy, counts, Brier score, sample counts,
and evaluation dates. Coverage is (BUY+SELL)/eligible observations. HOLD does not
enter BUY/SELL accuracy. Brier score is mean squared probability error; lower is
better. Undefined metrics remain null/undefined. Momentum has no probability,
ROC AUC, or Brier score. Always read accuracy beside counts and coverage; fewer
than 30 actionable observations produces a warning.

Trying many configurations can produce a high score by luck. Use validation for
candidate comparisons. The original test period has already been inspected.
Any final improvement claim needs fresh forward data or an untouched period.
Small development samples establish software behavior only. There are no
profitability calculations, automatic deployment, trading, or live feeds.

### Saved results, cache, rerun, and deletion

Results are stored separately under `results/experiments/<id>/`:

- `metadata.json`: normalized configuration, dataset/cohort identity, software
  hashes and library versions, partitions, elapsed time, warnings, and status.
- `features.csv`, `predictions.csv`, and `metrics.json`: identified artifacts.
- Optional `candidate.json`: a local research-candidate label.

Stable IDs include configuration, dataset, sample/cohort, evaluation rules, and
software versions. Same inputs reuse a completed result without training.
**Rerun experiment** explicitly recomputes only that result. It preserves the
old result if calculation fails and validates the replacement before publication.
Changed data or software requires a new experiment. Partial or altered artifacts
are rejected. Failed runs remain visible with an error and can be rerun.

To remove one result, select it, enable **Confirm deletion of the selected
experiment**, then press **Delete selected experiment**. Deletion rejects foreign
files and links and removes only that experiment. Source data and original
research outputs are preserved. No delete-all action exists.

### CLI and file-based batches

The CLI uses the same backend as Streamlit. In `config.yaml`,
`experiment_lab.configurations` is a list of configurations. One entry runs one
experiment; multiple entries run a common-cohort batch. For example:

```yaml
experiment_lab:
  output_dir: results/experiments
  max_batch: 30
  sample_candles: null
  configurations:
    - {rows: 8, columns: 8, steps: 512, direction: north, start: newest_return}
    - {rows: 8, columns: 8, steps: 1024, direction: north, start: newest_return}
    - {rows: 16, columns: 6, steps: 512, direction: north, start: newest_return}
    - {rows: 16, columns: 6, steps: 1024, direction: north, start: newest_return}
```

With the local Python 3.13 virtual environment active:

```powershell
python -m anttrader.experiments --config config.yaml
python -m anttrader.experiments --config config.yaml --sample-candles 650
python -m anttrader.experiments --config config.yaml --sample-candles 650 --rerun
streamlit run app.py
python -m pytest -q
ruff check .
```

A relative source path is resolved against the configuration file. Custom output
paths must stay inside its `results/experiments` directory. Full historical
experiments and optimization are user-run. Ordinary tests use small synthetic
fixtures and existing data checks; they do not run full historical experiments.
