# AntTrader

Experimental Langton's Ant research for spot gold (XAU/USD) on the **1-hour**
timeframe. Profitability is unknown. This project uses historical data only.

Implemented: project setup, validated CSV ingestion, binary log-return encoding,
rolling serpentine grids, a command-line demonstration, and a Streamlit scaffold.
Pending: ant simulation, feature extraction, signals, backtesting, and full charts.

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
No additional dependencies or services are needed for Topic 2.

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
4. Disable creation of flat/no-tick or forward-filled candles. Export only source
   candles. Keep normal closures as gaps. Do not use bid/ask averages or ask prices.
5. Save the CSV as `data/xauusd_h1_bid.csv`.
6. Check the actual header and timestamp format before running the pipeline.

If the web widget is unavailable, use **JForex -> Tools -> Historical Data Manager**
with the same settings. Confirm the opening-time convention and timezone in the
export settings. Do not infer them from a filename or your computer timezone.

The official [export page](https://www.dukascopy.com/api/data/get/historical-data-export)
describes bid/ask data and volume. Direct download checks on 2026-10-22 encountered
HTTP 429 from the binary archive and an AWS WAF challenge from the Dukascopy JSON
endpoint. **No real history has been obtained or tested yet. Manual export is
pending.** Do not bypass an access challenge or silently substitute another feed.

### CSV contract

Use comma-separated columns:

```text
timestamp,open,high,low,close,volume
```

`volume` is optional. Use ISO 8601 timestamps with `Z` or a UTC offset, for example
`2024-01-05T20:00:00Z`. Columns are case-insensitive. `Gmt time` and `Local time`
are supported timestamp header aliases. A naive timestamp requires an explicit
`timestamp_timezone` in config; its default `UTC` describes the required export
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
as supplied; it is not assumed to represent exchange-wide gold trading volume.

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
streamlit run app.py
python -m pytest -q
ruff check .
```

The pipeline resolves the CSV path relative to the configuration file. It prints
the candle count, date range, validation report, number of historical grids, and
latest grid with its decision time and window. If the CSV is absent, it prints
setup instructions. It never supplies fake history. The Streamlit application
shows project status; it does not run ant simulations or trading logic.

Tests use small **synthetic** inputs in temporary directories. They validate OHLC,
timezones, completion times, gaps, sign encoding, layout, rolling alignment, and
no look-ahead. They are not a real-data integration test.

Read [SPEC.md](SPEC.md) for approved rules and open decisions, [TASKS.md](TASKS.md)
for verified progress, and [AGENTS.md](AGENTS.md) for agent instructions. Preserve
provisional future defaults in `config.yaml`: one ant, 2,048 steps, toroidal
boundaries, logistic regression, and a four-bar prediction horizon. These future
settings do not drive Topic 2 behavior.

## Disclaimer

This is experimental research, not financial advice. No profitable strategy or
verified performance result is claimed. No license is granted yet.
