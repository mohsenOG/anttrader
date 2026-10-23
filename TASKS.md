# AntTrader tasks

## Topic 1: GitHub, Codex & Agent Setup

- [x] Create or initialize GitHub repository.
- [x] Create Python project structure.
- [x] Create AGENTS.md.
- [x] Create SPEC.md.
- [x] Create TASKS.md.
- [x] Create README.md.
- [x] Create config.yaml.
- [x] Configure Python dependencies.
- [x] Add basic pytest and Ruff setup.
- [x] Create minimal runnable application scaffolding.
- [x] Verify installation and tests.
- [x] Commit and push the completed foundation.

## Topic 2: XAU/USD Market Data & Grid Encoding

- [x] Document approved specification.
- [x] Configure historical data settings.
- [x] Obtain or prepare Dukascopy data.
- [x] Implement CSV loader.
- [x] Implement data validation.
- [x] Implement return calculation.
- [x] Implement binary encoding.
- [x] Implement serpentine grid mapping.
- [x] Implement historical rolling-window grid generation.
- [x] Write tests.
- [x] Verify integration with synthetic candles and the Streamlit scaffold.
- [x] Verify real Dukascopy data integration and the latest historical grid.
- [x] Update README.
- [x] Commit and push changes (completion fa895b2, verified on origin/main).

### Verification status (2026-10-23)

Topic 2 is complete. No manual data-acquisition or verification step remains.

- Python 3.13.11 in the local .venv: 77 pytest tests pass; Ruff passes.
- Synthetic and real-data integration pass, including explicit no-look-ahead
  checks, candle completion times, chronological windows, and serpentine layout.
- Streamlit renders in AppTest and launches with HTTP 200 / health=ok.
- Genuine user-supplied Dukascopy XAU/USD H1 BID history spans opening times
  2021-10-01 00:00 UTC through 2026-09-30 23:00 UTC. All 43,824 CSV rows match
  the 60 cached monthly archives for timestamps, OHLC / 1,000, and rounded volume.
- The configured loader excludes 14,257 flat zero-volume closure rows and retains
  29,567 completed candles. It keeps traded flat candles. The original CSV is
  unchanged, and all local data, cache files, and reports stay outside Git.
- Filtering exposes 1,290 gaps over one hour, all accounted for by excluded source
  rows. The maximum opening-time interval is 74 hours. No trading-calendar audit
  is claimed; no invalid OHLC, duplicates, or missing prices were found.
- Generated 29,311 grids. The latest decision is 2026-10-01 00:00 UTC, from a
  257-close window opening 2026-09-15 19:00 UTC through 2026-09-30 23:00 UTC.
  Its 16 x 16 grid contains 130 zeros and 126 ones; independent encoding passes.
- README.md records the source format, SHA-256, reproduction commands, and the
  local results/topic2_validation.json report. A fresh clone needs its own CSV;
  the optional real-data test skips when that local dataset is absent.
- The missing-CSV message remains tested. Dependencies and infrastructure are
  unchanged. At Topic 2 completion, no ant or trading logic was implemented.

## Topic 3: Langton's Ant Simulation & Feature Extraction

- [x] Document approved simulation specification.
- [x] Update simulation configuration.
- [x] Implement Langton's Ant movement rules.
- [x] Implement toroidal boundaries.
- [x] Implement simulation checkpoints.
- [x] Track movement and visited cells.
- [x] Implement behavioral feature extraction.
- [x] Implement historical batch simulation.
- [x] Export historical features to CSV.
- [x] Implement optional trajectory recording.
- [x] Add automated tests.
- [x] Benchmark full-dataset performance.
- [x] Integrate with existing pipeline.
- [x] Update README.
- [x] Verify and commit changes.

### Topic 3 verification (2026-10-23)

- Local Python 3.13.11: 131 pytest tests pass; Ruff and git diff --check pass.
- Approved rules verified: one ant, binary 16 x 16 grid, start (15, 0) North,
  READ -> TURN -> FLIP -> MOVE, 0=right, 1=left, toroidal wrapping, 2,048 steps,
  checkpoints after 256/512/1,024/2,048 transitions. Independent hand-calculated
  trajectory, all four edges, cumulative visits, and replay reconstruction pass.
- Full source history processed: 29,567 retained candles, 29,311 independent
  simulations/CSV rows, 60,028,928 transitions, 60 feature columns. All decision
  timestamps match completed-candle windows and are strictly chronological.
  Zero missing/nonfinite values, duplicate timestamps, or simulation failures.
- Full CLI wall time: 23.378 seconds. Peak process working set: 101.574 MiB,
  measured with Windows GetProcessMemoryInfo. Windows 11 build 26200,
  Python 3.13.11, NumPy 2.5.4, pandas 3.1.0. A separate 100-grid sample after
  loading took 0.0654 seconds and reproduced CSV output exactly; the initial
  100-window CLI run including loading took 3.093 seconds.
- Source SHA-256 is unchanged. Input grids remain unchanged. Future-candle
  changes do not change earlier features. Generated CSV, metadata, and
  results/topic3_validation.json remain local and ignored by Git.
- Topic 2 CLI still produces all 29,311 grids. Streamlit passes AppTest and
  launches with HTTP 200 / health=ok. No new dependencies or infrastructure.
- Topic 4 design can begin. No labels, signals, historical prediction evaluation, profitability
  findings, or interactive dashboard animation were added. No manual data step
  or implementation blocker remains.

## Topic 4: Signal Decoder and Historical Evaluation

The original decoder and dashboard were implemented and verified before this
refactor. Their historical trading scope is superseded by the signals-only
specification below. The approved model, features, thresholds, and chronological
research methodology remain unchanged.

## Signals-only refactor

- [x] Inspect dependencies and save historical prediction/source baselines.
- [x] Remove trade, position, forced-exit, P&L, equity, and drawdown code/config.
- [x] Retain the four-hour horizon only for forecast labels.
- [x] Add the signals CLI and signal-only outputs with strict validation.
- [x] Keep the ant simulation, 60 features, models, and thresholds unchanged.
- [x] Provide Ant Simulation and Gold & Signals tabs.
- [x] Enlarge the gold chart and add visible past-only decision markers.
- [x] Keep separate validation/test predictive statistics and signal history.
- [x] Verify dashboard loading without trading output files.
- [x] Update tests and documentation.
- [x] Run pytest, Ruff, and historical prediction parity checks.
- [x] Launch Streamlit and visually verify the chart and controls.
- [x] Obtain user confirmation before commit and push.

### Verification (2026-10-23)

- Before refactoring: local Python 3.13.11, 203 pytest tests pass; Ruff passes.
- After refactoring: 192 applicable pytest tests pass in 32.29 seconds. Ruff and
  git diff --check pass. Trading tests were removed with their deleted code;
  predictive metric tests were retained and extended.
- Full signal stage validates 29,567 candles and 29,311 feature rows. Its 7,119
  saved predictions retain 3,558 validation and 3,561 test rows. Both models'
  probabilities, all three classifications, decisions, partitions, targets, and
  forecast returns match the pre-refactor CSV exactly. Maximum probability
  difference is 0.0. A second signal run reproduces all three outputs byte for byte.
- Source CSV, feature CSV/metadata, grid, simulation, and feature-code SHA-256
  hashes are unchanged. Model fitting, inference, threshold decoding, historical
  input construction, splitting, and feature validation functions are AST-identical
  to their pre-refactor versions. No model settings or thresholds were changed.
- Dashboard tests pass without trade/equity files. Legacy files do not affect
  loading or cache identity. Dashboard loading is tested with model fitting
  disabled. Strict signal schemas/hashes, eligible timestamps, label boundaries,
  momentum, reference prices, predictive metrics, and exclusions are reconciled.
- Browser verification passes: exactly two tabs, validation/test separation,
  BUY-only filtering, replay step 2,048, readable 800-pixel candles and 16-pixel
  BUY/SELL markers, and correct UTC/probability/reference/partition hover data.
  Both observed hover records match the saved signals. No browser errors were
  reported. Streamlit starts and its health endpoint returns ok.
- Future-candle changes cannot change earlier grids, features, model inputs, or
  marker coordinates. Automated checks preserve threshold and boundary behavior.
- Local ignored evidence: results/signals_refactor_validation.json,
  results/signals.before_refactor.csv, results/signals_refactor_baseline.json,
  results/signals_refactor_gold.png, and results/signals_refactor_hover.png.
- No dependency, infrastructure, model optimization, or new research rule was
  added. No implementation or verification blocker remains. The user approved
  committing and pushing the verified implementation.
