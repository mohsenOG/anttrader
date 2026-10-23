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
- Topic 4 design can begin. No labels, signals, backtesting, profitability
  findings, or interactive dashboard animation were added. No manual data step
  or implementation blocker remains.

## Future topics

- [x] Signal decoder and backtester.
- [ ] Dashboard and integration.

## Topic 4: Signal Decoder & Backtesting

- [x] Document approved decoder and trading rules.
- [x] Update configuration.
- [x] Validate feature and price-data alignment.
- [x] Implement future-return targets.
- [x] Implement chronological dataset splitting.
- [x] Implement leakage-safe model training.
- [x] Implement BUY/SELL/HOLD decoding.
- [x] Implement historical trade simulation.
- [x] Implement theoretical P&L calculations.
- [x] Implement performance metrics.
- [x] Implement benchmark strategies.
- [x] Export experiment results.
- [x] Add automated tests.
- [x] Run full historical experiment.
- [x] Update README.
- [x] Verify implementation.
- [ ] Commit and push verified changes.

### Topic 4 verification (2026-10-23)

- Python 3.13.11 in the local .venv: 162 pytest tests pass; Ruff passes.
  Existing Topic 1-3 tests remain passing. git diff --check passes.
- Complete unchanged source/feature history validated: 29,567 candles,
  29,311 feature observations, 60 finite ant columns, exact UTC alignment,
  matching source SHA-256 and simulation/encoding metadata. Partial exports,
  invalid columns/timestamps, metadata mismatches, and input-overwrite paths
  are rejected. Sixty sampled features reproduce across the entire period.
- 22,914 valid four-hour labels; 6,393 forward-gap exclusions and four
  unavailable final outcomes. Another 5,112 past-momentum gap exclusions and
  four boundary purges leave 17,798 common eligible observations:
  training 10,679, validation 3,558, test 3,561. Exact original/eligible dates
  and counts are recorded in README.md and experiment metadata.
- Both StandardScaler/logistic models fit the same training rows only and
  remain frozen. Ant uses 60 features; raw input uses 256 original binary
  states. Four-interval momentum uses five completed, consecutive closes.
  All strategies use identical eligible dates, four-hour execution windows,
  fixed one-ounce quantities, one position at a time, and zero costs.
- Validation gross price-only P&L (USD): ant -211.655, momentum -387.864,
  raw 372.207. Test: ant 161.969, momentum 1,475.739, raw -1,082.725.
  Ant test accuracy/AUC: 51.81% / 0.52596; raw: 50.94% / 0.50495.
  Ant test HOLD fraction: 77.42%. Ant underperforms momentum on test P&L.
  Validation findings are weaker; no period sensitivity or real trading edge
  is established. No parameters were tuned after observing test results.
- Two complete historical reruns reproduce all five exports byte for byte.
  Independent label, execution, position-state, equity, P&L, metric-count,
  shared-date, and partition-cutoff checks pass. Exports contain 7,119 signal
  rows, 4,526 trades across six strategy/partition combinations, and 4,532
  equity rows including six zero baselines. Strict JSON contains no NaN/Inf
  numeric values; undefined metrics are explicit nulls or infinity strings.
- Streamlit passes AppTest and starts with HTTP 200 / health=ok. Its test
  server was stopped. Local results/topic4_validation.json records hashes
  and reconciliation checks; generated files remain ignored by Git.
- README, SPEC, configuration, and scaffold status are updated. No simulation
  or encoding change, new dependency, live trading, broker integration, or
  full dashboard was added. Topic 5 can proceed after publication completes.
