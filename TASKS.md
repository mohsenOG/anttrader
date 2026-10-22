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
  unchanged. Topic 3 design can begin; no ant or trading logic was implemented.

## Future topics

- [ ] Ant simulation and feature extraction.
- [ ] Signal decoder and backtester.
- [ ] Dashboard and integration.
