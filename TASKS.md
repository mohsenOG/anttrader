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
- [ ] Obtain or prepare Dukascopy data.
- [x] Implement CSV loader.
- [x] Implement data validation.
- [x] Implement return calculation.
- [x] Implement binary encoding.
- [x] Implement serpentine grid mapping.
- [x] Implement historical rolling-window grid generation.
- [x] Write tests.
- [x] Verify integration with synthetic candles and the Streamlit scaffold.
- [ ] Verify real Dukascopy data integration and the latest historical grid.
- [x] Update README.
- [x] Commit and push changes (implementation 2534686, verified on origin/main).

### Verification status (2026-10-22)

- Python 3.13.11 in the local .venv: 69 pytest tests pass; Ruff passes.
- Synthetic data-to-grid integration passes. The no-look-ahead test passes.
- Streamlit renders in AppTest and launches with HTTP 200 / health=ok.
- The missing-CSV pipeline prints the source, settings, target path, and README
  instructions. Existing .gitignore excludes full datasets; dependencies unchanged.
- No real historical dataset is present. Archive requests returned HTTP 429 and
  the Dukascopy JSON endpoint returned an AWS WAF challenge. Manual export and
  real-data verification remain pending. See README.md for the exact export step.
- Topic 2 is not fully complete. Real candle dates, counts, irregularities, and
  latest-grid results cannot be reported until genuine data is loaded.

## Future topics

- [ ] Ant simulation and feature extraction.
- [ ] Signal decoder and backtester.
- [ ] Dashboard and integration.
