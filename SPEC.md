# AntTrader specification

## Purpose

Investigate whether Langton's Ant can produce useful predictive information for
Gold (XAU/USD) on the 1-hour timeframe. This is experimental historical research.
Profitability is unknown. No trading strategy or research result is implemented.

## Confirmed scope

The eventual application will load historical candles, encode market information
as a grid, simulate Langton's Ant, extract numerical features, generate BUY, SELL,
and HOLD signals, backtest them, and display results in Streamlit.

Topic 1 is the current implementation milestone. It provides project setup,
documentation, configuration, tests, and clearly labeled runnable placeholders.
Topics 2-5 are pending detailed user approval. The module layout does not finalize
their interfaces or algorithms.

## Agreed technology stack

- Python 3.13 with a standard local .venv.
- NumPy, pandas, scikit-learn, Streamlit, Plotly, and PyYAML.
- pytest and Ruff for local verification.
- pyproject.toml for packaging and dependency declarations.
- GitHub repository: mohsenOG/anttrader; public; default branch main.
- No license initially. No GitHub Actions, CI/CD, Docker, or deployment pipeline.
- One Codex agent with shared instructions in AGENTS.md.
- Numba may be considered later. It is not a dependency now.

## High-level architecture

Planned flow:

Historical candles -> data -> grid -> simulation -> features -> decoder -> backtest

`anttrader.pipeline` will coordinate this flow. `app.py` will provide the Streamlit
view. `config.yaml` holds configuration. `data/` holds local historical datasets.
`results/` holds generated research outputs. These local artifacts are ignored by
Git. The current pipeline and dashboard only display setup status.

## Planning topics

1. **GitHub, Codex & Agent Setup**: current implementation milestone.
2. **XAU/USD Data & Grid Encoding**: pending detailed approval.
3. **Langton's Ant Simulation & Feature Extraction**: pending detailed approval.
4. **Signal Decoder & Backtesting**: pending detailed approval.
5. **Dashboard, Integration & Final Codex Instructions**: pending detailed approval.

## Configuration

Confirmed parameters are XAU/USD, 1H, and historical research only.

Provisional research defaults are one ant, 256 returns of lookback, a 16 x 16 grid,
2,048 simulation steps, toroidal boundaries, logistic regression, and a prediction
horizon of four bars. These are experimental starting values, not optimized or
final parameters. They are recorded in config.yaml and do not yet drive behavior.

## Explicitly out of scope

- Live trading, broker integration, and order execution.
- Profitability claims, fabricated prices, signals, or performance results.
- Full research algorithms during Topic 1.
- Additional markets or timeframes without approval.
- Production infrastructure or complex GitHub administration.
- Final algorithm choices for Topics 2-5 before detailed approval.

## Unresolved design decisions

- Topic 2: data provider, access, licensing, candle format, time zone, missing data,
  return definition, grid encoding, and lookback ordering.
- Topic 3: initial ant state, grid cell rules, feature definitions, deterministic
  initialization, and simulation conventions beyond the provisional defaults.
- Topic 4: target labels, decoder training, chronological splits, BUY/SELL/HOLD
  rules, execution timing, position rules, spread, slippage, and other costs.
- Topic 5: dashboard controls, charts, integration interfaces, and final workflows.

These decisions require approval before implementation. No default trading or
scientific rule is implied by a placeholder module or a configuration key.
