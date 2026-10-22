# AntTrader agent instructions

## Project philosophy

- AntTrader is a small hobby project. Prefer simplicity, readability, and modularity.
- Avoid overengineering, unnecessary packages, and unnecessary infrastructure.
- Follow SPEC.md as the source of truth.
- Use one Codex agent. Do not use subagents unless the user explicitly requests them.

## Before coding

- Read AGENTS.md, SPEC.md, and TASKS.md.
- Understand the relevant existing code and callers before changing behavior.
- Ask the user about important unresolved design decisions.
- Do not invent missing trading rules or scientific assumptions.
- Make minimal, in-scope changes. Reuse existing code and standard library features.

## Python development

- Write clean, maintainable Python. Use type hints where practical.
- Keep modules focused. Avoid unnecessary abstractions and dependencies.
- Make simulations deterministic.
- Make configuration changes through config.yaml.
- Use Python 3.13 and the local .venv for development.

## Quantitative research

- Never introduce look-ahead bias.
- Use only market data available at the signal timestamp.
- Preserve chronological separation between training and testing.
- Account for trading costs when implementing backtests.
- Make experiments reproducible.
- Never fabricate research results or performance metrics.
- Do not claim that the strategy is profitable without verified evidence.
- Keep the project limited to historical research unless the user approves a change.

## Testing

- Write meaningful tests for implemented functionality.
- Leave one small runnable check for non-trivial logic. Do not add redundant tests.
- Run `python -m pytest -q` and `ruff check .` after relevant changes.
- Fix failing tests where possible.
- Report tests that cannot be executed or remain failing.

## Task completion

- Update TASKS.md. Mark a task complete only after verification.
- Update documentation when relevant.
- Summarize changes and report actual verification results.
- Mention remaining blockers or open questions.
- Use short, direct English and consistent terms.
- Do not commit or push unless the user authorizes it.
- Include a simple English commit message in the completion report.
