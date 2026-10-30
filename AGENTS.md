# test — Swedish trading-signals system (this branch)

This branch runs the system on Nasdaq Stockholm. `main` holds the frozen
Brazilian (B3) version; do not port changes between them.

A genetic algorithm tunes a 38-gene trading policy for 120 Swedish Large Cap
tickers. One engine executes that policy everywhere: GA training, the
walk-forward replays that validate it, and the daily orders. Costs model an
ISK account (0.25% brokerage per order, 0.10% slippage per side, no per-trade
tax).

Code comments, log lines and user-facing strings are predominantly Portuguese.
Match the surrounding language when editing a file.

## Repo guides

### Swedish plan and guide

Decisions, success criteria, ramp-up and phase status: `docs/sweden/plano.md`.
How the engine decides, commands, and how to read the replays:
`docs/sweden/guia.md`. Read both before changing the engine, the universe,
costs, the walk-forward or the daily run.

### Pipeline and commands

Entry points, env knobs, CI topology, and which artifacts are generated versus
committed, written for the Brazilian pipeline on `main`; the Swedish entry
points (`walkforward.py`, `replay_se.py`, `portfolio_se.py`, `daily_se.py`,
`paper_se.py`) are in the guide: `docs/agents/pipeline.md`.

### Promotion gates

The four-layer gate stack of the Brazilian improvement loop, and why it
stalled: `docs/agents/promotion-gates.md`. Sweden does not use it (ADR 0002).

### Gotchas

Traps measured in this repo: stale README numbers, shallow clone, tracked
files under ignored paths, sweep-JSON accumulation, empty-sample holdout:
`docs/agents/gotchas.md`. Read when a number, a path, or git history surprises
you.

## Invariants

- Market definitions live only in `market_se.py`: universe, suffix, index,
  benchmark, costs, tax, order validity, account size.
- One engine: `ga_run_modular_final._backtest_stats_global_intraday`.
  `ga_run.backtest_stats_global_intraday` only applies the optional stage gate
  and delegates. A rule that is not inside the engine is not validated.
- The engine follows the real execution rule (ADR 0001) and uses no future
  data: decisions for bar i read bars up to i-1, stops use the previous ATR,
  signal strength uses a trailing-year percentile. `tests/test_execution_rule.py`
  checks causality; keep it green.
- Daily orders are exactly the engine's orders (ADR 0004); a genome reaches
  daily use only through `walkforward.py accept` after a replay (ADR 0002).
- The universe is the frozen snapshot in `data/universe_se.json`; model
  results never add or remove a ticker (ADR 0003).
- Feed the GA only causal features: the allowlist in `ga_run.py` blocks model
  outputs and snapshot fundamentals; quarterly fundamentals stay off in the ETL.
- Compare the system with `^OMXSBGI` (dividends reinvested), never with the
  price index `^OMXSPI`.
- State the data vintage whenever you quote a fitness number.
- Keep `output/` out of new commits; the ETL regenerates it.

## Agent skills

### Issue tracker

Issues live as GitHub issues in `gcallj/test`, driven by the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, using the default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` plus `docs/adr/` at the repo root. See `docs/agents/domain.md`.
