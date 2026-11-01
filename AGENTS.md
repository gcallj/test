# test — Swedish trading-signals system (this branch)

This branch runs the system on Nasdaq Stockholm. `main` holds the frozen
Brazilian (B3) version; do not port changes between them.

**Decision (ADR 0012): the owner invests in a broad Swedish index fund; active
strategy development is closed.** Findings, lessons and how to invest are in
`docs/sweden/conclusao.md` — read it first. Nordea fund comparison (fees,
annualized returns, accumulating share classes, withdrawals):
`docs/sweden/fundos_nordea.md`.

Seven Swedish strategies have been built and judged out of sample, and all
seven failed:
- v1, swing GA (`ga_run*.py`);
- v2, monthly momentum rotation (`rotation_se.py`, ADR 0005);
- v3, active selection by forecasts (`selective_se.py`, ADR 0007);
- v4, public-information events (`events_se.py` on `altdata_se.py`, ADR 0008);
- v5, analyst revisions from Börsvärlden (`revisions_se.py` on `analyst_se.py`
  and `analyst_parse.py`, ADR 0009);
- v6, a v1-style swing on one account with a CMA-ES search and the v4/v5
  inputs (`swing_se.py`, ADR 0010);
- v7, calendar effects with no tunable parameter: Halloween timing of the
  index fund and the dividend month (`seasonal_se.py`, ADR 0011).

v7's winter-minus-summer gap on the index was +0.66%/month (t 0.91 over
13.5 years); the dividend month was negative. Being in cash each summer
would have cost 24 points against the index in 2023–2026.

v6 returned +54.5% against +63.6% for the index in 2023–2026, with lower
drawdown. Its per-trade edge decayed from fold 1 to negative in folds 4–6, and
its input weights flip sign between folds.

Unused options are in `docs/sweden/possibilidades_v5.md`.

v4 traded insider purchases, disclosed short positions, announcement tone and
earnings reactions, from the Finansinspektionen registers and Nasdaq Nordic
announcements scored by FinBERT. Its mean abnormal return per event stayed
between −1.2% and +0.05%.

v2 is the reference strategy: a monthly
rotation into the six Large Cap names with the best 12-1 month momentum, a
20-name buffer and a 20% volatility target, on dividend-adjusted prices
(`prices_se.py`). Costs model an ISK account (0.25% brokerage per order,
0.10% slippage per side, no per-trade tax). Both v1 (the 38-gene GA swing
engine, `ga_run*.py`) and v2 failed the return criterion out of sample; the
results and the reasons are in `docs/sweden/plano.md` and
`docs/sweden/diagnostico.md`.

Code comments, log lines and user-facing strings are predominantly Portuguese.
Match the surrounding language when editing a file.

## Repo guides

### Evaluation protocol

All Swedish history from 2010 to 2026-10-30 has now been used to develop or
judge v1–v3 (ADR 0006, 0007). The insider, short-position and announcement
data from 2016-07 to 2026-10-30 have also been joined to Large Cap returns,
by v4 (ADR 0008), and so have the Börsvärlden analyst revisions from 2020 to
2026-10-30, by v5 (ADR 0009). Changing a strategy and re-running it on that history until
it passes manufactures a result. A new idea needs data that has not been used
yet (paper trading, future data, another universe) or information prices do
not already contain, with its spec and tests frozen before the first run. Run
`events_se.py dry-run`, which gives counts only, to find data bugs before
the frozen run.
Read before proposing any strategy change.

### Swedish plan and guide

Decisions, success criteria, ramp-up and phase status: `docs/sweden/plano.md`.
How the engine decides, commands, and how to read the replays:
`docs/sweden/guia.md`. Read both before changing the engine, the universe,
costs, the walk-forward or the daily run.

### Pipeline and commands

Entry points, env knobs, CI topology, and which artifacts are generated versus
committed, written for the Brazilian pipeline on `main`; the Swedish entry
points — v7: `seasonal_se.py` (dry-run, evaluate; Riksbank rate cached under `output/`);
v6: `swing_se.py` (download highs/lows, smoke, evaluate, report);
v5: `analyst_se.py` (Börsvärlden, rate-limited: cached, 2 workers),
`analyst_parse.py`, `revisions_se.py`; v4: `altdata_se.py` (FI registers, Nasdaq announcements, FinBERT
tone), `events_se.py`; v3: `selective_se.py`; v2: `prices_se.py`, `rotation_se.py`; v1: `walkforward.py`,
`replay_se.py`, `portfolio_se.py`, `daily_se.py`, `paper_se.py` — are in the
guide: `docs/agents/pipeline.md`.

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
- v2 decisions come only from `rotation_se.py`; the frozen spec is
  `data/strategy_se_v2.json`. `tests/test_rotation_se.py` checks that no
  decision reads a future price; keep it green.
- v1 (historical) has one engine: `ga_run_modular_final._backtest_stats_global_intraday`.
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
- Alternative data is usable only from the first open after it became
  public (`altdata_se.next_session_after`). Short positions dated T are used
  from T+2. Issuer→ticker mapping needs two of ISIN, LEI and issuer name to
  agree, because the FI insider register has mistyped LEIs.
  `tests/test_events_se.py` checks this; keep it green.
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
