# test — B3 trading-signals pipeline

Python pipeline that produces daily buy/hold/sell signals for ~116 Brazilian
tickers (equities, FIIs, crypto). A genetic algorithm tunes a 38-gene trading
policy against walk-forward backtests with modeled costs (brokerage, B3 fees,
slippage, 15% IR). GitHub Actions runs two loops: the daily signal job and a
continuous gene-tuning loop.

Code comments, log lines and user-facing strings are predominantly Portuguese.
Match the surrounding language when editing a file.

## Repo guides

### Pipeline and commands

Entry points, env knobs, CI topology, and which artifacts are generated versus
committed: `docs/agents/pipeline.md`. Read before running, scheduling, or
changing any pipeline step or workflow.

### Promotion gates

The four-layer gate stack that decides whether a gene change reaches
`global_ga_checkpoint.json`, its exact thresholds, and the measured reason the
loop has promoted nothing since 2026-04-22: `docs/agents/promotion-gates.md`.
Read before touching the GA, a sweep, a gate, or the checkpoint.

### Gotchas

Traps measured in this repo: stale README numbers, shallow clone, tracked
files under ignored paths, sweep-JSON accumulation, degenerate holdout:
`docs/agents/gotchas.md`. Read when a number, a path, or git history surprises
you.

## Invariants

- Feed the GA only OHLCV-derived and ETL features. `ga_run.py` holds the causal
  allowlist (blocked prefixes `p_`, `pred_`, `EV_`, `fund_`, plus named model
  outputs and snapshot fundamentals) and caps the pool at 80 features by
  variance. `analysis/leakage_audit.py` audits a built history for survivors.
- Treat `global_ga_checkpoint.json` as the single persisted model artifact.
  Change it through `apply_fullmetric_sweep_best.py` or
  `run_local_ga_staged.py`, which write a timestamped `.bak` and record the
  gate verdicts inside the checkpoint.
- Judge a candidate on `*_operable` metrics (the PREMIUM + HIGH_QUAL tiers the
  system would actually trade). That is what the gates read; `*_all` metrics
  include tickers it would not trade.
- State the data vintage whenever you quote a fitness number. The value stored
  in the checkpoint (26.46, April 2026) and the value the same genome scores on
  current data (~5-7) measure different datasets.
- Keep `output/` out of new commits; the ETL regenerates it and the workflows
  fail the job when those paths are staged.

## Agent skills

### Issue tracker

Issues live as GitHub issues in `gcallj/test`, driven by the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, using the default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` plus `docs/adr/` at the repo root. See `docs/agents/domain.md`.
