# Promotion gates

A gene change reaches `global_ga_checkpoint.json` only by passing four
independent layers. `apply_fullmetric_sweep_best.py` runs them in this order
and exits non-zero at the first failure, which the sweep workflow records as
`rejected`.

Read this before relaxing a threshold. Each layer was added to stop a specific
failure mode, and the current stall (below) comes from the interaction between
layers, not from any one being too strict.

## Layer 1 — staged guardrails

`run_local_ga_staged._candidate_beats_incumbent` promotes only when **all** of
these hold:

- `acceptance_vs_main.all_pass` — `_compute_acceptance`, candidate versus the
  `git:main` baseline re-evaluated on current data
- `incumbent_wr_guardrail.all_pass` — candidate versus the incumbent's WR
- `incumbent_safety_guardrail.all_pass` — MDD and exposure versus the incumbent
- `priority_better` — `_is_better_by_priority`, a lexicographic comparison
- no directional drift, when `GA_DRIFT_GUARD=1`

The acceptance checks that matter most:

| Check | Requirement |
|---|---|
| `wr_med_all_up` | candidate WR **strictly greater** than baseline WR |
| `wr_target_med_all_up` | candidate WR alvo **strictly greater** than baseline WR alvo |
| `wr_med_all_floor` | WR >= `max(0.60, base_wr - 0.0025)` |
| `wr_target_med_all_floor` | WR alvo >= `max(0.55, base_wr_target - 0.005)` |
| `mdd_med_not_worse` | MDD <= `base_mdd + max(0.04, 0.20 * base_mdd)` plus up to 1.2pp of relaxation earned by alpha/WR gains |
| alpha positivity | `alpha_ann_pos_rate` >= `max(base + 0.008, 0.17)` |
| trades floor | >= 70% of baseline `trades_med` |

`_is_better_by_priority` compares with tolerance bands, in order: swing-hold
profile, WR (0.25pp), WR alvo (0.5pp), alpha (0.10pp), MDD (0.25pp), then
fitness. A tie inside a band falls through to the next key; fitness alone never
wins.

## Layer 2 — long-term regression guard

`analysis/best_ever_guard.check_regression_vs_best` compares the candidate
against `analysis/best_ever_tracker.json`, rejecting if any of:

- `fit < best_fit * 0.96` (`FIT_TOL_RELATIVE = 0.04`)
- `wr_med_all < best_wr - 0.015`
- `mean_alpha_ann < best_alpha - 0.010`
- `n_ge_70 < best_n_ge_70 - 5`

It exists to stop death by a thousand cuts: every promotion passing the gate
versus the *current* main while main degrades step by step. A crash inside this
layer is non-fatal and logged as a warning.

## Layer 3 — fail-closed overfit guard

`analysis/overfitting_stage` evaluates a 90-day pristine holdout for both
candidate and incumbent, 4-fold rolling OOS with a 5-day embargo, and a
200-sample bootstrap WR CI over operable tickers only. Promotion is blocked
when the verdict is OVERFIT or any of:

| Threshold | Limit |
|---|---|
| `operable_oos_wr_drop_critical` | candidate OOS WR more than 3pp below incumbent |
| `operable_oos_alpha_drop_critical` | candidate OOS alpha more than 0.5pp below incumbent |
| `operable_min_tickers` | fewer than 20 tickers in the bootstrap |
| `operable_wr_wilson_lo_min` | Wilson lower bound below 55% |
| `operable_bootstrap_ci_width_max` | bootstrap WR CI wider than 15pp |

This layer is fail-closed: any exception raises `SystemExit`, so an
infrastructure problem reads as a rejection, not a pass.

## Layer 4 — write with provenance

On success `staged._write_checkpoint_with_metadata` writes the new checkpoint
plus a timestamped `global_ga_checkpoint_*.json.bak`, embedding
`acceptance_vs_main`, `incumbent_wr_guardrail`, `incumbent_safety_guardrail`
and `accepted_reason`. Then `update_best_ever` advances the tracker. Read those
fields to learn why the live genome was admitted.

## The stall

The checkpoint's `accepted_on` is **2026-04-22**. Every `[sweep-matrix]` commit
since reports `applied=0`. Measured on 2026-10-30 over the 24 most recent sweep
JSONs (556 candidates):

- **210** candidates beat the baseline on `fit`
- **8** clear both strict WR requirements
- **0** became `best_promoted`

Two independent blockers, both verifiable from committed data:

**1. The strict WR requirement meets a quantised median.** `trades_med` is ~35,
so per-ticker win rates land on coarse fractions and the median across tickers
collapses onto them. 316 of 556 candidates scored `wr_med_all` of exactly
0.6667 — the same value as the baseline. `wr_med_all_up` demands strictly
greater, so a tie is a rejection, and ties are the modal outcome.

**2. `best_ever_tracker.json` is frozen at an April vintage.** It holds
`best_fit = 26.46`, recorded 2026-04-22. The same genome re-evaluated on
current data scores `fit` 5.4-6.8. Layer 2 requires `fit >= 25.40`, so every
current candidate fails it even if Layer 1 passes. Fitness is not comparable
across data vintages, and this tracker compares across them.

Re-verify before acting on any of this:

```bash
python - <<'PY'
import json, glob
files = sorted(glob.glob('local_fullmetric_sweep_continuous_*.json'))[-24:]
tot = up = fitup = 0
for f in files:
    d = json.load(open(f)); b = d.get('base_metrics') or {}
    for c in (d.get('single_results') or []) + (d.get('combo_results') or []):
        m = c.get('metrics') or {}; tot += 1
        if m.get('fit', -9e9) > b.get('fit', 9e9): fitup += 1
        if (m.get('wr_med_all', 0) > b.get('wr_med_all', 1)
                and m.get('wr_target_med_all', 0) > b.get('wr_target_med_all', 1)): up += 1
    print(f[-44:], 'base_fit=%.3f' % b.get('fit', 0), 'promoted' if d.get('best_promoted') else '-')
print(f'candidates={tot} fit_up={fitup} both_wr_strictly_up={up}')
PY
```

## What the repo's own diagnostics say

`analysis/overfitting_diagnostic_report.md` returns **OVERFIT PROVAVEL**:
`alpha_ann` is -7.9% (the policy underperforms buy-and-hold annualised),
`alpha_ann_pos_rate` is 0.20, the Wilson 95% CI on WR is [52.7%, 82.0%] at
n=35 trades, the signal-to-noise ratio of per-sweep fitness gains is 0.01, and
three genes show one-directional drift (`consecutive_loss_cooldown` up,
`momentum_confirm_days` down, `partial_take_pct` down).

`analysis/pristine_holdout_report.md` returns OVERFIT MARGINAL, but its holdout
slice produced **0 trades** for both genomes (WR 0.00%, alpha -49.4%), so that
comparison carries no information. Treat a 0-trade holdout as a broken
evaluation window, not as a result — and note Layer 3 depends on the same
holdout machinery.

Raising promotion rate by loosening a gate therefore trades against a model the
repo already diagnoses as overfit. Prefer fixing the measurement: a trade
count per ticker that makes WR medians move, a `best_ever` baseline re-scored
on the current dataset, and a holdout window that produces trades.
