# Gotchas

Traps measured in this repo on 2026-10-30. Each one cost real time to find;
each is checkable with the command beside it.

## README numbers are an April snapshot

`README.md` is stamped "Ultima atualizacao: 10/Abr/2026" and its headline
metrics are older than the live checkpoint. Quote the checkpoint, not the
README:

| Claim in README | Value in the repo |
|---|---|
| Fitness 18.45 | `global_ga_checkpoint.json` fitness 26.46 (April vintage; ~5-7 on current data) |
| Win Rate 64.2% | checkpoint `wr_med_all` 69.2% |
| Trades/ticker 61 | checkpoint `trades_med` 35 |
| 119 tickers | recent runs report `n_tickers_total` 116, `n_tickers_active` ~105 |
| GA "30-param" | `GLOBAL_PARAM_SPECS` has 38 genes; the checkpoint genome is 33 long |
| PREMIUM = "alpha > +2pp" | `classify_operational_universe_tier` uses trades >= 30, WR >= 0.74, WR alvo >= 0.72, alpha >= -0.12, MDD <= 0.13 |
| `.github/workflows/weekly_sweep.yml` | absent; the sweep loop is `build_store.yml` + `sweep_matrix.yml` |

```bash
python -c "import json;d=json.load(open('global_ga_checkpoint.json'));print(d['fitness'],len(d['genome']),d['accepted_on'])"
```

`predict_daily.py`'s docstring says "48 allowed, 48 model outputs blocked";
`ga_run.py` caps the pool at `MAX_CAUSAL_FEATURES = 80`.

## The checkout is shallow

Clones here come in at depth 50, so `git log` shows ~50 commits — almost all
`[sweep-matrix]` bot commits — and the project looks two months old. It is not.

```bash
git rev-parse --is-shallow-repository      # true
git fetch --unshallow                       # when you genuinely need history
```

Blame, bisect and "when did this change" all mislead until you unshallow.

## Files are tracked under ignored paths

`.gitignore` lists `output/` and `.local_ga_backups/`, but 32 files under
`output/data/models/` and 22 under `.local_ga_backups/` were committed before
the rules landed. Ignore rules do not untrack. Several are large (two LightGBM
text models exceed 18 MB each), which is most of the 149 MB `.git`.

```bash
git ls-files output/ .local_ga_backups/ | wc -l
```

Consequence: `git add -A` can restage them. `sweep_matrix.yml` guards against
it by failing the job when `output/` or `__pycache__/` appear staged.

## Sweep JSONs accumulate, and consolidate re-reads all of them

The repo root holds **1918** `local_fullmetric_sweep_continuous_*.json` files,
~320 per category, each 3k-7k lines, all committed. One matrix cycle adds six
and ~26k lines.

The `consolidate` job intends to apply only the cycle's new sweeps — its
comment says so — but the staging step globs
`local_fullmetric_sweep_continuous_${category}_*.json` from a fresh checkout of
`main`, which already contains every historical file for that category. So each
artifact carries ~320 files, and consolidate loops over all of them. That is
why the commit messages read `rejected=1918`: the number equals the file count,
not six.

```bash
ls local_fullmetric_sweep_continuous_*.json | wc -l      # 1918
git log -1 --format=%s HEAD~1                             # ... rejected=1918
```

Most of those rejections are instant — `apply_fullmetric_sweep_best.py` exits
on "no `best_promoted` candidate" before evaluating anything — so the job still
fits its 60-minute budget, and `applied=0` is not caused by this. But the
`rejected` counter means nothing as a signal, and the glob is the thing to fix
if you want it to.

## Fitness is not comparable across data vintages

A sweep JSON's `baseline_info.fitness` and `seed_fit` echo the number stored in
the checkpoint (26.46). Its `base_metrics.fit` and `seed_metrics.fit` are the
*same genome re-scored on today's data* (5.4-6.8). Both appear in one file.
Compare only within a run, and always say which vintage a number came from.

## The holdout window produces zero trades

`analysis/pristine_holdout_report.md` reports 0 trades, WR 0.00% and alpha
-49.4% for both baseline and incumbent across the 90-day holdout. A 0-trade
window is a broken evaluation, not a bad result, and the fail-closed overfit
guard in `apply_fullmetric_sweep_best.py` leans on the same machinery. Check
`n_tickers_active` and the trade count before believing any holdout verdict.

The mechanism is in `evaluate_genome_full` (`run_local_ga_staged.py`): every
median is `np.median(xs) if xs else 0`, over tickers that traded in the
window. An empty list scores WR 0, which the overfit stage reads as a
catastrophic drop. A genome that simply trades rarely in a 90-day slice looks
identical to one that lost every trade.

## Five genes are never exercised

`GENE_CATEGORIES` defines seven categories; `sweep_matrix.yml` sweeps A-F. The
`G_exit_quality` genes (`min_hold_bars`, `early_exit_block_bars`,
`profit_take_min_bars`, `early_take_quality_gate`,
`early_take_score_decay_max`) are never tuned in CI, and because the committed
genome is 33 long while the specs are 38, `sanitize_global_genome` supplies
them from `_LEGACY_GENE_DEFAULTS` on every load. The v10 swing-hold controls
are effectively off, which fits the measured `avg_hold_bars_med` of ~2 bars
against a 3-10 day swing target.

## Rotation state is stale while CI keeps sweeping

`continuous_improvement_state.json` and the rotation table in
`analysis/sweep_learnings_summary.md` still show April timestamps, so
`--status` and the "next suggested sweep" line point at a cycle count (34) far
below the ~320 cycles the matrix has actually run. Treat local rotation state
as advisory.

## Telegram sends are deduplicated

`.telegram_last_sent.json` suppresses a second send on the same day. Set
`GA_FORCE_TELEGRAM_SEND=1` to override, and keep `GA_DISABLE_TELEGRAM_SEND=1`
on anything that is not the single daily send — `ga_run.py` sends on its own
otherwise, and the daily workflow relies on the richer caption from
`send_telegram.py`.

## CI cadence is deliberately aggressive

`build_store.yml` runs every 15 minutes with `cancel-in-progress: false`, and
each run takes ~35 minutes; the comment explains this is throughput tuning for
a public repo with unlimited Actions minutes, not an accident. `sweep_matrix`
chains off it and `gene_monitor` runs every 2 hours. Expect the default branch
to move under you, and rebase rather than force-push: the consolidate job
pushes to `main` with three rebase retries.

## Dependencies are heavy and not preinstalled

A bare container here runs Python 3.13 with pandas, numpy, openpyxl and
xlsxwriter only. CI pins 3.11. `pip install -r requirements.txt` pulls numba,
catboost, lightgbm, xgboost and optuna, which is the slow part of any setup.
The test modules import production code, so a partial install fails at import
rather than at assert.
