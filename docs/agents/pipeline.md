# Pipeline and commands

Two loops run over the same code. Know which one you are in before changing
anything: the daily loop ships signals to a human, the improvement loop mutates
the model.

## Loop 1 — daily signals

```bash
python predict_daily.py     # ETL + GA load, ~30-40 min
python send_telegram.py     # rich caption, dedup-guarded
```

`predict_daily.py` runs, in order: `stock_etl.run_etl` (yfinance download +
technical features), GA in load mode (`GA_RUN_MODE=load`, scores the checkpoint
genome and writes `summary_latest.xlsx`), `refresh_close.refresh` (patches
`Apply.close`/`Date` from live yfinance quotes), `guide_overlay.apply_overlay`
(no-op unless a PDF sits in `guides/`), and `analysis.log_predictions.append_log`
(appends to `analysis/predictions_log.jsonl` to build future ground truth).
The last three are non-fatal: they warn and continue.

`python predict_daily.py --full` adds BIN, REG and FINAL (~3.5 h). Those stages
produce model outputs (`p_*`, `pred_*`, `EV_*`, `up20_bin`, ...) which the
causal allowlist blocks from the GA feature pool, so the daily loop skips them.
They are research-only.

CI: `.github/workflows/daily_pipeline.yml`, cron `30 21 * * *` (18:30 BRT),
every day. It sets `GA_DISABLE_TELEGRAM_SEND=1` for `predict_daily.py` so only
`send_telegram.py` sends, runs `refresh_close.py` again as a safety net, runs
`analysis/holdout_operability_diagnostic.py` over a 90-day holdout, and uploads
the xlsx, the CSV, the checkpoint and the diagnostics as artifacts (30 days).

## Loop 2 — continuous improvement

Local, one cycle at a time:

```bash
python run_continuous_improvement.py --status          # which categories are stale
python run_continuous_improvement.py                   # least-recently-swept category
python run_continuous_improvement.py --category D      # a specific category
python run_continuous_improvement.py --max-cycles 6    # a full A-F turn, 3-6 h
python run_continuous_improvement.py --until 06:00     # until a wall-clock time
python analysis/sweep_learnings.py                     # regenerate the aggregator
```

A cycle picks a category, sweeps its genes with
`run_local_fullmetric_sweep.py`, and if a candidate clears the gates promotes
it with `apply_fullmetric_sweep_best.py`, then appends an entry — success or
failure — to `analysis/optimization_log.md`. The failures are the point: they
are what builds the graveyard.

In CI the same loop is parallelised across three workflows:

| Workflow | Trigger | Does |
|---|---|---|
| `build_store.yml` | cron `*/15 * * * *`, concurrency group `build-store` | ETL + GA load, uploads `output/ga_memmap/` and the parquets as the `payload-store` artifact (1-day retention) |
| `sweep_matrix.yml` | `workflow_run` on **Build PayloadStore** completing | 6 parallel jobs, one per category A-F, `--combo-budget 24 --no-codex-sync`; then a `consolidate` job re-applies candidates, regenerates the aggregator reports, and makes one commit `[sweep-matrix] <ts> applied=N rejected=M` |
| `gene_monitor.yml` | cron `7 */2 * * *` plus after `sweep_matrix` | flags genes with zero variance across recent sweeps; after 3 consecutive cycles it quarantines one in `analysis/gene_quarantine.json`, appends a warning to the optimization log, and opens a GitHub issue. That file does not exist, so nothing has been quarantined yet |

`G_exit_quality` exists in `GENE_CATEGORIES` but the matrix only covers A-F, so
its five genes (`min_hold_bars`, `early_exit_block_bars`,
`profit_take_min_bars`, `early_take_quality_gate`,
`early_take_score_decay_max`) are never swept in CI. The committed checkpoint
is 33 genes long, so those five run at `_LEGACY_GENE_DEFAULTS`.

## Heavy retrain

```bash
python ga_run.py                     # train from scratch, 4-8 h
python run_local_ga_staged.py        # seeded staged retrain (chunks x gens x pop)
python retrain_with_wr_target.py     # 3 chunks x 3 gens x 24 pop, 1-2 h
```

CI: `retrain_ga.yml` and `retrain_ga_only.yml` (manual). Both back up the
checkpoint first, then run `analysis/overfitting_stage.py`, which exits 0 CLEAN
/ 1 MARGINAL / 2 OVERFIT and gates the commit.

## Tests

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py' -v
```

`.github/workflows/test.yml` runs exactly that on every push and PR, on Python
3.11, and optionally fetches `history_consolidated.csv` from the
`HISTORY_CSV_URL` secret. Install the whole `requirements.txt`: the test
modules import production code, which transitively needs numba, catboost and
the rest.

21 test modules under `tests/`, the slowest in `tests/perf/`. Several skip
themselves when their data file is absent, so a green run on a bare checkout
covers less than it looks.

## Environment

CI pins Python 3.11; the README says 3.10+. A bare cloud container here has
3.13 with only pandas, numpy, openpyxl and xlsxwriter present — install
`requirements.txt` before running or testing anything, and expect numba and
catboost wheels to be the slow part.

Variables that change behaviour:

| Variable | Effect |
|---|---|
| `GA_RUN_MODE=load` | score the checkpoint genome and emit signals, no evolution |
| `GA_FAST_MODE=true` | daily-job speed profile |
| `GA_KEEP_STORE=1` | keep `output/ga_memmap/` for reuse by later jobs |
| `GA_MEMMAP_DIR` | relocate the PayloadStore |
| `GA_DISABLE_TELEGRAM_SEND=1` | suppress `ga_run`'s own Telegram send |
| `GA_FORCE_TELEGRAM_SEND=1` | bypass the `.telegram_last_sent.json` dedup guard |
| `GA_TELEGRAM_PREFIX` | prepend a tag to the caption |
| `GA_ALPHA_FOCUS=on\|off` | weight the sweep toward alpha repair |
| `GA_DRIFT_GUARD=1` | enable the B4 directional-drift block (CI sets it) |
| `GA_DRIFT_MIN_CONSECUTIVE` | moves in one direction before drift trips (default 3) |
| `GA_ALPHA_FLOOR_TOLERANCE_PP` | slack on the B2 alpha floor; CI sets `0.0` |
| `GA_EVAL_WORKERS`, `GA_WORKERS_OVERRIDE` | evaluation parallelism |
| `GA_TIME_BUDGET_HOURS`, `GA_STAGE1_*`, `GA_STAGE2_*` | retrain budget and stage shape |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | required by `send_telegram.py` |
| `HISTORY_CSV_URL` | optional test-data source in CI |

`ga_run.py` reads ~40 `GA_*` variables; grep it when you need one that is not
listed here.

## Artifacts

Committed, and therefore the state an agent inherits:

- `global_ga_checkpoint.json` — the live genome, its fitness, and the gate
  verdicts that admitted it
- `continuous_improvement_state.json` — category rotation timestamps
- `local_fullmetric_sweep_continuous_*.json` — every sweep ever run (1918 of
  them in the repo root)
- `analysis/optimization_log.md`, `analysis/sweep_learnings_summary.md`,
  `analysis/overfitting_diagnostic_report.md`, `analysis/best_ever_tracker.json`
  — the regenerated reports
- `.local_ga_backups/` and `output/data/models/` — tracked despite matching
  `.gitignore`; see `docs/agents/gotchas.md`

Generated and ignored: `output/` (parquets, memmap store), `*.parquet`,
`summary_latest.xlsx`, `*.log`, `catboost_info/`.
