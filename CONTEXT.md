# CONTEXT

Domain glossary for this repo. Use these terms as spelled here — in issue
titles, test names, log lines, and commit messages — so the vocabulary stays
one language across code and docs.

Column names, metric keys and tier names are literal identifiers: they appear
exactly like this in the code, the sweep JSONs, and the `Apply` sheet.

## Instrument universe

- **ticker** — a B3 symbol in yfinance spelling (`PETR4.SA`, `HGRE11.SA`).
  `ticker_metadata.py` maps 119 tickers to 11 sectors; recent runs report
  `n_tickers_total` 116 and `n_tickers_active` ~105, because a ticker needs
  enough history and trades to be evaluated.
- **universe tier** — a per-ticker live-trading classification recomputed from
  current backtest stats by `classify_operational_universe_tier` in
  `ga_run.py`:
  - **PREMIUM** — `n_trades >= 30`, WR >= 0.74, WR alvo >= 0.72,
    `alpha_ann >= -0.12`, MDD <= 0.13
  - **HIGH_QUAL** — `n_trades >= 20`, WR >= 0.67, WR alvo >= 0.65,
    `alpha_ann >= -0.20`, MDD <= 0.16
  - **REGULAR** — everything else; its signals are downgraded to HOLD/OBSERVAR
- **operable** — PREMIUM or HIGH_QUAL (`is_operational_stat`). Metric keys
  suffixed `_operable` aggregate only these tickers and are the ones the
  promotion gates read; `_all` variants cover every active ticker.

## Policy and search

- **genome** — the ordered list of floats encoding one trading policy.
  `GLOBAL_PARAM_SPECS` in `ga_run.py` defines 38 genes as
  `(name, lo, hi, step, is_int)`. `sanitize_global_genome` clips to range,
  snaps to step, and pads a short genome from `_LEGACY_GENE_DEFAULTS`, so an
  older 33-gene checkpoint still loads.
- **gene** — one named parameter of the genome (`regime_threshold`,
  `consecutive_loss_cooldown`, `partial_take_level`, ...).
- **gene category** — the A-G grouping in `run_continuous_improvement.py`
  (`GENE_CATEGORIES`) that one sweep cycle explores: `A_risk_management`,
  `B_take_profit`, `C_timing`, `D_entry_filter`, `E_regime_vol`, `F_trailing`,
  `G_exit_quality`.
- **sweep** — one neighbourhood search over a category: each focus gene moved
  +/- one step (`single_results`), plus multi-gene combinations up to
  `combo_budget` (`combo_results`). Output is one
  `local_fullmetric_sweep_continuous_<CATEGORY>_<timestamp>.json`.
- **promotion** — a candidate passing every gate and being written into
  `global_ga_checkpoint.json`. See `docs/agents/promotion-gates.md`.
- **graveyard** — gene moves attempted repeatedly with zero promotions. The
  table in `analysis/sweep_learnings_summary.md` is the live list; skip those
  genes unless the surrounding context changed.
- **hot zone** — a gene that has promoted at least once, worth revisiting with
  new neighbours. Same file.
- **drift** — a gene promoted repeatedly in one direction (e.g. three
  consecutive increases). `_detect_parameter_drift` flags it and, with
  `GA_DRIFT_GUARD=1`, blocks the promotion.

## Comparison points

A sweep and an apply both compare four genomes. Keep the names distinct:

- **baseline** — the genome on `git:main`, re-evaluated on current data
  (`base_metrics`, `baseline_info.ref`).
- **seed** / **incumbent** — the genome currently in
  `global_ga_checkpoint.json` (`seed_metrics`).
- **candidate** — a genome produced by moving one or more genes from the seed.
- **best_ever** — the best metrics ever recorded, in
  `analysis/best_ever_tracker.json`, used as a long-term regression floor.

## Metrics

- **fit** / **fitness** — the GA objective, a composite of WR, WR alvo, alpha,
  MDD, trade count and consistency. Comparable only between genomes scored on
  the same dataset in the same run.
- **WR** (`wr_med_all`, `wr_med_operable`) — median per-ticker share of
  profitable trades, trailing-stop exits included.
- **WR alvo** (`wr_target_med_all`, `wr_target_med_operable`) — median share of
  trades that reached the take-profit target. Stricter than WR.
- **alpha_ann** (`mean_alpha_ann`, `alpha_ann_mean_operable`) — annualised
  excess return over buy-and-hold for the same ticker. Negative across this
  repo's recorded history.
- **alpha_ann_pos_rate** — share of tickers with positive alpha.
- **MDD** (`mdd_med`, `mdd_p75`, `mdd_duration_med`) — max drawdown, as a
  positive fraction.
- **trades_med** — median trades per ticker. Small values (~35) make WR
  medians coarse; see `docs/agents/promotion-gates.md`.
- **expectancy_net** — mean net result per trade, after modeled costs.
- **n_ge_70** — count of tickers with WR >= 70%.
- **expected_hold_score** — 0-100 score for whether a ticker's edge survives a
  3-day-plus hold (`expected_hold_score_from_stat`).

## Signals

- **Apply** — the sheet in `summary_latest.xlsx` carrying one row per ticker
  per date, and the `apply_last_5d__H5.csv` export.
- **signal** — final `buy` / `hold` / `sell` after overrides and quality gates;
  **signal_ga** is the raw GA output before them.
- **Minervini Stage** — trend classification from price versus MA50/MA200,
  driving the `recomendacao` text and the hold-to-buy promotion.
- **pivot levels** — Woodie pivot, S1 (support) and R1 (resistance), from which
  `entrada` (entry), `stop` and `alvo` (target) are derived.
- **confidence** — 0-100 blend of 11 factors; **rank** — weighted score used to
  order the daily BUY list.
- **causal feature** — an input derived only from OHLCV or the ETL, never from
  a model output or a retroactively applied fundamental snapshot.

## Infrastructure

- **PayloadStore** — per-ticker arrays in a memmap under `output/ga_memmap/`
  (`payload_store.py`), built once and reused so parallel sweeps skip the ETL.
  `meta.json` present means the store is ready.
- **load mode** — `GA_RUN_MODE=load`: score the checkpoint genome and emit
  signals, no evolution. The daily job and the store build both use it.
