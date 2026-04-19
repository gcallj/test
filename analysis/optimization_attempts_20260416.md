# Optimization Attempts Through 2026-04-18

This note documents the optimizer experiments added after the GA-first metric refactor, with the current acceptance logic aimed at swing trading:

- prioritize `WR_med_all` and `WR_target_med_all`
- keep explicit risk guardrails on drawdown magnitude and duration
- improve `mean_alpha_ann` only when it does not materially weaken operational safety
- compare every promotion against `git:main`

## Important note on `fit`

The top-level `fitness` field stored in [`global_ga_checkpoint.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/global_ga_checkpoint.json) can differ from the re-evaluated `fit` in the experiment result JSON files because the fitness function evolved during this cycle. For comparisons in this document, the authoritative values are the re-evaluated metrics saved in each result artifact.

## Best automated result so far

Best result produced by the automation stack as of `2026-04-18`:

- source: `Attempt 10: Full-metric sweep (signal precision + vol regime)`
- promoted change: `vol_regime_mode: 1 -> 2`
- acceptance reason: the candidate improved both hit-rate priorities and reduced drawdown while passing all incumbent and `git:main` guardrails
- authoritative checkpoint: [`global_ga_checkpoint.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/global_ga_checkpoint.json)

Current best automated checkpoint vs `main`:

| Metric | `main` baseline | Current incumbent | Delta |
| --- | ---: | ---: | ---: |
| `fit` | `-31.7891` | `19.3981` | `+51.1872` |
| `WR_med_all` | `33.33%` | `68.97%` | `+35.63pp` |
| `WR_target_med_all` | `16.67%` | `69.09%` | `+52.42pp` |
| `mean_alpha_ann` | `-9.57%` | `-8.24%` | `+1.32pp` |
| `alpha_ann_pos_rate` | `15.13%` | `19.33%` | `+4.20pp` |
| `MDD_med` | `8.44%` | `11.75%` | `+3.31pp` |
| `MDD_p75` | `12.46%` | `16.77%` | `+4.31pp` |
| `MDD_duration_med` | `463.0` | `263.0` | `-200.0` |
| `trades_med` | `11.0` | `33.0` | `+22.0` |
| `# >= 70% WR` | `28` | `56` | `+28` |

Reading this in trading terms:

- the automation found a checkpoint that is materially stronger than `main` on hit rate, `WR_target`, and breadth of high-WR tickers
- alpha improved, but remains the main open problem because it is still negative overall
- risk magnitude is still above `main`, but the latest promoted automation result reduced `MDD_med` versus the prior incumbent and shortened drawdown duration materially

## Current incumbent vs prior incumbent

The current automation winner also improved on the immediately previous promoted checkpoint:

| Metric | Prior incumbent | Current incumbent | Delta |
| --- | ---: | ---: | ---: |
| `fit` | `8.7203` | `19.3981` | `+10.6778` |
| `WR_med_all` | `68.24%` | `68.97%` | `+0.73pp` |
| `WR_target_med_all` | `68.42%` | `69.09%` | `+0.67pp` |
| `mean_alpha_ann` | `-8.26%` | `-8.24%` | `+0.02pp` |
| `MDD_med` | `12.65%` | `11.75%` | `-0.90pp` |

Why this matters:

- it was not just a cosmetic promotion; the automation improved the two most important swing-trade hit-rate metrics while also lowering median drawdown
- this is the clearest example in the cycle of the automation learning from prior attempts: later full-metric sweeps targeted operational safety knobs rather than repeating optimizer-only searches

## Daily Refresh on 2026-04-17

The best model was re-run in daily fast mode with fresh ETL data through `2026-04-17`, and the operational spreadsheet was rebuilt and sent to Telegram.

Artifacts refreshed today:

- [`summary_latest.xlsx`](/C:/Users/gabri/.codex/worktrees/0b30/test/summary_latest.xlsx)
- [`apply_last_5d__H5.csv`](/C:/Users/gabri/.codex/worktrees/0b30/test/apply_last_5d__H5.csv)

Daily delivery status:

- spreadsheet refreshed with `Apply.Date = 2026-04-17`
- Telegram send completed successfully for `summary_latest.xlsx`

Live load-mode snapshot from the `2026-04-17` daily run:

| Metric | Value |
| --- | ---: |
| `GA fitness` | `12.9813` |
| `WR mean` | `63.0%` |
| `WR median` | `66.7%` |
| `median MDD` | `-11.8%` |
| `mean alpha_ann` | `-8.26pp` |
| `median alpha_ann` | `-7.74pp` |
| `mean trades/ticker` | `37.8` |
| `beat buy&hold rate` | `18.5%` |
| `test_cagr` | `+1.54% a.a.` |
| `buy_hold_cagr` | `+9.81% a.a.` |
| `BUY / HOLD / SELL` | `10 / 88 / 21` |

Quality buckets from the same run:

- High-Quality (`n=15`, `alpha>=0` and `WR>=65%`): `WR 70.3%`, `Ret +1.6% a.a.`, `Alpha +5.0pp`, `MDD 12.9%`
- Premium (`n=7`, `alpha>=+2pp` and `WR>=68%`): `WR 72.2%`, `Ret +3.4% a.a.`, `Alpha +5.5pp`, `MDD 13.6%`

Top BUY names in the `2026-04-17` sheet:

- `CSMG3.SA`
- `BBSE3.SA`
- `BPAC11.SA`
- `BRSR6.SA`
- `PSSA3.SA`
- `ABEV3.SA`
- `WEGE3.SA`
- `DIRR3.SA`

## Recovered Earlier Attempts (pre-formal log)

Before this Markdown log existed, several exploratory attempts were already saved as scripts and JSON artifacts under [`analysis`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis). They used older fitness formulations and looser acceptance discipline, so the absolute `fit` values below are not directly comparable with the current incumbent. They are still valuable because they show what was explored, what failed, and which ideas later fed into the automation flow.

Primary recovered artifacts:

- [`analysis/broad_sweep_results.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/broad_sweep_results.json)
- [`analysis/chunks_vs_base.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/chunks_vs_base.json)
- [`analysis/tune_rr_results.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/tune_rr_results.json)
- [`analysis/wr_target_retrain_results.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/wr_target_retrain_results.json)
- [`analysis/wr_target_fitness_sensitivity.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/wr_target_fitness_sensitivity.json)
- [`analysis/win_rate_deep_dive.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/win_rate_deep_dive.json)

### Earlier Attempt A: Broad parameter sweep

Recovered from:

- [`analysis/broad_param_sweep.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/broad_param_sweep.py)
- [`analysis/broad_sweep_results.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/broad_sweep_results.json)

What was tested:

- trailing-stop variants (`trail0`, `trail1`, `trail2`)
- time-stop variations (`timestop10`, `timestop15`, `timestop20`, `timestop25`)
- regime threshold changes (`0.25` through `0.65`)
- vote-threshold changes (`voteL0.2` through `voteL0.4`)
- entry confirmation variants (`conf1d`, `conf2d`, `conf3d`)
- several hand-built combinations of these knobs

Recovered result snapshot under the old objective:

| Label | `fit` | `wr_med` | `ret_med` | `mdd_med` | `trades_med` | `n_ge_70` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | `-175.3548` | `67.14%` | `0.5399` | `19.62%` | `147` | `29` |
| `voteL0.25` | `-174.4197` | `67.12%` | `0.5336` | `19.14%` | `145` | `31` |
| `trail1` | `-175.8196` | `67.31%` | `0.5408` | `19.62%` | `147` | `30` |
| `combo1` | `-180.4150` | `67.57%` | `0.4968` | `18.95%` | `149` | `34` |
| `combo4` | `-203.4743` | `54.21%` | `0.0919` | `20.42%` | `83` | `1` |

What was learned:

- small changes in `vote_threshold_long` had some promise even early, especially around `0.25`, but the gains were marginal under the old objective
- disabling/loosening some trailing-stop behavior often hurt hit rate more than it helped returns
- more aggressive combo edits quickly collapsed operational quality, especially `WR` breadth and trade count
- these early sweeps hinted that simple one-gene operational knobs could matter more than broad “optimizer cleverness”, which later showed up clearly in the full-metric sweeps

### Earlier Attempt B: chunk-vs-base checkpoint comparisons

Recovered from:

- [`analysis/compare_chunks_no_gate.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/compare_chunks_no_gate.py)
- [`analysis/chunks_vs_base.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/chunks_vs_base.json)

What was tested:

- direct comparison of early GA chunk outputs against the then-current base checkpoint, before the current acceptance gate structure was in place

Recovered result snapshot:

| Label | `fit` | `wr_med` | `ret_med` | `mdd_med` | `trades_med` | `n_ge_70` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | `-175.3548` | `67.14%` | `0.5399` | `19.62%` | `147` | `29` |
| `chunk_1` | `-211.0659` | `63.64%` | `0.0500` | `15.41%` | `44` | `24` |
| `chunk_2` | `-184.0639` | `67.12%` | `0.4065` | `20.02%` | `131` | `35` |

What was learned:

- early chunk candidates could look attractive on isolated dimensions like `MDD` or `n_ge_70`, but they often got there by sacrificing too much return quality or trade participation
- this was an early sign that “pick the visually nicer chunk” was unsafe; it motivated the later shared guardrail logic for incumbent safety

### Earlier Attempt C: reward/risk and partial-take sweeps

Recovered from:

- [`analysis/tune_rr_and_take.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/tune_rr_and_take.py)
- [`analysis/tune_rr_results.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/tune_rr_results.json)

What was tested:

- `reward_risk_ratio` values from `1.5` to `5.0`
- `partial_take_pct` and `partial_take_pct_2` combinations

Recovered result snapshot:

| Label | `fit` | `wr_med` | `ret_med` | `mdd_med` | `trades_med` | `n_ge_70` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | `12.8404` | `67.14%` | `0.5399` | `19.62%` | `147` | `29` |
| `rr1.5_pt0.1_pt20.2` | `10.5887` | `63.04%` | `0.3478` | `20.13%` | `127` | `10` |
| `rr3.0_pt0.1_pt20.2` | `9.9376` | `57.80%` | `0.5040` | `19.62%` | `110` | `5` |
| `rr5.0_pt0.0_pt20.0` | `9.2359` | `56.36%` | `0.4595` | `20.07%` | `92` | `3` |

What was learned:

- increasing `reward_risk_ratio` looked intuitively attractive but consistently damaged hit rate
- removing or shrinking partial exits did not rescue the trade-off enough
- this strongly reinforced the user’s preference for swing-trade safety with hit rate first, and later acceptance logic kept that priority explicit

### Earlier Attempt D: WR-target-focused retrain analysis

Recovered from:

- [`analysis/evaluate_wr_target_retrain.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/evaluate_wr_target_retrain.py)
- [`analysis/wr_target_retrain_results.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/wr_target_retrain_results.json)
- [`analysis/wr_target_fitness_sensitivity.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/wr_target_fitness_sensitivity.json)

What was tested:

- alternative chunk checkpoints chosen not by old global fitness, but by `WR_target`
- sensitivity of the fitness function to stronger `WR_target` bonuses

Recovered result snapshot:

| Label | `fit` | `wr_med` | `wr_target_med` | `mdd_med` | `trades_med` | `n_ge_60_target` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | `-173.8618` | `67.14%` | `45.50%` | `19.62%` | `147` | `27` |
| `chunk_2` | `-182.1705` | `65.08%` | `54.26%` | `20.26%` | `119` | `37` |
| `chunk_3` | `-200.4072` | `64.15%` | `61.11%` | `13.05%` | `61` | `65` |

Supporting sensitivity analysis:

- old `fit` -> new `fit` adjustment from `WR_target` bonus: about `+1.49`
- bonus at `50/50` (`mean/median WR_target`): `+3.3`
- bonus at `60/60`: `+11.1`
- bonus at `70/70`: `+26.9`

What was learned:

- it was possible to push `WR_target` much higher, but only by giving back too much `WR`, return quality, or trade breadth
- this is where the project first made the trade-off explicit: optimizing for target hits alone could damage the operational profile the user actually wanted
- later ranked acceptance logic (`WR_med_all -> WR_target_med_all -> alpha -> risk`) is effectively the production answer to this failure mode

### Earlier Attempt E: win-rate / exit-behavior audit

Recovered from:

- [`analysis/win_rate_deep_dive.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/win_rate_deep_dive.py)
- [`analysis/win_rate_deep_dive.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/analysis/win_rate_deep_dive.json)

Recovered checkpoint audit:

- `checkpoint_fit = 18.4532`
- `win_rate_median = 64.15%`
- `win_rate_target_median = 61.11%`
- `pct_exit_take_median = 61.11%`
- `pct_exit_stop_median = 38.64%`
- `pct_exit_time_median = 0.0%`

What was learned:

- by this point, exits were already overwhelmingly split between take-profit and stop-loss, with time-stop largely irrelevant in median behavior
- this made later focused sweeps over partial-take behavior, trailing-stop mode, and volatility-regime handling much more justified than generic broad exploration

## Attempt 1: Pattern Search

Artifact:

- [`run_local_pattern_search.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/run_local_pattern_search.py)
- [`local_pattern_search_result.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/local_pattern_search_result.json)

Outcome:

- the best local-search candidate did **not** beat the incumbent
- it kept `WR_med_all` flat at `66.67%`, but weakened `WR_target_med_all`
- it slightly worsened alpha and increased drawdown

Pattern-search comparison:

| Metric | Incumbent seed | Best candidate | Delta |
| --- | ---: | ---: | ---: |
| `fit` | `7.5986` | `5.6658` | `-1.9328` |
| `WR_med_all` | `66.67%` | `66.67%` | `0.00pp` |
| `WR_target_med_all` | `68.13%` | `66.67%` | `-1.47pp` |
| `mean_alpha_ann` | `-8.18%` | `-8.19%` | `-0.01pp` |
| `MDD_med` | `12.90%` | `13.50%` | `+0.60pp` |
| `MDD_duration_med` | `288.0` | `298.0` | `+10.0` |
| `trades_med` | `32.0` | `32.0` | `0.0` |

Decision:

- `promote = false`
- rejection was driven mainly by weaker `WR_target` and worse risk tail / duration guardrails

## Attempt 2: Optuna / TPE

Artifact:

- [`run_local_optuna_search.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/run_local_optuna_search.py)
- [`local_optuna_search_result.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/local_optuna_search_result.json)

Outcome:

- the TPE study improved the cheaper train objective
- once the best trials were validated with full staged metrics, the winner was still the incumbent itself

Optuna notes:

- `study_best_value = -150.3546` vs seed-train `-157.1813`
- validated candidate metrics matched the incumbent exactly
- no new parameter set cleared the incumbent priority gates

Decision:

- `promote = false`
- useful as proof that the train objective can improve without creating a better swing-trade operating point

## Attempt 3: Hybrid Mini-GA -> TPE

Artifacts:

- [`run_local_hybrid_search.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/run_local_hybrid_search.py)
- [`local_hybrid_search_result.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/local_hybrid_search_result.json)

Representative command:

```powershell
python run_local_hybrid_search.py --windows 4 --window-offset 23 --ga-pop 14 --ga-ngen 2 --ga-topk 5 --tpe-anchors 2 --tpe-trials 6 --tpe-radius-steps 4 --tpe-topk-validate 3 --seed 42
```

Outcome:

- the stronger hybrid also failed to beat the incumbent
- the best validated candidate remained the incumbent seed
- several non-seed candidates improved alpha or drawdown, but only by giving up too much hit rate

Hybrid result:

| Metric | Incumbent seed | Best candidate | Delta |
| --- | ---: | ---: | ---: |
| `fit` | `7.5647` | `7.5647` | `0.0000` |
| `WR_med_all` | `66.67%` | `66.67%` | `0.00pp` |
| `WR_target_med_all` | `68.18%` | `68.18%` | `0.00pp` |
| `mean_alpha_ann` | `-8.16%` | `-8.16%` | `0.00pp` |
| `MDD_med` | `12.76%` | `12.76%` | `0.00pp` |

Notable rejected candidates:

- `ga_2`: `fit=1.3019`, `WR_med_all=64.58%`, `WR_target_med_all=62.50%`, `mean_alpha_ann=-7.92%`, `MDD_med=12.48%`
- `ga_1_tpe_2`: `fit=1.0359`, `WR_med_all=56.41%`, `WR_target_med_all=66.67%`, `mean_alpha_ann=-7.45%`, `MDD_med=10.53%`

Decision:

- `promote = false`
- the hybrid was valuable because it showed the local frontier clearly: alpha can improve, but the current search space tends to pay for that with too much `WR` degradation

## Automated Attempts

The manual experiments above were not the only search path in this cycle. The main incumbent gains actually came from repeated automated staged runs and overnight refinement loops that reused the same swing-trade acceptance rules.

Tracked automation-related code:

- [`run_local_ga_staged.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/run_local_ga_staged.py)
- [`overnight_alpha_until_0600.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/overnight_alpha_until_0600.py)
- [`ensure_daily_telegram_file.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/ensure_daily_telegram_file.py)

Current tracked state artifacts:

- [`global_ga_checkpoint.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/global_ga_checkpoint.json)
- [`local_ga_best_so_far.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/local_ga_best_so_far.json)
- [`local_ga_chunk_progress.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/local_ga_chunk_progress.json)

### Promotion ladder from automated search

The automatic staged / overnight search promoted the checkpoint repeatedly. The saved promotion snapshots show this monotonic fitness climb:

| Promotion snapshot | `fitness` |
| --- | ---: |
| `2026-04-15 14:11` | `1.6316` |
| `2026-04-15 16:48` | `1.9077` |
| `2026-04-15 20:13` | `3.4112` |
| `2026-04-15 20:49` | `9.0748` |
| `2026-04-15 23:32` | `11.1951` |
| `2026-04-16 02:21` | `12.5071` |
| `2026-04-16 16:48` | `12.5656` |

This is the clearest evidence that the automated runner was the most effective optimizer in practice during this cycle.

### Key automated milestones

Pre-automation and early-automation snapshots preserved in the backup area show the progression of the incumbent:

| Snapshot | `fit` | `WR_med_all` |
| --- | ---: | ---: |
| automation start / pre-stage1 | `0.0773` | `64.96%` |
| continue pre-run | `6.0867` | `66.67%` |
| pre-alpha-on phase | `12.5071` | `66.67%` |
| current tracked incumbent | `12.5656` | `66.67%` |

Reading this operationally:

- the automated search delivered the big jump from roughly `fit ~= 0.08` to `fit ~= 12.57`
- hit rate improved early and then stayed stable while later rounds mainly refined alpha, risk shape, and guardrail compliance
- later manual alternative optimizers were tested against this much stronger incumbent and could not beat it

### What is committed vs left out

The repository commit includes the useful, replayable artifacts from automated search:

- the current checkpoint and staged progress files
- the automation scripts that produced and governed those runs
- this human-readable summary

The following are intentionally not committed:

- `.local_ga_backups/**`
- raw promotion backup pairs
- transient `.bak` snapshots
- large operational logs

Those files are useful locally for forensic replay, but they add a lot of noise to repository history. The committed checkpoint/progress JSON plus this summary preserve the meaningful automated-attempt record without carrying the entire scratch space.

## Operational Changes Added During These Attempts

These attempts were not only about alternative optimizers. The search/evaluation stack was also hardened:

- [`ga_run.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/ga_run.py) and [`ga_run_modular_final.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/ga_run_modular_final.py)
  - added swing-oriented metric pressure and acceptance support
  - exposed additional metrics such as `sortino`, `profit_factor`, `expectancy`, `payoff_ratio`, `max_drawdown_duration`, `cagr`, and hold-profile data
- [`run_local_ga_staged.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/run_local_ga_staged.py)
  - compares all promotions against `git:main`
  - applies explicit acceptance gates for swing safety, including drawdown duration and incumbent guardrails
- [`overnight_alpha_until_0600.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/overnight_alpha_until_0600.py)
  - continues staged improvement rounds automatically
  - now reuses the same promotion discipline as the staged runner
- [`ensure_daily_telegram_file.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/ensure_daily_telegram_file.py)
  - guarantees that, on update/promotion, Telegram receives the spreadsheet for the local trading day
- [`send_telegram.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/send_telegram.py)
  - exits non-zero when Telegram sending fails, so automation can detect delivery problems
- [`stock_etl.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/stock_etl.py)
  - pins yfinance cache inside the workspace and disables threaded downloads to avoid sqlite lock/cache failures on Windows
- [`tests/test_ga_metrics.py`](/C:/Users/gabri/.codex/worktrees/0b30/test/tests/test_ga_metrics.py)
  - extended to cover the new acceptance logic, optimizer result handling, and Telegram-daily-file checks

## Artifacts Committed for This Cycle

Code and result artifacts included with this documentation:

- [`local_pattern_search_result.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/local_pattern_search_result.json)
- [`local_optuna_search_result.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/local_optuna_search_result.json)
- [`local_hybrid_search_result.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/local_hybrid_search_result.json)
- current staged state:
  - [`global_ga_checkpoint.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/global_ga_checkpoint.json)
  - [`local_ga_best_so_far.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/local_ga_best_so_far.json)
  - [`local_ga_chunk_progress.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/local_ga_chunk_progress.json)

Backups, `.bak` snapshots, and large transient logs are intentionally left out of the commit because the result JSON files and this note already preserve the useful experiment history without adding replay noise.

## Conclusion

Across the alternative optimizers tested in this cycle:

- pure pattern search: no improvement
- pure TPE / Optuna: no validated improvement
- stronger hybrid GA -> TPE: no improvement over the incumbent
- automated staged / overnight GA refinement: the only search path that consistently improved the incumbent

The current incumbent remains the best operating point for the present swing-trade objective:

- very strong improvement in hit rate vs `main`
- modest but real improvement in alpha vs `main`
- still negative alpha overall, which remains the main open optimization problem
- risk is higher than `main`, so future search should keep focusing on alpha improvement without giving back `WR`

## Follow-up Attempts on 2026-04-18

These runs were executed on `2026-04-18` (Saturday) with the latest consolidated data through `2026-04-17`. Both runs re-evaluated:

- `git:main` baseline checkpoint at `main@865fdd894f0f21ae5244807ce73c68f77af4fa0d`
- the current worktree incumbent checkpoint in [`global_ga_checkpoint.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/global_ga_checkpoint.json)

Authoritative `eval-only` snapshot under the current fitness (hit-rate-first, `--alpha-focus off`):

| Metric | `main` baseline | Current incumbent | Delta |
| --- | ---: | ---: | ---: |
| `fit` | `-31.7891` | `11.2307` | `+43.0198` |
| `WR_med_all` | `33.33%` | `66.67%` | `+33.33pp` |
| `WR_target_med_all` | `16.67%` | `67.78%` | `+51.11pp` |
| `mean_alpha_ann` | `-9.57%` | `-8.26%` | `+1.30pp` |
| `MDD_med` | `8.44%` | `12.38%` | `+3.94pp` |

### Attempt 3: Staged GA (Stage 2 refine, `WR`-biased)

Timestamp: `2026-04-18T03:01` local (Europe/Berlin)

Method:

- `py run_local_ga_staged.py --stage 2 --alpha-focus off --chunks 2 --ngen-per-chunk 2 --pop-size 20 --windows 5 --window-offset 18 --apply-best`

Windows tested:

- `windows=[4, 11, 18, 25, 32]` (offset `18`, `5` windows)

Best candidate observed (chunk 1, full metrics):

| Metric | Incumbent | Candidate | Δ vs incumbent | Δ vs `main` |
| --- | ---: | ---: | ---: | ---: |
| `fit` | `11.2307` | `12.4853` | `+1.2546` | `+44.2744` |
| `WR_med_all` | `66.67%` | `66.67%` | `0.00pp` | `+33.33pp` |
| `WR_target_med_all` | `67.78%` | `67.08%` | `-0.70pp` | `+50.42pp` |
| `mean_alpha_ann` | `-8.26%` | `-8.20%` | `+0.06pp` | `+1.37pp` |
| `MDD_med` | `12.38%` | `12.20%` | `-0.18pp` | `+3.76pp` |

Outcome:

- `promote = false`
- rejection reason: **`WR_target_med_all` regressed vs the incumbent**, tripping the shared incumbent hit-rate guardrail (even though alpha and MDD marginally improved).

What was learned:

- local Stage-2 refinement can trade small alpha/MDD improvements for small `WR_target` losses; with the current incumbent guardrails this is not promotable.
- next attempts should either (a) widen search diversity (Stage 1 / more mutations) or (b) explicitly bias the *search* objective harder toward `WR_target` so improvements do not come at `WR_target`’s expense.

### Attempt 4: Staged GA (Stage 1 explore, higher `eval_topk`)

Timestamp: `2026-04-18T03:30` local (Europe/Berlin)

Method:

- `py run_local_ga_staged.py --stage 1 --alpha-focus off --chunks 1 --ngen-per-chunk 2 --pop-size 22 --windows 4 --window-offset 21 --eval-topk 8 --apply-best`

Windows tested:

- `windows=[3, 12, 21, 30]` (offset `21`, `4` windows)

Outcome:

- `promote = false`
- none of the top-8 validated candidates beat the incumbent under the prioritized objective (`WR_med_all` → `WR_target_med_all` → alpha recovery) while keeping drawdown/tail-risk guardrails satisfied.

What was learned:

- increasing `eval_topk` surfaced several “nearby” candidates with similar `WR_med_all`, but none improved `WR_target` without giving back the incumbent’s safety profile.
- the incumbent remains the best operating point under the current objective; further progress likely requires either a larger Stage-1 budget or a targeted wr-target-focused retrain (e.g., the wr-target wrapper / stronger `WR_target` pressure in the train objective).

### Attempt 5: Staged GA (Stage 1 explore, bigger budget, new offset)

Timestamp: `2026-04-18T03:49` local (Europe/Berlin), resumed and completed by ~`04:30`.

Method:

- `py run_local_ga_staged.py --stage 1 --alpha-focus off --chunks 1 --ngen-per-chunk 3 --pop-size 34 --windows 5 --window-offset 25 --eval-topk 12`

Windows tested:

- `windows=[3, 11, 18, 25, 32]` (offset `25`, `5` windows)

Best candidate observed (chunk 2, full metrics):

| Metric | Incumbent | Candidate | Δ vs incumbent | Δ vs `main` |
| --- | ---: | ---: | ---: | ---: |
| `fit` | `11.2307` | `10.4634` | `-0.7673` | `+42.2525` |
| `WR_med_all` | `66.67%` | `66.67%` | `0.00pp` | `+33.33pp` |
| `WR_target_med_all` | `67.78%` | `66.67%` | `-1.11pp` | `+50.00pp` |
| `mean_alpha_ann` | `-8.26%` | `-8.20%` | `+0.06pp` | `+1.37pp` |
| `MDD_med` | `12.38%` | `12.10%` | `-0.28pp` | `+3.66pp` |

Outcome:

- `promote = false`
- rejection reason: **`WR_target_med_all` dropped below the incumbent WR-target floor**, failing the shared incumbent hit-rate guardrail.

What was learned:

- even with a larger Stage-1 budget + broader validation (`eval_topk=12`), the best full-metric candidates tended to trade small alpha/MDD gains for meaningful `WR_target` losses.
- the bottleneck appears to be search objective alignment (top train fitness ≠ best swing-priority); next attempt should directly optimize the priority tuple under full metrics.

### Attempt 6: Full-metric local sweep (WR-target gene neighborhood) — PROMOTED

Timestamp: `2026-04-18T04:37` local (Europe/Berlin)

Method:

- `py run_local_fullmetric_sweep.py --alpha-focus off --combo-budget 12`
- evaluated small +/- step moves on the most WR-target-relevant genes (`stop_*`, `time_stop_bars`, `partial_take_*`, `reward_risk_ratio`, `trailing_stop_mode`) using full staged metrics on all tickers
- ranked candidates with the shared staged priority comparator (`WR_med_all` → `WR_target_med_all` → `mean_alpha_ann` → risk)

Promoted candidate:

- change: `partial_take_pct: 0.30 -> 0.20` (single-gene step)

Promotion deltas (authoritative from the incumbent snapshot in [`global_ga_checkpoint.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/global_ga_checkpoint.json)):

| Metric | `main` baseline | Prior incumbent | New incumbent | Δ vs prior incumbent |
| --- | ---: | ---: | ---: | ---: |
| `fit` | `-31.7891` | `11.2307` | `8.1675` | `-3.0632` |
| `WR_med_all` | `33.33%` | `66.67%` | `67.12%` | `+0.46pp` |
| `WR_target_med_all` | `16.67%` | `67.78%` | `67.78%` | `0.00pp` |
| `mean_alpha_ann` | `-9.57%` | `-8.26%` | `-8.29%` | `-0.02pp` |
| `MDD_med` | `8.44%` | `12.38%` | `12.58%` | `+0.19pp` |

Outcome:

- `promote = true`
- rationale: improved `WR_med_all` (primary operational objective) while keeping `WR_target` flat and staying inside the drawdown/tail-risk guardrails.

Daily workbook + Telegram status after promotion:

- rebuilt the workbook in GA load-only mode (no ETL) and saved [`summary_latest.xlsx`](/C:/Users/gabri/.codex/worktrees/0b30/test/summary_latest.xlsx)
- Telegram send was attempted earlier but failed in this environment due to a socket/network permissions error (`WinError 10013`), so delivery could not be confirmed here.

### Attempt 7: Optuna/TPE local refinement (train-fit proxy) — NOT PROMOTED

Timestamp: `2026-04-18T05:49` local (Europe/Berlin)

Method:

- `GA_ALPHA_FOCUS=off py run_local_optuna_search.py --windows 5 --window-offset 22 --trials 24 --radius-steps 4 --seed 42 --topk-validate 6`
- optimized the **train-phase objective** on a 5-window subset, then full-metric-validated the top-6 trials against the staged-runner guardrails

Windows tested:

- window indices: `[0, 8, 15, 22, 29]` (selected from `36` walk-forward windows)

Outcome:

- `promote = false`
- best validated candidate under full metrics was the **seed itself** (no change), i.e. the TPE-proposed genomes did not beat the incumbent under `WR_med_all → WR_target_med_all → alpha recovery` while staying inside the incumbent WR + safety guardrails.

Why it failed:

- the Optuna objective being optimized (train-fit proxy on a sparse window subset) did not correlate well with the full-metric staged priorities; suggested trials scored poorly on the proxy and did not produce any guardrail-passing improvements on full metrics.

What was learned:

- for this checkpoint, **objective alignment matters more than optimizer sophistication**: TPE over a misaligned proxy is not productive.
- next searches should either optimize a priority-aligned proxy (e.g., full-metric on a reduced ticker set) or use direct full-metric neighborhood search.

### Attempt 8: Expanded full-metric sweep (exit/risk neighborhood) — PROMOTED

Timestamp: `2026-04-18T05:58` local (Europe/Berlin) (promotion applied at ~`06:06`)

Method:

- `py run_local_fullmetric_sweep.py --alpha-focus off --genes "stop_atr_mult,stop_tighten_factor,time_stop_bars,max_loss_per_trade_pct,equity_drawdown_stop_pct,partial_take_level,partial_take_level_2,trailing_stop_mode,score_percentile_trigger,entry_discount_atr_frac" --combo-budget 24 --seed 43 --out local_fullmetric_sweep_result_20260418_055831.json`
- evaluated small +/- step moves with **full staged metrics on all tickers**, ranking by the shared staged priority comparator and requiring guardrail pass to promote.

Windows tested:

- full-metric evaluation across all tickers (data range: `2005-01-04 → 2026-04-17`)

Promoted candidate:

- change: `trailing_stop_mode: 2 -> 1` (single-gene step)

Promotion deltas (authoritative from the checkpoint metadata in [`global_ga_checkpoint.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/global_ga_checkpoint.json)):

| Metric | `main` baseline | Prior incumbent | New incumbent | Î” vs prior incumbent |
| --- | ---: | ---: | ---: | ---: |
| `fit` | `-31.7891` | `8.1675` | `8.8572` | `+0.6897` |
| `WR_med_all` | `33.33%` | `67.12%` | `67.78%` | `+0.65pp` |
| `WR_target_med_all` | `16.67%` | `67.78%` | `68.42%` | `+0.64pp` |
| `mean_alpha_ann` | `-9.57%` | `-8.29%` | `-8.29%` | `+0.00pp` |
| `MDD_med` | `8.44%` | `12.58%` | `12.65%` | `+0.07pp` |

Outcome:

- `promote = true`
- rationale: improved both primary operational hit rate (`WR_med_all`) and `WR_target_med_all` without any material regression and while staying inside the incumbent WR + safety guardrails.

Daily workbook + Telegram status after promotion:

- refreshed workbook via `py ensure_daily_telegram_file.py --date 2026-04-17` and rebuilt [`summary_latest.xlsx`](/C:/Users/gabri/.codex/worktrees/0b30/test/summary_latest.xlsx)
- attempted to send exactly one Telegram document (`summary_latest.xlsx`, date `2026-04-17`), but the send failed in this environment due to socket/network permissions (`WinError 10013`), so delivery could not be confirmed here.

### Attempt 9: Full-metric sweep (entry/regime neighborhood) — PROMOTED

Timestamp: `2026-04-18T07:50` local (Europe/Berlin) (promotion applied at ~`07:57`)

Method:

- `py run_local_fullmetric_sweep.py --alpha-focus off --genes "entry_score_threshold,min_signal_strength,score_percentile_trigger,regime_threshold,volatility_filter_percentile,volume_confirm_mode,momentum_confirm_days" --combo-budget 24 --seed 44 --out local_fullmetric_sweep_result_20260418_075045.json`
- evaluated small +/- step moves with full staged metrics (all tickers), ranked by the staged priority comparator and requiring guardrail pass to promote.

Windows tested:

- full-metric evaluation across all tickers (data range: `2005-01-04 → 2026-04-17`, `36` walk-forward windows)

Promoted candidate:

- change: `regime_threshold: 0.30 -> 0.25` (single-gene step)

Promotion deltas (authoritative from the checkpoint metadata in [`global_ga_checkpoint.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/global_ga_checkpoint.json)):

| Metric | `main` baseline | Prior incumbent | New incumbent | Δ vs prior incumbent |
| --- | ---: | ---: | ---: | ---: |
| `fit` | `-31.7891` | `8.8572` | `8.7203` | `-0.1369` |
| `WR_med_all` | `33.33%` | `67.78%` | `68.24%` | `+0.46pp` |
| `WR_target_med_all` | `16.67%` | `68.42%` | `68.42%` | `0.00pp` |
| `mean_alpha_ann` | `-9.57%` | `-8.29%` | `-8.26%` | `+0.03pp` |
| `MDD_med` | `8.44%` | `12.65%` | `12.65%` | `0.00pp` |

Outcome:

- `promote = true`
- rationale: improved the primary operational metric (`WR_med_all`) with no drawdown regression and a small alpha improvement; passed the incumbent WR + safety guardrails.

What was learned:

- the regime-gating threshold still has a clean local improvement left: a slightly looser regime threshold reduced false positives enough to lift `WR_med_all` without sacrificing `WR_target`.

Daily workbook + Telegram status after promotion:

- rebuilt the workbook and attempted a single Telegram send via `py ensure_daily_telegram_file.py` for `summary_latest.xlsx` (`Apply.Date = 2026-04-17`)
- Telegram send failed in this environment due to socket/network permissions (`WinError 10013`), so delivery could not be confirmed here.

### Attempt 10: Full-metric sweep (signal precision + vol regime) — PROMOTED

Timestamp: `2026-04-18T09:50` local (Europe/Berlin) (promotion applied at ~`09:55`)

Method:

- `py run_local_fullmetric_sweep.py --alpha-focus off --genes "vote_threshold_long,vote_threshold_short,z_threshold,signal_ema_span,entry_confirmation_days,score_strength_scaling,consecutive_loss_cooldown,volatility_filter_percentile,vol_regime_mode" --combo-budget 24 --seed 45 --out local_fullmetric_sweep_result_20260418_095044.json`
- `py apply_fullmetric_sweep_best.py --sweep local_fullmetric_sweep_result_20260418_095044.json`

Windows tested:

- full-metric evaluation across all tickers (data range: `2005-01-04 → 2026-04-17`, `36` walk-forward windows built)

Promoted candidate:

- change: `vol_regime_mode: 1 -> 2` (single-gene step; skip high-vol regime)

Promotion deltas (authoritative from the checkpoint metadata in [`global_ga_checkpoint.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/global_ga_checkpoint.json)):

| Metric | `main` baseline | Prior incumbent | New incumbent | Delta vs prior incumbent |
| --- | ---: | ---: | ---: | ---: |
| `fit` | `-31.7891` | `8.7203` | `19.3981` | `+10.6778` |
| `WR_med_all` | `33.33%` | `68.24%` | `68.97%` | `+0.73pp` |
| `WR_target_med_all` | `16.67%` | `68.42%` | `69.09%` | `+0.67pp` |
| `mean_alpha_ann` | `-9.57%` | `-8.26%` | `-8.24%` | `+0.02pp` |
| `MDD_med` | `8.44%` | `12.65%` | `11.75%` | `-0.90pp` |

Outcome:

- `promote = true`
- rationale: improved both hit-rate priorities (`WR_med_all`, `WR_target_med_all`) while also reducing drawdown magnitude; passed incumbent WR + safety guardrails.

What was learned:

- `vol_regime_mode` is a high-leverage operational safety knob: skipping high-vol regimes can simultaneously reduce drawdown and slightly improve hit rates without collapsing trade frequency.

Daily workbook + Telegram status after promotion:

- rebuilt [`summary_latest.xlsx`](/C:/Users/gabri/.codex/worktrees/0b30/test/summary_latest.xlsx) because the checkpoint became newer than the workbook (latest store day `2026-04-17`)
- attempted to send exactly one Telegram document (`summary_latest.xlsx`, date `2026-04-17`), but the send failed in this environment due to socket/network permissions (`WinError 10013`), so delivery could not be confirmed here.

### Attempt 11: Full-metric sweep (safety knobs: vol filter / DD stop / cooldown) â€” PROMOTED

Timestamp: `2026-04-18T11:50` local (Europe/Berlin) (promotion applied at ~`11:56`)

Method:

- refreshed incumbent/baseline under the latest fitness: `py run_local_ga_staged.py --eval-only --alpha-focus off`
- `py run_local_fullmetric_sweep.py --alpha-focus off --genes "volatility_filter_percentile,regime_threshold,max_loss_per_trade_pct,equity_drawdown_stop_pct,consecutive_loss_cooldown,trailing_stop_mode,stop_atr_mult,reward_risk_ratio" --combo-budget 10 --seed 42 --out local_fullmetric_sweep_result_20260418_115025.json`
- `py apply_fullmetric_sweep_best.py --sweep local_fullmetric_sweep_result_20260418_115025.json`

Windows tested:

- full-metric evaluation across all tickers (data range: `2005-01-04 â†’ 2026-04-17`, `36` walk-forward windows built)

Promoted candidate:

- change: `volatility_filter_percentile: 0.00 -> 0.05` (single-gene step)

Promotion deltas (authoritative from the checkpoint metadata in [`global_ga_checkpoint.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/global_ga_checkpoint.json)):

| Metric | `main` baseline | Prior incumbent | New incumbent | Delta vs prior incumbent |
| --- | ---: | ---: | ---: | ---: |
| `fit` | `-31.7891` | `19.3981` | `20.7172` | `+1.3191` |
| `WR_med_all` | `33.33%` | `68.97%` | `69.70%` | `+0.73pp` |
| `WR_target_med_all` | `16.67%` | `69.09%` | `69.44%` | `+0.35pp` |
| `mean_alpha_ann` | `-9.57%` | `-8.24%` | `-8.27%` | `-0.03pp` |
| `MDD_med` | `8.44%` | `11.75%` | `11.59%` | `-0.16pp` |

Outcome:

- `promote = true`
- rationale: improved `WR_med_all` (primary) and slightly improved `WR_target_med_all` while also reducing `MDD_med`; small alpha regression was within tolerance and did not block promotion under hit-rate-first priorities.

What was learned:

- `volatility_filter_percentile` appears to be another high-leverage operational safety knob: a tiny filter (`+0.05`) improved hit rate and reduced drawdown without collapsing trade counts.

Daily workbook + Telegram status after promotion:

- ran `py ensure_daily_telegram_file.py` to send [`summary_latest.xlsx`](/C:/Users/gabri/.codex/worktrees/0b30/test/summary_latest.xlsx) for the latest store day (`2026-04-17`)
- Telegram send failed in this environment due to socket/network permissions (`WinError 10013`), so delivery could not be confirmed here.
- follow-up fix: updated `ensure_daily_telegram_file.py` to **avoid** best-effort workbook rebuilds purely due to checkpoint mtime changes (can OOM in this environment); it now rebuilds only when the workbook is stale/missing, unless `--refresh-model` is explicitly requested.

### Attempt 26: Full-metric sweep (E_regime_vol revisit) - NOT PROMOTED

Timestamp: `2026-04-19T16:54:22Z`

Method:

- `py run_local_fullmetric_sweep.py --alpha-focus off --genes "ma_filter_period,ma_filter_mode,vol_regime_mode,volatility_filter_percentile,regime_threshold,volume_confirm_mode,momentum_confirm_days" --combo-budget 10 --seed 59 --out local_fullmetric_sweep_result_20260419_1647_regimevol_revisit.json`

Genes touched: `[ma_filter_period, ma_filter_mode, vol_regime_mode, volatility_filter_percentile, regime_threshold, volume_confirm_mode, momentum_confirm_days]`

Windows tested: full-metric across all tickers (data range: `2005-01-04 -> 2026-04-17`, `36` walk-forward windows)

Candidate summary:
| Metric | main baseline | Prior incumbent | Candidate | delta vs incumbent |
|---|---:|---:|---:|---:|
| fit | 20.9898 | 20.9898 | 20.7059 | -0.2838 |
| WR_med_all | 69.70% | 69.70% | 69.70% | +0.00pp |
| WR_target_med_all | 69.44% | 69.44% | 69.44% | +0.00pp |
| mean_alpha_ann | -8.26% | -8.26% | -8.25% | +0.01pp |
| MDD_med | 11.31% | 11.31% | 11.31% | +0.00pp |
| MDD_duration_med | 226.5 | 226.5 | 228.5 | +2.0 |
| trades_med | 31.5 | 31.5 | 32.0 | +0.5 |

Outcome:

- `promote = false`
- rationale: the best guardrail-safe candidate (`regime_threshold: 0.25 -> 0.20`) kept `WR_med_all` and `WR_target_med_all` flat and recovered a tiny amount of alpha, but `MDD_duration_med` lengthened from `226.5` to `228.5`, so the staged priority comparator left `priority_better = false`. Higher-fit variants were not promotable: `momentum_confirm_days: 5 -> 4` lifted `WR_target_med_all` to `70.00%`, improved alpha to `-8.22%`, and reduced `MDD_med` to `11.04%`, but `WR_med_all` slipped to `69.44%`, missing the incumbent floor (`69.45%`) and failing the shared hit-rate guardrail.

What was learned:

- `momentum_confirm_days: 5 -> 4` is the strongest uncovered E-category direction: it improves `WR_target`, alpha, and drawdown simultaneously, but it gives back about `0.25pp` of `WR_med_all` on its own and therefore needs a paired WR-restoring lever.
- extra volatility filtering (`volatility_filter_percentile: 0.05 -> 0.10`) is over-defensive here: fit and risk improve, but alpha regresses to `-8.34%` and `alpha_ann_pos_rate` falls below the acceptance floor.
- `regime_threshold: 0.25 -> 0.20` is still guardrail-safe, but only as a tie-preserving nudge; with drawdown duration worsening and no hit-rate gain, this local E frontier looks saturated for single-step regime loosening.
- Proxima direcao sugerida: revisit `C_timing` or a hot-zone combo around `momentum_confirm_days` plus a WR restorer such as `consecutive_loss_cooldown` after `D_entry_filter` ages out of the 4h exclusion window (`2026-04-19T19:25:06Z`).

### Attempt 27: Full-metric sweep (momentum WR-restore combo probe) - PROMOTED

Timestamp: `2026-04-19T18:54:45Z`

Method:

- `py run_local_fullmetric_sweep.py --alpha-focus off --genes "momentum_confirm_days,consecutive_loss_cooldown,partial_take_level,regime_threshold,ma_filter_mode,volatility_filter_percentile" --combo-budget 12 --seed 60 --out local_fullmetric_sweep_result_20260419_2047_momentum_wrrestore.json`
- `py apply_fullmetric_sweep_best.py --sweep local_fullmetric_sweep_result_20260419_2047_momentum_wrrestore.json`
- `py run_local_ga_staged.py --eval-only --alpha-focus off`

Genes touched: `[momentum_confirm_days, consecutive_loss_cooldown, partial_take_level, regime_threshold, ma_filter_mode, volatility_filter_percentile]`

Windows tested: full-metric across all tickers (data range: `2005-01-04 -> 2026-04-17`, `36` walk-forward windows)

Candidate summary:
| Metric | main baseline | Prior incumbent | Candidate | delta vs incumbent |
|---|---:|---:|---:|---:|
| fit | 20.9898 | 20.9898 | 21.2051 | +0.2154 |
| WR_med_all | 69.70% | 69.70% | 69.70% | +0.00pp |
| WR_target_med_all | 69.44% | 69.44% | 69.70% | +0.25pp |
| mean_alpha_ann | -8.26% | -8.26% | -8.25% | +0.01pp |
| MDD_med | 11.31% | 11.31% | 11.31% | +0.00pp |
| MDD_duration_med | 226.5 | 226.5 | 228.5 | +2.0 |
| trades_med | 31.5 | 31.5 | 31.0 | -0.5 |

Outcome:

- `promote = true`
- rationale: the sweep was designed to rescue the uncovered `momentum_confirm_days:-1` direction with WR-restoring levers, but the clean winner was still a single-gene timing move: `consecutive_loss_cooldown: 7 -> 8`. It preserved incumbent `WR_med_all` exactly, lifted `WR_target_med_all` by `0.25pp`, improved annualized alpha slightly, and kept `MDD_med` flat while staying inside all shared staged-runner guardrails. The small `MDD_duration_med` increase (`226.5 -> 228.5`) did not block promotion because the primary hit-rate metric did not regress and the candidate was priority-better on the staged comparator.

What was learned:

- `consecutive_loss_cooldown` still has one more safe step beyond the prior `6 -> 7` promotion: `7 -> 8` adds one more `>=70%` ticker (`57 -> 58`) and upgrades `WR_target_med_all` without giving back `WR_med_all`.
- The intended `momentum_confirm_days:-1` rescue still did not materialize in this budget. Even with multiple hot-zone WR restorers in the sweep, no momentum-including variant beat the incumbent guardrails, so the WR-restoration burden remains the binding constraint.
- `partial_take_level:+1` remained guardrail-safe and improved alpha/MDD, but because it left `WR_target_med_all` flat at `69.44%`, it was dominated by the cooldown step in the hit-rate-first priority order.
- Proxima direcao sugerida: after `D_entry_filter` clears the 4h exclusion window (`2026-04-19T19:25:06Z`), test `entry_score_threshold` as the WR stabilizer paired with `momentum_confirm_days:-1`; if staying outside `D`, try a narrower cross-category sweep around `momentum_confirm_days, partial_take_level, consecutive_loss_cooldown`.
