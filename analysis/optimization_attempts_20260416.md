# Optimization Attempts Through 2026-04-16

This note documents the optimizer experiments added after the GA-first metric refactor, with the current acceptance logic aimed at swing trading:

- prioritize `WR_med_all` and `WR_target_med_all`
- keep explicit risk guardrails on drawdown magnitude and duration
- improve `mean_alpha_ann` only when it does not materially weaken operational safety
- compare every promotion against `git:main`

## Important note on `fit`

The top-level `fitness` field stored in [`global_ga_checkpoint.json`](/C:/Users/gabri/.codex/worktrees/0b30/test/global_ga_checkpoint.json) can differ from the re-evaluated `fit` in the experiment result JSON files because the fitness function evolved during this cycle. For comparisons in this document, the authoritative values are the re-evaluated metrics saved in each result artifact.

## Current incumbent vs `main`

Latest full re-evaluation snapshot used by the stronger hybrid run:

| Metric | `main` baseline | Current incumbent | Delta |
| --- | ---: | ---: | ---: |
| `fit` | `-37.5492` | `7.5647` | `+45.1138` |
| `WR_med_all` | `33.33%` | `66.67%` | `+33.33pp` |
| `WR_mean_all` | `34.70%` | `63.12%` | `+28.42pp` |
| `WR_target_med_all` | `0.00%` | `68.18%` | `+68.18pp` |
| `mean_alpha_ann` | `-9.57%` | `-8.16%` | `+1.41pp` |
| `alpha_ann_pos_rate` | `15.97%` | `19.33%` | `+3.36pp` |
| `MDD_med` | `8.03%` | `12.76%` | `+4.73pp` |
| `MDD_p75` | `12.47%` | `18.57%` | `+6.09pp` |
| `MDD_duration_med` | `445.5` | `283.5` | `-162.0` |
| `trades_med` | `11.0` | `32.5` | `+21.5` |

Reading this in trading terms:

- the current incumbent is materially better than `main` on hit rate and active participation
- alpha improved, but is still negative
- risk magnitude is higher than `main`, even though drawdown duration improved

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

The current incumbent remains the best operating point for the present swing-trade objective:

- very strong improvement in hit rate vs `main`
- modest but real improvement in alpha vs `main`
- still negative alpha overall, which remains the main open optimization problem
- risk is higher than `main`, so future search should keep focusing on alpha improvement without giving back `WR`
