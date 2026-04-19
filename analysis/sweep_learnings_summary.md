# Sweep Learnings Summary

_Generated: 2026-04-19T12:30:10+00:00_

Aggregates all `local_fullmetric_sweep_result*.json` files plus the
continuous-improvement state. Use this file to plan the next cycle.

## Current snapshot

- Checkpoint fitness: `21.01378551704503`
- Total sweeps seen: **24**
- Total promotions: **10** (including 2 from codex log)
- Promotion rate: **26.3%** (over 38 total attempts)
- Continuous cycles run: 16
- Continuous promotions: 3
- Codex attempts synced: **14** (7 promoted) from `optimization_attempts_20260416.md`

## Next suggested sweep

categoria mais antiga: **D_entry_filter** (ultima vez: 2026-04-19T07:56:24+00:00) -> `python run_continuous_improvement.py --category D`

## Promotion timeline (chronological)

| Date | Source | Sweep file | Move | Resulting metrics |
|---|---|---|---|---|
| 2026-04-18 | local | `local_fullmetric_sweep_result.json` | `partial_take_pct 0.30000000000000004->0.2` | fit=8.168 WR=67.1% alpha=-8.29% MDD=12.6% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_055831.json` | `trailing_stop_mode 2->1` | fit=8.857 WR=67.8% alpha=-8.29% MDD=12.6% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_075045.json` | `regime_threshold 0.30000000000000004->0.25` | fit=8.720 WR=68.2% alpha=-8.26% MDD=12.6% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_095044.json` | `vol_regime_mode 1->2` | fit=19.398 WR=69.0% alpha=-8.24% MDD=11.7% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_115025.json` | `volatility_filter_percentile 0.0->0.05` | fit=20.717 WR=69.7% alpha=-8.27% MDD=11.6% |
| 2026-04-18 | codex | `local_fullmetric_sweep_result_20260418_135553.json` | `consecutive_loss_cooldown 6.0->7.0; entry_score_threshold 0.15->0.2` | fit=21.007 WR=70.0% alpha=-8.27% MDD=11.4% |
| 2026-04-18 | codex | `local_fullmetric_sweep_result_20260418_1620_alphaexit.json` | `partial_take_level 0.75->1.0` | fit=21.021 WR=70.0% alpha=-8.24% MDD=11.2% |
| 2026-04-19 | local | `local_fullmetric_sweep_continuous_C_timing_20260419_093647.json` | `consecutive_loss_cooldown 6->7` | fit=19.758 WR=70.0% alpha=-8.31% MDD=10.7% |
| 2026-04-19 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260419_104733.json` | `ma_filter_mode 1->0` | fit=19.835 WR=69.7% alpha=-8.30% MDD=11.2% |
| 2026-04-19 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260419_122638.json` | `stop_tighten_factor 0.45->0.4` | fit=21.014 WR=69.7% alpha=-8.29% MDD=10.4% |

## Graveyard (genes que falharam multiplas vezes)

Genes tentados >= 2 vezes sem nenhuma promocao. Considere SKIP no proximo ciclo.
(Colunas `codex` contam tentativas importadas do log do GPT codex — dedup por sweep file.)

| Gene | Attempts | Promotions | (codex) Attempts | (codex) Promotions |
|---|---:|---:|---:|---:|
| `reward_risk_ratio` | 9 | 0 | 1 | 0 |
| `partial_take_level_2` | 9 | 0 | 1 | 0 |
| `partial_take_pct_2` | 8 | 0 | 1 | 0 |
| `stop_atr_mult` | 6 | 0 | 0 | 0 |
| `max_loss_per_trade_pct` | 5 | 0 | 0 | 0 |
| `equity_drawdown_stop_pct` | 5 | 0 | 0 | 0 |
| `score_percentile_trigger` | 5 | 0 | 0 | 0 |
| `entry_discount_atr_frac` | 5 | 0 | 1 | 0 |
| `score_strength_scaling` | 5 | 0 | 1 | 0 |
| `stop_tighten_after_bars` | 4 | 0 | 0 | 0 |
| `time_stop_bars` | 4 | 0 | 0 | 0 |
| `min_signal_strength` | 4 | 0 | 0 | 0 |
| `volume_confirm_mode` | 4 | 0 | 1 | 0 |
| `momentum_confirm_days` | 4 | 0 | 1 | 0 |
| `vote_threshold_long` | 4 | 0 | 0 | 0 |
| `vote_threshold_short` | 4 | 0 | 0 | 0 |
| `z_threshold` | 4 | 0 | 0 | 0 |
| `signal_ema_span` | 4 | 0 | 0 | 0 |
| `entry_confirmation_days` | 4 | 0 | 0 | 0 |
| `ma_filter_period` | 2 | 0 | 0 | 0 |

## Hot zones (genes que ja promoveram)

Genes onde alguma combinacao funcionou. Vale revisitar com novos vizinhos.

| Gene | Promotions | Attempts | Hit rate | (codex) Promotions |
|---|---:|---:|---:|---:|
| `consecutive_loss_cooldown` | 2 | 5 | 40% | 1 |
| `ma_filter_mode` | 1 | 2 | 50% | 0 |
| `vol_regime_mode` | 1 | 3 | 33% | 0 |
| `stop_tighten_factor` | 1 | 5 | 20% | 0 |
| `entry_score_threshold` | 1 | 5 | 20% | 1 |
| `regime_threshold` | 1 | 5 | 20% | 0 |
| `volatility_filter_percentile` | 1 | 5 | 20% | 0 |
| `trailing_stop_mode` | 1 | 6 | 17% | 0 |
| `partial_take_pct` | 1 | 8 | 12% | 0 |
| `partial_take_level` | 1 | 9 | 11% | 1 |

## Category rotation status

| Category | Last swept | Runs | Promotions |
|---|---|---:|---:|
| A_risk_management | 2026-04-19T12:29:52+00:00 | 3 | 1 |
| B_take_profit | 2026-04-19T08:12:23+00:00 | 4 | 0 |
| C_timing | 2026-04-19T09:39:50+00:00 | 2 | 1 |
| D_entry_filter | 2026-04-19T07:56:24+00:00 | 3 | 0 |
| E_regime_vol | 2026-04-19T10:51:45+00:00 | 2 | 1 |
| F_trailing | 2026-04-19T11:29:21+00:00 | 2 | 0 |

## Codex sync history (latest attempts)

Parsed `optimization_attempts_20260416.md` at 2026-04-18T16:33:25+02:00. Total attempts: **14**, promoted: **7**. Updates via `python analysis/codex_attempts_sync.py` at the start of every continuous cycle (auto) or on demand.

| Attempt | Timestamp | Promoted | Move(s) | Learning (snippet) |
|---|---|:-:|---|---|
| #13 | 2026-04-18T15:56 | YES | `partial_take_level 0.75->1.0` | `partial_take_level` is a useful local refinement lever: raising the first partial take threshold can improve median dra... |
| #12 | 2026-04-18T13:55 | YES | `consecutive_loss_cooldown 6.0->7.0; entry_score_threshold 0.15->0.2` | a slightly stricter entry gating (`entry_score_threshold`) plus a longer loss cooldown can lift WR and reduce drawdown s... |
| #11 | 2026-04-18T11:50 | YES | `volatility_filter_percentile 0.0->0.05` | `volatility_filter_percentile` appears to be another high-leverage operational safety knob: a tiny filter (`+0.05`) impr... |
| #10 | 2026-04-18T09:50 | YES | `vol_regime_mode 1.0->2.0` | `vol_regime_mode` is a high-leverage operational safety knob: skipping high-vol regimes can simultaneously reduce drawdo... |
| #9 | 2026-04-18T07:50 | YES | `regime_threshold 0.3->0.25` | the regime-gating threshold still has a clean local improvement left: a slightly looser regime threshold reduced false p... |

---

_Run again with: `python analysis/sweep_learnings.py`_
