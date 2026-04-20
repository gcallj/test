# Sweep Learnings Summary

_Generated: 2026-04-20T08:24:53+00:00_

Aggregates all `local_fullmetric_sweep_result*.json` files plus the
continuous-improvement state. Use this file to plan the next cycle.

## Current snapshot

- Checkpoint fitness: `25.520597384933595`
- Total sweeps seen: **36**
- Total promotions: **21** (including 6 from codex log)
- Promotion rate: **35.6%** (over 59 total attempts)
- Continuous cycles run: 28
- Continuous promotions: 10
- Codex attempts synced: **23** (11 promoted) from `codex_attempts_source.md`

## Next suggested sweep

categoria mais antiga: **D_entry_filter** (ultima vez: 2026-04-20T01:36:41+00:00) -> `python run_continuous_improvement.py --category D`

## Promotion timeline (chronological)

| Date | Source | Sweep file | Move | Resulting metrics |
|---|---|---|---|---|
| 2026-04-18 | local | `local_fullmetric_sweep_result.json` | `partial_take_pct 0.30000000000000004->0.2` | fit=8.168 WR=67.1% alpha=-8.29% MDD=12.6% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_055831.json` | `trailing_stop_mode 2->1` | fit=8.857 WR=67.8% alpha=-8.29% MDD=12.6% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_075045.json` | `regime_threshold 0.30000000000000004->0.25` | fit=8.720 WR=68.2% alpha=-8.26% MDD=12.6% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_095044.json` | `vol_regime_mode 1->2` | fit=19.398 WR=69.0% alpha=-8.24% MDD=11.7% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_115025.json` | `volatility_filter_percentile 0.0->0.05` | fit=20.717 WR=69.7% alpha=-8.27% MDD=11.6% |
| 2026-04-19 | local | `local_fullmetric_sweep_continuous_C_timing_20260419_093647.json` | `consecutive_loss_cooldown 6->7` | fit=19.758 WR=70.0% alpha=-8.31% MDD=10.7% |
| 2026-04-19 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260419_104733.json` | `ma_filter_mode 1->0` | fit=19.835 WR=69.7% alpha=-8.30% MDD=11.2% |
| 2026-04-19 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260419_122638.json` | `stop_tighten_factor 0.45->0.4` | fit=21.014 WR=69.7% alpha=-8.29% MDD=10.4% |
| 2026-04-19 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260419_151906.json` | `entry_score_threshold 0.15000000000000002->0.2` | fit=19.694 WR=69.7% alpha=-8.33% MDD=11.2% |
| 2026-04-19 | codex | `local_fullmetric_sweep_result_20260419_2047_momentum_wrrestore.json` | `` | fit=21.205 WR=69.7% alpha=-8.25% MDD=11.3% |
| 2026-04-19 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260419_212524.json` | `regime_threshold 0.25->0.2` | fit=21.907 WR=69.9% alpha=-8.28% MDD=11.2% |
| 2026-04-19 | codex | `local_fullmetric_sweep_result_20260419_2248_entry_momentum_anchor.json` | `momentum_confirm_days 5.0->4.0` | fit=23.529 WR=70.0% alpha=-8.15% MDD=11.0% |
| 2026-04-19 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260419_235131.json` | `max_loss_per_trade_pct 0.08->0.09` | fit=20.967 WR=69.7% alpha=-8.27% MDD=10.9% |
| 2026-04-20 | codex | `local_fullmetric_sweep_result_20260420_001_hotzone_rebalance.json` | `regime_threshold 0.25->0.3` | fit=23.924 WR=70.0% alpha=-8.17% MDD=11.0% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_B_take_profit_20260420_023246.json` | `partial_take_level_2 1.75->1.5` | fit=21.018 WR=69.7% alpha=-8.24% MDD=11.0% |
| 2026-04-20 | codex | `local_pattern_search_result_20260420_0612_partialtake_cooldown_regime.json` | `partial_take_level 0.75->0.5; consecutive_loss_cooldown 8.0->9.0; regime_threshold 0.3->0.25` | fit=23.817 WR=70.0% alpha=-8.23% MDD=10.3% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_C_timing_20260420_043330.json` | `consecutive_loss_cooldown 7->8` | fit=20.196 WR=69.7% alpha=-8.26% MDD=10.8% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260420_054005.json` | `momentum_confirm_days 5->4` | fit=22.220 WR=70.0% alpha=-8.15% MDD=10.6% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260420_082054.json` | `equity_drawdown_stop_pct 0.18->0.2` | fit=25.521 WR=70.0% alpha=-8.04% MDD=10.6% |
| 2026-04-20 | codex | `local_fullmetric_sweep_result_20260420_085552_takeprofit_anchor_recovery.json` | `partial_take_pct 0.2->0.1` | fit=23.586 WR=70.4% alpha=-8.22% MDD=11.0% |
| 2026-04-20 | codex | `local_fullmetric_sweep_result_20260420_0918_postpartialtake_repair.json` | `consecutive_loss_cooldown 9.0->10.0` | fit=23.636 WR=70.8% alpha=-8.23% MDD=11.0% |

## Graveyard (genes que falharam multiplas vezes)

Genes tentados >= 2 vezes sem nenhuma promocao. Considere SKIP no proximo ciclo.
(Colunas `codex` contam tentativas importadas do log do GPT codex — dedup por sweep file.)

| Gene | Attempts | Promotions | (codex) Attempts | (codex) Promotions |
|---|---:|---:|---:|---:|
| `reward_risk_ratio` | 11 | 0 | 1 | 0 |
| `partial_take_pct_2` | 10 | 0 | 1 | 0 |
| `stop_atr_mult` | 8 | 0 | 0 | 0 |
| `score_percentile_trigger` | 8 | 0 | 1 | 0 |
| `volume_confirm_mode` | 7 | 0 | 1 | 0 |
| `vote_threshold_short` | 7 | 0 | 1 | 0 |
| `signal_ema_span` | 7 | 0 | 1 | 0 |
| `score_strength_scaling` | 7 | 0 | 0 | 0 |
| `stop_tighten_after_bars` | 6 | 0 | 0 | 0 |
| `time_stop_bars` | 6 | 0 | 0 | 0 |
| `entry_discount_atr_frac` | 6 | 0 | 0 | 0 |
| `min_signal_strength` | 6 | 0 | 0 | 0 |
| `vote_threshold_long` | 6 | 0 | 0 | 0 |
| `z_threshold` | 6 | 0 | 0 | 0 |
| `entry_confirmation_days` | 6 | 0 | 0 | 0 |
| `ma_filter_period` | 5 | 0 | 1 | 0 |

## Hot zones (genes que ja promoveram)

Genes onde alguma combinacao funcionou. Vale revisitar com novos vizinhos.

| Gene | Promotions | Attempts | Hit rate | (codex) Promotions |
|---|---:|---:|---:|---:|
| `consecutive_loss_cooldown` | 4 | 14 | 29% | 2 |
| `regime_threshold` | 4 | 17 | 24% | 2 |
| `partial_take_pct` | 2 | 10 | 20% | 1 |
| `momentum_confirm_days` | 2 | 10 | 20% | 1 |
| `trailing_stop_mode` | 1 | 7 | 14% | 0 |
| `max_loss_per_trade_pct` | 1 | 7 | 14% | 0 |
| `equity_drawdown_stop_pct` | 1 | 7 | 14% | 0 |
| `vol_regime_mode` | 1 | 8 | 12% | 0 |
| `stop_tighten_factor` | 1 | 10 | 10% | 0 |
| `partial_take_level_2` | 1 | 11 | 9% | 0 |
| `ma_filter_mode` | 1 | 11 | 9% | 0 |
| `volatility_filter_percentile` | 1 | 12 | 8% | 0 |
| `entry_score_threshold` | 1 | 13 | 8% | 0 |
| `partial_take_level` | 1 | 16 | 6% | 1 |

## Category rotation status

| Category | Last swept | Runs | Promotions |
|---|---|---:|---:|
| A_risk_management | 2026-04-20T08:24:30+00:00 | 5 | 3 |
| B_take_profit | 2026-04-20T02:36:23+00:00 | 6 | 1 |
| C_timing | 2026-04-20T04:36:50+00:00 | 4 | 2 |
| D_entry_filter | 2026-04-20T01:36:41+00:00 | 5 | 1 |
| E_regime_vol | 2026-04-20T05:42:53+00:00 | 4 | 3 |
| F_trailing | 2026-04-20T07:43:17+00:00 | 4 | 0 |

## Codex sync history (latest attempts)

Parsed `codex_attempts_source.md` at 2026-04-20T08:20:54+00:00. Total attempts: **23**, promoted: **11**. Updates via `python analysis/codex_attempts_sync.py` at the start of every continuous cycle (auto) or on demand.

| Attempt | Timestamp | Promoted | Move(s) | Learning (snippet) |
|---|---|:-:|---|---|
| #36 | 2026-04-20T09:19:32+02:00 | YES | `consecutive_loss_cooldown 9.0->10.0` | Attempt 34 materially changed the frontier. Once `partial_take_pct` moved to `0.10`, the system could absorb one more co... |
| #35 | 2026-04-20T09:09:41+02:00 | no | `vote_threshold_short 0.15->0.1550698474225143; partial_take_pct_2 0.3->0.2982570491636789` | the first post-Attempt-33 staged refine slice did not open a new frontier. On windows `[1, 10, 19, 28]`, stage 2 mostly ... |
| #34 | 2026-04-20T08:55:59+02:00 | YES | `partial_take_pct 0.2->0.1` | the B-side search space is not globally dead; it was only dead under the older frontiers. On the current `partial_take_l... |
| #33 | 2026-04-20T07:08:00+02:00 | no | `regime_threshold 0.25->0.3` | the strict-regime branch is now locally exhausted: `regime_threshold: 0.25 -> 0.30` remains the best tail-risk shape on ... |
| #32 | 2026-04-20T06:54:31+02:00 | no | `regime_threshold 0.25->0.3` | the current frontier now has two clear but incomplete branches: `regime_threshold: 0.25 -> 0.30` repairs tail risk (`MDD... |

---

_Run again with: `python analysis/sweep_learnings.py`_
