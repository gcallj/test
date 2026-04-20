# Sweep Learnings Summary

_Generated: 2026-04-20T16:30:40+00:00_

Aggregates all `local_fullmetric_sweep_result*.json` files plus the
continuous-improvement state. Use this file to plan the next cycle.

## Current snapshot

- Checkpoint fitness: `22.714907200870954`
- Total sweeps seen: **64**
- Total promotions: **28** (including 1 from codex log)
- Promotion rate: **31.8%** (over 88 total attempts)
- Continuous cycles run: 32
- Continuous promotions: 13
- Codex attempts synced: **24** (11 promoted) from `codex_attempts_source.md`

## Next suggested sweep

categoria mais antiga: **F_trailing** (ultima vez: 2026-04-20T07:43:17+00:00) -> `python run_continuous_improvement.py --category F`

## Promotion timeline (chronological)

| Date | Source | Sweep file | Move | Resulting metrics |
|---|---|---|---|---|
| 2026-04-18 | local | `local_fullmetric_sweep_result.json` | `partial_take_pct 0.30000000000000004->0.2` | fit=8.168 WR=67.1% alpha=-8.29% MDD=12.6% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_055831.json` | `trailing_stop_mode 2->1` | fit=8.857 WR=67.8% alpha=-8.29% MDD=12.6% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_075045.json` | `regime_threshold 0.30000000000000004->0.25` | fit=8.720 WR=68.2% alpha=-8.26% MDD=12.6% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_095044.json` | `vol_regime_mode 1->2` | fit=19.398 WR=69.0% alpha=-8.24% MDD=11.7% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_115025.json` | `volatility_filter_percentile 0.0->0.05` | fit=20.717 WR=69.7% alpha=-8.27% MDD=11.6% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_1620_alphaexit.json` | `partial_take_level 0.75->1.0` | fit=21.021 WR=70.0% alpha=-8.24% MDD=11.2% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_180019_volregime.json` | `regime_threshold 0.25->0.30000000000000004` | fit=21.152 WR=70.0% alpha=-8.26% MDD=11.2% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_195309_entryquality.json` | `entry_score_threshold 0.2->0.25` | fit=21.248 WR=70.0% alpha=-8.27% MDD=11.2% |
| 2026-04-18 | local | `local_fullmetric_sweep_result_20260418_215034_hitrate_combo.json` | `entry_score_threshold 0.25->0.30000000000000004; regime_threshold 0.30000000000000004->0.25` | fit=21.576 WR=70.0% alpha=-8.26% MDD=11.2% |
| 2026-04-19 | local | `local_fullmetric_sweep_continuous_C_timing_20260419_093647.json` | `consecutive_loss_cooldown 6->7` | fit=19.758 WR=70.0% alpha=-8.31% MDD=10.7% |
| 2026-04-19 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260419_104733.json` | `ma_filter_mode 1->0` | fit=19.835 WR=69.7% alpha=-8.30% MDD=11.2% |
| 2026-04-19 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260419_122638.json` | `stop_tighten_factor 0.45->0.4` | fit=21.014 WR=69.7% alpha=-8.29% MDD=10.4% |
| 2026-04-19 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260419_151906.json` | `entry_score_threshold 0.15000000000000002->0.2` | fit=19.694 WR=69.7% alpha=-8.33% MDD=11.2% |
| 2026-04-19 | local | `local_fullmetric_sweep_result_20260419_2047_momentum_wrrestore.json` | `consecutive_loss_cooldown 7->8` | fit=21.205 WR=69.7% alpha=-8.25% MDD=11.3% |
| 2026-04-19 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260419_212524.json` | `regime_threshold 0.25->0.2` | fit=21.907 WR=69.9% alpha=-8.28% MDD=11.2% |
| 2026-04-19 | local | `local_fullmetric_sweep_result_20260419_2248_entry_momentum_anchor.json` | `momentum_confirm_days 5->4` | fit=23.529 WR=70.0% alpha=-8.15% MDD=11.0% |
| 2026-04-19 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260419_235131.json` | `max_loss_per_trade_pct 0.08->0.09` | fit=20.967 WR=69.7% alpha=-8.27% MDD=10.9% |
| 2026-04-20 | local | `local_fullmetric_sweep_result_20260420_001_hotzone_rebalance.json` | `regime_threshold 0.25->0.30000000000000004` | fit=23.924 WR=70.0% alpha=-8.17% MDD=11.0% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_B_take_profit_20260420_023246.json` | `partial_take_level_2 1.75->1.5` | fit=21.018 WR=69.7% alpha=-8.24% MDD=11.0% |
| 2026-04-20 | codex | `local_pattern_search_result_20260420_0612_partialtake_cooldown_regime.json` | `partial_take_level 0.75->0.5; consecutive_loss_cooldown 8.0->9.0; regime_threshold 0.3->0.25` | fit=23.817 WR=70.0% alpha=-8.23% MDD=10.3% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_C_timing_20260420_043330.json` | `consecutive_loss_cooldown 7->8` | fit=20.196 WR=69.7% alpha=-8.26% MDD=10.8% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260420_054005.json` | `momentum_confirm_days 5->4` | fit=22.220 WR=70.0% alpha=-8.15% MDD=10.6% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260420_082054.json` | `equity_drawdown_stop_pct 0.18->0.2` | fit=25.521 WR=70.0% alpha=-8.04% MDD=10.6% |
| 2026-04-20 | local | `local_fullmetric_sweep_result_20260420_085552_takeprofit_anchor_recovery.json` | `partial_take_pct 0.2->0.1` | fit=23.586 WR=70.4% alpha=-8.22% MDD=11.0% |
| 2026-04-20 | local | `local_fullmetric_sweep_result_20260420_0918_postpartialtake_repair.json` | `consecutive_loss_cooldown 9->10` | fit=23.636 WR=70.7% alpha=-8.23% MDD=11.0% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260420_102407.json` | `score_percentile_trigger 0.6000000000000001->0.55` | fit=23.898 WR=69.8% alpha=-8.13% MDD=10.6% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_B_take_profit_20260420_113233.json` | `partial_take_pct 0.2->0.1` | fit=24.174 WR=70.0% alpha=-8.13% MDD=11.1% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260420_162649.json` | `momentum_confirm_days 4->3` | fit=22.715 WR=70.2% alpha=-8.26% MDD=10.6% |

## Graveyard (genes que falharam multiplas vezes)

Genes tentados >= 2 vezes sem nenhuma promocao. Considere SKIP no proximo ciclo.
(Colunas `codex` contam tentativas importadas do log do GPT codex — dedup por sweep file.)

| Gene | Attempts | Promotions | (codex) Attempts | (codex) Promotions |
|---|---:|---:|---:|---:|
| `reward_risk_ratio` | 13 | 0 | 0 | 0 |
| `partial_take_pct_2` | 12 | 0 | 1 | 0 |
| `signal_ema_span` | 12 | 0 | 0 | 0 |
| `vote_threshold_long` | 11 | 0 | 0 | 0 |
| `score_strength_scaling` | 11 | 0 | 0 | 0 |
| `time_stop_bars` | 10 | 0 | 0 | 0 |
| `min_signal_strength` | 10 | 0 | 0 | 0 |
| `stop_atr_mult` | 9 | 0 | 0 | 0 |
| `volume_confirm_mode` | 9 | 0 | 1 | 0 |
| `entry_confirmation_days` | 9 | 0 | 0 | 0 |
| `ma_filter_period` | 9 | 0 | 1 | 0 |
| `entry_discount_atr_frac` | 8 | 0 | 0 | 0 |
| `vote_threshold_short` | 8 | 0 | 1 | 0 |
| `stop_tighten_after_bars` | 7 | 0 | 0 | 0 |
| `z_threshold` | 7 | 0 | 0 | 0 |

## Hot zones (genes que ja promoveram)

Genes onde alguma combinacao funcionou. Vale revisitar com novos vizinhos.

| Gene | Promotions | Attempts | Hit rate | (codex) Promotions |
|---|---:|---:|---:|---:|
| `regime_threshold` | 6 | 29 | 21% | 1 |
| `consecutive_loss_cooldown` | 5 | 21 | 24% | 1 |
| `partial_take_pct` | 3 | 13 | 23% | 0 |
| `momentum_confirm_days` | 3 | 14 | 21% | 0 |
| `entry_score_threshold` | 3 | 22 | 14% | 0 |
| `partial_take_level` | 2 | 20 | 10% | 1 |
| `equity_drawdown_stop_pct` | 1 | 8 | 12% | 0 |
| `trailing_stop_mode` | 1 | 9 | 11% | 0 |
| `max_loss_per_trade_pct` | 1 | 11 | 9% | 0 |
| `vol_regime_mode` | 1 | 11 | 9% | 0 |
| `score_percentile_trigger` | 1 | 13 | 8% | 0 |
| `stop_tighten_factor` | 1 | 14 | 7% | 0 |
| `partial_take_level_2` | 1 | 15 | 7% | 0 |
| `volatility_filter_percentile` | 1 | 19 | 5% | 0 |
| `ma_filter_mode` | 1 | 19 | 5% | 0 |

## Category rotation status

| Category | Last swept | Runs | Promotions |
|---|---|---:|---:|
| A_risk_management | 2026-04-20T08:24:30+00:00 | 5 | 3 |
| B_take_profit | 2026-04-20T11:36:36+00:00 | 7 | 2 |
| C_timing | 2026-04-20T14:16:03+00:00 | 5 | 2 |
| D_entry_filter | 2026-04-20T10:29:24+00:00 | 6 | 2 |
| E_regime_vol | 2026-04-20T16:30:07+00:00 | 5 | 4 |
| F_trailing | 2026-04-20T07:43:17+00:00 | 4 | 0 |

## Codex sync history (latest attempts)

Parsed `codex_attempts_source.md` at 2026-04-20T11:32:33+00:00. Total attempts: **24**, promoted: **11**. Updates via `python analysis/codex_attempts_sync.py` at the start of every continuous cycle (auto) or on demand.

| Attempt | Timestamp | Promoted | Move(s) | Learning (snippet) |
|---|---|:-:|---|---|
| #37 | 2026-04-20T10:56:55+02:00 | no | `regime_threshold 0.25->0.2` | the main problem is no longer only local tail-risk repair. The refreshed `git:main` now beats the current local checkpoi... |
| #36 | 2026-04-20T09:19:32+02:00 | YES | `consecutive_loss_cooldown 9.0->10.0` | Attempt 34 materially changed the frontier. Once `partial_take_pct` moved to `0.10`, the system could absorb one more co... |
| #35 | 2026-04-20T09:09:41+02:00 | no | `vote_threshold_short 0.15->0.1550698474225143; partial_take_pct_2 0.3->0.2982570491636789` | the first post-Attempt-33 staged refine slice did not open a new frontier. On windows `[1, 10, 19, 28]`, stage 2 mostly ... |
| #34 | 2026-04-20T08:55:59+02:00 | YES | `partial_take_pct 0.2->0.1` | the B-side search space is not globally dead; it was only dead under the older frontiers. On the current `partial_take_l... |
| #33 | 2026-04-20T07:08:00+02:00 | no | `regime_threshold 0.25->0.3` | the strict-regime branch is now locally exhausted: `regime_threshold: 0.25 -> 0.30` remains the best tail-risk shape on ... |

---

_Run again with: `python analysis/sweep_learnings.py`_
