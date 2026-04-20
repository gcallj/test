# Sweep Learnings Summary

_Generated: 2026-04-20T15:57:13+02:00_

Aggregates all `local_fullmetric_sweep_result*.json` files plus the
continuous-improvement state. Use this file to plan the next cycle.

## Current snapshot

- Checkpoint fitness: `23.63581384973892`
- Total sweeps seen: **49**
- Total promotions: **19** (including 1 from codex log)
- Promotion rate: **25.0%** (over 76 total attempts)
- Continuous cycles run: 17
- Continuous promotions: 4
- Codex attempts synced: **27** (11 promoted) from `optimization_attempts_20260416.md`

## Next suggested sweep

categoria mais antiga: **B_take_profit** (ultima vez: 2026-04-19T08:12:23+00:00) -> `python run_continuous_improvement.py --category B`

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
| 2026-04-19 | local | `local_fullmetric_sweep_result_20260419_2248_entry_momentum_anchor.json` | `momentum_confirm_days 5->4` | fit=23.529 WR=70.0% alpha=-8.15% MDD=11.0% |
| 2026-04-20 | local | `local_fullmetric_sweep_result_20260420_001_hotzone_rebalance.json` | `regime_threshold 0.25->0.30000000000000004` | fit=23.924 WR=70.0% alpha=-8.17% MDD=11.0% |
| 2026-04-20 | codex | `local_pattern_search_result_20260420_0612_partialtake_cooldown_regime.json` | `partial_take_level 0.75->0.5; consecutive_loss_cooldown 8.0->9.0; regime_threshold 0.3->0.25` | fit=23.817 WR=70.0% alpha=-8.23% MDD=10.3% |
| 2026-04-20 | local | `local_fullmetric_sweep_result_20260420_085552_takeprofit_anchor_recovery.json` | `partial_take_pct 0.2->0.1` | fit=23.586 WR=70.4% alpha=-8.22% MDD=11.0% |
| 2026-04-20 | local | `local_fullmetric_sweep_result_20260420_0918_postpartialtake_repair.json` | `consecutive_loss_cooldown 9->10` | fit=23.636 WR=70.7% alpha=-8.23% MDD=11.0% |

## Graveyard (genes que falharam multiplas vezes)

Genes tentados >= 2 vezes sem nenhuma promocao. Considere SKIP no proximo ciclo.
(Colunas `codex` contam tentativas importadas do log do GPT codex — dedup por sweep file.)

| Gene | Attempts | Promotions | (codex) Attempts | (codex) Promotions |
|---|---:|---:|---:|---:|
| `partial_take_level_2` | 12 | 0 | 2 | 0 |
| `score_percentile_trigger` | 11 | 0 | 1 | 0 |
| `reward_risk_ratio` | 10 | 0 | 0 | 0 |
| `max_loss_per_trade_pct` | 10 | 0 | 3 | 0 |
| `signal_ema_span` | 10 | 0 | 0 | 0 |
| `partial_take_pct_2` | 9 | 0 | 1 | 0 |
| `vote_threshold_long` | 9 | 0 | 0 | 0 |
| `score_strength_scaling` | 9 | 0 | 0 | 0 |
| `min_signal_strength` | 8 | 0 | 0 | 0 |
| `stop_atr_mult` | 7 | 0 | 0 | 0 |
| `time_stop_bars` | 7 | 0 | 0 | 0 |
| `entry_confirmation_days` | 7 | 0 | 0 | 0 |
| `equity_drawdown_stop_pct` | 6 | 0 | 1 | 0 |
| `entry_discount_atr_frac` | 6 | 0 | 0 | 0 |
| `volume_confirm_mode` | 6 | 0 | 1 | 0 |
| `vote_threshold_short` | 6 | 0 | 1 | 0 |
| `ma_filter_period` | 6 | 0 | 1 | 0 |
| `stop_tighten_after_bars` | 5 | 0 | 0 | 0 |
| `z_threshold` | 5 | 0 | 0 | 0 |

## Hot zones (genes que ja promoveram)

Genes onde alguma combinacao funcionou. Vale revisitar com novos vizinhos.

| Gene | Promotions | Attempts | Hit rate | (codex) Promotions |
|---|---:|---:|---:|---:|
| `regime_threshold` | 5 | 26 | 19% | 1 |
| `consecutive_loss_cooldown` | 4 | 18 | 22% | 1 |
| `entry_score_threshold` | 3 | 20 | 15% | 0 |
| `partial_take_pct` | 2 | 10 | 20% | 0 |
| `partial_take_level` | 2 | 17 | 12% | 1 |
| `trailing_stop_mode` | 1 | 7 | 14% | 0 |
| `vol_regime_mode` | 1 | 8 | 12% | 0 |
| `momentum_confirm_days` | 1 | 11 | 9% | 0 |
| `stop_tighten_factor` | 1 | 12 | 8% | 0 |
| `volatility_filter_percentile` | 1 | 16 | 6% | 0 |
| `ma_filter_mode` | 1 | 16 | 6% | 0 |

## Category rotation status

| Category | Last swept | Runs | Promotions |
|---|---|---:|---:|
| A_risk_management | 2026-04-19T12:29:52+00:00 | 3 | 1 |
| B_take_profit | 2026-04-19T08:12:23+00:00 | 4 | 0 |
| C_timing | 2026-04-19T09:39:50+00:00 | 2 | 1 |
| D_entry_filter | 2026-04-19T15:25:06+00:00 | 4 | 1 |
| E_regime_vol | 2026-04-19T10:51:45+00:00 | 2 | 1 |
| F_trailing | 2026-04-19T11:29:21+00:00 | 2 | 0 |

## Codex sync history (latest attempts)

Parsed `optimization_attempts_20260416.md` at 2026-04-20T15:57:12+02:00. Total attempts: **27**, promoted: **11**. Updates via `python analysis/codex_attempts_sync.py` at the start of every continuous cycle (auto) or on demand.

| Attempt | Timestamp | Promoted | Move(s) | Learning (snippet) |
|---|---|:-:|---|---|
| #39 | 2026-04-20T15:53:41+02:00 | no | `max_loss_per_trade_pct 0.08->0.09` | the current local-vs-main bridge is now exhausted at the small-pattern level. Main-like take-profit/risk settings can re... |
| #39 | 2026-04-20T15:07:48+02:00 | no | `max_loss_per_trade_pct 0.08->0.09` | the `max_loss_per_trade_pct:+1` anchor is now locally exhausted under the refreshed `git:main` comparator. Pairing it wi... |
| #38 | 2026-04-20T12:59:59+02:00 | no | `max_loss_per_trade_pct 0.08->0.09` | the refreshed frontier is now explicitly split: `git:main` is the alpha/fit leader and the local checkpoint is the WR le... |
| #37 | 2026-04-20T10:56:55+02:00 | no | `regime_threshold 0.25->0.2` | the main problem is no longer only local tail-risk repair. The refreshed `git:main` now beats the current local checkpoi... |
| #36 | 2026-04-20T09:19:32+02:00 | YES | `consecutive_loss_cooldown 9.0->10.0` | Attempt 34 materially changed the frontier. Once `partial_take_pct` moved to `0.10`, the system could absorb one more co... |

---

_Run again with: `python analysis/sweep_learnings.py`_
