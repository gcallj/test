# Sweep Learnings Summary

_Generated: 2026-04-19T20:57:24+02:00_

Aggregates all `local_fullmetric_sweep_result*.json` files plus the
continuous-improvement state. Use this file to plan the next cycle.

## Current snapshot

- Checkpoint fitness: `21.20513393040993`
- Total sweeps seen: **37**
- Total promotions: **14** (including 0 from codex log)
- Promotion rate: **27.5%** (over 51 total attempts)
- Continuous cycles run: 17
- Continuous promotions: 4
- Codex attempts synced: **14** (6 promoted) from `optimization_attempts_20260416.md`

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

## Graveyard (genes que falharam multiplas vezes)

Genes tentados >= 2 vezes sem nenhuma promocao. Considere SKIP no proximo ciclo.
(Colunas `codex` contam tentativas importadas do log do GPT codex — dedup por sweep file.)

| Gene | Attempts | Promotions | (codex) Attempts | (codex) Promotions |
|---|---:|---:|---:|---:|
| `partial_take_level_2` | 10 | 0 | 0 | 0 |
| `reward_risk_ratio` | 9 | 0 | 0 | 0 |
| `vote_threshold_long` | 9 | 0 | 0 | 0 |
| `signal_ema_span` | 9 | 0 | 0 | 0 |
| `score_strength_scaling` | 9 | 0 | 0 | 0 |
| `partial_take_pct_2` | 8 | 0 | 0 | 0 |
| `score_percentile_trigger` | 8 | 0 | 0 | 0 |
| `min_signal_strength` | 8 | 0 | 0 | 0 |
| `stop_atr_mult` | 7 | 0 | 0 | 0 |
| `time_stop_bars` | 7 | 0 | 0 | 0 |
| `momentum_confirm_days` | 7 | 0 | 1 | 0 |
| `entry_confirmation_days` | 7 | 0 | 0 | 0 |
| `max_loss_per_trade_pct` | 6 | 0 | 0 | 0 |
| `entry_discount_atr_frac` | 6 | 0 | 0 | 0 |
| `volume_confirm_mode` | 6 | 0 | 1 | 0 |
| `ma_filter_period` | 6 | 0 | 1 | 0 |
| `stop_tighten_after_bars` | 5 | 0 | 0 | 0 |
| `equity_drawdown_stop_pct` | 5 | 0 | 0 | 0 |
| `vote_threshold_short` | 5 | 0 | 0 | 0 |
| `z_threshold` | 5 | 0 | 0 | 0 |

## Hot zones (genes que ja promoveram)

Genes onde alguma combinacao funcionou. Vale revisitar com novos vizinhos.

| Gene | Promotions | Attempts | Hit rate | (codex) Promotions |
|---|---:|---:|---:|---:|
| `entry_score_threshold` | 3 | 11 | 27% | 0 |
| `regime_threshold` | 3 | 13 | 23% | 0 |
| `consecutive_loss_cooldown` | 2 | 9 | 22% | 0 |
| `vol_regime_mode` | 1 | 5 | 20% | 0 |
| `stop_tighten_factor` | 1 | 6 | 17% | 0 |
| `trailing_stop_mode` | 1 | 7 | 14% | 0 |
| `ma_filter_mode` | 1 | 7 | 14% | 0 |
| `partial_take_pct` | 1 | 8 | 12% | 0 |
| `partial_take_level` | 1 | 10 | 10% | 0 |
| `volatility_filter_percentile` | 1 | 10 | 10% | 0 |

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

Parsed `optimization_attempts_20260416.md` at 2026-04-19T20:57:24+02:00. Total attempts: **14**, promoted: **6**. Updates via `python analysis/codex_attempts_sync.py` at the start of every continuous cycle (auto) or on demand.

| Attempt | Timestamp | Promoted | Move(s) | Learning (snippet) |
|---|---|:-:|---|---|
| #27 | 2026-04-19T18:54:45Z | YES | `(no moves)` | `consecutive_loss_cooldown` still has one more safe step beyond the prior `6 -> 7` promotion: `7 -> 8` adds one more `>=... |
| #26 | 2026-04-19T16:54:22Z | no | `(no moves)` | `momentum_confirm_days: 5 -> 4` is the strongest uncovered E-category direction: it improves `WR_target`, alpha, and dra... |
| #11 | 2026-04-18T11:50 | YES | `volatility_filter_percentile 0.0->0.05` | `volatility_filter_percentile` appears to be another high-leverage operational safety knob: a tiny filter (`+0.05`) impr... |
| #10 | 2026-04-18T09:50 | YES | `vol_regime_mode 1.0->2.0` | `vol_regime_mode` is a high-leverage operational safety knob: skipping high-vol regimes can simultaneously reduce drawdo... |
| #9 | 2026-04-18T07:50 | YES | `regime_threshold 0.3->0.25` | the regime-gating threshold still has a clean local improvement left: a slightly looser regime threshold reduced false p... |

---

_Run again with: `python analysis/sweep_learnings.py`_
