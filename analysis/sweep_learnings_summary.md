# Sweep Learnings Summary

_Generated: 2026-04-19T02:37:40+00:00_

Aggregates all `local_fullmetric_sweep_result*.json` files plus the
continuous-improvement state. Use this file to plan the next cycle.

## Current snapshot

- Checkpoint fitness: `20.717201265906507`
- Total sweeps seen: **15**
- Total promotions: **7** (including 2 from codex log)
- Promotion rate: **24.1%** (over 29 total attempts)
- Continuous cycles run: 8
- Continuous promotions: 0
- Codex attempts synced: **14** (7 promoted) from `optimization_attempts_20260416.md`

## Next suggested sweep

categoria mais antiga: **B_take_profit** (ultima vez: 2026-04-18T19:15:37+00:00) -> `python run_continuous_improvement.py --category B`

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

## Graveyard (genes que falharam multiplas vezes)

Genes tentados >= 2 vezes sem nenhuma promocao. Considere SKIP no proximo ciclo.
(Colunas `codex` contam tentativas importadas do log do GPT codex — dedup por sweep file.)

| Gene | Attempts | Promotions | (codex) Attempts | (codex) Promotions |
|---|---:|---:|---:|---:|
| `stop_atr_mult` | 5 | 0 | 0 | 0 |
| `reward_risk_ratio` | 5 | 0 | 1 | 0 |
| `partial_take_level_2` | 5 | 0 | 1 | 0 |
| `stop_tighten_factor` | 4 | 0 | 0 | 0 |
| `partial_take_pct_2` | 4 | 0 | 1 | 0 |
| `max_loss_per_trade_pct` | 4 | 0 | 0 | 0 |
| `equity_drawdown_stop_pct` | 4 | 0 | 0 | 0 |
| `score_percentile_trigger` | 4 | 0 | 0 | 0 |
| `entry_discount_atr_frac` | 4 | 0 | 1 | 0 |
| `score_strength_scaling` | 4 | 0 | 1 | 0 |
| `stop_tighten_after_bars` | 3 | 0 | 0 | 0 |
| `time_stop_bars` | 3 | 0 | 0 | 0 |
| `min_signal_strength` | 3 | 0 | 0 | 0 |
| `volume_confirm_mode` | 3 | 0 | 1 | 0 |
| `momentum_confirm_days` | 3 | 0 | 1 | 0 |
| `vote_threshold_long` | 3 | 0 | 0 | 0 |
| `vote_threshold_short` | 3 | 0 | 0 | 0 |
| `z_threshold` | 3 | 0 | 0 | 0 |
| `signal_ema_span` | 3 | 0 | 0 | 0 |
| `entry_confirmation_days` | 3 | 0 | 0 | 0 |

## Hot zones (genes que ja promoveram)

Genes onde alguma combinacao funcionou. Vale revisitar com novos vizinhos.

| Gene | Promotions | Attempts | Hit rate | (codex) Promotions |
|---|---:|---:|---:|---:|
| `vol_regime_mode` | 1 | 2 | 50% | 0 |
| `partial_take_pct` | 1 | 4 | 25% | 0 |
| `entry_score_threshold` | 1 | 4 | 25% | 1 |
| `regime_threshold` | 1 | 4 | 25% | 0 |
| `volatility_filter_percentile` | 1 | 4 | 25% | 0 |
| `consecutive_loss_cooldown` | 1 | 4 | 25% | 1 |
| `partial_take_level` | 1 | 5 | 20% | 1 |
| `trailing_stop_mode` | 1 | 5 | 20% | 0 |

## Category rotation status

| Category | Last swept | Runs | Promotions |
|---|---|---:|---:|
| A_risk_management | 2026-04-19T02:37:22+00:00 | 2 | 0 |
| B_take_profit | 2026-04-18T19:15:37+00:00 | 1 | 0 |
| C_timing | 2026-04-18T20:39:53+00:00 | 1 | 0 |
| D_entry_filter | 2026-04-19T01:44:00+00:00 | 2 | 0 |
| E_regime_vol | 2026-04-18T21:31:21+00:00 | 1 | 0 |
| F_trailing | 2026-04-18T22:36:36+00:00 | 1 | 0 |

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
