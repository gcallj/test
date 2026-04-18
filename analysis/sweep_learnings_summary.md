# Sweep Learnings Summary

_Generated: 2026-04-18T14:38:03+02:00_

Aggregates all `local_fullmetric_sweep_result*.json` files plus the
continuous-improvement state. Use this file to plan the next cycle.

## Current snapshot

- Checkpoint fitness: `20.717201265906507`
- Total sweeps seen: **7**
- Total promotions: **5**
- Promotion rate: **71.4%**
- Continuous cycles run: 1
- Continuous promotions: 0

## Next suggested sweep

categoria nao explorada: **A_risk_management** (rodar `python run_continuous_improvement.py --category A`)

## Promotion timeline (chronological)

| Date | Sweep file | Move | Resulting metrics |
|---|---|---|---|
| 2026-04-18 | `local_fullmetric_sweep_result.json` | `partial_take_pct 0.30000000000000004->0.2` | fit=8.168 WR=67.1% alpha=-8.29% MDD=12.6% |
| 2026-04-18 | `local_fullmetric_sweep_result_20260418_055831.json` | `trailing_stop_mode 2->1` | fit=8.857 WR=67.8% alpha=-8.29% MDD=12.6% |
| 2026-04-18 | `local_fullmetric_sweep_result_20260418_075045.json` | `regime_threshold 0.30000000000000004->0.25` | fit=8.720 WR=68.2% alpha=-8.26% MDD=12.6% |
| 2026-04-18 | `local_fullmetric_sweep_result_20260418_095044.json` | `vol_regime_mode 1->2` | fit=19.398 WR=69.0% alpha=-8.24% MDD=11.7% |
| 2026-04-18 | `local_fullmetric_sweep_result_20260418_115025.json` | `volatility_filter_percentile 0.0->0.05` | fit=20.717 WR=69.7% alpha=-8.27% MDD=11.6% |

## Graveyard (genes que falharam multiplas vezes)

Genes tentados >= 2 vezes sem nenhuma promocao. Considere SKIP no proximo ciclo.

| Gene | Attempts | Promotions |
|---|---:|---:|
| `stop_atr_mult` | 3 | 0 |
| `score_percentile_trigger` | 3 | 0 |
| `entry_score_threshold` | 3 | 0 |
| `score_strength_scaling` | 3 | 0 |
| `consecutive_loss_cooldown` | 3 | 0 |
| `stop_tighten_factor` | 2 | 0 |
| `time_stop_bars` | 2 | 0 |
| `reward_risk_ratio` | 2 | 0 |
| `partial_take_level` | 2 | 0 |
| `partial_take_level_2` | 2 | 0 |
| `max_loss_per_trade_pct` | 2 | 0 |
| `equity_drawdown_stop_pct` | 2 | 0 |
| `entry_discount_atr_frac` | 2 | 0 |
| `min_signal_strength` | 2 | 0 |
| `volume_confirm_mode` | 2 | 0 |
| `momentum_confirm_days` | 2 | 0 |
| `vote_threshold_long` | 2 | 0 |
| `vote_threshold_short` | 2 | 0 |
| `z_threshold` | 2 | 0 |
| `signal_ema_span` | 2 | 0 |

## Hot zones (genes que ja promoveram)

Genes onde alguma combinacao funcionou. Vale revisitar com novos vizinhos.

| Gene | Promotions | Attempts | Hit rate |
|---|---:|---:|---:|
| `partial_take_pct` | 1 | 1 | 100% |
| `vol_regime_mode` | 1 | 1 | 100% |
| `trailing_stop_mode` | 1 | 3 | 33% |
| `regime_threshold` | 1 | 3 | 33% |
| `volatility_filter_percentile` | 1 | 3 | 33% |

## Category rotation status

| Category | Last swept | Runs | Promotions |
|---|---|---:|---:|
| A_risk_management | never | 0 | 0 |
| B_take_profit | never | 0 | 0 |
| C_timing | never | 0 | 0 |
| D_entry_filter | 2026-04-18T14:37:35+02:00 | 1 | 0 |
| E_regime_vol | never | 0 | 0 |
| F_trailing | never | 0 | 0 |

---

_Run again with: `python analysis/sweep_learnings.py`_
