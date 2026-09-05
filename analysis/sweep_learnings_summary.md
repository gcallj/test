# Sweep Learnings Summary

_Generated: 2026-09-05T04:48:11+00:00_

Aggregates all `local_fullmetric_sweep_result*.json` files plus the
continuous-improvement state. Use this file to plan the next cycle.

## Current snapshot

- Checkpoint fitness: `26.459605239382256`
- Total sweeps seen: **1738**
- Total promotions: **121** (including 1 from codex log)
- Promotion rate: **6.9%** (over 1762 total attempts)
- Continuous cycles run: 34
- Continuous promotions: 14
- Codex attempts synced: **24** (11 promoted) from `codex_attempts_source.md`

## Next suggested sweep

categoria mais antiga: **D_entry_filter** (ultima vez: 2026-04-20T10:29:24+00:00) -> `python run_continuous_improvement.py --category D`

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
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260420_193942.json` | `max_loss_per_trade_pct 0.09->0.1` | fit=24.037 WR=70.4% alpha=-8.20% MDD=10.5% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260420_205608.json` | `momentum_confirm_days 3->4` | fit=24.174 WR=69.8% alpha=-8.16% MDD=11.0% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260420_205610.json` | `equity_drawdown_stop_pct 0.2->0.22` | fit=23.864 WR=69.7% alpha=-8.23% MDD=11.1% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_B_take_profit_20260420_205620.json` | `partial_take_pct_2 0.30000000000000004->0.4` | fit=23.735 WR=69.7% alpha=-8.13% MDD=11.1% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260420_205625.json` | `signal_ema_span 8->7` | fit=23.590 WR=70.0% alpha=-8.21% MDD=11.0% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260420_221658.json` | `equity_drawdown_stop_pct 0.2->0.22` | fit=24.404 WR=69.7% alpha=-8.16% MDD=11.0% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260420_221658.json` | `score_percentile_trigger 0.55->0.5` | fit=23.744 WR=69.7% alpha=-8.15% MDD=10.7% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260420_221713.json` | `momentum_confirm_days 3->4` | fit=24.469 WR=70.0% alpha=-8.09% MDD=10.8% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_C_timing_20260420_221716.json` | `consecutive_loss_cooldown 8->9` | fit=23.707 WR=69.7% alpha=-8.17% MDD=10.7% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_B_take_profit_20260420_231834.json` | `partial_take_level 0.75->1.0` | fit=24.632 WR=70.2% alpha=-8.15% MDD=11.0% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260420_231835.json` | `momentum_confirm_days 3->4` | fit=25.200 WR=70.0% alpha=-8.10% MDD=10.9% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260420_231838.json` | `score_percentile_trigger 0.55->0.5` | fit=24.811 WR=70.3% alpha=-8.15% MDD=10.5% |
| 2026-04-20 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260420_231841.json` | `equity_drawdown_stop_pct 0.2->0.22` | fit=25.014 WR=70.2% alpha=-8.17% MDD=10.9% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260421_001340.json` | `signal_ema_span 8->7` | fit=22.898 WR=70.0% alpha=-8.14% MDD=10.9% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260421_001400.json` | `momentum_confirm_days 3->4` | fit=24.323 WR=70.2% alpha=-8.07% MDD=11.1% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260421_035944.json` | `score_percentile_trigger 0.55->0.5` | fit=23.658 WR=70.6% alpha=-8.14% MDD=10.5% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_B_take_profit_20260421_035951.json` | `partial_take_level 0.75->1.0` | fit=23.975 WR=70.2% alpha=-8.15% MDD=11.0% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260421_040019.json` | `ma_filter_mode 0->1; momentum_confirm_days 3->4` | fit=24.341 WR=70.0% alpha=-8.12% MDD=11.1% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260421_040018.json` | `equity_drawdown_stop_pct 0.2->0.22` | fit=24.285 WR=70.2% alpha=-8.16% MDD=11.0% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_B_take_profit_20260421_043340.json` | `partial_take_pct_2 0.30000000000000004->0.4` | fit=23.488 WR=70.0% alpha=-8.09% MDD=10.6% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260421_043341.json` | `score_percentile_trigger 0.55->0.6000000000000001` | fit=23.020 WR=70.6% alpha=-8.19% MDD=10.6% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260421_062701.json` | `ma_filter_mode 1->0` | fit=24.923 WR=70.0% alpha=-7.92% MDD=11.1% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260421_062710.json` | `entry_score_threshold 0.2->0.25` | fit=25.226 WR=70.0% alpha=-7.94% MDD=11.1% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260421_062719.json` | `max_loss_per_trade_pct 0.09->0.1` | fit=25.481 WR=70.0% alpha=-7.96% MDD=11.1% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260421_100656.json` | `entry_discount_atr_frac 0.45->0.5` | fit=28.385 WR=70.0% alpha=-7.81% MDD=11.0% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_C_timing_20260421_100703.json` | `consecutive_loss_cooldown 9->8` | fit=27.782 WR=70.0% alpha=-7.89% MDD=11.1% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260421_114702.json` | `stop_tighten_factor 0.4->0.45` | fit=27.689 WR=70.0% alpha=-7.79% MDD=11.1% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260421_142259.json` | `momentum_confirm_days 3->4` | fit=25.052 WR=70.0% alpha=-8.10% MDD=11.2% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260421_155752.json` | `signal_ema_span 8->9` | fit=26.292 WR=70.5% alpha=-8.05% MDD=11.2% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260421_155755.json` | `max_loss_per_trade_pct 0.09->0.1` | fit=26.656 WR=70.5% alpha=-8.06% MDD=11.2% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_C_timing_20260421_173202.json` | `consecutive_loss_cooldown 8->9` | fit=28.241 WR=70.0% alpha=-7.87% MDD=11.6% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260421_173225.json` | `entry_discount_atr_frac 0.45->0.5` | fit=27.791 WR=70.0% alpha=-7.83% MDD=11.4% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260421_195202.json` | `signal_ema_span 7->8` | fit=26.291 WR=69.5% alpha=-7.88% MDD=10.9% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260421_195206.json` | `max_loss_per_trade_pct 0.09->0.1` | fit=26.092 WR=69.4% alpha=-7.89% MDD=10.9% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_C_timing_20260421_195209.json` | `consecutive_loss_cooldown 9->10` | fit=25.607 WR=69.2% alpha=-7.88% MDD=10.9% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260421_201301.json` | `score_percentile_trigger 0.55->0.6000000000000001` | fit=27.180 WR=69.9% alpha=-7.87% MDD=10.9% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260421_210841.json` | `score_percentile_trigger 0.55->0.6000000000000001` | fit=26.919 WR=69.6% alpha=-7.88% MDD=10.9% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260421_220001.json` | `signal_ema_span 8->7` | fit=27.367 WR=69.4% alpha=-7.87% MDD=10.9% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_A_risk_management_20260421_220012.json` | `max_loss_per_trade_pct 0.09->0.1` | fit=27.613 WR=69.5% alpha=-7.92% MDD=10.9% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_C_timing_20260421_220040.json` | `consecutive_loss_cooldown 9->8` | fit=26.425 WR=69.4% alpha=-7.92% MDD=10.9% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260421_225236.json` | `signal_ema_span 8->9` | fit=29.717 WR=70.0% alpha=-7.88% MDD=10.9% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260421_225241.json` | `support_broken_gate 0.0->1.0` | fit=29.619 WR=70.0% alpha=-7.88% MDD=10.9% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_C_timing_20260421_225243.json` | `consecutive_loss_cooldown 8->9` | fit=29.047 WR=70.0% alpha=-7.88% MDD=10.9% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_B_take_profit_20260421_225244.json` | `partial_take_level 1.0->1.25` | fit=29.111 WR=70.0% alpha=-7.89% MDD=11.0% |
| 2026-04-21 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260421_233743.json` | `resistance_overext_gate 0.0->2.0` | fit=24.039 WR=69.0% alpha=-8.09% MDD=11.0% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_C_timing_20260422_001520.json` | `consecutive_loss_cooldown 9->10` | fit=28.013 WR=70.0% alpha=-7.92% MDD=11.5% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260422_001521.json` | `support_broken_gate 0.0->1.0` | fit=27.585 WR=69.8% alpha=-7.92% MDD=11.5% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260422_001526.json` | `score_percentile_trigger 0.6000000000000001->0.65` | fit=28.885 WR=69.8% alpha=-7.91% MDD=11.3% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260422_035623.json` | `ma_filter_mode 1->0` | fit=27.924 WR=69.2% alpha=-7.89% MDD=10.9% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260422_035625.json` | `entry_score_threshold 0.2->0.25` | fit=28.793 WR=69.7% alpha=-7.90% MDD=10.9% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_C_timing_20260422_035633.json` | `consecutive_loss_cooldown 9->8` | fit=28.082 WR=69.2% alpha=-7.91% MDD=11.0% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260422_060646.json` | `entry_score_threshold 0.2->0.25` | fit=27.311 WR=69.2% alpha=-7.89% MDD=10.6% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260422_080243.json` | `volatility_filter_percentile 0.05->0.0` | fit=19.814 WR=66.7% alpha=-8.11% MDD=12.2% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_C_timing_20260422_080247.json` | `consecutive_loss_cooldown 7->6` | fit=18.988 WR=66.7% alpha=-8.20% MDD=11.8% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260422_080308.json` | `score_percentile_trigger 0.55->0.5` | fit=18.931 WR=66.7% alpha=-8.20% MDD=11.8% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260422_093539.json` | `momentum_confirm_days 4->5` | fit=18.974 WR=66.7% alpha=-8.16% MDD=12.5% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260422_093554.json` | `score_percentile_trigger 0.55->0.5` | fit=18.573 WR=66.7% alpha=-8.19% MDD=12.5% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260422_110152.json` | `signal_ema_span 7->6` | fit=19.921 WR=66.7% alpha=-8.13% MDD=11.5% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260422_110205.json` | `momentum_confirm_days 4->5` | fit=20.068 WR=67.4% alpha=-8.11% MDD=12.5% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260422_121333.json` | `volatility_filter_percentile 0.05->0.0` | fit=21.244 WR=67.3% alpha=-7.99% MDD=12.5% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260422_121402.json` | `score_percentile_trigger 0.5->0.45` | fit=21.050 WR=66.7% alpha=-8.07% MDD=12.2% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260422_141648.json` | `ma_filter_mode 1->0` | fit=9.580 WR=66.1% alpha=-8.12% MDD=14.3% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_B_take_profit_20260422_141648.json` | `partial_take_level 1.0->0.75` | fit=10.280 WR=65.7% alpha=-8.11% MDD=13.3% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_C_timing_20260422_141655.json` | `consecutive_loss_cooldown 6->5` | fit=10.256 WR=65.7% alpha=-8.08% MDD=13.7% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260422_141721.json` | `score_percentile_trigger 0.45->0.4` | fit=9.760 WR=66.1% alpha=-8.12% MDD=13.7% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260422_155455.json` | `score_percentile_trigger 0.4->0.35000000000000003` | fit=22.566 WR=66.7% alpha=-7.88% MDD=11.6% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_B_take_profit_20260422_155514.json` | `partial_take_pct 0.1->0.2` | fit=23.353 WR=66.7% alpha=-7.76% MDD=11.1% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260422_165716.json` | `momentum_confirm_days 4->5` | fit=19.548 WR=66.7% alpha=-8.05% MDD=13.6% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_F_trailing_20260422_165727.json` | `trailing_stop_mode 1->2` | fit=18.094 WR=65.9% alpha=-8.14% MDD=13.6% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_C_timing_20260422_165735.json` | `consecutive_loss_cooldown 8->9` | fit=19.074 WR=66.0% alpha=-8.14% MDD=12.9% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_B_take_profit_20260422_165746.json` | `partial_take_level_2 1.5->1.25` | fit=18.252 WR=66.0% alpha=-8.14% MDD=13.5% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_C_timing_20260422_175845.json` | `consecutive_loss_cooldown 8->9` | fit=20.283 WR=66.7% alpha=-8.08% MDD=12.0% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260422_175845.json` | `score_percentile_trigger 0.55->0.5` | fit=21.152 WR=66.7% alpha=-8.02% MDD=12.0% |
| 2026-04-22 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260422_175856.json` | `momentum_confirm_days 4->5` | fit=20.509 WR=66.7% alpha=-8.02% MDD=12.3% |
| 2026-04-23 | local | `local_fullmetric_sweep_continuous_C_timing_20260423_001610.json` | `consecutive_loss_cooldown 8->9` | fit=19.911 WR=66.7% alpha=-8.12% MDD=11.6% |
| 2026-04-23 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260423_001613.json` | `score_percentile_trigger 0.55->0.5` | fit=20.478 WR=66.7% alpha=-8.12% MDD=11.6% |
| 2026-04-23 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260423_001620.json` | `momentum_confirm_days 4->5` | fit=20.362 WR=66.7% alpha=-8.06% MDD=11.5% |
| 2026-04-23 | local | `local_fullmetric_sweep_continuous_C_timing_20260423_040003.json` | `consecutive_loss_cooldown 8->9` | fit=20.013 WR=66.7% alpha=-8.19% MDD=11.9% |
| 2026-04-23 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260423_040022.json` | `score_percentile_trigger 0.55->0.5` | fit=19.510 WR=66.7% alpha=-8.18% MDD=12.0% |
| 2026-04-23 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260423_040040.json` | `momentum_confirm_days 4->5` | fit=19.629 WR=66.7% alpha=-8.14% MDD=11.7% |
| 2026-04-23 | local | `local_fullmetric_sweep_continuous_B_take_profit_20260423_101031.json` | `partial_take_level 1.0->1.25` | fit=18.629 WR=66.7% alpha=-8.13% MDD=12.2% |
| 2026-04-23 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260423_101039.json` | `momentum_confirm_days 4->5` | fit=19.111 WR=66.9% alpha=-8.06% MDD=11.7% |
| 2026-04-23 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260423_101049.json` | `score_percentile_trigger 0.55->0.5` | fit=19.130 WR=66.7% alpha=-8.09% MDD=12.2% |
| 2026-04-23 | local | `local_fullmetric_sweep_continuous_C_timing_20260423_112844.json` | `consecutive_loss_cooldown 8->9` | fit=19.393 WR=66.7% alpha=-8.12% MDD=11.7% |
| 2026-04-23 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260423_112844.json` | `momentum_confirm_days 4->5` | fit=19.189 WR=67.1% alpha=-8.06% MDD=11.7% |
| 2026-04-23 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260423_112917.json` | `score_percentile_trigger 0.55->0.5` | fit=20.082 WR=66.7% alpha=-8.09% MDD=11.7% |
| 2026-04-25 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260425_132007.json` | `momentum_confirm_days 4->5` | fit=19.187 WR=66.7% alpha=-7.99% MDD=12.6% |
| 2026-04-25 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260425_132044.json` | `score_percentile_trigger 0.55->0.5` | fit=18.934 WR=66.7% alpha=-8.02% MDD=12.8% |
| 2026-04-26 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260426_061324.json` | `volatility_filter_percentile 0.05->0.1` | fit=19.812 WR=66.7% alpha=-8.05% MDD=12.1% |
| 2026-04-26 | local | `local_fullmetric_sweep_continuous_B_take_profit_20260426_061331.json` | `partial_take_level 1.0->1.25` | fit=17.490 WR=66.7% alpha=-8.06% MDD=12.2% |
| 2026-04-26 | local | `local_fullmetric_sweep_continuous_C_timing_20260426_061342.json` | `consecutive_loss_cooldown 8->9` | fit=17.952 WR=66.7% alpha=-8.04% MDD=12.1% |
| 2026-04-26 | local | `local_fullmetric_sweep_continuous_E_regime_vol_20260426_070113.json` | `volatility_filter_percentile 0.05->0.0` | fit=19.215 WR=66.7% alpha=-7.92% MDD=12.0% |
| 2026-04-26 | local | `local_fullmetric_sweep_continuous_D_entry_filter_20260426_070114.json` | `score_percentile_trigger 0.55->0.5` | fit=18.795 WR=66.7% alpha=-7.99% MDD=12.0% |

## Graveyard (genes que falharam multiplas vezes)

Genes tentados >= 2 vezes sem nenhuma promocao. Considere SKIP no proximo ciclo.
(Colunas `codex` contam tentativas importadas do log do GPT codex — dedup por sweep file.)

| Gene | Attempts | Promotions | (codex) Attempts | (codex) Promotions |
|---|---:|---:|---:|---:|
| `vote_threshold_long` | 292 | 0 | 0 | 0 |
| `score_strength_scaling` | 292 | 0 | 0 | 0 |
| `reward_risk_ratio` | 292 | 0 | 0 | 0 |
| `min_signal_strength` | 291 | 0 | 0 | 0 |
| `entry_confirmation_days` | 290 | 0 | 0 | 0 |
| `vote_threshold_short` | 289 | 0 | 1 | 0 |
| `z_threshold` | 288 | 0 | 0 | 0 |
| `time_stop_bars` | 288 | 0 | 0 | 0 |
| `volume_confirm_mode` | 288 | 0 | 1 | 0 |
| `ma_filter_period` | 288 | 0 | 1 | 0 |
| `stop_atr_mult` | 286 | 0 | 0 | 0 |
| `stop_tighten_after_bars` | 284 | 0 | 0 | 0 |
| `entry_aggressiveness` | 271 | 0 | 0 | 0 |

## Hot zones (genes que ja promoveram)

Genes onde alguma combinacao funcionou. Vale revisitar com novos vizinhos.

| Gene | Promotions | Attempts | Hit rate | (codex) Promotions |
|---|---:|---:|---:|---:|
| `consecutive_loss_cooldown` | 21 | 299 | 7% | 1 |
| `score_percentile_trigger` | 20 | 294 | 7% | 0 |
| `momentum_confirm_days` | 18 | 293 | 6% | 0 |
| `partial_take_level` | 8 | 299 | 3% | 1 |
| `signal_ema_span` | 7 | 293 | 2% | 0 |
| `max_loss_per_trade_pct` | 6 | 288 | 2% | 0 |
| `entry_score_threshold` | 6 | 303 | 2% | 0 |
| `regime_threshold` | 6 | 308 | 2% | 1 |
| `equity_drawdown_stop_pct` | 5 | 285 | 2% | 0 |
| `volatility_filter_percentile` | 5 | 298 | 2% | 0 |
| `ma_filter_mode` | 5 | 298 | 2% | 0 |
| `partial_take_pct` | 4 | 292 | 1% | 0 |
| `support_broken_gate` | 2 | 267 | 1% | 0 |
| `trailing_stop_mode` | 2 | 289 | 1% | 0 |
| `entry_discount_atr_frac` | 2 | 289 | 1% | 0 |
| `stop_tighten_factor` | 2 | 291 | 1% | 0 |
| `partial_take_pct_2` | 2 | 291 | 1% | 0 |
| `partial_take_level_2` | 2 | 294 | 1% | 0 |
| `resistance_overext_gate` | 1 | 267 | 0% | 0 |
| `vol_regime_mode` | 1 | 290 | 0% | 0 |

## Category rotation status

| Category | Last swept | Runs | Promotions |
|---|---|---:|---:|
| A_risk_management | 2026-04-20T19:42:51+00:00 | 6 | 4 |
| B_take_profit | 2026-04-20T11:36:36+00:00 | 7 | 2 |
| C_timing | 2026-04-20T14:16:03+00:00 | 5 | 2 |
| D_entry_filter | 2026-04-20T10:29:24+00:00 | 6 | 2 |
| E_regime_vol | 2026-04-20T16:30:07+00:00 | 5 | 4 |
| F_trailing | 2026-04-20T18:25:20+00:00 | 5 | 0 |

## Codex sync history (latest attempts)

Parsed `codex_attempts_source.md` at 2026-09-05T04:48:10+00:00. Total attempts: **24**, promoted: **11**. Updates via `python analysis/codex_attempts_sync.py` at the start of every continuous cycle (auto) or on demand.

| Attempt | Timestamp | Promoted | Move(s) | Learning (snippet) |
|---|---|:-:|---|---|
| #37 | 2026-04-20T10:56:55+02:00 | no | `regime_threshold 0.25->0.2` | the main problem is no longer only local tail-risk repair. The refreshed `git:main` now beats the current local checkpoi... |
| #36 | 2026-04-20T09:19:32+02:00 | YES | `consecutive_loss_cooldown 9.0->10.0` | Attempt 34 materially changed the frontier. Once `partial_take_pct` moved to `0.10`, the system could absorb one more co... |
| #35 | 2026-04-20T09:09:41+02:00 | no | `vote_threshold_short 0.15->0.1550698474225143; partial_take_pct_2 0.3->0.2982570491636789` | the first post-Attempt-33 staged refine slice did not open a new frontier. On windows `[1, 10, 19, 28]`, stage 2 mostly ... |
| #34 | 2026-04-20T08:55:59+02:00 | YES | `partial_take_pct 0.2->0.1` | the B-side search space is not globally dead; it was only dead under the older frontiers. On the current `partial_take_l... |
| #33 | 2026-04-20T07:08:00+02:00 | no | `regime_threshold 0.25->0.3` | the strict-regime branch is now locally exhausted: `regime_threshold: 0.25 -> 0.30` remains the best tail-risk shape on ... |

---

_Run again with: `python analysis/sweep_learnings.py`_
