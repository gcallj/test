# Optimization Log (continuous improvement)

Append-only log of every full-metric sweep cycle executed by
`run_continuous_improvement.py`. Each entry records:

- timestamp, category, genes swept
- baseline (`git:main`) and incumbent metrics snapshot
- best candidate found (promoted or not)
- promotion decision rationale (which guardrail passed/failed)
- learning extracted from the result

Failures are kept on purpose: they form the graveyard that
tells future iterations which gene moves are dead ends.

---

## 2026-04-18T14:37:35+02:00 - D_entry_filter
**Cycle**: #1  |  **Promoted**: NO
**Genes swept**: `vote_threshold_long, vote_threshold_short, z_threshold, signal_ema_span, entry_confirmation_days, entry_discount_atr_frac, score_strength_scaling, min_signal_strength, entry_score_threshold, score_percentile_trigger`
**Sweep file**: `local_fullmetric_sweep_continuous_D_entry_filter_20260418_141423.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | -189.0316 | 65.28% | 67.46% | -3.54% | 18.13% | 85 |
| incumbent (worktree) | -189.0316 | 65.28% | 67.46% | -3.54% | 18.13% | 85 |
| **candidate** | -185.4229 | 65.98% | 67.93% | -3.38% | 18.03% | 86 |

**Best candidate label**: `entry_discount_atr_frac:+1`

**Moves applied to incumbent**:

- `entry_discount_atr_frac`: `0.45` -> `0.5` (step 0.05)

**Learning**: candidato `entry_discount_atr_frac: 0.45->0.5` parecia bom mas falhou nos gates: acceptance_vs_main. Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto.

---

## 2026-04-18T15:42:25+00:00 - A_risk_management
**Cycle**: #2  |  **Promoted**: NO  |  **Codex sync**: YES
**Genes swept**: `stop_atr_mult, stop_tighten_after_bars, stop_tighten_factor, max_loss_per_trade_pct, equity_drawdown_stop_pct`
**Sweep file**: `local_fullmetric_sweep_continuous_A_risk_management_20260418_153817.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | -75.1949 | 70.13% | 69.70% | -8.28% | 10.72% | 32 |
| incumbent (worktree) | -75.1949 | 70.13% | 69.70% | -8.28% | 10.72% | 32 |
| **candidate** | -74.8961 | 70.13% | 70.00% | -8.32% | 10.72% | 31 |

**Best candidate label**: `max_loss_per_trade_pct:+1`

**Moves applied to incumbent**:

- `max_loss_per_trade_pct`: `0.08` -> `0.09` (step 0.01)

**Learning**: candidato `max_loss_per_trade_pct: 0.08->0.09` parecia bom mas falhou nos gates: acceptance_vs_main. Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto.

---

## 2026-04-18T18:50:09+00:00 - B_take_profit
**Cycle**: #3  |  **Promoted**: NO  |  **Codex sync**: YES
**Genes swept**: `reward_risk_ratio, partial_take_pct, partial_take_level, partial_take_pct_2, partial_take_level_2`
**Sweep file**: `local_fullmetric_sweep_continuous_B_take_profit_20260418_184523.json`
## 2026-04-18T19:15:37+00:00 - B_take_profit
**Cycle**: #3  |  **Promoted**: NO  |  **Codex sync**: YES
**Genes swept**: `reward_risk_ratio, partial_take_pct, partial_take_level, partial_take_pct_2, partial_take_level_2`
**Sweep file**: `local_fullmetric_sweep_continuous_B_take_profit_20260418_191042.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | -80.1094 | 67.27% | 66.67% | -8.42% | 11.08% | 26 |
| incumbent (worktree) | -80.1094 | 67.27% | 66.67% | -8.42% | 11.08% | 26 |
| **candidate** | -79.3785 | 67.74% | 66.67% | -8.35% | 10.66% | 26 |

**Best candidate label**: `partial_take_level_2:-1`

**Moves applied to incumbent**:

- `partial_take_level_2`: `1.75` -> `1.5` (step 0.25)

**Learning**: candidato `partial_take_level_2: 1.75->1.5` parecia bom mas falhou nos gates: acceptance_vs_main. Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto.
| baseline (git:main) | -75.2106 | 70.00% | 69.70% | -8.29% | 10.55% | 32 |
| incumbent (worktree) | -75.2106 | 70.00% | 69.70% | -8.29% | 10.55% | 32 |
| **candidate** | -75.1232 | 70.00% | 69.70% | -8.26% | 10.91% | 31 |

**Best candidate label**: `partial_take_level:+1`

**Moves applied to incumbent**:

- `partial_take_level`: `0.75` -> `1.0` (step 0.25)

**Learning**: candidato `partial_take_level: 0.75->1.0` parecia bom mas falhou nos gates: acceptance_vs_main, priority_better. Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto.

---

## 2026-04-18T20:39:53+00:00 - C_timing
**Cycle**: #4  |  **Promoted**: NO  |  **Codex sync**: YES
**Genes swept**: `time_stop_bars, consecutive_loss_cooldown`
**Sweep file**: `local_fullmetric_sweep_continuous_C_timing_20260418_203623.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | -75.4866 | 69.88% | 69.70% | -8.33% | 11.09% | 32 |
| incumbent (worktree) | -75.4866 | 69.88% | 69.70% | -8.33% | 11.09% | 32 |
| **candidate** | -75.6432 | 70.00% | 69.70% | -8.34% | 11.09% | 31 |

**Best candidate label**: `consecutive_loss_cooldown:+1`

**Moves applied to incumbent**:

- `consecutive_loss_cooldown`: `6` -> `7` (step 1.0)

**Learning**: candidato `consecutive_loss_cooldown: 6->7` parecia bom mas falhou nos gates: acceptance_vs_main, priority_better. Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto.

---

## 2026-04-18T21:31:21+00:00 - E_regime_vol
**Cycle**: #5  |  **Promoted**: NO  |  **Codex sync**: YES
**Genes swept**: `ma_filter_period, ma_filter_mode, vol_regime_mode, volatility_filter_percentile, regime_threshold, volume_confirm_mode, momentum_confirm_days`
**Sweep file**: `local_fullmetric_sweep_continuous_E_regime_vol_20260418_212640.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 21.1254 | 69.91% | 69.70% | -8.28% | 10.79% | 32 |
| incumbent (worktree) | 21.1254 | 69.91% | 69.70% | -8.28% | 10.79% | 32 |
| **candidate** | 22.4735 | 70.00% | 70.00% | -8.24% | 10.76% | 32 |

**Best candidate label**: `momentum_confirm_days:-1`

**Moves applied to incumbent**:

- `momentum_confirm_days`: `5` -> `4` (step 1.0)

**Learning**: candidato `momentum_confirm_days: 5->4` parecia bom mas falhou nos gates: acceptance_vs_main. Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto.

---

## 2026-04-18T22:36:36+00:00 - F_trailing
**Cycle**: #6  |  **Promoted**: NO  |  **Codex sync**: YES
**Genes swept**: `trailing_stop_mode`
**Sweep file**: `local_fullmetric_sweep_continuous_F_trailing_20260418_223530.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 19.6541 | 69.18% | 68.83% | -8.30% | 11.25% | 32 |
| incumbent (worktree) | 19.6541 | 69.18% | 68.83% | -8.30% | 11.25% | 32 |

*(no candidate produced metrics — sweep returned empty)*

**Learning**: nenhum candidato cruzou os guardrails de WR ou de seguranca; o incumbent ja esta no Pareto local desse conjunto de genes. Considerar marcar essas genes como 'exploradas recentemente' por 30 dias.

---

## 2026-04-19T01:44:00+00:00 - D_entry_filter
**Cycle**: #7  |  **Promoted**: NO  |  **Codex sync**: YES
**Genes swept**: `vote_threshold_long, vote_threshold_short, z_threshold, signal_ema_span, entry_confirmation_days, entry_discount_atr_frac, score_strength_scaling, min_signal_strength, entry_score_threshold, score_percentile_trigger`
**Sweep file**: `local_fullmetric_sweep_continuous_D_entry_filter_20260419_013636.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 19.6338 | 69.57% | 68.83% | -8.30% | 10.26% | 31 |
| incumbent (worktree) | 19.6338 | 69.57% | 68.83% | -8.30% | 10.26% | 31 |
| **candidate** | 20.0916 | 69.70% | 69.52% | -8.32% | 10.68% | 31 |

**Best candidate label**: `score_percentile_trigger:+1`

**Moves applied to incumbent**:

- `score_percentile_trigger`: `0.6000000000000001` -> `0.65` (step 0.05)

**Learning**: candidato `score_percentile_trigger: 0.6000000000000001->0.65` parecia bom mas falhou nos gates: acceptance_vs_main. Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto.

---

## 2026-04-19T03:31:30+00:00 - A_risk_management
**Cycle**: #8  |  **Promoted**: NO  |  **Codex sync**: YES
**Genes swept**: `stop_atr_mult, stop_tighten_after_bars, stop_tighten_factor, max_loss_per_trade_pct, equity_drawdown_stop_pct`
**Sweep file**: `local_fullmetric_sweep_continuous_A_risk_management_20260419_032748.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 17.6724 | 67.21% | 66.67% | -8.47% | 10.48% | 27 |
| incumbent (worktree) | 17.6724 | 67.21% | 66.67% | -8.47% | 10.48% | 27 |
| **candidate** | 21.3929 | 68.50% | 68.42% | -8.35% | 10.48% | 28 |

**Best candidate label**: `equity_drawdown_stop_pct:+1`

**Moves applied to incumbent**:

- `equity_drawdown_stop_pct`: `0.18` -> `0.2` (step 0.02)

**Learning**: candidato `equity_drawdown_stop_pct: 0.18->0.2` parecia bom mas falhou nos gates: acceptance_vs_main. Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto.

---

## 2026-04-19T07:42:34+00:00 - B_take_profit
**Cycle**: #9  |  **Promoted**: NO  |  **Codex sync**: YES
**Genes swept**: `reward_risk_ratio, partial_take_pct, partial_take_level, partial_take_pct_2, partial_take_level_2`
**Sweep file**: `local_fullmetric_sweep_continuous_B_take_profit_20260419_073804.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 20.0425 | 69.44% | 68.75% | -8.31% | 11.20% | 33 |
| incumbent (worktree) | 20.0425 | 69.44% | 68.75% | -8.31% | 11.20% | 33 |
| **candidate** | 20.2170 | 69.57% | 68.75% | -8.38% | 10.58% | 33 |

**Best candidate label**: `partial_take_level:-1`

**Moves applied to incumbent**:

- `partial_take_level`: `0.75` -> `0.5` (step 0.25)

**Learning**: candidato `partial_take_level: 0.75->0.5` parecia bom mas falhou nos gates: acceptance_vs_main. Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto.

---

## 2026-04-19T07:47:29+00:00 - B_take_profit
**Cycle**: #9  |  **Promoted**: NO  |  **Codex sync**: YES
**Genes swept**: `reward_risk_ratio, partial_take_pct, partial_take_level, partial_take_pct_2, partial_take_level_2`
**Sweep file**: `local_fullmetric_sweep_continuous_B_take_profit_20260419_074247.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 20.7213 | 69.70% | 69.44% | -8.28% | 10.79% | 31 |
| incumbent (worktree) | 20.7213 | 69.70% | 69.44% | -8.28% | 10.79% | 31 |
| **candidate** | 21.3109 | 70.00% | 69.44% | -8.35% | 10.58% | 31 |

**Best candidate label**: `partial_take_level:-1`

**Moves applied to incumbent**:

- `partial_take_level`: `0.75` -> `0.5` (step 0.25)

**Learning**: candidato `partial_take_level: 0.75->0.5` parecia bom mas falhou nos gates: acceptance_vs_main. Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto.

---


## 2026-04-19T07:56:24+00:00 - D_entry_filter
**Cycle**: #11  |  **Promoted**: NO  |  **Codex sync**: YES
**Genes swept**: `vote_threshold_long, vote_threshold_short, z_threshold, signal_ema_span, entry_confirmation_days, entry_discount_atr_frac, score_strength_scaling, min_signal_strength, entry_score_threshold, score_percentile_trigger`
**Sweep file**: `local_fullmetric_sweep_continuous_D_entry_filter_20260419_075026.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 18.1659 | 67.27% | 66.67% | -8.42% | 10.45% | 26 |
| incumbent (worktree) | 18.1659 | 67.27% | 66.67% | -8.42% | 10.45% | 26 |
| **candidate** | 18.2429 | 67.90% | 66.67% | -8.40% | 10.60% | 26 |

**Best candidate label**: `signal_ema_span:+1`

**Moves applied to incumbent**:

- `signal_ema_span`: `8` -> `9` (step 1.0)

**Learning**: candidato `signal_ema_span: 8->9` parecia bom mas falhou nos gates: acceptance_vs_main. Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto.

---
## 2026-04-19T08:12:23+00:00 - B_take_profit
**Cycle**: #12  |  **Promoted**: NO  |  **Codex sync**: YES
**Genes swept**: `reward_risk_ratio, partial_take_pct, partial_take_level, partial_take_pct_2, partial_take_level_2`
**Sweep file**: `local_fullmetric_sweep_continuous_B_take_profit_20260419_080739.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 21.3465 | 69.70% | 69.44% | -8.28% | 10.79% | 31 |
| incumbent (worktree) | 21.3465 | 69.70% | 69.44% | -8.28% | 10.79% | 31 |
| **candidate** | 20.9070 | 70.00% | 69.44% | -8.34% | 10.91% | 31 |

**Best candidate label**: `partial_take_pct:-1`

**Moves applied to incumbent**:

- `partial_take_pct`: `0.2` -> `0.1` (step 0.1)

**Learning**: candidato `partial_take_pct: 0.2->0.1` parecia bom mas falhou nos gates: acceptance_vs_main. Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto.

---

## 2026-04-19T09:39:50+00:00 - C_timing
**Cycle**: #13  |  **Promoted**: YES  |  **Codex sync**: YES
**Genes swept**: `time_stop_bars, consecutive_loss_cooldown`
**Sweep file**: `local_fullmetric_sweep_continuous_C_timing_20260419_093647.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 19.8977 | 69.44% | 69.12% | -8.32% | 10.46% | 30 |
| incumbent (worktree) | 19.8977 | 69.44% | 69.12% | -8.32% | 10.46% | 30 |
| **candidate** | 19.7585 | 70.00% | 69.12% | -8.31% | 10.72% | 29 |

**Best candidate label**: `consecutive_loss_cooldown:+1`

**Moves applied to incumbent**:

- `consecutive_loss_cooldown`: `6` -> `7` (step 1.0)

**Learning**: o move `consecutive_loss_cooldown: 6->7` foi promovido. Categoria/genes adjacentes podem render melhorias semelhantes; considerar sweep daquela categoria no proximo ciclo.

---

## 2026-04-19T10:51:45+00:00 - E_regime_vol
**Cycle**: #14  |  **Promoted**: YES  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `ma_filter_period, ma_filter_mode, vol_regime_mode, volatility_filter_percentile, regime_threshold, volume_confirm_mode, momentum_confirm_days`
**Sweep file**: `local_fullmetric_sweep_continuous_E_regime_vol_20260419_104733.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 19.6525 | 69.12% | 69.12% | -8.32% | 11.09% | 31 |
| incumbent (worktree) | 19.5709 | 69.70% | 69.12% | -8.32% | 11.09% | 31 |
| **candidate** | 19.8349 | 69.70% | 69.57% | -8.30% | 11.16% | 32 |

**Best candidate label**: `ma_filter_mode:-1`

**Moves applied to incumbent**:

- `ma_filter_mode`: `1` -> `0` (step 1.0)

**Learning**: o move `ma_filter_mode: 1->0` foi promovido. Categoria/genes adjacentes podem render melhorias semelhantes; considerar sweep daquela categoria no proximo ciclo.

---

## 2026-04-19T11:29:21+00:00 - F_trailing
**Cycle**: #15  |  **Promoted**: NO  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `trailing_stop_mode`
**Sweep file**: `local_fullmetric_sweep_continuous_F_trailing_20260419_112824.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 19.8770 | 69.70% | 69.12% | -8.33% | 10.85% | 31 |
| incumbent (worktree) | 19.7775 | 70.00% | 69.44% | -8.32% | 10.85% | 31 |

*(no candidate produced metrics — sweep returned empty)*

**Learning**: nenhum candidato cruzou os guardrails de WR ou de seguranca; o incumbent ja esta no Pareto local desse conjunto de genes. Considerar marcar essas genes como 'exploradas recentemente' por 30 dias.

---

## 2026-04-19T12:29:52+00:00 - A_risk_management
**Cycle**: #16  |  **Promoted**: YES  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `stop_atr_mult, stop_tighten_after_bars, stop_tighten_factor, max_loss_per_trade_pct, equity_drawdown_stop_pct`
**Sweep file**: `local_fullmetric_sweep_continuous_A_risk_management_20260419_122638.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 20.6723 | 69.23% | 68.75% | -8.29% | 10.12% | 31 |
| incumbent (worktree) | 20.3000 | 69.70% | 69.44% | -8.30% | 10.46% | 31 |
| **candidate** | 21.0138 | 69.74% | 69.70% | -8.29% | 10.41% | 31 |

**Best candidate label**: `stop_tighten_factor:-1`

**Moves applied to incumbent**:

- `stop_tighten_factor`: `0.45` -> `0.4` (step 0.05)

**Learning**: o move `stop_tighten_factor: 0.45->0.4` foi promovido. Categoria/genes adjacentes podem render melhorias semelhantes; considerar sweep daquela categoria no proximo ciclo.

---

## 2026-04-19T15:25:06+00:00 - D_entry_filter
**Cycle**: #17  |  **Promoted**: YES  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `vote_threshold_long, vote_threshold_short, z_threshold, signal_ema_span, entry_confirmation_days, entry_discount_atr_frac, score_strength_scaling, min_signal_strength, entry_score_threshold, score_percentile_trigger`
**Sweep file**: `local_fullmetric_sweep_continuous_D_entry_filter_20260419_151906.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 19.2118 | 69.70% | 69.70% | -8.33% | 11.18% | 31 |
| incumbent (worktree) | 19.2118 | 69.70% | 69.70% | -8.33% | 11.18% | 31 |
| **candidate** | 19.6941 | 69.70% | 69.70% | -8.33% | 11.18% | 31 |

**Best candidate label**: `entry_score_threshold:+1`

**Moves applied to incumbent**:

- `entry_score_threshold`: `0.15000000000000002` -> `0.2` (step 0.05)

**Learning**: o move `entry_score_threshold: 0.15000000000000002->0.2` foi promovido. Categoria/genes adjacentes podem render melhorias semelhantes; considerar sweep daquela categoria no proximo ciclo.

---

## 2026-04-19T16:29:41+00:00 - B_take_profit
**Cycle**: #18  |  **Promoted**: NO  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `reward_risk_ratio, partial_take_pct, partial_take_level, partial_take_pct_2, partial_take_level_2`
**Sweep file**: `local_fullmetric_sweep_continuous_B_take_profit_20260419_162527.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 19.4036 | 68.83% | 68.52% | -8.33% | 11.20% | 31 |
| incumbent (worktree) | 19.6381 | 69.23% | 68.69% | -8.31% | 10.93% | 31 |
| **candidate** | 19.9562 | 69.92% | 68.69% | -8.34% | 11.04% | 31 |

**Best candidate label**: `partial_take_pct:-1`

**Moves applied to incumbent**:

- `partial_take_pct`: `0.2` -> `0.1` (step 0.1)

**Learning**: candidato `partial_take_pct: 0.2->0.1` parecia bom mas falhou nos gates: acceptance_vs_main. Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto.

---

## 2026-04-19T19:30:20+00:00 - C_timing
**Cycle**: #19  |  **Promoted**: NO  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `time_stop_bars, consecutive_loss_cooldown`
**Sweep file**: `local_fullmetric_sweep_continuous_C_timing_20260419_192718.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 20.4057 | 69.70% | 69.88% | -8.31% | 11.17% | 33 |
| incumbent (worktree) | 20.4358 | 69.57% | 69.70% | -8.30% | 10.60% | 32 |
| **candidate** | 20.4000 | 69.70% | 70.00% | -8.30% | 10.60% | 32 |

**Best candidate label**: `consecutive_loss_cooldown:+1`

**Moves applied to incumbent**:

- `consecutive_loss_cooldown`: `7` -> `8` (step 1.0)

**Learning**: candidato `consecutive_loss_cooldown: 7->8` parecia bom mas falhou nos gates: priority_better. Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto.

---

## 2026-04-19T21:28:53+00:00 - E_regime_vol
**Cycle**: #20  |  **Promoted**: YES  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `ma_filter_period, ma_filter_mode, vol_regime_mode, volatility_filter_percentile, regime_threshold, volume_confirm_mode, momentum_confirm_days`
**Sweep file**: `local_fullmetric_sweep_continuous_E_regime_vol_20260419_212524.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 21.8884 | 69.92% | 70.45% | -8.30% | 11.20% | 32 |
| incumbent (worktree) | 21.8884 | 69.92% | 70.45% | -8.30% | 11.20% | 32 |
| **candidate** | 21.9074 | 69.92% | 70.45% | -8.28% | 11.20% | 32 |

**Best candidate label**: `regime_threshold:-1`

**Moves applied to incumbent**:

- `regime_threshold`: `0.25` -> `0.2` (step 0.05)

**Learning**: o move `regime_threshold: 0.25->0.2` foi promovido. Categoria/genes adjacentes podem render melhorias semelhantes; considerar sweep daquela categoria no proximo ciclo.

---

## 2026-04-19T22:35:21+00:00 - F_trailing
**Cycle**: #21  |  **Promoted**: NO  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `trailing_stop_mode`
**Sweep file**: `local_fullmetric_sweep_continuous_F_trailing_20260419_223419.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 19.2618 | 69.44% | 68.59% | -8.32% | 10.80% | 33 |
| incumbent (worktree) | 19.6063 | 69.44% | 68.75% | -8.28% | 10.80% | 32 |

*(no candidate produced metrics — sweep returned empty)*

**Learning**: nenhum candidato cruzou os guardrails de WR ou de seguranca; o incumbent ja esta no Pareto local desse conjunto de genes. Considerar marcar essas genes como 'exploradas recentemente' por 30 dias.

---

## 2026-04-19T23:55:12+00:00 - A_risk_management
**Cycle**: #22  |  **Promoted**: YES  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `stop_atr_mult, stop_tighten_after_bars, stop_tighten_factor, max_loss_per_trade_pct, equity_drawdown_stop_pct`
**Sweep file**: `local_fullmetric_sweep_continuous_A_risk_management_20260419_235131.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 19.3035 | 69.23% | 68.25% | -8.31% | 11.09% | 32 |
| incumbent (worktree) | 20.5747 | 69.70% | 69.57% | -8.25% | 10.90% | 32 |
| **candidate** | 20.9670 | 69.70% | 69.70% | -8.27% | 10.90% | 32 |

**Best candidate label**: `max_loss_per_trade_pct:+1`

**Moves applied to incumbent**:

- `max_loss_per_trade_pct`: `0.08` -> `0.09` (step 0.01)

**Learning**: o move `max_loss_per_trade_pct: 0.08->0.09` foi promovido. Categoria/genes adjacentes podem render melhorias semelhantes; considerar sweep daquela categoria no proximo ciclo.

---

## 2026-04-20T01:36:41+00:00 - D_entry_filter
**Cycle**: #23  |  **Promoted**: NO  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `vote_threshold_long, vote_threshold_short, z_threshold, signal_ema_span, entry_confirmation_days, entry_discount_atr_frac, score_strength_scaling, min_signal_strength, entry_score_threshold, score_percentile_trigger`
**Sweep file**: `local_fullmetric_sweep_continuous_D_entry_filter_20260420_013020.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 19.0320 | 69.12% | 68.57% | -8.35% | 10.99% | 31 |
| incumbent (worktree) | 20.3067 | 69.18% | 69.44% | -8.33% | 11.09% | 32 |
| **candidate** | 19.7627 | 69.23% | 69.44% | -8.35% | 11.09% | 32 |

**Best candidate label**: `signal_ema_span:-1`

**Moves applied to incumbent**:

- `signal_ema_span`: `8` -> `7` (step 1.0)

**Learning**: candidato `signal_ema_span: 8->7` parecia bom mas falhou nos gates: acceptance_vs_main, priority_better. Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto.

---

## 2026-04-20T02:36:23+00:00 - B_take_profit
**Cycle**: #24  |  **Promoted**: YES  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `reward_risk_ratio, partial_take_pct, partial_take_level, partial_take_pct_2, partial_take_level_2`
**Sweep file**: `local_fullmetric_sweep_continuous_B_take_profit_20260420_023246.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 19.5777 | 69.91% | 69.05% | -8.33% | 11.09% | 31 |
| incumbent (worktree) | 20.7359 | 69.92% | 70.00% | -8.30% | 10.90% | 29 |
| **candidate** | 21.0176 | 69.74% | 70.00% | -8.24% | 11.00% | 29 |

**Best candidate label**: `partial_take_level_2:-1`

**Moves applied to incumbent**:

- `partial_take_level_2`: `1.75` -> `1.5` (step 0.25)

**Learning**: o move `partial_take_level_2: 1.75->1.5` foi promovido. Categoria/genes adjacentes podem render melhorias semelhantes; considerar sweep daquela categoria no proximo ciclo.

---

## 2026-04-20T04:36:50+00:00 - C_timing
**Cycle**: #25  |  **Promoted**: YES  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `time_stop_bars, consecutive_loss_cooldown`
**Sweep file**: `local_fullmetric_sweep_continuous_C_timing_20260420_043330.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 20.1996 | 69.43% | 69.23% | -8.28% | 10.49% | 33 |
| incumbent (worktree) | 20.1668 | 69.70% | 69.70% | -8.26% | 10.78% | 32 |
| **candidate** | 20.1963 | 69.70% | 69.70% | -8.26% | 10.78% | 32 |

**Best candidate label**: `consecutive_loss_cooldown:+1`

**Moves applied to incumbent**:

- `consecutive_loss_cooldown`: `7` -> `8` (step 1.0)

**Learning**: o move `consecutive_loss_cooldown: 7->8` foi promovido. Categoria/genes adjacentes podem render melhorias semelhantes; considerar sweep daquela categoria no proximo ciclo.

---

## 2026-04-20T05:42:53+00:00 - E_regime_vol
**Cycle**: #26  |  **Promoted**: YES  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `ma_filter_period, ma_filter_mode, vol_regime_mode, volatility_filter_percentile, regime_threshold, volume_confirm_mode, momentum_confirm_days`
**Sweep file**: `local_fullmetric_sweep_continuous_E_regime_vol_20260420_054005.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 20.3611 | 69.57% | 69.70% | -8.26% | 10.53% | 31 |
| incumbent (worktree) | 20.3611 | 69.57% | 69.70% | -8.26% | 10.53% | 31 |
| **candidate** | 22.2202 | 70.00% | 70.00% | -8.15% | 10.56% | 32 |

**Best candidate label**: `momentum_confirm_days:-1`

**Moves applied to incumbent**:

- `momentum_confirm_days`: `5` -> `4` (step 1.0)

**Learning**: o move `momentum_confirm_days: 5->4` foi promovido. Categoria/genes adjacentes podem render melhorias semelhantes; considerar sweep daquela categoria no proximo ciclo.

---

## 2026-04-20T07:43:17+00:00 - F_trailing
**Cycle**: #27  |  **Promoted**: NO  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `trailing_stop_mode`
**Sweep file**: `local_fullmetric_sweep_continuous_F_trailing_20260420_074220.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 21.1938 | 69.12% | 69.23% | -8.19% | 11.15% | 36 |
| incumbent (worktree) | 22.9224 | 69.23% | 69.57% | -8.04% | 11.26% | 35 |

*(no candidate produced metrics — sweep returned empty)*

**Learning**: nenhum candidato cruzou os guardrails de WR ou de seguranca; o incumbent ja esta no Pareto local desse conjunto de genes. Considerar marcar essas genes como 'exploradas recentemente' por 30 dias.

---

## 2026-04-20T08:24:30+00:00 - A_risk_management
**Cycle**: #28  |  **Promoted**: YES  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `stop_atr_mult, stop_tighten_after_bars, stop_tighten_factor, max_loss_per_trade_pct, equity_drawdown_stop_pct`
**Sweep file**: `local_fullmetric_sweep_continuous_A_risk_management_20260420_082054.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 19.7032 | 69.23% | 68.66% | -8.28% | 10.49% | 33 |
| incumbent (worktree) | 24.3079 | 70.00% | 70.59% | -8.05% | 10.37% | 33 |
| **candidate** | 25.5206 | 70.00% | 70.59% | -8.04% | 10.56% | 33 |

**Best candidate label**: `equity_drawdown_stop_pct:+1`

**Moves applied to incumbent**:

- `equity_drawdown_stop_pct`: `0.18` -> `0.2` (step 0.02)

**Learning**: o move `equity_drawdown_stop_pct: 0.18->0.2` foi promovido. Categoria/genes adjacentes podem render melhorias semelhantes; considerar sweep daquela categoria no proximo ciclo.

---

## 2026-04-20T10:29:24+00:00 - D_entry_filter
**Cycle**: #29  |  **Promoted**: YES  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `vote_threshold_long, vote_threshold_short, z_threshold, signal_ema_span, entry_confirmation_days, entry_discount_atr_frac, score_strength_scaling, min_signal_strength, entry_score_threshold, score_percentile_trigger`
**Sweep file**: `local_fullmetric_sweep_continuous_D_entry_filter_20260420_102407.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 19.7890 | 69.05% | 68.75% | -8.31% | 10.95% | 31 |
| incumbent (worktree) | 24.3437 | 69.44% | 70.00% | -8.11% | 10.59% | 33 |
| **candidate** | 23.8979 | 69.84% | 70.00% | -8.13% | 10.59% | 33 |

**Best candidate label**: `score_percentile_trigger:-1`

**Moves applied to incumbent**:

- `score_percentile_trigger`: `0.6000000000000001` -> `0.55` (step 0.05)

**Learning**: o move `score_percentile_trigger: 0.6000000000000001->0.55` foi promovido. Categoria/genes adjacentes podem render melhorias semelhantes; considerar sweep daquela categoria no proximo ciclo.

---

## 2026-04-20T11:36:36+00:00 - B_take_profit
**Cycle**: #30  |  **Promoted**: YES  |  **Codex sync**: YES  |  **combo_budget**: 8
**Genes swept**: `reward_risk_ratio, partial_take_pct, partial_take_level, partial_take_pct_2, partial_take_level_2`
**Sweep file**: `local_fullmetric_sweep_continuous_B_take_profit_20260420_113233.json`

### Metrics

| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |
|---|---:|---:|---:|---:|---:|---:|
| baseline (git:main) | 20.7554 | 70.00% | 69.44% | -8.29% | 11.09% | 32 |
| incumbent (worktree) | 24.0138 | 69.70% | 70.00% | -8.08% | 10.99% | 33 |
| **candidate** | 24.1736 | 70.00% | 70.00% | -8.13% | 11.09% | 33 |

**Best candidate label**: `partial_take_pct:-1`

**Moves applied to incumbent**:

- `partial_take_pct`: `0.2` -> `0.1` (step 0.1)

**Learning**: o move `partial_take_pct: 0.2->0.1` foi promovido. Categoria/genes adjacentes podem render melhorias semelhantes; considerar sweep daquela categoria no proximo ciclo.

---

