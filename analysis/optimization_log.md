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

