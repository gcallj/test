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

