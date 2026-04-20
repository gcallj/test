# Overfitting diagnostic report

Generated: overfitting_diagnostic.py sobre 39 attempts (14 promoted) + 32 sweeps

## Executive summary

- Incumbent atual: fit=21.02, WR=69.7%, alpha=-8.24%, trades=29
- Historico: 39 attempts, 14 promoted (35.9%)

## 1. Wilson 95% CI para WR

- **WR observado**: 0.6974 (69.74%), com n=29 trades
- **Wilson 95% IC**: [0.5155, 0.8331] = [51.55%, 83.31%]
- **Veredito**: significativamente melhor que 50%
- **Interpretacao**: com n=29 trades, o IC tem largura 31.76pp — amostra pequena, alta variancia

## 2. Multiple testing correction

- **N attempts totais**: 39
- **N promovidos**: 14 (taxa 35.9%)
- **Bonferroni alpha=0.05/N**: p < 0.001282 exigido por attempt individual
- **Binomial vs null rate 10%**: z=5.39, one-sided p=3.504e-08
- **Interpretacao**: Taxa ALTA — forte evidencia contra null (mas pode ser skill OU overfit a mesma janela)

## 3. Trajetoria do alpha ao longo dos sweeps

- **Alpha inicial** (primeiro sweep): -8.28%
- **Alpha atual** (ultimo sweep): -8.27%
- **Delta total**: +0.01pp (melhorou)
- **RED FLAG**: alpha atual (-8.27%) e NEGATIVO — sistema subperforma buy-and-hold em termos anualizados

**Amostragem cronologica**:

| idx | timestamp | fit | alpha_ann |
|---:|---|---:|---:|
| 0 | 2026-04-18T15:38:22 | -75.19 | -8.28% |
| 10 | 2026-04-19T16:25:31 | 19.64 | -8.31% |
| 21 | 2026-04-19T10:47:38 | 19.83 | -8.30% |
| 31 | 2026-04-18T13:56:00 | 20.72 | -8.27% |

## 4. Gene drift detection (retrospective B4 preview)


**Genes movidos multiplas vezes (red flag se >=3 na mesma direcao)**:

| gene | #promos | movimentos | direcao |
|---|---:|---|---|
| `consecutive_loss_cooldown` | 2 | 6.000→7.000 → 6.000→7.000 | ALL UP |
| `entry_score_threshold` | 2 | 0.150→0.200 → 0.150→0.200 | ALL UP |
| `regime_threshold` | 2 | 0.250→0.200 → 0.300→0.250 | ALL DOWN |

## 5. Signal-to-noise do ganho por attempt

- **Ganho medio de fit por sweep**: +3.0939
- **Desvio padrao dos deltas**: 66.7694
- **Signal-to-noise ratio**: 0.05
- **Interpretacao**: SNR < 1 sugere que os ganhos sao indistinguiveis de ruido — overfit provavel

## Vereditos e recomendacoes

- 🔴 Alpha negativo atual (subperforma buy-and-hold)
- 🔴 Poucos trades por ticker (< 40, amostra fraca)
- 🟡 Taxa de promocao alta (36%) sem correcao multiple testing

**VEREDITO**: OVERFIT PROVAVEL. Recomenda-se:
1. Rodar Fase A1 completa (pristine holdout temporal — veja `docs/run_pristine_holdout.md`)
2. Ativar `GA_ALPHA_FLOOR_TOLERANCE_PP=0.0` (ja default apos B2)
3. Considerar reverter incumbent para um commit pre-tuning recente
4. Implementar Fase B3 (bootstrap CI) no apply

---

## Como rodar Fase A1 completa (pristine holdout temporal)

Esse script NAO reavalia o GA em uma janela temporal cortada. Para fazer isso corretamente:

```bash
# 1. Adicione ao ga_run.py: parametro GA_EVAL_CUTOFF_DATE no build_temporal_windows
# 2. Rode load mode com cutoff para baseline e incumbent:
GA_RUN_MODE=load GA_EVAL_CUTOFF_DATE=2026-01-31 \
  python -c 'from ga_run import run; run()'
# 3. Compare fit/WR/alpha/MDD: se incumbent piora mais que baseline no holdout,
#    esta overfit.
```

