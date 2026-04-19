# Overfitting diagnostic report

Generated: overfitting_diagnostic.py sobre 31 attempts (10 promoted) + 24 sweeps

## Executive summary

- Incumbent atual: fit=21.01, WR=69.7%, alpha=-8.29%, trades=31
- Historico: 31 attempts, 10 promoted (32.3%)

## 1. Wilson 95% CI para WR

- **WR observado**: 0.6974 (69.74%), com n=31 trades
- **Wilson 95% IC**: [0.5215, 0.8297] = [52.15%, 82.97%]
- **Veredito**: significativamente melhor que 50%
- **Interpretacao**: com n=31 trades, o IC tem largura 30.82pp — amostra pequena, alta variancia

## 2. Multiple testing correction

- **N attempts totais**: 31
- **N promovidos**: 10 (taxa 32.3%)
- **Bonferroni alpha=0.05/N**: p < 0.001613 exigido por attempt individual
- **Binomial vs null rate 10%**: z=4.13, one-sided p=1.807e-05
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
| 8 | 2026-04-19T08:07:44 | 21.35 | -8.28% |
| 16 | 2026-04-18T22:35:34 | 19.65 | -8.30% |
| 23 | 2026-04-18T13:56:00 | 20.72 | -8.27% |

## 4. Gene drift detection (retrospective B4 preview)


**Genes movidos multiplas vezes (red flag se >=3 na mesma direcao)**:

| gene | #promos | movimentos | direcao |
|---|---:|---|---|
| `consecutive_loss_cooldown` | 2 | 6.000→7.000 → 6.000→7.000 | ALL UP |

## 5. Signal-to-noise do ganho por attempt

- **Ganho medio de fit por sweep**: +4.1701
- **Desvio padrao dos deltas**: 77.8769
- **Signal-to-noise ratio**: 0.05
- **Interpretacao**: SNR < 1 sugere que os ganhos sao indistinguiveis de ruido — overfit provavel

## Vereditos e recomendacoes

- 🔴 Alpha negativo atual (subperforma buy-and-hold)
- 🔴 Poucos trades por ticker (< 40, amostra fraca)
- 🟡 Taxa de promocao alta (32%) sem correcao multiple testing

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

