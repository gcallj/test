# Overfitting diagnostic report

Generated: overfitting_diagnostic.py sobre 41 attempts (16 promoted) + 34 sweeps

## Executive summary

- Incumbent atual: fit=22.22, WR=70.0%, alpha=-8.15%, trades=32
- Historico: 41 attempts, 16 promoted (39.0%)

## 1. Wilson 95% CI para WR

- **WR observado**: 0.7000 (70.00%), com n=32 trades
- **Wilson 95% IC**: [0.5270, 0.8301] = [52.70%, 83.01%]
- **Veredito**: significativamente melhor que 50%
- **Interpretacao**: com n=32 trades, o IC tem largura 30.31pp — amostra pequena, alta variancia

## 2. Multiple testing correction

- **N attempts totais**: 41
- **N promovidos**: 16 (taxa 39.0%)
- **Bonferroni alpha=0.05/N**: p < 0.00122 exigido por attempt individual
- **Binomial vs null rate 10%**: z=6.19, one-sided p=2.916e-10
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
| 11 | 2026-04-20T02:32:51 | 21.02 | -8.24% |
| 22 | 2026-04-19T10:47:38 | 19.83 | -8.30% |
| 33 | 2026-04-18T13:56:00 | 20.72 | -8.27% |

## 4. Gene drift detection (retrospective B4 preview)


**Genes movidos multiplas vezes (red flag se >=3 na mesma direcao)**:

| gene | #promos | movimentos | direcao |
|---|---:|---|---|
| `consecutive_loss_cooldown` | 3 | 6.000→7.000 → 7.000→8.000 → 6.000→7.000 | ALL UP 🚩 |
| `entry_score_threshold` | 2 | 0.150→0.200 → 0.150→0.200 | ALL UP |
| `regime_threshold` | 2 | 0.250→0.200 → 0.300→0.250 | ALL DOWN |

**1 gene(s) com drift suspeito** — veja Fase B4 do plano.

## 5. Signal-to-noise do ganho por attempt

- **Ganho medio de fit por sweep**: +2.9064
- **Desvio padrao dos deltas**: 64.6297
- **Signal-to-noise ratio**: 0.04
- **Interpretacao**: SNR < 1 sugere que os ganhos sao indistinguiveis de ruido — overfit provavel

## Vereditos e recomendacoes

- 🔴 Alpha negativo atual (subperforma buy-and-hold)
- 🔴 Poucos trades por ticker (< 40, amostra fraca)
- 🟡 Taxa de promocao alta (39%) sem correcao multiple testing
- 🟡 Genes com drift (>=3 promos consecutivas)

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

