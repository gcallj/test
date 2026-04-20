# Overfitting diagnostic report

Generated: overfitting_diagnostic.py sobre 55 attempts (23 promoted) + 59 sweeps

## Executive summary

- Incumbent atual: fit=24.17, WR=70.0%, alpha=-8.13%, trades=33
- Historico: 55 attempts, 23 promoted (41.8%)

## 1. Wilson 95% CI para WR

- **WR observado**: 0.7000 (70.00%), com n=33 trades
- **Wilson 95% IC**: [0.5297, 0.8286] = [52.97%, 82.86%]
- **Veredito**: significativamente melhor que 50%
- **Interpretacao**: com n=33 trades, o IC tem largura 29.89pp — amostra pequena, alta variancia

## 2. Multiple testing correction

- **N attempts totais**: 55
- **N promovidos**: 23 (taxa 41.8%)
- **Bonferroni alpha=0.05/N**: p < 0.0009091 exigido por attempt individual
- **Binomial vs null rate 10%**: z=7.87, one-sided p=1.832e-15
- **Interpretacao**: Taxa ALTA — forte evidencia contra null (mas pode ser skill OU overfit a mesma janela)

## 3. Trajetoria do alpha ao longo dos sweeps

- **Alpha inicial** (primeiro sweep): -8.28%
- **Alpha atual** (ultimo sweep): -8.23%
- **Delta total**: +0.06pp (melhorou)
- **RED FLAG**: alpha atual (-8.23%) e NEGATIVO — sistema subperforma buy-and-hold em termos anualizados

**Amostragem cronologica**:

| idx | timestamp | fit | alpha_ann |
|---:|---|---:|---:|
| 0 | 2026-04-18T15:38:22 | -75.19 | -8.28% |
| 19 | 2026-04-19T01:36:39 | 19.63 | -8.30% |
| 39 | 2026-04-18T17:52:20 | 21.02 | -8.24% |
| 58 | 2026-04-20T10:56:55 | 23.64 | -8.23% |

## 4. Gene drift detection (retrospective B4 preview)


**Genes movidos multiplas vezes (red flag se >=3 na mesma direcao)**:

| gene | #promos | movimentos | direcao |
|---|---:|---|---|
| `consecutive_loss_cooldown` | 4 | 7.000→8.000 → 9.000→10.000 → 8.000→9.000 | ALL UP 🚩 |
| `regime_threshold` | 4 | 0.300→0.250 → 0.300→0.250 → 0.250→0.300 | mixed |
| `partial_take_pct` | 3 | 0.200→0.100 → 0.300→0.200 → 0.200→0.100 | ALL DOWN 🚩 |
| `momentum_confirm_days` | 2 | 5.000→4.000 → 5.000→4.000 | ALL DOWN |

**2 gene(s) com drift suspeito** — veja Fase B4 do plano.

## 5. Signal-to-noise do ganho por attempt

- **Ganho medio de fit por sweep**: +1.7040
- **Desvio padrao dos deltas**: 48.7546
- **Signal-to-noise ratio**: 0.03
- **Interpretacao**: SNR < 1 sugere que os ganhos sao indistinguiveis de ruido — overfit provavel

## Vereditos e recomendacoes

- 🔴 Alpha negativo atual (subperforma buy-and-hold)
- 🔴 Poucos trades por ticker (< 40, amostra fraca)
- 🟡 Taxa de promocao alta (42%) sem correcao multiple testing
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

