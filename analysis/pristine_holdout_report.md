# Pristine holdout report

- **Cutoff date** (final do treino, inicio do holdout): 2026-01-06
- **Holdout days**: 90
- **Baseline**: `git:1bd4217`
- **Incumbent**: `file:global_ga_checkpoint.json`

## 1. Full dataset (sanity check)

### Full

| Metric | Baseline | Incumbent | Δ (incb-base) |
|---|---:|---:|---:|
| fit | -15.6529 | -17.8510 | -2.198 |
| WR | 65.28% | 63.64% | -1.64pp |
| WR_target | 67.46% | 67.95% | +0.49pp |
| alpha_ann | -3.54% | -3.74% | -0.20pp |
| MDD | 18.13% | 18.11% | -0.02pp |
| trades | 85.5000 | 80.5000 | -5.000 |
| n_tickers_active | 118.0000 | 118.0000 | +0.000 |

## 2. Pre-holdout (training range, dataset minus last N days)

### Pre-holdout

| Metric | Baseline | Incumbent | Δ (incb-base) |
|---|---:|---:|---:|
| fit | -13.3964 | -14.2265 | -0.830 |
| WR | 65.26% | 63.54% | -1.72pp |
| WR_target | 67.16% | 67.71% | +0.55pp |
| alpha_ann | -3.51% | -3.68% | -0.17pp |
| MDD | 18.13% | 18.10% | -0.03pp |
| trades | 84.5000 | 79.5000 | -5.000 |
| n_tickers_active | 118.0000 | 118.0000 | +0.000 |

## 3. Holdout only (last 90 days — UNSEEN)

### Holdout (90d)

| Metric | Baseline | Incumbent | Δ (incb-base) |
|---|---:|---:|---:|
| fit | -15.0921 | -17.7102 | -2.618 |
| WR | 50.00% | 50.00% | +0.00pp |
| WR_target | 60.00% | 60.00% | +0.00pp |
| alpha_ann | -16.79% | -17.26% | -0.48pp |
| MDD | 4.20% | 4.37% | +0.17pp |
| trades | 7.0000 | 7.0000 | +0.000 |
| n_tickers_active | 6.0000 | 7.0000 | +1.000 |

## Veredito

### Ganho do incumbent vs baseline em cada slice

| Metric | Δ pre-holdout | Δ holdout | Gap (pre - holdout) |
|---|---:|---:|---:|
| fit | -0.830 | -2.618 | +1.788 |
| WR | -1.72pp | +0.00pp | -1.72pp |
| alpha_ann | -0.17pp | -0.48pp | +0.31pp |
| MDD | -0.03pp | +0.17pp | -0.20pp |

- 🔴 Fit gap `1.788` — incumbent teve vantagem grande em pre-holdout que NAO sustenta no holdout
- 🔴 Alpha holdout: incumbent -17.26% < baseline -16.79% (incumbent pior no unseen)

**VEREDITO**: OVERFIT DETECTADO. Multiplas metricas degradam no holdout. Recomenda-se: ativar Fase B1, considerar reverter para um commit pre-tuning.
