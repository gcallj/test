# Pristine holdout report

- **Cutoff date** (final do treino, inicio do holdout): 2026-01-10
- **Holdout days**: 90
- **Baseline**: `git:1bd4217`
- **Incumbent**: `file:global_ga_checkpoint.json`

## 1. Full dataset (sanity check)

### Full

| Metric | Baseline | Incumbent | Δ (incb-base) |
|---|---:|---:|---:|
| fit | 18.2217 | 18.5353 | +0.314 |
| WR | 68.66% | 68.42% | -0.24pp |
| WR_target | 68.00% | 68.00% | +0.00pp |
| alpha_ann | -8.22% | -8.22% | -0.00pp |
| MDD | 11.57% | 11.53% | -0.04pp |
| trades | 31.0000 | 29.0000 | -2.000 |
| n_tickers_active | 101.0000 | 101.0000 | +0.000 |

## 2. Pre-holdout (training range, dataset minus last N days)

### Pre-holdout

| Metric | Baseline | Incumbent | Δ (incb-base) |
|---|---:|---:|---:|
| fit | 16.8212 | 16.9877 | +0.166 |
| WR | 68.38% | 67.09% | -1.29pp |
| WR_target | 67.11% | 67.21% | +0.11pp |
| alpha_ann | -8.26% | -8.26% | +0.00pp |
| MDD | 11.75% | 11.62% | -0.13pp |
| trades | 29.0000 | 29.0000 | +0.000 |
| n_tickers_active | 98.0000 | 98.0000 | +0.000 |

## 3. Holdout only (last 90 days — UNSEEN)

### Holdout (90d)

| Metric | Baseline | Incumbent | Δ (incb-base) |
|---|---:|---:|---:|
| fit | -126.4845 | -127.4540 | -0.969 |
| WR | 0.00% | 0.00% | +0.00pp |
| WR_target | 0.00% | 0.00% | +0.00pp |
| alpha_ann | -49.43% | -49.43% | -0.00pp |
| MDD | 0.00% | 0.00% | +0.00pp |
| trades | 0.0000 | 0.0000 | +0.000 |
| n_tickers_active | 0.0000 | 0.0000 | +0.000 |

## Veredito

### Ganho do incumbent vs baseline em cada slice

| Metric | Δ pre-holdout | Δ holdout | Gap (pre - holdout) |
|---|---:|---:|---:|
| fit | +0.166 | -0.969 | +1.136 |
| WR | -1.29pp | +0.00pp | -1.29pp |
| alpha_ann | +0.00pp | -0.00pp | +0.01pp |
| MDD | -0.13pp | +0.00pp | -0.13pp |

- 🔴 Fit gap `1.136` — incumbent teve vantagem grande em pre-holdout que NAO sustenta no holdout

**VEREDITO**: OVERFIT MARGINAL. Uma metrica degrada no holdout. Monitorar; considerar ativar Fase B1 (holdout no acceptance gate).
