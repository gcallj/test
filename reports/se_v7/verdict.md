# Avaliacao da estrategia v7 — calendario (Halloween e mes do dividendo)

Especificacao congelada em `data/strategy_se_v7.json` antes de qualquer retorno (ADR 0011). Periodo: 2013-04-01 a 2026-10-30. Nenhum parametro foi ajustado.

## Testes

| Hipotese | Amostra | Efeito | t NW | p bootstrap | Holm | |
|---|---:|---:|---:|---:|---|---|
| T1 Halloween: inverno (nov-abr) - verao (mai-out), ^OMXSPI mensal | 162 meses | +0.66%/mes (inverno +1.03%, verao +0.36%) | 0.91 | 0.161 | NAO PASSA | **NAO PASSA** |
| T2 mes do dividendo: retorno anormal vs mesmo quintil | 484 eventos em 120 meses | -0.67%/evento (acerto +42.56%) | -1.87 | 0.970 | NAO PASSA | **NAO PASSA** (tamanho NAO PASSA) |

Diagnostico T1 no universo de peso igual desde 2006-07-01: +1.49%/mes, t 2.19, p 0.016 (243 meses).

Sinais usados na carteira: nenhum.

## Criterio 1 (2023-10-31 a 2026-10-30)

- Retorno +63.62% vs indice +63.60% -> PASSA
- Folds positivos: 0 -> NAO PASSA
- Max drawdown -21.29% vs indice -21.29% -> NAO PASSA
- Trades: 0

**REPROVADA**

## Periodo inteiro

- CAGR +11.27% vs indice +11.27%; max drawdown -33.16% vs -33.16%; 0 trades; 1 trocas de fundo

## Diagnosticos anunciados (nao decidem)

| Carteira | 2023-2026 | Folds | MDD | CAGR periodo inteiro | Trades (periodo) |
|---|---:|---:|---:|---:|---:|
| T1 | +39.14% vs +63.60% | 2 | -21.29% vs -21.29% | +8.08% vs +11.27% | 0 |
| T2 | +64.67% vs +63.60% | 4 | -21.55% vs -21.29% | +3.40% vs +11.27% | 484 |
| T1+T2 | +55.55% vs +63.60% | 4 | -21.55% vs -21.29% | +0.85% vs +11.27% | 484 |

## T1 por ano (soma dos retornos mensais do ^OMXSPI)

| Ano | Inverno | Verao |
|---|---:|---:|
| 2013 | +4.33% | +7.74% |
| 2014 | +8.63% | +2.58% |
| 2015 | +12.02% | -5.64% |
| 2016 | -0.66% | +6.33% |
| 2017 | +3.76% | +2.45% |
| 2018 | -6.41% | -1.57% |
| 2019 | +21.96% | +4.00% |
| 2020 | -0.62% | +12.72% |
| 2021 | +22.20% | +7.80% |
| 2022 | -16.32% | -11.93% |
| 2023 | +25.90% | -11.41% |
| 2024 | +3.88% | +1.59% |
| 2025 | -0.39% | +9.50% |
| 2026 | +1.76% | +6.29% |

## T2 por ano (retorno anormal medio, eventos)

| Ano | T2 |
|---|---:|
| 2013 | -1.59% (13) |
| 2014 | +0.35% (20) |
| 2015 | -1.85% (23) |
| 2016 | +0.47% (26) |
| 2017 | +0.71% (29) |
| 2018 | +0.47% (31) |
| 2019 | -1.29% (33) |
| 2020 | -0.89% (33) |
| 2021 | -2.33% (49) |
| 2022 | -2.47% (48) |
| 2023 | -2.10% (50) |
| 2024 | +1.60% (50) |
| 2025 | +0.24% (44) |
| 2026 | +0.12% (35) |

T2, eventos por mes do calendario: {1: 8, 2: 15, 3: 78, 4: 84, 5: 84, 6: 27, 7: 8, 8: 26, 9: 43, 10: 56, 11: 42, 12: 13}

Versao dos dados (sha256): `prices_se.parquet` a773d44491513e6e, `riksbank_policy_rate.parquet` c33d615e2633ec3b
