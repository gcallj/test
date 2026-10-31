# Avaliacao da estrategia v5 — revisoes de analistas

Especificacao congelada em `data/strategy_se_v5.json` antes de ligar as revisoes a retornos (ADR 0009). Periodo dos testes: 2020-01-01 a 2026-10-30.

Retorno anormal = abertura a abertura menos o das acoes do mesmo quintil de retorno de 63 dias. Serie semanal; t de Newey-West e p de bootstrap no sentido previsto; Holm sobre as tres hipoteses.

| Hipotese | Eventos | Retorno anormal medio | Acerto | t NW | p bootstrap | Holm | Tamanho | |
|---|---:|---:|---:|---:|---:|---|---|---|
| R1 elevacao de recomendacao (20 pregoes) | 347 | +0.41% | +50.43% | 1.31 | 0.093 | NAO PASSA | NAO PASSA | **NAO PASSA** |
| R2 rebaixamento (60 pregoes, previsto negativo) | 244 | -0.16% | +52.87% | -0.42 | 0.674 | NAO PASSA | PASSA | **NAO PASSA** |
| R3 mais precos-alvo elevados que cortados (20 pregoes) | 432 | -0.82% | +43.98% | -1.31 | 0.909 | NAO PASSA | NAO PASSA | **NAO PASSA** |

Sinais usados: nenhum; filtro de rebaixamento: nao.

## Criterio 1 (2023-10-31 a 2026-10-30)

- Retorno +63.60% vs indice +63.60% -> PASSA
- Folds com expectativa positiva: 0/6 -> NAO PASSA
- Max drawdown -21.29% vs indice -21.29% -> NAO PASSA
- Trades: 0

**REPROVADA**

## Periodo inteiro

- CAGR +10.67% vs indice +10.67%; max drawdown -33.16% vs -33.16%; 0 trades

## Diagnostico anunciado (nao decide)

- R1+R3 com filtro de rebaixamento, 2023-2026: retorno +17.64% vs +63.60%, MDD -25.97% vs -21.29%, folds 2/6, 184 trades

## Por ano (retorno anormal medio, eventos)

| Ano | R1 | R2 | R3 |
|---|---:|---:|---:|
| 2020 | +1.62% (20) | +2.10% (11) | -0.06% (49) |
| 2021 | -0.44% (39) | -1.17% (25) | -1.24% (54) |
| 2022 | -1.00% (39) | +6.71% (25) | -1.25% (28) |
| 2023 | +0.85% (36) | -2.16% (30) | +0.53% (32) |
| 2024 | -0.67% (45) | +0.19% (47) | -0.09% (61) |
| 2025 | +0.56% (74) | -0.29% (64) | -1.28% (103) |
| 2026 | +1.31% (94) | -2.99% (42) | -1.22% (105) |

Versao dos dados (sha256): `actions.parquet` 93f2ac0076ed0ea6, `articles.parquet` 63ce06125de31c93, `prices_se.parquet` a773d44491513e6e
