# Avaliacao da estrategia v6 — swing derivado da v1 com CMA-ES e entradas novas

Especificacao congelada em `data/strategy_se_v6.json` antes da rodada (ADR 0010). Walk-forward de 6 folds (2023-10-31 a 2026-10-30), cada genoma treinado so nos 4 anos anteriores ao fold.

| Teste | Resultado | Exigido | |
|---|---:|---:|---|
| C1 criterio 1 | retorno +54.46% vs +63.60%; folds 3/6; MDD -18.97% vs -21.29% | os 3 | NAO PASSA |
| S excesso diario sobre o indice (bootstrap) | p = 0.545 | p < 0,05 | NAO PASSA |

**REPROVADA**

## Comportamento

- Trades: 124 (41/ano); prazo medio 32.0 pregoes; exposicao media em acoes 89%
- Resultado medio por trade: bruto +2.65%, liquido +1.93%
- Saidas: time stop 62, score 33, stop 27, alvo 2

## Treino por fold

| Fold | Treino | Fitness | Robusta | Tempo |
|---|---|---:|---:|---:|
| 1 | 2019-10-30 a 2023-10-30 | +0.084 | +0.058 | 30s |
| 2 | 2020-04-29 a 2024-04-29 | +0.109 | +0.034 | 27s |
| 3 | 2020-10-30 a 2024-10-30 | +0.140 | +0.064 | 27s |
| 4 | 2021-04-29 a 2025-04-29 | +0.164 | +0.127 | 26s |
| 5 | 2021-10-30 a 2025-10-30 | +0.114 | +0.044 | 26s |
| 6 | 2022-04-29 a 2026-04-29 | +0.102 | +0.048 | 26s |

## Genomas (pesos das entradas e genes de operacao)

| | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | fold 6 |
|---|---:|---:|---:|---:|---:|---:|
| w_mom12_1 | -0.128 | 0.841 | 0.789 | 0.252 | 0.056 | 0.827 |
| w_resid5 | 0.159 | 0.209 | 0.287 | 0.044 | -0.382 | 0.290 |
| w_resid21 | 0.570 | 0.737 | 0.411 | 0.056 | -0.457 | 0.572 |
| w_hi52 | -0.276 | -0.134 | 0.102 | -0.364 | -0.379 | -0.013 |
| w_vol63 | -0.002 | 0.124 | -0.299 | -0.495 | -0.374 | -0.077 |
| w_gap200 | -0.338 | 0.278 | -0.173 | -0.478 | -0.278 | -0.212 |
| w_insider60 | -0.439 | 0.580 | 0.034 | 0.651 | 0.319 | -0.168 |
| w_short_si | -0.358 | -0.355 | -0.111 | -0.764 | -0.759 | -0.555 |
| w_short_chg20 | 0.268 | -0.473 | -0.211 | 0.218 | -0.615 | -0.069 |
| w_analyst_rating20 | -0.509 | 0.326 | 0.268 | -0.826 | 0.238 | -0.739 |
| w_analyst_target20 | -0.507 | 0.535 | 0.349 | -0.164 | 0.177 | -0.023 |
| w_tone60 | -0.744 | 0.510 | 0.042 | 0.544 | 0.532 | -0.240 |
| w_ear60 | -0.706 | -0.775 | -0.667 | -0.914 | -0.434 | -0.476 |
| entry_q | 0.937 | 0.882 | 0.917 | 0.895 | 0.981 | 0.819 |
| exit_q | 0.569 | 0.352 | 0.346 | 0.565 | 0.294 | 0.411 |
| discount_atr | 0.431 | 0.486 | 0.194 | 0.193 | 0.661 | 0.880 |
| stop_atr | 3.323 | 4.842 | 4.030 | 4.574 | 3.939 | 3.618 |
| reward_risk | 5.168 | 4.184 | 5.075 | 4.823 | 3.000 | 4.071 |
| time_stop | 39.000 | 39.000 | 44.000 | 48.000 | 56.000 | 46.000 |
## Nota sobre a geracao deste relatorio

A rodada unica (`python swing_se.py evaluate`, 2026-10-31) gravou `verdict.json`,
`genomes.csv`, `equity.csv` e `trades.csv` e quebrou em seguida, ao montar a tabela
dos genomas (faltava o pacote `tabulate`). Este texto e o grafico foram gerados por
`python swing_se.py report` a partir desses arquivos, sem otimizar de novo.

## Leitura

| | v1 (2023-2026) | v6 (2023-2026) |
|---|---:|---:|
| Retorno em 3 anos | -26,2% | +54,5% (indice +63,6%) |
| Trades por ano | 183 | 41 |
| Prazo medio | 4 pregoes | 32 pregoes |
| Exposicao em acoes | ~48% | 89% (o resto no fundo) |
| Resultado bruto por trade | +0,42% | +2,65% |
| Resultado liquido por trade | negativo | +1,93% |

As correcoes estruturais funcionaram: os trades passaram a pagar o custo. Mas a
carteira ainda rende menos que o indice (CAGR 15,6% contra 17,8%), e a vantagem
some com o tempo. A expectativa por trade foi +10,2% no fold 1, +2,5% e +1,6% nos
folds 2 e 3, e ficou negativa nos folds 4 a 6.

Os pesos trocam de sinal entre folds: momentum 12-1 vai de -0,13 a +0,84, revisoes
de analistas de -0,83 a +0,33 e tom de -0,74 a +0,54. Isso indica que o otimizador
ajusta ruido de cada janela de treino. So dois pesos tem o mesmo sinal nos 6 folds:
posicao vendida alta (sempre negativo) e reacao forte ao ultimo balanco (sempre
negativo). Os dois concordam com a v4 (H2 -0,29%, H4 -1,21%), mas la nao foram
significativos.
