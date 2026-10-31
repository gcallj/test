# Avaliacao da estrategia v4 — eventos de informacao publica

Especificacao congelada em `data/strategy_se_v4.json` antes de ligar os dados alternativos a retornos (ADR 0008). Periodo dos testes: 2016-07-01 a 2026-10-30.

## Testes por hipotese

Retorno anormal = retorno da acao (abertura a abertura) menos o das acoes do mesmo quintil de retorno de 63 dias, nas mesmas datas. Serie semanal da media dos eventos; t de Newey-West e p de bootstrap estacionario no sentido previsto; Holm sobre as quatro hipoteses.

| Hipotese | Eventos | Retorno anormal medio | Acerto | t NW | p bootstrap | Holm | Tamanho | |
|---|---:|---:|---:|---:|---:|---|---|---|
| H1 compras de insiders (60 pregoes) | 1253 | +0.05% | +48.04% | 1.45 | 0.068 | NAO PASSA | NAO PASSA | **NAO PASSA** |
| H2 posicao vendida alta (20 pregoes, previsto negativo) | 4770 | -0.29% | +53.75% | 0.98 | 0.153 | NAO PASSA | PASSA | **NAO PASSA** |
| H3 tom positivo dos comunicados (20 pregoes) | 1349 | -0.32% | +48.63% | -0.09 | 0.546 | NAO PASSA | NAO PASSA | **NAO PASSA** |
| H4 reacao positiva ao balanco (60 pregoes) | 1167 | -1.21% | +43.87% | -0.77 | 0.787 | NAO PASSA | NAO PASSA | **NAO PASSA** |

Sinais usados na carteira: nenhum; filtro H2: nao.

## Criterio 1 (2023-10-31 a 2026-10-30)

- Retorno +63.60% vs indice +63.60% -> PASSA
- Folds com expectativa positiva: 0/6 -> NAO PASSA
- Max drawdown -21.29% vs indice -21.29% -> NAO PASSA
- Trades: 0

**REPROVADA**

## Periodo inteiro

- CAGR +11.79% vs indice +11.79%; max drawdown -33.16% vs -33.16%; 0 trades; excesso diario p bootstrap 0.428

## Diagnostico anunciado (nao decide)

- Todas as familias longas com filtro H2, 2023-2026: retorno +76.65% vs +63.60%, MDD -19.99% vs -21.29%, folds 6/6, 93 trades

## Por ano (retorno anormal medio, eventos)

| Ano | H1 | H2 | H3 | H4 |
|---|---:|---:|---:|---:|
| 2016 | +0.18% (22) | +0.99% (157) | -1.17% (28) | +0.44% (27) |
| 2017 | +0.31% (73) | -1.08% (355) | +0.01% (95) | -0.18% (75) |
| 2018 | +0.66% (99) | +1.11% (366) | -0.14% (118) | -2.14% (113) |
| 2019 | -1.15% (85) | +0.04% (418) | +0.18% (101) | +0.61% (103) |
| 2020 | +0.94% (117) | -0.89% (449) | -2.15% (96) | -3.55% (113) |
| 2021 | +0.56% (126) | -0.56% (487) | -0.66% (188) | +0.24% (103) |
| 2022 | -1.31% (144) | -0.30% (511) | +0.07% (140) | -2.73% (128) |
| 2023 | -0.86% (146) | -0.54% (519) | -0.14% (155) | -0.15% (123) |
| 2024 | +0.81% (145) | +0.87% (531) | +0.91% (135) | -1.09% (135) |
| 2025 | +1.03% (180) | -2.74% (547) | -0.48% (159) | -1.82% (136) |
| 2026 | -0.88% (116) | +1.27% (430) | -0.81% (134) | -0.87% (111) |

Versao dos dados (sha256): `fi_insiders.parquet` ad28eb3aecb10d69, `fi_shorts.parquet` be89d06da5c04f65, `news.parquet` ff3f2bcbbd6345cd, `news_tone.parquet` 651a99d86a0ca647, `prices_se.parquet` a773d44491513e6e

## Nota posterior a rodada (2026-10-31)

O veredito acima e o da rodada unica e nao muda. Ao conferir o diagnostico, achei
uma inconsistencia no codigo: a carteira diagnostica usou a ordem de prioridade
H1 > H3 > H4, e a especificacao diz H1 > H4 > H3. Com a ordem da especificacao, a
mesma carteira rende **+56,6%** contra +63,6% do indice. O codigo foi corrigido
para as proximas rodadas.

A ordem de prioridade so decide quem fica com a vaga quando ha mais eventos que
vagas, mas muda o resultado de 3 anos de +18% a +81%:

| Prioridade | Com filtro H2 | Sem filtro |
|---|---:|---:|
| H1 > H3 > H4 | +75,4% | +78,5% |
| H1 > H4 > H3 (especificacao) | +56,6% | +49,3% |
| H3 > H1 > H4 | +64,7% | +39,7% |
| H3 > H4 > H1 | +69,1% | +43,0% |
| H4 > H1 > H3 | +67,3% | +80,7% |
| H4 > H3 > H1 | +41,1% | +18,2% |
| Indice ^OMXSBGI | +63,6% | +63,6% |

(retorno de 2023-10-31 a 2026-10-30, do primeiro fechamento ao ultimo; a primeira
linha difere do +76,65% acima porque aquele numero parte dos 30.000 SEK
anteriores aos custos do primeiro dia.)

Com cerca de 90 trades em 3 anos, o resultado da carteira e dominado pela sorte
de trajetoria; por isso a regra registrada decide pelos testes por evento. Por
trade, na carteira da especificacao: retorno bruto medio +3,69%, o fundo de
indice nas mesmas janelas +3,60%, o controle de mesmo quintil +2,78%. Nao ha
vantagem sobre simplesmente ficar no indice.
