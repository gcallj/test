# Holdout da estrategia v3 — selecao ativa por previsao

Especificacao congelada em `data/strategy_se_v3.json` (`combo5-H20-q3-K6`), escolhida so com 2010-2019 e avaliada uma unica vez em 2020-01-01 a 2026-10-30 (ADR 0007).

## Testes registrados antes da rodada

| Teste | Resultado | Exigido | |
|---|---:|---:|---|
| S1 previsao: IC semanal medio (t Newey-West) | 0.0040 (t = 0.35, 353 semanas) | t >= 2 | NAO PASSA |
| S2 excesso diario sobre o indice (p bootstrap) | p = 0.882 | p < 0,05 | NAO PASSA |
| S3 Deflated Sharpe (40 tentativas) | 0.000 | > 0,95 | NAO PASSA |
| C1 criterio 1 nos 6 folds (2023-2026) | retorno +34.2% vs +63.6%; folds 5/6; MDD -20.8% vs -21.3% | os 3 | NAO PASSA |

**REPROVADA**

## Holdout inteiro (2020-01 a 2026-10)

- CAGR +4.8% · indice +10.7% · universo peso igual +12.0%
- Max drawdown -36.7% · indice -33.2%
- Trades por ano: 46 · exposicao media 98%
- Trades fechados: 306 · resultado medio +0.9% (p bootstrap 0.129)
- Diagnostico: excesso sobre o universo de peso igual, p bootstrap 0.940 (o quanto vem do sinal, e nao de segurar o universo)
- IC positivo em 50% das semanas

Arquivos: `equity.csv`, `equity.png`, `trades.csv`, `decisions.csv`, `ic_weekly.csv`, `verdict.json`.
