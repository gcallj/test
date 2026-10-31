# Avaliacao da estrategia v2 — rotacao mensal por momentum

Especificacao congelada em `data/strategy_se_v2.json` (`L252-S21-K6-B20-vt20-index`), escolhida so com dados ate 2023-10-30 e avaliada aqui uma unica vez (ADR 0006).

Periodo: 2023-10-31 a 2026-10-30 · conta ISK de 30,000 SEK · index benchmark `^OMXSBGI` (com dividendos)

## Veredito (criterio 1 do plano)

| Criterio | Sistema | Indice | Resultado |
|---|---:|---:|---|
| Retorno total depois de custos | +35.7% | +63.6% | NAO PASSA |
| Folds com expectancy positiva | 5 de 6 | precisa >= 4 | PASSA |
| Max drawdown da conta | -20.7% | -21.3% | PASSA |

**REPROVADO**

## Conta

- Patrimonio final: 40,710 SEK (indice: 49,080 SEK)
- CAGR: +10.7% (indice: +17.8%)
- Posicoes abertas por ano: 11.0 · exposicao media em acoes: 90%

## Por fold (trades pela data de compra; posicoes abertas no fim marcadas a mercado)

| Fold | Trades | Win rate | Expectancy | Max drawdown da conta |
|---:|---:|---:|---:|---:|
| 1 | 13 | +84.6% | +30.6% | -6.0% |
| 2 | 3 | +66.7% | +1.0% | -10.2% |
| 3 | 14 | +42.9% | +16.3% | -19.3% |
| 4 | 8 | +75.0% | +20.5% | -6.6% |
| 5 | 5 | +80.0% | +16.0% | -9.2% |
| 6 | 7 | +0.0% | -12.4% | -17.0% |

## Cenario com courtage zero

Retorno +39.4% · folds positivos 5 de 6 · MDD -20.1% · nao passaria no criterio 1.

Arquivos: `equity.csv`, `equity.png`, `trades.csv`, `rebalances.csv` (o que foi decidido em cada fim de mes e por que).
