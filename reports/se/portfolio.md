# Portfolio replay — mercado sueco (walk-forward)

Periodo: 2023-10-31 a 2026-10-30 · conta ISK de 30.000 SEK · ate 6 posicoes de 5,000 SEK · index benchmark `^OMXSBGI` (com dividendos)

## Veredito (criterio 1 do plano)

| Criterio | Sistema | Indice | Resultado |
|---|---:|---:|---|
| Retorno total depois de custos | -26.2% | +63.6% | NAO PASSA |
| Folds com expectancy positiva | 2 de 6 | precisa >= 4 | NAO PASSA |
| Max drawdown da conta | -45.6% | -21.3% | NAO PASSA |

**REPROVADO: a resposta honesta e o fundo de indice**

## Conta

- Patrimonio final: 22,137 SEK (indice: 49,080 SEK)
- CAGR: -9.6% (indice: +17.8%)
- Trades executados: 550 · ordens sem vaga: 1990 · vagas ocupadas em media: 4.4 de 6

## Por fold

| Fold | Trades | Win rate | Expectancy | Max drawdown da conta |
|---:|---:|---:|---:|---:|
| 1 | 62 | +75.8% | +1.5% | -4.0% |
| 2 | 115 | +53.0% | -0.5% | -11.2% |
| 3 | 97 | +42.3% | -1.1% | -22.9% |
| 4 | 103 | +58.3% | -0.5% | -15.5% |
| 5 | 86 | +55.8% | -0.8% | -17.7% |
| 6 | 87 | +64.4% | +0.5% | -4.1% |

## Cenario otimista (courtage zero)

Retorno +16.4% vs indice +63.6% · folds positivos 3 de 6 · MDD -20.5% · tambem nao passaria no criterio 1.

Se o sistema so passa com courtage zero, isso e sinal de alerta (plano, Q19).

Gerado por `python portfolio_se.py`. Curva da conta: `portfolio_equity.csv`, `portfolio.png`.
