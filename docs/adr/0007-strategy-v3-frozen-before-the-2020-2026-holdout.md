# Strategy v3 is frozen before a single 2020–2026 holdout

The user asked for a strategy as active as v1 that anticipates moves, trades
little in practice and is statistically relevant. No pristine history is left:
2014–2026 was used to choose and judge v1 and v2. v3 was therefore developed
only on 2010–2019 and judged once on 2020-01 to 2026-10, with four tests fixed
in `data/strategy_se_v3.json` before that run: forecast skill (weekly IC,
Newey-West t ≥ 2, over all eligible stock-weeks, so few trades do not starve
the test), excess over the index significant by stationary bootstrap
(p < 0.05), deflated Sharpe above 0.95 across the 40 development variants, and
criterion 1 on the six folds. All four must pass.

## Considered Options

Fitted models (ridge, LightGBM) were the first choice and failed the
development forecast test (t below 2; LightGBM near zero). Fixed-sign
composites of published effects — one-week and one-month reversal, 12-1
momentum, low volatility, low turnover surge — passed it and were kept. A
market-timing filter was tested again and dropped: it lowered every variant.

## Consequences

2020–2023 was already seen while choosing v2 and 2023–2026 while judging v1
and v2, so even a pass here is weaker evidence than a pass on unseen data;
paper trading remains the final test. The survivorship bias of today's
universe favours reversal signals ("buy recent losers"), more so in the
development period than in the holdout.
