# The Swedish strategy is a monthly momentum rotation

The swing-trading engine failed its walk-forward replay for structural
reasons (docs/sweden/diagnostico.md): a 0.70% round trip ate 166% of its
+0.42% gross edge per four-day trade, it sat in cash half the time, and its
GA optimised per-ticker win rates rather than the portfolio against the
index. Sweden now runs a rule-based rotation (`rotation_se.py`): at each month
end, rank the liquid universe by 12-1 month momentum, hold up to six names
with equal weight, keep a holding while it stays in the top 20, scale stock
exposure down when the basket's 63-day volatility exceeds 20% a year, execute
at the next open. Few, discrete parameters were chosen on 2014–2023
development data by robustness across sub-periods, not by a GA.

## Considered Options

Retuning the GA with a portfolio-level fitness was rejected: the horizon and
cost problem is in the trading style, not in the optimiser. Market-regime
filters (200-day average, 12-month absolute momentum) were tested and dropped:
on development data they cut excess return more than drawdown.

## Consequences

ADR 0001's limit-order rule applied to swing entries; rebalances are
market-on-open orders, with slippage and brokerage charged per side. The GA,
`daily_se.py` and the swing replays remain for reference but are no longer the
Swedish strategy.
