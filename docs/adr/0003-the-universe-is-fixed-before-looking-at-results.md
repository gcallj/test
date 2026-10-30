# The universe is fixed before looking at results

The first Swedish experiment removed eight tickers because the model did badly
on them in-sample, which inflates every later metric. The Swedish universe is
instead a frozen snapshot of today's Nasdaq Stockholm Large Cap list, keeping
only the most liquid share class per company, requiring a median daily
turnover of at least 10 MSEK over the last 12 months, and excluding crypto.
Model results never add or remove a ticker.

## Consequences

The snapshot carries survivorship bias, since the data source has neither
delisted companies nor historical index membership. It is accepted and
documented rather than corrected, and results from the most recent folds carry
the most weight.
