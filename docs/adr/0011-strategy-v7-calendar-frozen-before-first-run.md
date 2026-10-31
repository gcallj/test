# Strategy v7 trades the calendar, with no tunable parameter, frozen before its run

After six failed strategies the owner asked for an alternative, even one with
fewer trades. A calendar strategy is the lowest-frequency family with
published evidence, and it has no parameter to fit, which removes the
noise-fitting that undid v1 and v6.

## Hypotheses

Both are taken from the literature and fixed in `data/strategy_se_v7.json`
before any return is computed:

- **T1, Halloween.** The index earns more from November to April than from
  May to October (Bouman & Jacobsen 2002 report Sweden among the strongest
  markets). The strategy holds the index fund in winter and cash at the
  Riksbank policy rate in summer: two fund switches a year, no brokerage.
- **T2, dividend month.** Stocks earn more in the month they are expected
  to go ex-dividend (Hartzmark & Solomon 2013). The strategy holds, each
  month, the six eligible names that went ex in the same calendar month a
  year earlier, by trailing yield, and keeps the rest in the index fund.

## Tests

- T1: monthly log returns of `^OMXSPI` from 2013-04 to 2026-10, signed +1 in
  winter and −1 in summer; Newey-West t and stationary bootstrap on the
  mean. The price index leaves out dividends, which fall mostly in winter,
  so the test is conservative.
- T2: v4's matched control (same 63-day-return quintile) on the monthly
  mean abnormal return, with the 0.70% round-trip floor.
- Holm across T1 and T2; t ≥ 2.
- Criterion 1 for the resulting portfolio. T1 has no trades, so its fold
  rule is adapted before the run: a fold counts as positive when the excess
  over the index in the sessions spent in cash is positive, and two of the
  three summer folds must pass.

## Considered Options

- Turn-of-the-month and pre-holiday effects trade too often for the cost.
- Annual momentum rotation is a variant of v2 on data v2 already used.
- Insider purchases in Mid/Small Cap remain the best-supported untested
  idea, but need paid prices of delisted companies.

## Consequences

Yahoo serves the Swedish indices only from March 2013, so T1 is tested on
13.5 years: only a large effect can pass. Dichtl & Drobetz (2015) found the
Halloween effect weakening after 2000, so a failure here is the likelier
outcome. A pass earns paper trading, not real money, because the price
history has been seen by v1–v6 even though nothing here was fitted to it.
