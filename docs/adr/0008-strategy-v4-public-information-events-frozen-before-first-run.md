# Strategy v4 trades public-information events, frozen before its first run

v1–v3 used every Swedish price from 2010 to 2026-10-30, so another price-only
idea tested on that history would only measure how well it was fitted to it.
v4 instead uses information that prices do not already contain and that no
earlier strategy has touched: the Finansinspektionen insider register
(PDMR trades since July 2016), its short-position register (positions of
0.5% or more since November 2012), and the companies' own announcements on
Nasdaq Nordic (timestamps, categories and text, scored for tone by FinBERT).

Four hypotheses are taken from published work, with horizons, thresholds
and the pass rule fixed in `data/strategy_se_v4.json` before any of this
data is joined to a return:

- H1: open-market purchases by insiders;
- H2: high disclosed short interest, predicted to underperform, used only to
  exclude names because an ISK cannot sell short;
- H3: positive tone in reports and inside-information releases;
- H4: post-earnings drift after a strong three-day reaction.

Each event trades at the first open after the information was public, and is
measured against stocks in the same 63-day-return quintile on the same
dates. This matched control is chosen because insiders buy after falls, and
in a survivor-only universe "bought after a fall" already looks good. The
tests run once, from 2016-07-01 to 2026-10-30, with the rules fixed in the
spec:

- Holm correction across the four hypotheses;
- Newey-West t of at least 2 on the weekly series;
- a mean event return above the 0.70% round trip.

The portfolio then uses only the families that passed, keeps idle slots in
the index fund, and must still meet criterion 1 on the six folds.

## Considered Options

Fund holdings (FI quarterly files from 2018 Q4) were left out because the
files were republished in 2022 and the original publication dates are
unknown, so a backtest could not be made causal. GDELT news tone and Google
Trends were left out because their coverage and mapping of Swedish Large
Cap companies is poor next to the companies' own announcements. Macro series
were left out because market-timing filters lowered every v2 and v3 variant.
Paid fundamentals, analyst estimates and ownership data (Börsdata, Modular
Finance, LSEG) remain options if the owner wants to pay for them.

## Consequences

The registers are free, official and timestamped, so the daily run can read
them the same way the backtest does. The price side of the backtest is still
history that has already been seen, and the universe is still today's
survivors; a pass here is a reason to paper trade, not proof. Changing a
threshold after this run and trying again would manufacture a result, as in
ADR 0006.
