# The backtest uses the real execution rule

The backtest used to fill a buy at a limit of `open - discount * ATR` when the
day's low touched it, and at the open otherwise. That choice needs the day's
low before it is known, so the simulation never missed an entry, while a real
trader must choose market-at-open or a limit order in advance. We changed the
backtest to the rule that is actually traded: a limit order at the emitted
entry price (computed from data up to the signal day's close), valid for three
trading days, cancelled if the price closes below the stop; an expired order
is a missed entry and counts as no trade.

## Consequences

Fitness from before this change is not comparable with fitness after it, for
either market. Every genome evaluated under the old rule must be retrained
before its numbers mean anything under the new one.
