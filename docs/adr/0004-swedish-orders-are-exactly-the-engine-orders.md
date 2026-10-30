# Swedish orders are exactly the engine's orders

The Brazilian daily pipeline rewrote the backtest's output after the fact: it
promoted holds to buys by Minervini stage, demoted buys by win rate, target
distance, confidence and universe tier, and recomputed entry, stop and target
from pivot levels. None of that was simulated, so no backtest or replay could
say whether it helped. For Sweden, the orders sent each day are the engine's
own orders, with the engine's own levels (`daily_se.py` runs the same engine
as the backtest and the replays, one empty bar ahead), and the post-hoc
overrides are not used. A filter earns its way back only by moving into the
engine, where the walk-forward replay can measure it.

## Consequences

The Brazilian `Apply` sheet columns (`recomendacao`, `confidence`, `rank`,
Minervini text) do not exist for Sweden; the daily report lists orders, live
orders and open positions with tomorrow's stop and target. When there are more
orders than free slots, signal strength decides, as in the portfolio replay.
