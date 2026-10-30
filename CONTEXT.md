# Trading signals

Daily buy/hold/sell signals for one stock market, produced by a tuned trading
policy and validated by simulation before anyone trades on them. Brazil (B3) is
frozen as it is; Sweden (Nasdaq Stockholm) is the market under active
development.

## Language

### Markets and instruments

**Market**:
An exchange together with its currency, costs, tax regime and index, run as
one independent instance of the system.
_Avoid_: bolsa, exchange (when costs or tax are meant)

**Ticker**:
One listed share class, identified by its market-data symbol.
_Avoid_: ativo, papel, stock

**Universe**:
The set of tickers a market evaluates, fixed by objective criteria decided
before looking at any model result.
_Avoid_: curated list, watchlist

**Universe tier**:
A ticker's live-trading grade — PREMIUM, HIGH_QUAL or REGULAR — derived from
the quality of its recent backtest.

**Operable**:
A ticker whose universe tier is PREMIUM or HIGH_QUAL: one the system would
actually trade. REGULAR tickers are downgraded to hold.

### Account and costs

**ISK**:
The Swedish investment savings account assumed for Sweden: taxed by a yearly
charge on account value, never per trade.
_Avoid_: depå, AF-konto (different tax regime)

**Position size**:
The amount of currency committed to one trade; 10,000 SEK for Sweden.

**Brokerage**:
The broker's fee per order, paid on entry and again on exit.
_Avoid_: courtage, corretagem

**Slippage**:
The gap between the price a trade was planned at and the price it filled at.

**Round-trip cost**:
Brokerage on both orders plus exchange fees plus slippage on both sides,
expressed as a fraction of position size.

### Policy and search

**Genome**:
One complete trading policy, expressed as an ordered set of gene values.

**Gene**:
One tunable parameter of the policy, such as a stop distance or a cooldown
after losses.

**Gene category**:
A group of related genes searched together in one sweep (risk management,
take profit, timing, entry filter, regime/volatility, trailing, exit quality).

**Sweep**:
A neighbourhood search that moves the genes of one category a step at a time,
alone and in combinations, and scores every result.

**Promotion**:
Replacing the incumbent with a candidate after it passes every gate.

**Gate**:
A pass/fail condition a candidate must meet to be promoted.

**Graveyard**:
Genes tried many times without a single promotion.

**Hot zone**:
Genes that have produced at least one promotion.

**Drift**:
A gene promoted repeatedly in the same direction, a sign of fitting noise.

### Comparison points

**Baseline**:
The genome on the main line of history, re-scored on current data.

**Incumbent**:
The genome currently in use for signals.
_Avoid_: seed, current model

**Candidate**:
A genome derived from the incumbent by moving one or more genes.

**Best ever**:
The best metrics ever recorded, used as a long-term floor against slow
degradation.

### Metrics

**Fitness**:
The single composite score the optimizer maximises.
_Avoid_: score (overloaded)

**Data vintage**:
The dataset snapshot a metric was computed on. Fitness is comparable only
within one vintage.

**Win rate**:
Median across tickers of the share of profitable trades.
_Avoid_: hit rate, accuracy

**Target win rate**:
Median across tickers of the share of trades that reached the take-profit
target. Stricter than win rate.
_Avoid_: WR alvo (in prose)

**Alpha**:
Annualised return in excess of simply holding the same ticker.

**Max drawdown**:
The largest peak-to-trough fall of an equity curve.
_Avoid_: MDD (in prose)

**Expectancy**:
The mean net result per trade, after costs.

### Signals

**Daily run**:
The once-per-trading-day job that refreshes market data and emits that day's
signals.

**Signal**:
The final buy, hold or sell for a ticker on a date, after every override and
quality gate.

**Raw signal**:
The policy's output before overrides and quality gates.

**Entry, stop, target**:
The planned purchase price, the price that exits at a loss, and the price that
takes profit, emitted with every buy signal.
_Avoid_: entrada/alvo (in prose)

**Execution rule**:
How a signal becomes an order and an order becomes a fill: order type, price,
validity, and what happens when it does not fill.

**Order validity**:
The number of trading days a limit buy order stays live after its signal
before it is cancelled.

**Missed entry**:
A buy signal whose limit order expired without filling. It counts as no trade,
never as a fill at another price.
_Avoid_: skipped trade, unfilled signal

**Minervini stage**:
A ticker's trend phase from its price against its 50- and 200-day averages.

**Pivot levels**:
Support and resistance derived from the previous bar, from which entry, stop
and target are placed.

**Causal feature**:
An input derived only from prices, volumes and other market data available on
the day, never from a model output or a fact learned later.

### Simulation

**Backtest**:
The per-ticker historical simulation the optimizer scores genomes with; every
ticker trades its own independent equity.
_Avoid_: simulação (too broad)

**Replay**:
A day-by-day reconstruction of the signals the daily run would have emitted,
using only information available on each day, traded under the real
execution rule. Run per ticker unless stated otherwise.

**Portfolio replay**:
A replay in which all tickers share one account with fixed capital and a limit
on open positions.

**Paper trading**:
Forward recording of each day's real signals and the fills a trader would have
got, without real money. Its job is a mechanics check, not proof of edge.

**Mechanics check**:
Verifying that operating the system works in practice: signals arrive in
time, orders can be placed, and fill rate and slippage match what the replay
assumed.

**Index benchmark**:
Holding a broad Swedish index fund in the same account; the passive
alternative a portfolio replay must beat after costs.
_Avoid_: benchmark (alone), market return

**Ramp-up**:
The staged move from paper trading to reduced real capital to full real
capital.

**Stop rule**:
A measured condition that, when breached, sends live trading back to paper
trading for investigation.
_Avoid_: kill switch, circuit breaker

**Walk-forward replay**:
A replay split into consecutive folds, where each fold is traded by a genome
trained only on data from before that fold began.

**Fold**:
One train-then-trade segment of a walk-forward replay: a training span ending
at a cutoff date, followed by the trading span the resulting genome replays.

**Warmup**:
The bars before a period that only feed the indicators: no order is placed on
them and every statistic starts after them. Training windows, replay periods
and the live account all start this way.
_Avoid_: burn-in, lookback (when the no-trading rule is meant)

**Holdout**:
A span of dates withheld from the optimizer, used to judge a genome on data it
has never seen.
_Avoid_: pristine window, OOS window
