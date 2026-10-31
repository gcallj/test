# Strategy v6 is a v1-style swing system with a robust optimizer, frozen before its run

The owner asked for a new strategy derived from v1, searched by a more robust
optimization algorithm and fed with more relevant inputs.

## What v6 keeps from v1

- the execution rule (ADR 0001): a limit order below the close, valid for 3
  sessions, with ATR stop, target and time stop;
- parameters chosen by an optimizer;
- the same six-fold walk-forward, each fold trained on the four years before
  it.

## What v6 changes

Each change answers one of the causes of v1's failure in
`docs/sweden/diagnostico.md`:

- **One account.** Six slots share one account, and idle slots sit in the
  index fund. v1 judged each ticker alone and spent half its time in cash.
- **Cross-sectional score.** The score is a weighted sum of 13 inputs ranked
  across stocks each day:
  - six documented price factors;
  - seven series prices do not contain: insider purchases, disclosed short
    position and its change, analyst rating and target revisions, the tone of
    the last report, and the last earnings reaction.
- **Longer holding.** A minimum hold of 10 sessions applies before a score
  exit; v1 held trades for 4 days.
- **Fewer genes.** v6 has 19 genes (13 weights and 6 trading genes) instead
  of 38.
- **A robust optimizer.** CMA-ES runs 600 trials per fold.
  - The fitness is the account's yearly excess over the index, with a
    drawdown penalty.
  - It is taken as the mean minus one standard deviation over the four
    training years.
  - Of the ten best trials, the one with the best mean under random
    perturbation of its genes wins.

## The single run

`data/strategy_se_v6.json` fixes the gene ranges, budgets, seeds and the pass
rule before the run. To pass, v6 needs both:
- criterion 1;
- a bootstrap p below 0.05 for the daily excess over the index in the three
  walk-forward years.

## Consequences

The three walk-forward years (2023-10-31 to 2026-10-30) have already judged
v1–v5. The alternative inputs have already been joined to returns in v4 and
v5, where they showed no event-level edge. A pass here therefore earns paper
trading only, never real money; a failure closes the v1 line.

Optimization is fast enough (about 0.2 s per evaluation) to run locally. The
inputs reuse the v4 and v5 caches under `output/`.
