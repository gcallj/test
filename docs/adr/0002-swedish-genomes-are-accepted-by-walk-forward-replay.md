# Swedish genomes are accepted by walk-forward replay, not by promotion gates

The automated promotion gates were calibrated on ~116 Brazilian tickers. One of
them needs at least 20 operable tickers, which a filtered Swedish Large Cap
universe may never reach, and in Brazil the same stack has promoted nothing
since April 2026. For Sweden, a genome goes into daily use only when a person
reads its walk-forward replay report (six six-month folds over three years,
each traded by a genome trained only on the four years before it) and accepts
it. The continuous sweep loop is not run for Sweden.

## Considered Options

Recalibrating the Brazilian gates for Sweden was rejected: it would tune a
second automated stack before there is evidence the first one works, and the
replay already answers the question the gates approximate.
