# The strategy is frozen before a single evaluation run

The three evaluation years (2023-10-31 to 2026-10-30) had already been used
to reject the swing engine, so trying variants against them until one passes
would manufacture a result. The v2 specification and the rule that selected
it were committed in `data/strategy_se_v2.json` using only data up to
2023-10-30, before any v2 run touched the evaluation period, and the
evaluation is run once. A later change to the strategy needs new
out-of-sample data — paper trading or time that has not happened yet — not
another pass over the same three years.
