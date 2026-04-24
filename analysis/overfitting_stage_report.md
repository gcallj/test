# Overfitting Stage Report

Generated at: `2026-04-24T22:21:59+00:00`
Verdict: **OVERFIT**

## Checkpoint

- fit: `25.4701`
- WR: `0.6923`
- alpha_ann: `-0.0791`
- n_ge_70: `57`

## Pristine Holdout (90d)

| Metric | Train | Holdout | Delta |
|---|---:|---:|---:|
| fit | +23.1107 | -34.3604 | -57.4711 |
| wr_med_all | +0.6726 | +0.5833 | -0.0892 |
| mean_alpha_ann | -0.0785 | -0.2675 | -0.1890 |
| n_ge_70 | +52.0000 | +47.0000 | -5.0000 |
| trades_med | +32.0000 | +10.0000 | +0.0000 |

## Bootstrap CI (WR)

- Samples: 100
- Tickers: 103
- WR CI 5-95: [0.6733, 0.7169] (width=0.0436)

## Parameter Sensitivity

- Base fit: 0.0000
- Median abs delta: 0.0000
- Max abs delta: 0.0000

### Top 5 fragile genes

| Gene | Max delta_fit |
|---|---:|

## Reasons

- holdout_alpha_drop=0.1890 > critical (0.06)
- holdout_wr_drop=0.0892 > warn (0.05)
