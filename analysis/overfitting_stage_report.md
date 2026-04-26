# Overfitting Stage Report

Generated at: `2026-04-26T08:59:45+00:00`
Verdict: **CLEAN**

## Checkpoint

- fit: `-40.5758`
- WR: `0.6176`
- alpha_ann: `-0.1077`
- n_ge_70: `6`

## Multi-Window Holdout (3 janelas)

Train baseline: 49 tickers, WR=0.6222, alpha=-0.1113

| Window | Pristine | n_tickers | WR | alpha | trades_med | WR_drop | alpha_drop |
|---|---|---:|---:|---:|---:|---:|---:|
| W1_offset0_len90 | YES | 49 | 0.5714 | -0.0277 | 14.0 | +0.0508 | -0.0836 |
| W2_offset90_len90 | no | 49 | 0.6000 | -0.0911 | 14.0 | +0.0222 | -0.0203 |
| W3_offset270_len90 | no | 49 | 0.6190 | -0.0871 | 17.0 | +0.0032 | -0.0242 |

## Bootstrap CI (WR)

- Samples: 200
- Tickers: 49
- WR CI 5-95: [0.5882, 0.6418] (width=0.0536)

## Parameter Sensitivity

- Base fit: 0.0000
- Median abs delta: 0.0000
- Max abs delta: 0.0000

### Top 5 fragile genes

| Gene | Max delta_fit |
|---|---:|

## Reasons

- all_checks_passed
