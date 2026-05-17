# PayloadStore Audit (plan item R5)

**Status:** Read-only inventory. No code changes in this pass.

## Verdict

The cache is well-built for the memmap I/O pattern. Per-fold ticker payloads are
zero-copy views (no disk re-read inside the GA loop), `score_matrix` and `atr`
are pre-baked at store-build time, and `vol_rank` is correctly checked in the
precomputed-dict path inside `backtest_stats_global_intraday`.

**One actionable opportunity exists**: two fixed-window moving averages used
inside the backtest loop are recomputed per-genome despite being
genome-independent. Lifting them to window-scope precomputation should yield a
~5–10% speedup on GA generation wall-time.

## What PayloadStore caches

Per-ticker memmap arrays (`payload_store.py:95-98`):

- Market data: `open`, `high`, `low`, `close`, `atr`, `volume`
- Pre-baked indicators: `ma`, `vol_rank`, `ret_fwd`, `long_votes`, `short_votes`
- Feature matrix: `score_matrix` (2-D)
- Metadata: `dates`, `valid_mask`

Window slicing in `get_window_payloads()` (`payload_store.py:323-342`) returns
zero-copy views — no disk hit per fold.

## Already optimized (no action)

| Concern | Resolution |
|---|---|
| Per-fold disk re-reads | Zero-copy memmap views via `get_window_payloads`. |
| `score_matrix` recomputation | Baked at store-build (one-time cost). |
| `atr` recomputation | Baked at store-build. |
| `vol_rank` per-genome cost | `precomputed` dict path is hit (`ga_run.py:1523-1526`). |

## Findings worth a follow-up

Inside `backtest_stats_global_intraday` (`ga_run.py:~1505-1963`):

| Array | Genome-dependent? | Currently | Should be |
|---|---|---|---|
| `ma` (line 1522) | **Yes** — uses `gp.ma_filter_period` | Recomputed per call | Stays per-genome. No change. |
| `ma50_exit` (line 1538) | **No** — fixed window 50 | Recomputed per call | Precompute once per window. |
| `ma200` (line 1539) | **No** — fixed window 200 | Recomputed per call | Precompute once per window. |
| `pivot_prev`, `r1_prev`, `s1_prev` (lines 1560-1566) | No — derived from OHLC alone | Recomputed per call | Precompute once per window. |
| `long_votes`, `short_votes` (line 1529) | **Yes** — depends on `gp.z_threshold` | Recomputed per call | Stays per-genome. No change. |

### Suggested follow-up implementation sketch

In `precompute_global_payloads()` (`ga_run.py:~642-687`), augment the
per-window payload dict with:

```python
payload["ma50_exit"] = rolling_mean_np(close, 50, max(20, 50 // 2))
payload["ma200"]   = rolling_mean_np(close, 200, max(20, 200 // 2))
# pivot/r1/s1 require shifted highs/lows/closes — same precompute pattern.
```

Then in `backtest_stats_global_intraday` (lines 1538-1539, 1560-1566), guard
the recomputation with a `precomputed is not None and "ma200" in precomputed`
check (mirrors the existing `vol_rank` pattern at line 1523).

**Expected payoff:** ~5–10% per GA generation. On a 25-minute sweep matrix
cycle that's ~1.5–2.5 minutes shaved per cycle, ~30 minutes per day across the
12 daily auto-improvement cycles. Modest but free.

**Expected effort:** ~100 LOC + parity test mirroring R1's `test_vectorize_parity.py`
to confirm `ma200_precomputed == ma200_recomputed` bit-for-bit.

### Risk

Low. The pattern mirrors `vol_rank` which is already in production. Bit-parity
test guards against accidental drift. Cached arrays are read-only views, so
threading isn't a concern.

## Items confirmed as no-op

- `ret_fwd` is stored in the payload but unused in the backtest loop. No
  recomputation, no benefit to caching. Leave alone.
- `score_matrix`, `atr`, `vol_rank`: already optimal.

## When to act on this

The plan flagged R5 as `BAIXO-MÉDIO (incerto)`. The audit confirms `BAIXO`
with a precise finding. Recommended priority order:

1. **Now**: do nothing — R2 (profile + Numba) is upstream and may reveal
   larger wins. The two MA recomputes are loud in a profile.
2. **After R2 profile**: if `ma200`/`ma50_exit` show up as top-10 hot spots,
   land this fix. Otherwise defer indefinitely.

The R2 profile is the right gate for this change.
