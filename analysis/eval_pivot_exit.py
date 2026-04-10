"""
Testa exit rule baseado em R1/S1 (pivot levels) ao inves de ATR-based.

Hipotese: Se o backtest exitar em R1 (take) / S1 (stop) ao inves de
ATR-multiples, pode capturar melhor os movimentos que o modelo trade.

Simula isso modificando score_matrix: zera sinais quando close ja esta
proximo de r1 (sinal de exit natural).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

os.environ.setdefault("GA_RUN_MODE", "load")

import ga_run  # noqa: E402
from ga_run import (  # noqa: E402
    decode_global_params,
    backtest_stats_global_intraday,
    global_fitness_from_stats,
    buyhold_capped,
    rolling_mean_np,
    GA_MEMMAP_DIR,
)
from payload_store import PayloadStore  # noqa: E402


candidates = [
    os.path.abspath(GA_MEMMAP_DIR),
    os.path.abspath("output/ga_memmap_staged"),
]
store_path = None
for p in candidates:
    if os.path.exists(os.path.join(p, "meta.json")):
        store_path = p
        break
store = PayloadStore(store_path, mode="r")

cache = {}
for tk in store.tickers:
    a = store.get_ticker_arrays(tk)
    p = {}
    for k in ("open", "high", "low", "close", "atr", "ma", "vol_rank", "volume",
              "ret_fwd", "long_votes", "short_votes"):
        if k in a and a[k] is not None:
            p[k] = np.asarray(a[k], dtype=np.float64)
    if "score_matrix" in a and a["score_matrix"] is not None:
        p["score_matrix"] = np.asarray(a["score_matrix"], dtype=np.float64)
    if p.get("close") is not None and len(p["close"]) >= 252:
        cache[tk] = p

with open("global_ga_checkpoint.json") as f:
    ck = json.load(f)
gp = decode_global_params(ck["genome"])


def compute_rolling_pivot(h, l, c, window=20):
    """Rolling pivot = (high + low + close) / 3 over window."""
    n = len(c)
    pivot = np.full(n, np.nan)
    for i in range(window, n):
        hh = np.nanmax(h[i-window:i])
        ll = np.nanmin(l[i-window:i])
        cc = c[i-1]
        if np.isfinite(hh) and np.isfinite(ll) and np.isfinite(cc):
            pivot[i] = (hh + ll + cc) / 3.0
    return pivot


def eval_with_pivot_zero(subset_name, tickers):
    """Evaluate with score zeroed when close near r1 (intended exit)."""
    stats = []
    for tk in tickers:
        if tk not in cache:
            continue
        p = cache[tk]
        o = p["open"]
        h = p["high"]
        l = p["low"]
        c = p["close"]
        atr = p["atr"]
        sm = p["score_matrix"].copy()

        # Compute rolling pivot/r1/s1
        pivot = compute_rolling_pivot(h, l, c, window=20)
        r1_arr = 2 * pivot - np.roll(l, 1)  # approximate r1
        s1_arr = 2 * pivot - np.roll(h, 1)  # approximate s1

        # Zero score when close > 98% of r1 (trigger exit)
        for i in range(len(c)):
            if np.isfinite(r1_arr[i]) and c[i] >= r1_arr[i] * 0.98:
                sm[i, :] = 0.0

        st = backtest_stats_global_intraday(
            o, h, l, c, sm, atr, gp, precomputed=p,
        )
        bh = buyhold_capped(c)
        st["buy_hold_return"] = bh
        st["excess_return"] = st.get("total_return", 0.0) - bh
        stats.append(st)

    if not stats:
        return
    fit = global_fitness_from_stats(stats)
    wrs = [s["win_rate"] for s in stats]
    wr_tgts = [s.get("win_rate_target", 0.0) for s in stats]
    rets = [s["total_return"] for s in stats]
    excesses = [s["excess_return"] for s in stats]
    mdds = [abs(s["mdd"]) for s in stats]
    trades = [s.get("n_trades", 0) for s in stats]

    print(f"\n=== {subset_name} (n={len(stats)}) with pivot-zero ===", flush=True)
    print(f"  fit: {fit:.2f}", flush=True)
    print(f"  WR_med: {np.median(wrs)*100:.1f}%", flush=True)
    print(f"  WR_tgt_med: {np.median(wr_tgts)*100:.1f}%", flush=True)
    print(f"  ret_med: {np.median(rets):+.3f}", flush=True)
    print(f"  excess_med: {np.median(excesses):+.3f}", flush=True)
    print(f"  trades_med: {np.median(trades):.0f}", flush=True)


with open("analysis/ticker_alpha_analysis.json") as f:
    q_data = json.load(f)
premium = q_data["premium_subset"]["tickers"]
hq = q_data["high_quality_subset"]["tickers"]

print("Testing pivot-zero exit (close>0.98*r1 kills score):", flush=True)
eval_with_pivot_zero("ALL", list(cache.keys()))
eval_with_pivot_zero("HIGH_QUAL", hq)
eval_with_pivot_zero("PREMIUM", premium)
