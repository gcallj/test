"""
Identify the subset of tickers where the current checkpoint BEATS buy-and-hold.
This helps us understand where the strategy has real edge and define a
high-quality ticker universe for live trading.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

os.environ.setdefault("GA_RUN_MODE", "load")
os.environ["GA_STAGE_GATE"] = "off"

import ga_run  # noqa: E402
from ga_run import (  # noqa: E402
    decode_global_params,
    backtest_stats_global_intraday,
    global_fitness_from_stats,
    buyhold_capped,
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
print(f"[INIT] store: {store_path}", flush=True)
store = PayloadStore(store_path, mode="r")

print(f"[INIT] Building ticker cache...", flush=True)
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
print(f"[INIT] Cache: {len(cache)} tickers", flush=True)


# Load base genome
with open("global_ga_checkpoint.json") as f:
    ck = json.load(f)
base_genome = ck["genome"]
gp = decode_global_params(base_genome)

# Evaluate per-ticker
print(f"\n[EVAL] Evaluating per-ticker with base genome...", flush=True)
rows = []
for tk, p in cache.items():
    st = backtest_stats_global_intraday(
        p["open"], p["high"], p["low"], p["close"],
        p["score_matrix"], p["atr"], gp, precomputed=p,
    )
    bh = buyhold_capped(p["close"])
    n_years = len(p["close"]) / 252.0
    # Annualized
    ret_ann = (1 + st["total_return"]) ** (1/max(n_years, 0.1)) - 1
    bh_ann = (1 + bh) ** (1/max(n_years, 0.1)) - 1
    alpha_ann = ret_ann - bh_ann
    rows.append({
        "ticker": tk,
        "wr": st["win_rate"],
        "n_trades": st.get("n_trades", 0),
        "ret_total": st["total_return"],
        "bh_total": bh,
        "ret_ann": ret_ann,
        "bh_ann": bh_ann,
        "alpha_ann_pp": alpha_ann * 100,
        "mdd": abs(st["mdd"]),
        "sharpe": st.get("sharpe", 0),
        "n_years": n_years,
    })

rows.sort(key=lambda r: -r["alpha_ann_pp"])

print(f"\n=== TOP 30 TICKERS BY ALPHA (annualized) ===", flush=True)
print(f"{'#':<3} {'ticker':<12} {'WR':>7} {'trades':>7} {'ret_ann':>9} {'bh_ann':>9} {'alpha':>10} {'MDD':>7}", flush=True)
for i, r in enumerate(rows[:30]):
    print(f"{i+1:<3} {r['ticker']:<12} "
          f"{r['wr']*100:>6.2f}% {r['n_trades']:>7.0f} "
          f"{r['ret_ann']*100:>+7.2f}% {r['bh_ann']*100:>+7.2f}% "
          f"{r['alpha_ann_pp']:>+8.2f}pp {r['mdd']*100:>6.2f}%",
          flush=True)

# Subset: alpha >= 0 AND WR >= 65%
print(f"\n=== HIGH-QUALITY SUBSET (alpha >= 0pp AND WR >= 65%) ===", flush=True)
subset = [r for r in rows if r["alpha_ann_pp"] >= 0 and r["wr"] >= 0.65 and r["n_trades"] > 10]
print(f"Count: {len(subset)}/{len(rows)} ({len(subset)/len(rows)*100:.1f}%)", flush=True)
if subset:
    sub_wr = float(np.median([r["wr"] for r in subset]))
    sub_ret = float(np.median([r["ret_ann"] for r in subset]))
    sub_bh = float(np.median([r["bh_ann"] for r in subset]))
    sub_alpha = float(np.median([r["alpha_ann_pp"] for r in subset]))
    sub_mdd = float(np.median([r["mdd"] for r in subset]))
    sub_sharpe = float(np.median([r["sharpe"] for r in subset]))
    print(f"Subset median: WR={sub_wr*100:.2f}% ret={sub_ret*100:+.2f}% bh={sub_bh*100:+.2f}% "
          f"alpha={sub_alpha:+.2f}pp MDD={sub_mdd*100:.2f}% Sharpe={sub_sharpe:+.3f}", flush=True)

# Subset 2: alpha >= +2pp AND WR >= 68%
print(f"\n=== PREMIUM SUBSET (alpha >= +2pp AND WR >= 68%) ===", flush=True)
subset2 = [r for r in rows if r["alpha_ann_pp"] >= 2.0 and r["wr"] >= 0.68 and r["n_trades"] > 20]
print(f"Count: {len(subset2)}/{len(rows)} ({len(subset2)/len(rows)*100:.1f}%)", flush=True)
if subset2:
    sub_wr = float(np.median([r["wr"] for r in subset2]))
    sub_ret = float(np.median([r["ret_ann"] for r in subset2]))
    sub_bh = float(np.median([r["bh_ann"] for r in subset2]))
    sub_alpha = float(np.median([r["alpha_ann_pp"] for r in subset2]))
    sub_mdd = float(np.median([r["mdd"] for r in subset2]))
    sub_sharpe = float(np.median([r["sharpe"] for r in subset2]))
    print(f"Premium median: WR={sub_wr*100:.2f}% ret={sub_ret*100:+.2f}% bh={sub_bh*100:+.2f}% "
          f"alpha={sub_alpha:+.2f}pp MDD={sub_mdd*100:.2f}% Sharpe={sub_sharpe:+.3f}", flush=True)

# Save results
out = {
    "all_tickers": rows,
    "high_quality_subset": {
        "criteria": "alpha_ann >= 0pp AND WR >= 65% AND n_trades > 10",
        "tickers": [r["ticker"] for r in subset],
        "count": len(subset),
    },
    "premium_subset": {
        "criteria": "alpha_ann >= 2pp AND WR >= 68% AND n_trades > 20",
        "tickers": [r["ticker"] for r in subset2],
        "count": len(subset2),
    },
}
with open("analysis/ticker_alpha_analysis.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n[SAVE] analysis/ticker_alpha_analysis.json", flush=True)

# Print bottom 10 (worst alpha)
print(f"\n=== BOTTOM 10 TICKERS (alpha destroying) ===", flush=True)
for i, r in enumerate(rows[-10:]):
    print(f"{i+1:<3} {r['ticker']:<12} "
          f"WR={r['wr']*100:>6.2f}% "
          f"ret_ann={r['ret_ann']*100:>+7.2f}% bh_ann={r['bh_ann']*100:>+7.2f}% "
          f"alpha={r['alpha_ann_pp']:>+8.2f}pp",
          flush=True)
