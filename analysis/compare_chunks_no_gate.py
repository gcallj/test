"""
Compare chunk genomes from SOFT retrain vs base WITHOUT gate enabled.
Uses proper excess_return calculation (vs buy_hold) for accurate alpha metrics.
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


def evaluate_with_excess(genome, label):
    """Evaluate with proper excess_return (total - buy_hold)."""
    gp = decode_global_params(genome)
    stats = []
    for tk, p in cache.items():
        st = backtest_stats_global_intraday(
            p["open"], p["high"], p["low"], p["close"],
            p["score_matrix"], p["atr"], gp, precomputed=p,
        )
        # Add excess_return (what fitness function expects)
        bh = buyhold_capped(p["close"])
        st["buy_hold_return"] = bh
        st["excess_return"] = st.get("total_return", 0.0) - bh
        stats.append(st)

    fit = global_fitness_from_stats(stats)

    all_rets = [s["total_return"] for s in stats]
    all_bhs = [s["buy_hold_return"] for s in stats]
    all_excess = [s["excess_return"] for s in stats]
    all_wrs = [s["win_rate"] for s in stats]
    all_mdds = [abs(s["mdd"]) for s in stats]
    all_trades = [s.get("n_trades", 0) for s in stats]

    wr_med = float(np.median(all_wrs))
    wr_mean = float(np.mean(all_wrs))
    ret_med = float(np.median(all_rets))
    ret_mean = float(np.mean(all_rets))
    bh_med = float(np.median(all_bhs))
    excess_med = float(np.median(all_excess))
    excess_mean = float(np.mean(all_excess))
    pct_beat_bh = float(np.mean([e > 0 for e in all_excess]))
    mdd_med = float(np.median(all_mdds))
    trades_med = float(np.median(all_trades))
    n_ge_70 = sum(1 for w in all_wrs if w >= 0.70)

    row = {
        "label": label,
        "fit": float(fit),
        "wr_med": wr_med,
        "wr_mean": wr_mean,
        "ret_med": ret_med,
        "ret_mean": ret_mean,
        "bh_med": bh_med,
        "excess_med": excess_med,
        "excess_mean": excess_mean,
        "pct_beat_bh": pct_beat_bh,
        "mdd_med": mdd_med,
        "trades_med": trades_med,
        "n_ge_70": n_ge_70,
        "genome": list(genome),
    }
    print(f"[{label:<20}] fit={fit:>8.3f} WR={wr_med*100:>5.2f}% "
          f"ret={ret_med:>+6.3f} BH={bh_med:>+6.3f} "
          f"excess_med={excess_med:>+7.3f} pct_beat={pct_beat_bh*100:>4.1f}% "
          f"MDD={mdd_med*100:>5.2f}% >=70={n_ge_70}/{len(cache)}",
          flush=True)
    return row


# Base from global_ga_checkpoint.json
with open("global_ga_checkpoint.json") as f:
    base_ck = json.load(f)
base_genome = base_ck["genome"]

# Read all chunks from progress file
chunks_data = []
if os.path.exists("local_ga_chunk_progress.json"):
    with open("local_ga_chunk_progress.json") as f:
        progress = json.load(f)
    chunks_data = progress.get("chunks", [])
    print(f"[INIT] Found {len(chunks_data)} chunks in progress file", flush=True)

# Evaluate base
print("\n[EVAL] base genome (global_ga_checkpoint.json)", flush=True)
results = []
results.append(evaluate_with_excess(base_genome, "base"))

# Evaluate each chunk
for ch in chunks_data:
    label = f"chunk_{ch['chunk_id']}"
    print(f"\n[EVAL] {label}", flush=True)
    results.append(evaluate_with_excess(ch["best_genome"], label))

# Rank by fit
print("\n=== RANKING (by fit) ===", flush=True)
ranked = sorted(results, key=lambda r: -r["fit"])
print(f"{'#':<3} {'label':<15} {'fit':>9} {'WR':>7} {'ret':>8} {'BH':>8} "
      f"{'excess':>8} {'beatBH':>7} {'MDD':>7} {'trades':>7} {'>=70':>7}",
      flush=True)
for i, r in enumerate(ranked):
    star = " *" if i == 0 else ""
    print(f"{i+1:<3} {r['label']:<15} "
          f"{r['fit']:>9.3f} {r['wr_med']*100:>6.2f}% "
          f"{r['ret_med']:>+8.3f} {r['bh_med']:>+8.3f} "
          f"{r['excess_med']:>+8.3f} {r['pct_beat_bh']*100:>6.1f}% "
          f"{r['mdd_med']*100:>6.2f}% {r['trades_med']:>7.0f} "
          f"{r['n_ge_70']:>5d}/{len(cache)}{star}",
          flush=True)

# Save full results
out = {
    "results": [{k: v for k, v in r.items() if k != "genome"} for r in results],
    "best_label": ranked[0]["label"],
    "best_fit": ranked[0]["fit"],
    "best_genome": ranked[0]["genome"],
}
with open("analysis/chunks_vs_base.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n[SAVE] analysis/chunks_vs_base.json", flush=True)
