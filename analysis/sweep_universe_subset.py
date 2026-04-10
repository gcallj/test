"""
Evaluate base genome on RESTRICTED UNIVERSE (premium subset only).

Hypothesis: modelo underperforms B&H overall because some tickers dragam
(CPTS11, ISAE4, RADL3 have strong B&H that nothing beats). Se rodarmos o
modelo apenas nos 13-35 tickers do premium subset, pode ter alpha positivo.
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
os.environ["GA_ALPHA_FOCUS"] = "off"

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

# Load subset lists from analysis
with open("analysis/ticker_alpha_analysis.json") as f:
    data = json.load(f)
premium = set(data["premium_subset"]["tickers"])
hq = set(data["high_quality_subset"]["tickers"])

with open("global_ga_checkpoint.json") as f:
    ck = json.load(f)
base_genome = ck["genome"]
gp = decode_global_params(base_genome)

def eval_subset(subset, label):
    stats = []
    for tk, p in cache.items():
        if subset is not None and tk not in subset:
            continue
        st = backtest_stats_global_intraday(
            p["open"], p["high"], p["low"], p["close"],
            p["score_matrix"], p["atr"], gp, precomputed=p,
        )
        bh = buyhold_capped(p["close"])
        st["buy_hold_return"] = bh
        st["excess_return"] = st.get("total_return", 0.0) - bh
        stats.append(st)

    if not stats:
        print(f"[{label}] no tickers")
        return
    fit = global_fitness_from_stats(stats)
    wrs = [s["win_rate"] for s in stats]
    wr_tgts = [s.get("win_rate_target", 0.0) for s in stats]
    rets = [s["total_return"] for s in stats]
    excesses = [s["excess_return"] for s in stats]
    mdds = [abs(s["mdd"]) for s in stats]
    trades = [s.get("n_trades", 0) for s in stats]

    print(f"\n=== {label} (n={len(stats)}) ===", flush=True)
    print(f"  fit: {fit:.2f}", flush=True)
    print(f"  WR: med={np.median(wrs)*100:.1f}% mean={np.mean(wrs)*100:.1f}%", flush=True)
    print(f"  WR_tgt: med={np.median(wr_tgts)*100:.1f}% mean={np.mean(wr_tgts)*100:.1f}%", flush=True)
    print(f"  ret_med: {np.median(rets):+.3f}", flush=True)
    print(f"  excess_med: {np.median(excesses):+.3f}", flush=True)
    print(f"  pct_beat_bh: {np.mean([e > 0 for e in excesses])*100:.1f}%", flush=True)
    print(f"  MDD_med: {np.median(mdds)*100:.1f}%", flush=True)
    print(f"  trades_med: {np.median(trades):.0f}", flush=True)
    print(f"  n_ge_70_wr: {sum(1 for w in wrs if w >= 0.70)}/{len(wrs)}", flush=True)


eval_subset(None, "ALL (119 tickers)")
eval_subset(hq, f"HIGH-QUALITY subset ({len(hq)} tickers)")
eval_subset(premium, f"PREMIUM subset ({len(premium)} tickers)")
