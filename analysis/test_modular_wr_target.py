"""Quick check: modular backtest + fitness now return win_rate_target."""
import os
import sys
import json
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

os.environ.setdefault("GA_RUN_MODE", "load")

import ga_run  # noqa: E402
from ga_run import decode_global_params, GA_MEMMAP_DIR  # noqa: E402
from ga_run_modular_final import (  # noqa: E402
    _backtest_stats_global_intraday,
    global_fitness_from_stats,
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

with open("global_ga_checkpoint.json") as f:
    ck = json.load(f)
base_genome = ck["genome"]
gp = decode_global_params(base_genome)

print(f"\n[EVAL] Running modular backtest on base genome...", flush=True)
stats = []
for tk, p in list(cache.items())[:10]:  # first 10 for speed check
    st = _backtest_stats_global_intraday(
        p["open"], p["high"], p["low"], p["close"],
        p["score_matrix"], p["atr"], gp, precomputed=p,
    )
    print(f"  {tk:<12} WR={st.get('win_rate', 0)*100:.1f}% "
          f"WR_tgt={st.get('win_rate_target', 0)*100:.1f}% "
          f"take={st.get('pct_exit_take', 0)*100:.1f}% "
          f"stop={st.get('pct_exit_stop', 0)*100:.1f}%",
          flush=True)
    stats.append(st)

# Now eval full 119 and check fitness
print(f"\n[EVAL] Full 119 tickers for fitness...", flush=True)
full_stats = []
for tk, p in cache.items():
    st = _backtest_stats_global_intraday(
        p["open"], p["high"], p["low"], p["close"],
        p["score_matrix"], p["atr"], gp, precomputed=p,
    )
    from ga_run import buyhold_capped
    bh = buyhold_capped(p["close"])
    st["buy_hold_return"] = bh
    st["excess_return"] = st.get("total_return", 0.0) - bh
    full_stats.append(st)

fit = global_fitness_from_stats(full_stats)
mean_wrt = float(np.mean([s.get("win_rate_target", 0.0) for s in full_stats]))
med_wrt = float(np.median([s.get("win_rate_target", 0.0) for s in full_stats]))
mean_wr = float(np.mean([s.get("win_rate", 0.0) for s in full_stats]))
med_wr = float(np.median([s.get("win_rate", 0.0) for s in full_stats]))

print(f"\n=== MODULAR FITNESS ===", flush=True)
print(f"fitness:         {fit:+.3f}", flush=True)
print(f"mean_win_rate:   {mean_wr*100:.2f}%", flush=True)
print(f"med_win_rate:    {med_wr*100:.2f}%", flush=True)
print(f"mean_wr_target:  {mean_wrt*100:.2f}%", flush=True)
print(f"med_wr_target:   {med_wrt*100:.2f}%", flush=True)
print(f"\nPrimer check: has win_rate_target in stats: "
      f"{bool(full_stats[0].get('win_rate_target') is not None)}",
      flush=True)
