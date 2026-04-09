"""
Avalia a base genome sob a nova fitness (com wr_target_bonus) para
quantificar o incentivo que o GA deve receber para encontrar genoma
que aumente o win_rate_target.

Compara valores antes e depois da adicao do bonus.
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

print(f"\n[EVAL] Running backtest on base genome...", flush=True)
stats = []
for tk, p in cache.items():
    st = backtest_stats_global_intraday(
        p["open"], p["high"], p["low"], p["close"],
        p["score_matrix"], p["atr"], gp, precomputed=p,
    )
    bh = buyhold_capped(p["close"])
    st["buy_hold_return"] = bh
    st["excess_return"] = st.get("total_return", 0.0) - bh
    stats.append(st)

# Full fitness (with new wr_target bonus)
fit_new = global_fitness_from_stats(stats)

# Compute what the wr_target bonus contributed
mean_wr_target = float(np.mean([s.get("win_rate_target", 0.0) for s in stats]))
med_wr_target = float(np.median([s.get("win_rate_target", 0.0) for s in stats]))

# Compute the wr_target_bonus standalone
wr_target_bonus = 0.0
if mean_wr_target < 0.35:
    wr_target_bonus -= (0.35 - mean_wr_target) * 25.0
if mean_wr_target > 0.35:
    wr_target_bonus += (mean_wr_target - 0.35) * 10.0
if mean_wr_target > 0.45:
    wr_target_bonus += (mean_wr_target - 0.45) * 20.0
if mean_wr_target > 0.55:
    wr_target_bonus += (mean_wr_target - 0.55) * 30.0
if mean_wr_target > 0.65:
    wr_target_bonus += (mean_wr_target - 0.65) * 40.0
if med_wr_target > 0.40:
    wr_target_bonus += (med_wr_target - 0.40) * 8.0
if med_wr_target > 0.50:
    wr_target_bonus += (med_wr_target - 0.50) * 25.0
if med_wr_target > 0.60:
    wr_target_bonus += (med_wr_target - 0.60) * 45.0
if med_wr_target > 0.70:
    wr_target_bonus += (med_wr_target - 0.70) * 60.0

fit_old = fit_new - wr_target_bonus

print(f"\n=== BASE GENOME EVAL ===", flush=True)
print(f"mean_wr_target:           {mean_wr_target*100:.2f}%", flush=True)
print(f"med_wr_target:            {med_wr_target*100:.2f}%", flush=True)
print(f"mean_win_rate (atual):    {float(np.mean([s.get('win_rate', 0.0) for s in stats]))*100:.2f}%", flush=True)
print(f"med_win_rate  (atual):    {float(np.median([s.get('win_rate', 0.0) for s in stats]))*100:.2f}%", flush=True)
print(f"", flush=True)
print(f"fit_old (without wr_target bonus): {fit_old:+.3f}", flush=True)
print(f"fit_new (with    wr_target bonus): {fit_new:+.3f}", flush=True)
print(f"wr_target_bonus contribution:      {wr_target_bonus:+.3f}", flush=True)
print(f"", flush=True)
print(f"=== SENSITIVIDADE ===", flush=True)
print(f"Se mean_wr_target subir de {mean_wr_target*100:.1f}% para 50%, bonus +=",
      flush=True)
# Compute bonus at 50%
m = 0.50
md = 0.50
b50 = 0.0
if m > 0.35:
    b50 += (m - 0.35) * 10.0
if m > 0.45:
    b50 += (m - 0.45) * 20.0
if m > 0.55:
    b50 += (m - 0.55) * 30.0
if m > 0.65:
    b50 += (m - 0.65) * 40.0
if md > 0.40:
    b50 += (md - 0.40) * 8.0
if md > 0.50:
    b50 += (md - 0.50) * 25.0
print(f"  bonus @ mean=50% med=50%: {b50:+.3f} (delta {b50 - wr_target_bonus:+.3f})",
      flush=True)

# Compute bonus at 60%
m = 0.60
md = 0.60
b60 = 0.0
if m > 0.35:
    b60 += (m - 0.35) * 10.0
if m > 0.45:
    b60 += (m - 0.45) * 20.0
if m > 0.55:
    b60 += (m - 0.55) * 30.0
if m > 0.65:
    b60 += (m - 0.65) * 40.0
if md > 0.40:
    b60 += (md - 0.40) * 8.0
if md > 0.50:
    b60 += (md - 0.50) * 25.0
if md > 0.60:
    b60 += (md - 0.60) * 45.0
print(f"  bonus @ mean=60% med=60%: {b60:+.3f} (delta {b60 - wr_target_bonus:+.3f})",
      flush=True)

# Compute bonus at 70%
m = 0.70
md = 0.70
b70 = 0.0
if m > 0.35:
    b70 += (m - 0.35) * 10.0
if m > 0.45:
    b70 += (m - 0.45) * 20.0
if m > 0.55:
    b70 += (m - 0.55) * 30.0
if m > 0.65:
    b70 += (m - 0.65) * 40.0
if md > 0.40:
    b70 += (md - 0.40) * 8.0
if md > 0.50:
    b70 += (md - 0.50) * 25.0
if md > 0.60:
    b70 += (md - 0.60) * 45.0
if md > 0.70:
    b70 += (md - 0.70) * 60.0
print(f"  bonus @ mean=70% med=70%: {b70:+.3f} (delta {b70 - wr_target_bonus:+.3f})",
      flush=True)

# Save diagnostic
diagnostic = {
    "base_fit_new": fit_new,
    "base_fit_old": fit_old,
    "wr_target_bonus": wr_target_bonus,
    "base_mean_wr_target": mean_wr_target,
    "base_med_wr_target": med_wr_target,
    "bonus_at_50_50": b50,
    "bonus_at_60_60": b60,
    "bonus_at_70_70": b70,
}
with open("analysis/wr_target_fitness_sensitivity.json", "w") as f:
    json.dump(diagnostic, f, indent=2)
print(f"\n[SAVE] analysis/wr_target_fitness_sensitivity.json", flush=True)
