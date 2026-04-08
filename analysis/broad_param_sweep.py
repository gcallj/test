"""
Broader parameter sweep: trailing_stop_mode, time_stop_bars, regime_threshold,
entry_confirmation_days, vote_threshold_long.

Baseline: current checkpoint (WR 67%, ret 0.54, alpha pct_beat 35%).

Focus: look for variations that IMPROVE per-ticker alpha without destroying WR.
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
    sanitize_global_genome,
    GLOBAL_PARAM_SPECS,
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
    base_genome = json.load(f)["genome"]

PARAM_IDX = {spec[0]: i for i, spec in enumerate(GLOBAL_PARAM_SPECS)}


def make_variant(base, overrides: dict):
    g = list(base)
    for k, v in overrides.items():
        idx = PARAM_IDX[k]
        g[idx] = v
    return sanitize_global_genome(g)


def evaluate(genome, label):
    gp = decode_global_params(genome)
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
    fit = global_fitness_from_stats(stats)

    all_rets = [s["total_return"] for s in stats]
    all_excess = [s["excess_return"] for s in stats]
    all_wrs = [s["win_rate"] for s in stats]
    all_mdds = [abs(s["mdd"]) for s in stats]
    all_trades = [s.get("n_trades", 0) for s in stats]

    wr_med = float(np.median(all_wrs))
    ret_med = float(np.median(all_rets))
    excess_med = float(np.median(all_excess))
    pct_beat_bh = float(np.mean([e > 0 for e in all_excess]))
    mdd_med = float(np.median(all_mdds))
    trades_med = float(np.median(all_trades))
    n_ge_70 = sum(1 for w in all_wrs if w >= 0.70)

    row = {
        "label": label,
        "fit": float(fit),
        "wr_med": wr_med,
        "ret_med": ret_med,
        "excess_med": excess_med,
        "pct_beat_bh": pct_beat_bh,
        "mdd_med": mdd_med,
        "trades_med": trades_med,
        "n_ge_70": n_ge_70,
        "genome": list(genome),
    }
    print(f"[{label:<35}] fit={fit:>9.3f} WR={wr_med*100:>5.2f}% "
          f"ret={ret_med:>+6.3f} excess={excess_med:>+7.3f} "
          f"beat={pct_beat_bh*100:>4.1f}% MDD={mdd_med*100:>5.2f}% "
          f"trades={trades_med:>5.0f} >=70={n_ge_70}/{len(cache)}",
          flush=True)
    return row


print(f"\n[EVAL] Baseline...", flush=True)
results = [evaluate(base_genome, "base")]

# --- Sweep 1: trailing stop mode ---
print(f"\n[SWEEP] trailing_stop_mode...", flush=True)
for ts in [0, 1, 2]:
    lbl = f"trail{ts}"
    g = make_variant(base_genome, {"trailing_stop_mode": ts})
    results.append(evaluate(g, lbl))

# --- Sweep 2: time_stop_bars ---
print(f"\n[SWEEP] time_stop_bars...", flush=True)
for t in [10, 15, 20, 25]:
    lbl = f"timestop{t}"
    g = make_variant(base_genome, {"time_stop_bars": t})
    results.append(evaluate(g, lbl))

# --- Sweep 3: regime_threshold ---
print(f"\n[SWEEP] regime_threshold...", flush=True)
for rt in [0.25, 0.35, 0.45, 0.55, 0.65]:
    lbl = f"regime{rt}"
    g = make_variant(base_genome, {"regime_threshold": rt})
    results.append(evaluate(g, lbl))

# --- Sweep 4: vote_threshold_long (more selective entries) ---
print(f"\n[SWEEP] vote_threshold_long...", flush=True)
for vt in [0.20, 0.25, 0.30, 0.40]:
    lbl = f"voteL{vt}"
    g = make_variant(base_genome, {"vote_threshold_long": vt})
    results.append(evaluate(g, lbl))

# --- Sweep 5: entry_confirmation_days ---
print(f"\n[SWEEP] entry_confirmation_days...", flush=True)
for ec in [1, 2, 3]:
    lbl = f"conf{ec}d"
    g = make_variant(base_genome, {"entry_confirmation_days": ec})
    results.append(evaluate(g, lbl))

# --- Sweep 6: combined (more selective) ---
print(f"\n[SWEEP] combined more-selective...", flush=True)
combos = [
    {"vote_threshold_long": 0.25, "regime_threshold": 0.45, "time_stop_bars": 30},
    {"vote_threshold_long": 0.30, "regime_threshold": 0.55, "entry_confirmation_days": 2},
    {"trailing_stop_mode": 1, "time_stop_bars": 35, "reward_risk_ratio": 2.0},
    {"trailing_stop_mode": 0, "reward_risk_ratio": 2.5, "partial_take_pct": 0.0},
    {"vote_threshold_long": 0.20, "regime_threshold": 0.45, "partial_take_pct": 0.0, "partial_take_pct_2": 0.0},
]
for i, ov in enumerate(combos):
    lbl = f"combo{i+1}"
    g = make_variant(base_genome, ov)
    r = evaluate(g, lbl)
    r["overrides"] = ov
    results.append(r)


# Rank by fit (excess-aware)
print(f"\n=== RANKING (by fit) ===", flush=True)
ranked = sorted(results, key=lambda r: -r["fit"])
for i, r in enumerate(ranked[:15]):
    star = " *" if i == 0 else ""
    print(f"{i+1:<3} {r['label'][:35]:<35} "
          f"fit={r['fit']:>9.3f} WR={r['wr_med']*100:>5.2f}% "
          f"ret={r['ret_med']:>+6.3f} excess={r['excess_med']:>+7.3f} "
          f"beat={r['pct_beat_bh']*100:>4.1f}% >=70={r['n_ge_70']:>3d}{star}",
          flush=True)

# Save all
out = {
    "results": [{k: v for k, v in r.items() if k != "genome"} for r in results],
    "best_label": ranked[0]["label"],
    "best_genome": ranked[0]["genome"],
    "best_fit": ranked[0]["fit"],
}
with open("analysis/broad_sweep_results.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n[SAVE] analysis/broad_sweep_results.json", flush=True)

# Check if any variant beats base on fit AND beats on pct_beat_bh
base_row = results[0]
print(f"\n=== VARIANTS BETTER THAN BASE ===", flush=True)
better = []
for r in results[1:]:
    if r["fit"] > base_row["fit"] and r["pct_beat_bh"] >= base_row["pct_beat_bh"] - 0.01:
        better.append(r)
        print(f"  {r['label']} fit={r['fit']:.3f} (+{r['fit']-base_row['fit']:.3f}) "
              f"beat_bh={r['pct_beat_bh']*100:.1f}% vs base {base_row['pct_beat_bh']*100:.1f}%",
              flush=True)
if not better:
    print("  (none)", flush=True)
