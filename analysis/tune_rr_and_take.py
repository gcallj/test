"""
Tune reward_risk_ratio and partial_take_pct on the current checkpoint genome
WITHOUT retraining. This is a post-hoc parameter sweep.

Hypothesis: The current genome has rr_ratio=1.0 and partial_take at 0.75x stop,
which caps winners at ~2% ATR move. Relaxing these should let trailing_stop_mode=2
capture more of the trend (improving alpha vs buy&hold).

Strategy: evaluate a grid of (rr_ratio, partial_take_pct, partial_take_pct_2)
on all 119 tickers and report fit, WR_med_all, ret_med, n_trades, mdd.
"""
import os
import sys
import json
import copy
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

os.environ.setdefault("GA_RUN_MODE", "load")
os.environ["GA_STAGE_GATE"] = "off"  # disable gate for this sweep

import ga_run  # noqa: E402
from ga_run import (  # noqa: E402
    decode_global_params,
    backtest_stats_global_intraday,
    global_fitness_from_stats,
    sanitize_global_genome,
    GLOBAL_PARAM_SPECS,
    GA_MEMMAP_DIR,
)
from payload_store import PayloadStore  # noqa: E402


# Open store (try both candidate paths)
candidates = [
    os.path.abspath(GA_MEMMAP_DIR),
    os.path.abspath("output/ga_memmap_staged"),
]
store_path = None
for p in candidates:
    if os.path.exists(os.path.join(p, "meta.json")):
        store_path = p
        break
if store_path is None:
    raise RuntimeError("No PayloadStore found")
print(f"[INIT] store: {store_path}", flush=True)
store = PayloadStore(store_path, mode="r")


# Build cache
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


# Load current checkpoint
with open("global_ga_checkpoint.json", "r") as f:
    ck = json.load(f)
base_genome = list(ck["genome"])
print(f"[BASE] Loaded checkpoint fit={ck['fitness']:.3f}", flush=True)
base_gp = decode_global_params(base_genome)
print(f"[BASE] rr_ratio={base_gp.reward_risk_ratio} "
      f"partial_take_pct={base_gp.partial_take_pct} "
      f"partial_take_level={base_gp.partial_take_level} "
      f"partial_take_pct_2={base_gp.partial_take_pct_2} "
      f"trailing_stop_mode={base_gp.trailing_stop_mode}", flush=True)


# Map param name -> genome index
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
        stats.append(st)
    fit = global_fitness_from_stats(stats)

    all_wrs = [s["win_rate"] for s in stats]
    all_rets = [s["total_return"] for s in stats]
    all_mdds = [abs(s["mdd"]) for s in stats]
    all_trades = [s.get("n_trades", 0) for s in stats]
    all_sharpes = [s.get("sharpe", 0) for s in stats]

    wr_med = float(np.median(all_wrs)) if all_wrs else 0.0
    ret_med = float(np.median(all_rets)) if all_rets else 0.0
    mdd_med = float(np.median(all_mdds)) if all_mdds else 1.0
    trades_med = float(np.median(all_trades)) if all_trades else 0.0
    sharpe_med = float(np.median(all_sharpes)) if all_sharpes else 0.0
    n_ge_70 = sum(1 for w in all_wrs if w >= 0.70)

    row = {
        "label": label,
        "fit": float(fit),
        "wr_med": wr_med,
        "ret_med": ret_med,
        "mdd_med": mdd_med,
        "trades_med": trades_med,
        "sharpe_med": sharpe_med,
        "n_ge_70": n_ge_70,
        "genome": list(genome),
    }
    print(f"[{label:<30}] fit={fit:>8.3f} WR={wr_med*100:>5.2f}% "
          f"ret={ret_med:>+7.3f} MDD={mdd_med*100:>5.2f}% "
          f"trades={trades_med:>5.0f} sharpe={sharpe_med:>+6.3f} "
          f">=70={n_ge_70}/{len(cache)}", flush=True)
    return row


# -- Baseline eval --
print(f"\n[EVAL] Baseline (current checkpoint)...", flush=True)
results = []
r0 = evaluate(base_genome, "base")
results.append(r0)

# -- Grid search --
print(f"\n[EVAL] Grid search: rr_ratio x partial_take_pct...", flush=True)
# Param combinations to try
grid = [
    # (rr_ratio, p_take_pct, p_take_pct_2)
    (1.5, 0.1, 0.2),
    (2.0, 0.1, 0.2),
    (2.5, 0.1, 0.2),
    (3.0, 0.1, 0.2),
    (4.0, 0.1, 0.2),
    (5.0, 0.1, 0.2),
    # No partial takes
    (2.0, 0.0, 0.0),
    (3.0, 0.0, 0.0),
    (4.0, 0.0, 0.0),
    (5.0, 0.0, 0.0),
    # Single partial only
    (3.0, 0.1, 0.0),
    (4.0, 0.1, 0.0),
    (3.0, 0.2, 0.0),
    (4.0, 0.2, 0.0),
]

for rr, ptp, ptp2 in grid:
    label = f"rr{rr}_pt{ptp}_pt2{ptp2}"
    overrides = {
        "reward_risk_ratio": rr,
        "partial_take_pct": ptp,
        "partial_take_pct_2": ptp2,
    }
    g = make_variant(base_genome, overrides)
    r = evaluate(g, label)
    r["overrides"] = overrides
    results.append(r)


# -- Rank by ret_med (primary alpha proxy), then wr_med, then fit --
print(f"\n=== RANKING (by ret_med, wr_med, fit) ===", flush=True)
def score_key(r):
    return (-r["ret_med"], -r["wr_med"], -r["fit"])

ranked = sorted(results, key=score_key)
print(f"{'#':<3} {'label':<30} {'fit':>9} {'WR':>7} {'ret':>8} {'MDD':>7} {'trades':>7} {'Sharpe':>8} {'>=70':>7}",
      flush=True)
for i, r in enumerate(ranked[:10]):
    print(f"{i+1:<3} {r['label'][:30]:<30} "
          f"{r['fit']:>9.3f} {r['wr_med']*100:>6.2f}% "
          f"{r['ret_med']:>+8.3f} {r['mdd_med']*100:>6.2f}% "
          f"{r['trades_med']:>7.0f} {r['sharpe_med']:>+8.3f} "
          f"{r['n_ge_70']:>5d}/{len(cache)}",
          flush=True)


# -- Save best candidate(s) --
# Rank by (wr_med + alpha_ret) / mdd
def composite_score(r):
    # Alpha proxy: ret_med should beat 0.5 (base)
    return r["ret_med"] * 2 + r["wr_med"] * 5 - r["mdd_med"] * 3

best = max(results[1:], key=composite_score)  # skip base
print(f"\n[BEST NON-BASE] {best['label']}", flush=True)
print(f"  fit={best['fit']:.3f} (base {r0['fit']:.3f} delta {best['fit']-r0['fit']:+.3f})", flush=True)
print(f"  wr_med={best['wr_med']*100:.2f}% (base {r0['wr_med']*100:.2f}% delta {(best['wr_med']-r0['wr_med'])*100:+.2f}pp)", flush=True)
print(f"  ret_med={best['ret_med']:+.3f} (base {r0['ret_med']:+.3f} delta {best['ret_med']-r0['ret_med']:+.3f})", flush=True)
print(f"  mdd_med={best['mdd_med']*100:.2f}% (base {r0['mdd_med']*100:.2f}% delta {(best['mdd_med']-r0['mdd_med'])*100:+.2f}pp)", flush=True)

# Save all results
out = {
    "base": {k: r0[k] for k in ("fit", "wr_med", "ret_med", "mdd_med", "trades_med", "sharpe_med", "n_ge_70")},
    "best_non_base": best,
    "all_results": [{k: v for k, v in r.items() if k != "genome"} for r in results],
    "best_genome": best["genome"],
}
with open("analysis/tune_rr_results.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n[SAVE] analysis/tune_rr_results.json", flush=True)

# If best is significantly better, also save as checkpoint candidate
if best["wr_med"] >= r0["wr_med"] - 0.01 and best["ret_med"] > r0["ret_med"] + 0.05:
    candidate = {
        "fitness": best["fit"],
        "genome": best["genome"],
        "parent": "global_ga_checkpoint.json (post-hoc tune)",
        "overrides": best.get("overrides", {}),
    }
    with open("analysis/tuned_candidate.json", "w") as f:
        json.dump(candidate, f, indent=2)
    print(f"[SAVE] analysis/tuned_candidate.json (WR preserved + ret improved)", flush=True)
