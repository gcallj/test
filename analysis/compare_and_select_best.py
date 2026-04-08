"""
Compara candidatos de checkpoint (incluindo atual main) e seleciona o melhor.

Criterios de ranking (lexicografico):
  1. WR_med_all (tickers com WR >= 65% filtro)
  2. Alpha anualizado vs buy&hold (Ret a.a. - BH a.a.)
  3. Fitness absoluto (global_fitness_from_stats)
  4. MDD menor (tiebreaker)

Uso:
  python analysis/compare_and_select_best.py <genome1.json> <genome2.json> ...

Cada arquivo deve ter {"fitness": float, "genome": [float, ...]}.

Se GA_STAGE_GATE=soft estiver setado no env, avaliacao usa gate.
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

import ga_run  # noqa: E402
from ga_run import (  # noqa: E402
    decode_global_params,
    backtest_stats_global_intraday,
    global_fitness_from_stats,
    GA_MEMMAP_DIR,
    GA_STAGE_GATE,
)
from payload_store import PayloadStore  # noqa: E402

print(f"[CONFIG] GA_STAGE_GATE={GA_STAGE_GATE}", flush=True)

# Open store
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

    # Unfiltered median WR (matches summary)
    all_wrs = [s["win_rate"] for s in stats]
    all_rets = [s["total_return"] for s in stats]
    all_bhs = [s.get("buy_hold_return", 0) for s in stats]
    # Filter n_trades > 5
    active = [s for s in stats if s.get("n_trades", 0) > 5]
    wrs = [s["win_rate"] for s in active]
    mdds = [abs(s["mdd"]) for s in active]
    rets = [s["total_return"] for s in active]

    wr_med_all = float(np.median(all_wrs)) if all_wrs else 0
    wr_med = float(np.median(wrs)) if wrs else 0
    mdd_med = float(np.median(mdds)) if mdds else 1
    ret_med = float(np.median(rets)) if rets else 0

    # Alpha annualized (assume 15 years)
    n_years = 15.0
    def annualize(r):
        return (max(1.0 + r, 0.001)) ** (1.0 / n_years) - 1.0
    wr_65_filtered = [(r, bh) for r, bh, wr in zip(all_rets, all_bhs, all_wrs) if wr > 0.65]
    if wr_65_filtered:
        ret_ann = np.median([annualize(r) for r, _ in wr_65_filtered])
        bh_ann = np.median([annualize(bh) for _, bh in wr_65_filtered])
        alpha_ann = (ret_ann - bh_ann) * 100
    else:
        alpha_ann = 0.0

    n_ge_70 = sum(1 for w in all_wrs if w >= 0.70)

    print(f"[{label}] fit={fit:.4f} WR_med_all={wr_med_all*100:.2f}% "
          f"MDD={mdd_med*100:.2f}% ret={ret_med:.3f} alpha_ann={alpha_ann:+.2f}pp "
          f"n>=70={n_ge_70}/{len(all_wrs)}", flush=True)

    return {
        "label": label,
        "genome": list(genome),
        "fit": float(fit),
        "wr_med_all": wr_med_all,
        "wr_med_active": wr_med,
        "mdd_med": mdd_med,
        "ret_med": ret_med,
        "alpha_ann": alpha_ann,
        "n_ge_70": n_ge_70,
    }


# Parse args
candidates_files = sys.argv[1:] if len(sys.argv) > 1 else [
    "global_ga_checkpoint.json",
]
print(f"[EVAL] Evaluating {len(candidates_files)} candidates...", flush=True)

results = []
for fp in candidates_files:
    if not os.path.exists(fp):
        print(f"[SKIP] {fp} not found", flush=True)
        continue
    try:
        ck = json.load(open(fp))
        if "genome" not in ck:
            # Could be a best_so_far with nested genome
            if "best_genome" in ck:
                genome = ck["best_genome"]
            else:
                print(f"[SKIP] {fp} no genome key", flush=True)
                continue
        else:
            genome = ck["genome"]
        r = evaluate(genome, os.path.basename(fp))
        r["file"] = fp
        results.append(r)
    except Exception as e:
        print(f"[ERR] {fp}: {e}", flush=True)

# Rank by: WR_med_all desc, alpha_ann desc, fit desc, mdd asc
def score_key(r):
    return (
        -r["wr_med_all"],
        -r["alpha_ann"],
        -r["fit"],
        r["mdd_med"],
    )

results.sort(key=score_key)
print("\n=== RANKING ===", flush=True)
print(f"{'#':<3} {'file':<40} {'fit':>8} {'WR':>7} {'MDD':>7} {'ret':>7} {'alpha':>9} {'>=70':>7}", flush=True)
for i, r in enumerate(results):
    star = " *" if i == 0 else ""
    print(f"{i+1:<3} {r['label'][:40]:<40} "
          f"{r['fit']:>8.3f} {r['wr_med_all']*100:>6.2f}% "
          f"{r['mdd_med']*100:>6.2f}% {r['ret_med']:>7.3f} "
          f"{r['alpha_ann']:>+8.2f}pp {r['n_ge_70']:>5d}/{len(cache):d}{star}", flush=True)

if results:
    best = results[0]
    print(f"\n[BEST] {best['label']} fit={best['fit']:.4f} WR={best['wr_med_all']*100:.2f}%", flush=True)
    # Save to analysis/best_selected.json
    with open("analysis/best_selected.json", "w") as f:
        json.dump({
            "fitness": best["fit"],
            "genome": best["genome"],
            "metrics": {
                "wr_med_all": best["wr_med_all"],
                "mdd_med": best["mdd_med"],
                "ret_med": best["ret_med"],
                "alpha_ann": best["alpha_ann"],
                "n_ge_70": best["n_ge_70"],
            },
            "source_file": best["file"],
            "stage_gate": GA_STAGE_GATE,
        }, f, indent=2)
    print(f"[SAVE] analysis/best_selected.json", flush=True)
