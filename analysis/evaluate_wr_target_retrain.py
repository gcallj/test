"""
Evaluate cada chunk do retrain wr_target vs base com TODOS os metricos
(WR, WR_target, alpha, MDD, fit completo).

O selector run_local_ga_staged rejeitou todos os chunks porque so olha
WR_med_all. Mas o objetivo era aumentar win_rate_target — preciso checar
se algum chunk realmente melhorou esse metric antes de descartar.
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

print(f"[INIT] Building cache...", flush=True)
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


def evaluate_full(genome, label):
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

    wrs = [s["win_rate"] for s in stats]
    wr_tgts = [s.get("win_rate_target", 0.0) for s in stats]
    pct_takes = [s.get("pct_exit_take", 0.0) for s in stats]
    rets = [s["total_return"] for s in stats]
    bhs = [s["buy_hold_return"] for s in stats]
    excesses = [s["excess_return"] for s in stats]
    mdds = [abs(s["mdd"]) for s in stats]
    trades = [s.get("n_trades", 0) for s in stats]
    sharpes = [s.get("sharpe", 0) for s in stats]

    row = {
        "label": label,
        "fit": float(fit),
        "wr_med": float(np.median(wrs)) * 100,
        "wr_mean": float(np.mean(wrs)) * 100,
        "wr_target_med": float(np.median(wr_tgts)) * 100,
        "wr_target_mean": float(np.mean(wr_tgts)) * 100,
        "pct_take_mean": float(np.mean(pct_takes)) * 100,
        "ret_med": float(np.median(rets)),
        "excess_med": float(np.median(excesses)),
        "pct_beat_bh": float(np.mean([e > 0 for e in excesses])) * 100,
        "mdd_med": float(np.median(mdds)) * 100,
        "sharpe_med": float(np.median(sharpes)),
        "trades_med": float(np.median(trades)),
        "n_ge_70": sum(1 for w in wrs if w >= 0.70),
        "n_ge_50_target": sum(1 for w in wr_tgts if w >= 0.50),
        "n_ge_60_target": sum(1 for w in wr_tgts if w >= 0.60),
        "genome": list(genome),
    }
    print(f"[{label:<15}] fit={fit:>9.3f} WR={row['wr_med']:>5.1f}% "
          f"WR_tgt={row['wr_target_med']:>5.1f}% "
          f"take={row['pct_take_mean']:>5.1f}% "
          f"ret={row['ret_med']:>+6.3f} MDD={row['mdd_med']:>5.1f}% "
          f"trades={row['trades_med']:>5.0f} "
          f">=70={row['n_ge_70']:>3d} >=50tgt={row['n_ge_50_target']:>3d}",
          flush=True)
    return row


# Load base and all chunks
with open("global_ga_checkpoint.json") as f:
    base_genome = json.load(f)["genome"]

with open("local_ga_chunk_progress.json") as f:
    progress = json.load(f)
chunks = progress.get("chunks", [])
print(f"[INIT] Found {len(chunks)} chunks in progress file\n", flush=True)

print("=" * 100, flush=True)
print("EVALUATION: base + all chunks with FULL fitness (includes wr_target_bonus)",
      flush=True)
print("=" * 100, flush=True)

results = []
results.append(evaluate_full(base_genome, "base"))
for ch in chunks:
    label = f"chunk_{ch['chunk_id']}"
    results.append(evaluate_full(ch["best_genome"], label))

# Rank by fit
print("\n" + "=" * 100, flush=True)
print("RANKING BY FITNESS", flush=True)
print("=" * 100, flush=True)
ranked = sorted(results, key=lambda r: -r["fit"])
print(f"{'#':<3} {'label':<12} {'fit':>10} {'WR':>7} {'WR_tgt':>8} {'take':>7} "
      f"{'ret':>7} {'MDD':>7} {'trades':>7} {'>=70':>5} {'>=50tgt':>7}",
      flush=True)
for i, r in enumerate(ranked):
    mark = " <-" if r["label"] == "base" else ""
    print(f"{i+1:<3} {r['label']:<12} "
          f"{r['fit']:>10.3f} {r['wr_med']:>6.1f}% {r['wr_target_med']:>7.1f}% "
          f"{r['pct_take_mean']:>6.1f}% "
          f"{r['ret_med']:>+7.3f} {r['mdd_med']:>6.1f}% "
          f"{r['trades_med']:>7.0f} {r['n_ge_70']:>5d} {r['n_ge_50_target']:>7d}{mark}",
          flush=True)

# Rank by wr_target_med (the goal metric)
print("\n" + "=" * 100, flush=True)
print("RANKING BY WR_TARGET_MED (objetivo do retrain)", flush=True)
print("=" * 100, flush=True)
ranked_tgt = sorted(results, key=lambda r: -r["wr_target_med"])
for i, r in enumerate(ranked_tgt):
    mark = " <-" if r["label"] == "base" else ""
    print(f"{i+1:<3} {r['label']:<12} WR_tgt={r['wr_target_med']:>5.1f}% "
          f"(WR {r['wr_med']:>5.1f}%, fit {r['fit']:>7.2f}, "
          f"MDD {r['mdd_med']:>5.1f}%, trades {r['trades_med']:>4.0f}){mark}",
          flush=True)

# Save results
out = {
    "results": [{k: v for k, v in r.items() if k != "genome"} for r in results],
    "best_by_fit": ranked[0]["label"],
    "best_by_wr_target": ranked_tgt[0]["label"],
    "best_by_fit_genome": ranked[0]["genome"],
    "best_by_wr_target_genome": ranked_tgt[0]["genome"],
}
with open("analysis/wr_target_retrain_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[SAVE] analysis/wr_target_retrain_results.json", flush=True)
