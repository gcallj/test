"""
Parameter sweep on chunk_3 genome: test variants of rr_ratio, partial_take,
and trailing_stop_mode to find better alpha without full retrain.

Hypothesis: chunk_3 has rr_ratio=1.0, partial_take=0.1, trailing=2.
With rr_ratio=1.0, take_hit dominates and winners are capped. Higher rr
might capture more alpha. Without partial takes, winners can run longer.
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

with open("global_ga_checkpoint.json") as f:
    ck = json.load(f)
base_genome = ck["genome"]

PARAM_IDX = {spec[0]: i for i, spec in enumerate(GLOBAL_PARAM_SPECS)}


def make_variant(base, overrides: dict):
    g = list(base)
    for k, v in overrides.items():
        if k in PARAM_IDX:
            g[PARAM_IDX[k]] = v
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

    wrs = [s["win_rate"] for s in stats]
    wr_tgts = [s.get("win_rate_target", 0.0) for s in stats]
    rets = [s["total_return"] for s in stats]
    bhs = [s["buy_hold_return"] for s in stats]
    excesses = [s["excess_return"] for s in stats]
    mdds = [abs(s["mdd"]) for s in stats]
    trades = [s.get("n_trades", 0) for s in stats]

    row = {
        "label": label,
        "fit": float(fit),
        "wr_med": float(np.median(wrs)) * 100,
        "wr_tgt_med": float(np.median(wr_tgts)) * 100,
        "ret_med": float(np.median(rets)),
        "excess_med": float(np.median(excesses)),
        "pct_beat_bh": float(np.mean([e > 0 for e in excesses])) * 100,
        "mdd_med": float(np.median(mdds)) * 100,
        "trades_med": float(np.median(trades)),
        "n_ge_70": sum(1 for w in wrs if w >= 0.70),
        "genome": list(genome),
    }
    print(
        f"[{label:<30}] fit={fit:>9.2f} WR={row['wr_med']:>5.1f}% "
        f"WR_tgt={row['wr_tgt_med']:>5.1f}% ret={row['ret_med']:>+6.3f} "
        f"excess={row['excess_med']:>+7.3f} beat={row['pct_beat_bh']:>4.1f}% "
        f"MDD={row['mdd_med']:>5.1f}% trades={row['trades_med']:>5.0f} "
        f">=70={row['n_ge_70']:>3d}",
        flush=True,
    )
    return row


print(f"\n[EVAL] Baseline...", flush=True)
results = [evaluate(base_genome, "base")]

print(f"\n[SWEEP] rr_ratio variants (partial_take unchanged)...", flush=True)
for rr in [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
    g = make_variant(base_genome, {"reward_risk_ratio": rr})
    results.append(evaluate(g, f"rr{rr}"))

print(f"\n[SWEEP] no partial_take variants (rr_ratio varied)...", flush=True)
for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
    g = make_variant(base_genome, {
        "reward_risk_ratio": rr,
        "partial_take_pct": 0.0,
        "partial_take_pct_2": 0.0,
    })
    results.append(evaluate(g, f"noPT_rr{rr}"))

print(f"\n[SWEEP] trailing_stop_mode variants...", flush=True)
for tm in [0, 1, 2]:
    g = make_variant(base_genome, {"trailing_stop_mode": tm})
    results.append(evaluate(g, f"trail{tm}"))

print(f"\n[SWEEP] combined: rr + no PT + trail1 ...", flush=True)
for rr in [1.5, 2.0, 2.5, 3.0]:
    g = make_variant(base_genome, {
        "reward_risk_ratio": rr,
        "partial_take_pct": 0.0,
        "partial_take_pct_2": 0.0,
        "trailing_stop_mode": 1,
    })
    results.append(evaluate(g, f"trail1_rr{rr}_noPT"))

print(f"\n[SWEEP] combined: rr + no PT + trail2 ...", flush=True)
for rr in [1.5, 2.0, 2.5, 3.0]:
    g = make_variant(base_genome, {
        "reward_risk_ratio": rr,
        "partial_take_pct": 0.0,
        "partial_take_pct_2": 0.0,
        "trailing_stop_mode": 2,
    })
    results.append(evaluate(g, f"trail2_rr{rr}_noPT"))


print("\n" + "=" * 100, flush=True)
print("RANKING BY FITNESS", flush=True)
print("=" * 100, flush=True)
ranked = sorted(results, key=lambda r: -r["fit"])
for i, r in enumerate(ranked[:20]):
    mark = " <-" if r["label"] == "base" else ""
    print(
        f"{i+1:<3} {r['label']:<30} fit={r['fit']:>8.2f} "
        f"WR={r['wr_med']:>5.1f}% WR_tgt={r['wr_tgt_med']:>5.1f}% "
        f"ret={r['ret_med']:>+6.3f} excess={r['excess_med']:>+7.3f} "
        f"trades={r['trades_med']:>5.0f}{mark}",
        flush=True,
    )

print("\n" + "=" * 100, flush=True)
print("RANKING BY WR_TARGET", flush=True)
print("=" * 100, flush=True)
ranked_wrt = sorted(results, key=lambda r: -r["wr_tgt_med"])
for i, r in enumerate(ranked_wrt[:10]):
    print(
        f"{i+1:<3} {r['label']:<30} WR_tgt={r['wr_tgt_med']:>5.1f}% "
        f"WR={r['wr_med']:>5.1f}% ret={r['ret_med']:>+6.3f} "
        f"fit={r['fit']:>8.2f}",
        flush=True,
    )

print("\n" + "=" * 100, flush=True)
print("RANKING BY EXCESS (alpha)", flush=True)
print("=" * 100, flush=True)
ranked_exc = sorted(results, key=lambda r: -r["excess_med"])
for i, r in enumerate(ranked_exc[:10]):
    print(
        f"{i+1:<3} {r['label']:<30} excess={r['excess_med']:>+7.3f} "
        f"ret={r['ret_med']:>+6.3f} WR_tgt={r['wr_tgt_med']:>5.1f}% "
        f"fit={r['fit']:>8.2f}",
        flush=True,
    )

out = {
    "results": [{k: v for k, v in r.items() if k != "genome"} for r in results],
    "best_by_fit": ranked[0]["label"],
    "best_by_fit_genome": ranked[0]["genome"],
    "best_by_wr_target": ranked_wrt[0]["label"],
    "best_by_wr_target_genome": ranked_wrt[0]["genome"],
    "best_by_excess": ranked_exc[0]["label"],
    "best_by_excess_genome": ranked_exc[0]["genome"],
}
with open("analysis/sweep_chunk3_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[SAVE] analysis/sweep_chunk3_results.json", flush=True)
