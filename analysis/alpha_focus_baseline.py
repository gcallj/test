"""
Quick eval: base genome fitness with GA_ALPHA_FOCUS=on (new fitness weights)
vs GA_ALPHA_FOCUS=off (default fitness). This shows the baseline the GA will
need to beat in the alpha-focused retrain.
"""
import os
import sys
import json
import importlib
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

os.environ["GA_RUN_MODE"] = "load"
os.environ["GA_STAGE_GATE"] = "off"


def eval_with_focus(focus_mode):
    os.environ["GA_ALPHA_FOCUS"] = focus_mode
    # Reimport ga_run to pick up the env var change
    if "ga_run" in sys.modules:
        del sys.modules["ga_run"]
    import ga_run
    from payload_store import PayloadStore

    print(f"\n[FOCUS={focus_mode}] GA_ALPHA_FOCUS={ga_run.GA_ALPHA_FOCUS}", flush=True)

    candidates = [
        os.path.abspath(ga_run.GA_MEMMAP_DIR),
        os.path.abspath("output/ga_memmap_staged"),
    ]
    store_path = None
    for p in candidates:
        if os.path.exists(os.path.join(p, "meta.json")):
            store_path = p
            break
    store = PayloadStore(store_path, mode="r")

    # Build cache
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

    with open("global_ga_checkpoint.json") as f:
        base_genome = json.load(f)["genome"]

    gp = ga_run.decode_global_params(base_genome)
    stats = []
    for tk, p in cache.items():
        st = ga_run.backtest_stats_global_intraday(
            p["open"], p["high"], p["low"], p["close"],
            p["score_matrix"], p["atr"], gp, precomputed=p,
        )
        bh = ga_run.buyhold_capped(p["close"])
        st["buy_hold_return"] = bh
        st["excess_return"] = st.get("total_return", 0.0) - bh
        stats.append(st)

    fit = ga_run.global_fitness_from_stats(stats)
    mean_excess = float(np.mean([s["excess_return"] for s in stats]))
    med_excess = float(np.median([s["excess_return"] for s in stats]))
    pct_beat = float(np.mean([s["excess_return"] > 0 for s in stats]))
    print(f"[FOCUS={focus_mode}] fit={fit:.3f} mean_excess={mean_excess:+.3f} "
          f"med_excess={med_excess:+.3f} pct_beat={pct_beat*100:.1f}%", flush=True)

    return fit


fit_off = eval_with_focus("off")
fit_on = eval_with_focus("on")
print(f"\n[DELTA] fit_on - fit_off = {fit_on - fit_off:+.3f}", flush=True)
print(f"[NOTE] fit_on is baseline GA must beat with seeded retrain", flush=True)
