"""
Verifica empiricamente que o rank novo (ga_run.py modificado) tem correlacao
muito melhor com win_rate do que o antigo. Simula a logica diretamente.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
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

# Load base checkpoint
with open("global_ga_checkpoint.json") as f:
    ck = json.load(f)
base_genome = ck["genome"]
gp = decode_global_params(base_genome)

# Simulate rank computation for each ticker using current code logic
rows = []
for tk, p in cache.items():
    st = backtest_stats_global_intraday(
        p["open"], p["high"], p["low"], p["close"],
        p["score_matrix"], p["atr"], gp, precomputed=p,
    )
    wr = st.get("win_rate", 0.0)
    wrt = st.get("win_rate_target", 0.0)
    n_trades = st.get("n_trades", 0)
    if n_trades < 10:
        continue

    # Approximate fixed factors (use placeholders)
    confidence = 60.0 + wr * 30  # rough
    upside_score = 50.0
    q_timing_pre = 0.70
    q_pivot_pre = 0.70
    rr_ratio = 2.0
    regime_val = 0.60
    q_stop = 0.80

    # OLD rank formula
    rank_old = round(
        0.20 * confidence +
        0.20 * upside_score +
        0.15 * q_timing_pre * 100 +
        0.15 * q_pivot_pre * 100 +
        0.10 * min(rr_ratio / 3.0, 1.0) * 100 +
        0.10 * wr * 100 +
        0.05 * regime_val * 100 +
        0.05 * q_stop * 100,
        1
    )

    # NEW rank formula (just modified in ga_run.py)
    rank_new = round(
        0.15 * confidence +
        0.15 * upside_score +
        0.10 * q_timing_pre * 100 +
        0.10 * q_pivot_pre * 100 +
        0.10 * min(rr_ratio / 3.0, 1.0) * 100 +
        0.25 * wr * 100 +
        0.05 * wrt * 100 +
        0.05 * regime_val * 100 +
        0.05 * q_stop * 100,
        1
    )

    rows.append({
        "ticker": tk,
        "n_trades": n_trades,
        "win_rate_pct": wr * 100,
        "wr_target_pct": wrt * 100,
        "rank_old": rank_old,
        "rank_new": rank_new,
    })

df = pd.DataFrame(rows)

def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])

print(f"\n=== RANK COHERENCE SIMULATION ({len(df)} tickers) ===", flush=True)
print(f"  OLD rank vs win_rate:     {corr(df['rank_old'], df['win_rate_pct']):+.3f}", flush=True)
print(f"  NEW rank vs win_rate:     {corr(df['rank_new'], df['win_rate_pct']):+.3f}", flush=True)
print(f"  OLD rank vs wr_target:    {corr(df['rank_old'], df['wr_target_pct']):+.3f}", flush=True)
print(f"  NEW rank vs wr_target:    {corr(df['rank_new'], df['wr_target_pct']):+.3f}", flush=True)
print(f"  OLD vs NEW rank:          {corr(df['rank_old'], df['rank_new']):+.3f}", flush=True)

print(f"\n=== TOP 10 by NEW rank ===", flush=True)
df_sorted = df.sort_values("rank_new", ascending=False).head(10)
for _, r in df_sorted.iterrows():
    print(f"  {r['ticker']:<12} WR={r['win_rate_pct']:>5.1f}% WR_tgt={r['wr_target_pct']:>5.1f}% "
          f"rank_old={r['rank_old']:>5.1f} rank_new={r['rank_new']:>5.1f}",
          flush=True)

print(f"\n=== BOTTOM 10 by NEW rank ===", flush=True)
df_sorted_asc = df.sort_values("rank_new", ascending=True).head(10)
for _, r in df_sorted_asc.iterrows():
    print(f"  {r['ticker']:<12} WR={r['win_rate_pct']:>5.1f}% WR_tgt={r['wr_target_pct']:>5.1f}% "
          f"rank_old={r['rank_old']:>5.1f} rank_new={r['rank_new']:>5.1f}",
          flush=True)

df.to_csv("analysis/rank_coherence_simulation.csv", index=False)
print(f"\n[SAVE] analysis/rank_coherence_simulation.csv", flush=True)
