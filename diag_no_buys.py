"""
Diagnostico: por que o GA nao esta produzindo buys no dia 08/04?

Replica a logica de vectorized_signals(apply_mode=True) e mostra
ponto a ponto onde cada ticker Stage 2 ideal/momentum eh bloqueado.
"""
import os
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

os.environ["GA_RUN_MODE"] = "load"
os.environ["GA_EVAL_WORKERS"] = "1"
import ga_run
from ga_run import (
    decode_global_params, classify_minervini_stage,
    compute_weighted_votes, compute_regime_score,
    rolling_mean_np, rolling_quantile_trigger, ewm_mean_np,
    SCORE_LOOKBACK, ATR_EPS,
)
from payload_store import PayloadStore

# Load genome
ck = json.load(open("global_ga_checkpoint.json"))
gp = decode_global_params(ck["genome"])
print(f"=== Checkpoint: fitness={ck['fitness']:.4f} ===")
print(f"Key genes:")
print(f"  vote_threshold_long  = {gp.vote_threshold_long:.3f}")
print(f"  vote_threshold_short = {gp.vote_threshold_short:.3f}")
print(f"  z_threshold          = {gp.z_threshold:.3f}")
print(f"  signal_ema_span      = {gp.signal_ema_span}")
print(f"  entry_confirmation_days = {gp.entry_confirmation_days}")
print(f"  score_percentile_trigger = {gp.score_percentile_trigger:.3f}")
print(f"  volatility_filter_pct = {gp.volatility_filter_percentile:.3f}")
print(f"  vol_regime_mode      = {gp.vol_regime_mode}")
print(f"  regime_threshold     = {gp.regime_threshold:.3f}")
print(f"  ma_filter_period     = {gp.ma_filter_period}")
print(f"  ma_filter_mode       = {gp.ma_filter_mode}")
print(f"  min_signal_strength  = {gp.min_signal_strength:.3f}")
print(f"  entry_score_threshold = {gp.entry_score_threshold:.3f}")
print(f"  volume_confirm_mode  = {gp.volume_confirm_mode}")
print(f"  momentum_confirm_days = {gp.momentum_confirm_days}")
print()

# Open store
# ga_run cleans up GA_MEMMAP_DIR after load, so use the staged version
_store_candidates = [
    os.path.abspath("output/ga_memmap"),
    os.path.abspath("output/ga_memmap_staged"),
]
store_path = None
for p in _store_candidates:
    if os.path.exists(os.path.join(p, "meta.json")):
        store_path = p
        break
if store_path is None:
    raise RuntimeError(f"No PayloadStore found in {_store_candidates}. Run ga_run first.")
print(f"Using store: {store_path}")
store = PayloadStore(store_path, mode="r")
print(f"Store: {len(store.tickers)} tickers")

# Analyze each ticker at the last day (i = N-1)
filter_reasons = Counter()
stage_dist = Counter()
stage_by_reason = defaultdict(Counter)

# For Stage 2 ideal/momentum blockers — show details
stage2_blocks = []

for tk in store.tickers:
    arr = store.get_ticker_arrays(tk)
    c = np.asarray(arr["close"], dtype=np.float64)
    o = np.asarray(arr["open"], dtype=np.float64)
    h = np.asarray(arr["high"], dtype=np.float64)
    l = np.asarray(arr["low"], dtype=np.float64)
    atr = np.asarray(arr["atr"], dtype=np.float64)
    vol = np.asarray(arr["volume"], dtype=np.float64) if "volume" in arr else None
    sm = np.asarray(arr["score_matrix"], dtype=np.float64)
    n = len(c)
    if n < 252:
        continue
    i = n - 1  # last day

    # Recompute indicators
    ma = rolling_mean_np(c, int(gp.ma_filter_period), max(20, int(gp.ma_filter_period // 2)))
    ma50 = rolling_mean_np(c, 50, 10)
    ma200 = rolling_mean_np(c, 200, 50)
    vol_rank = arr.get("vol_rank")
    if vol_rank is None:
        from ga_run import rolling_percentile_rank
        vol_rank = rolling_percentile_rank(atr / np.maximum(c, ATR_EPS), 252)
    else:
        vol_rank = np.asarray(vol_rank, dtype=np.float64)
    regime = compute_regime_score(c, ma200, vol_rank)

    # Votes
    feat_n = max(1, sm.shape[1])
    votes_long, votes_short = compute_weighted_votes(sm, gp.z_threshold, feat_n)
    score_raw = votes_long - votes_short
    score_ev = ewm_mean_np(score_raw, int(gp.signal_ema_span))
    score_pctl = rolling_quantile_trigger(score_ev, float(gp.score_percentile_trigger), max(63, SCORE_LOOKBACK))
    score95 = np.nanpercentile(np.abs(score_ev), 95) if np.isfinite(np.nanmax(np.abs(score_ev))) else 1.0
    score95 = max(score95, ATR_EPS)

    # Classify stage for this ticker
    ma50_now = ma50[i] if np.isfinite(ma50[i]) else np.nan
    ma200_now = ma200[i] if np.isfinite(ma200[i]) else np.nan
    ma50_prev = ma50[i-20] if i >= 20 and np.isfinite(ma50[i-20]) else np.nan
    ma200_prev = ma200[i-20] if i >= 20 and np.isfinite(ma200[i-20]) else np.nan

    stage_num, stage_label, timing_sub, _ = classify_minervini_stage(
        close=float(c[i]), ma50=ma50_now, ma200=ma200_now,
        ma50_prev=ma50_prev, ma200_prev=ma200_prev,
    )
    stage_key = f"{stage_label}_{timing_sub}"
    stage_dist[stage_key] += 1

    # ===== Aplicar filtros do vectorized_signals(apply_mode=True) =====
    reason = None

    # 1. volatility_filter_percentile
    if gp.volatility_filter_percentile > 0 and np.isfinite(vol_rank[i-1]) and vol_rank[i-1] < gp.volatility_filter_percentile:
        reason = "vol_filter_low"
    # 2. vol_regime_mode 2: skip high vol
    elif gp.vol_regime_mode == 2 and np.isfinite(vol_rank[i-1]) and vol_rank[i-1] > 0.85:
        reason = "vol_regime_high"
    # 3. atr liquidity
    elif atr[i-1] < 0.01:
        reason = "atr_low"
    # 4. regime_threshold (apply_mode still applies this as long_ok = False)
    elif gp.regime_threshold > 0 and np.isfinite(regime[i-1]) and regime[i-1] < gp.regime_threshold:
        reason = f"regime_threshold ({regime[i-1]:.2f} < {gp.regime_threshold:.2f})"
    # 5. vote thresholds
    else:
        vl = votes_long[i-1] if i >= 1 else 0.0
        vs = votes_short[i-1] if i >= 1 else 0.0
        score_now = score_ev[i-1] if i >= 1 and np.isfinite(score_ev[i-1]) else 0.0
        pctl_now = score_pctl[i-1] if i >= 1 and np.isfinite(score_pctl[i-1]) else 0.0

        long_cond = (vl >= gp.vote_threshold_long) and (score_now >= pctl_now)
        short_cond = (vs >= gp.vote_threshold_short) and (-score_now >= pctl_now)

        if not (long_cond or short_cond):
            reason = f"vote_fail (vl={vl:.3f}/{gp.vote_threshold_long:.3f} score={score_now:.3f}/pctl={pctl_now:.3f})"
        elif gp.entry_confirmation_days > 1:
            # Check consecutive days
            consec = 0
            for j in range(max(0, i - 5), i + 1):
                if j < 1:
                    continue
                vl_j = votes_long[j-1] if j >= 1 else 0.0
                score_j = score_ev[j-1] if np.isfinite(score_ev[j-1]) else 0.0
                pctl_j = score_pctl[j-1] if np.isfinite(score_pctl[j-1]) else 0.0
                if (vl_j >= gp.vote_threshold_long) and (score_j >= pctl_j):
                    consec += 1
                else:
                    consec = 0
            if consec < gp.entry_confirmation_days:
                reason = f"confirmation_days ({consec}/{int(gp.entry_confirmation_days)})"
            else:
                # 6. ma_filter_mode
                if gp.ma_filter_mode == 2 and np.isfinite(ma[i-1]):
                    if c[i-1] < ma[i-1]:
                        reason = "ma_filter_below"
                    else:
                        reason = "PASS"
                else:
                    reason = "PASS"
        else:
            # ma_filter check
            if gp.ma_filter_mode == 2 and np.isfinite(ma[i-1]):
                if c[i-1] < ma[i-1]:
                    reason = "ma_filter_below"
                else:
                    reason = "PASS"
            else:
                reason = "PASS"

    filter_reasons[reason.split()[0]] += 1
    stage_by_reason[reason.split()[0]][stage_key] += 1

    if stage_num == 2 and timing_sub in ("ideal", "momentum") and reason != "PASS":
        stage2_blocks.append({
            "ticker": tk,
            "stage": stage_key,
            "reason": reason,
            "vl": float(votes_long[i-1]) if i >= 1 else 0,
            "vs": float(votes_short[i-1]) if i >= 1 else 0,
            "score_ev": float(score_ev[i-1]) if i >= 1 and np.isfinite(score_ev[i-1]) else 0,
            "score_pctl": float(score_pctl[i-1]) if i >= 1 and np.isfinite(score_pctl[i-1]) else 0,
            "regime": float(regime[i-1]),
            "vol_rank": float(vol_rank[i-1]) if np.isfinite(vol_rank[i-1]) else 0,
        })

print(f"\n=== Distribuicao de filter reasons (last day, {sum(filter_reasons.values())} tickers) ===")
for k, v in sorted(filter_reasons.items(), key=lambda x: -x[1]):
    print(f"  {v:>4} {k}")
print()

print(f"=== Stage distribution ===")
for k, v in sorted(stage_dist.items(), key=lambda x: -x[1]):
    print(f"  {v:>4} {k}")
print()

print(f"=== Stage 2 ideal/momentum BLOCKED tickers ({len(stage2_blocks)}) ===")
for b in stage2_blocks[:20]:
    print(f"  {b['ticker']:<12} {b['stage']:<25} {b['reason']}")
    print(f"               vl={b['vl']:.3f} score_ev={b['score_ev']:.3f} pctl={b['score_pctl']:.3f} regime={b['regime']:.2f}")

store.close() if hasattr(store, 'close') else None
