"""Diagnose por que UGPA3.SA nao aparece no summary."""
import os
os.environ["GA_RUN_MODE"] = "load"

import json
import numpy as np
from payload_store import PayloadStore
import ga_run as gr
from ga_run import (
    decode_global_params, backtest_stats_global_intraday,
    classify_minervini_stage,
)

# Open store
candidates = ["output/ga_memmap", "output/ga_memmap_staged"]
store_path = None
for p in candidates:
    if os.path.exists(os.path.join(p, "meta.json")):
        store_path = os.path.abspath(p)
        break
print(f"Store: {store_path}")
store = PayloadStore(store_path, mode="r")

print(f"Total tickers in store: {len(store.tickers)}")
ugpa = [t for t in store.tickers if "UGPA" in t]
print(f"UGPA matches: {ugpa}")

if not ugpa:
    print("UGPA not in store - issue is in Phase 1 filter")
    raise SystemExit(0)

tk = ugpa[0]
arr = store.get_ticker_arrays(tk)
print(f"\n[{tk}]")
print(f"  rows: {len(arr['close'])}")
print(f"  open shape: {arr['open'].shape if arr.get('open') is not None else None}")
print(f"  score_matrix shape: {arr['score_matrix'].shape if arr.get('score_matrix') is not None else None}")
print(f"  has volume: {arr.get('volume') is not None}")
print(f"  has ma: {arr.get('ma') is not None}")
print(f"  has vol_rank: {arr.get('vol_rank') is not None}")

# Cast
payload = dict(arr)
for k in ("open", "high", "low", "close", "atr", "volume"):
    if k in payload and payload[k] is not None:
        payload[k] = np.asarray(payload[k], dtype=np.float64)
if "score_matrix" in payload:
    payload["score_matrix"] = np.asarray(payload["score_matrix"], dtype=np.float64)

import pandas as pd
if "dates" in payload:
    payload["dates"] = pd.to_datetime(np.asarray(payload["dates"], dtype=np.int64), unit="ns")

c = payload["close"]
n = len(c)
print(f"\n  n={n}")
print(f"  last 5 dates: {payload['dates'][-5:]}")
print(f"  last 5 closes: {c[-5:]}")
print(f"  last 5 ATR: {payload['atr'][-5:]}")
print(f"  any nan in last 30 close: {np.isnan(c[-30:]).sum()}")
print(f"  any nan in last 30 atr: {np.isnan(payload['atr'][-30:]).sum()}")

# Filter checks from apply loop
print("\n=== APPLY LOOP FILTER CHECKS ===")
print(f"  n < 252? {n < 252}")

# Genome
ck = json.load(open("global_ga_checkpoint.json"))
gp = decode_global_params(ck["genome"])

# Run backtest
st = backtest_stats_global_intraday(
    payload["open"], payload["high"], payload["low"], payload["close"],
    payload["score_matrix"], payload["atr"], gp, precomputed=payload
)
print(f"\n  backtest stats: {st}")

# Check stop_loss/take_profit calculation
from ga_run import (
    rolling_mean_np, ATR_EPS, MIN_STOP_PCT, MIN_RR_RATIO
)
i = n - 1
close_val = float(c[i])
atr_val = float(payload["atr"][i])
print(f"\n  close[{i}]={close_val}")
print(f"  atr[{i}]={atr_val}")
print(f"  atr finite: {np.isfinite(atr_val)}")

# stop calc
stop_atr_val = gp.stop_atr_mult * atr_val
stop_pct_from_atr = stop_atr_val / max(close_val, ATR_EPS)
effective_stop_pct = max(stop_pct_from_atr, MIN_STOP_PCT)
print(f"  stop_pct_from_atr: {stop_pct_from_atr:.4f}")
print(f"  effective_stop_pct: {effective_stop_pct:.4f}")

# Check if take_reward/stop_risk would be valid
stop_risk = close_val * effective_stop_pct
print(f"  stop_risk: {stop_risk:.4f}")
print(f"  > 0? {stop_risk > 0}")

# Stage
ma50 = rolling_mean_np(c, 50, 10)
ma200 = rolling_mean_np(c, 200, 50)
ma50_now = ma50[i] if np.isfinite(ma50[i]) else np.nan
ma200_now = ma200[i] if np.isfinite(ma200[i]) else np.nan
print(f"\n  MA50[{i}]={ma50_now}")
print(f"  MA200[{i}]={ma200_now}")
print(f"  finite ma50: {np.isfinite(ma50_now)}")
print(f"  finite ma200: {np.isfinite(ma200_now)}")

stage_num, stage_label, timing_sub, _ = classify_minervini_stage(
    close=close_val, ma50=ma50_now, ma200=ma200_now,
    ma50_prev=ma50[i-20] if i >= 20 else np.nan,
    ma200_prev=ma200[i-20] if i >= 20 else np.nan,
)
print(f"  stage: {stage_num} {stage_label} {timing_sub}")

# Try the WHOLE apply path
print("\n=== SIMULATING APPLY LOOP ===")
try:
    from ga_run import (
        compute_weighted_votes, ewm_mean_np, rolling_quantile_trigger,
        SCORE_LOOKBACK, build_direct_feature_signal,
    )
    score_matrix = payload["score_matrix"]
    feat_n = max(1, score_matrix.shape[1])
    print(f"  feat_n: {feat_n}")
    print(f"  score_matrix dtype: {score_matrix.dtype}")
    print(f"  any nan in last row: {np.isnan(score_matrix[i]).sum()}")

    votes_long, votes_short = compute_weighted_votes(score_matrix, gp.z_threshold, feat_n)
    score_raw = votes_long - votes_short
    score_ev = ewm_mean_np(score_raw, int(gp.signal_ema_span))
    score_pctl = rolling_quantile_trigger(score_ev, float(gp.score_percentile_trigger), max(63, SCORE_LOOKBACK))
    print(f"  votes_long[i]: {votes_long[i]}")
    print(f"  score_ev[i]: {score_ev[i]}")
    print(f"  score_pctl[i]: {score_pctl[i]}")

    # Stop / target
    take_reward = stop_risk * gp.reward_risk_ratio
    take_profit = close_val + take_reward
    stop_loss = close_val - stop_risk
    print(f"  entrada candidate: {close_val}")
    print(f"  stop_loss: {stop_loss}")
    print(f"  take_profit: {take_profit}")
    print(f"  rr: {take_reward / stop_risk}")
except Exception as e:
    import traceback
    traceback.print_exc()
