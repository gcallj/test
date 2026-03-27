#!/usr/bin/env python3
"""
Compare v3 (old genes OFF) vs v4 (new genes ON) on real data.
Uses ga_run internals directly, with reduced GA size for speed.
"""
import os
import sys
import time
import random
import numpy as np
import pandas as pd
import multiprocessing

# Patch GA params for speed before import
os.environ["GA_RUN_MODE"] = "train"

# Now import ga_run — this loads all the machinery
import ga_run
from ga_run import (
    GLOBAL_PARAM_SPECS, GlobalParams, decode_global_params,
    backtest_stats_global_intraday, global_fitness_from_stats,
    precompute_global_payloads, evaluate_global_walkforward,
    buyhold_capped, sanitize_global_genome,
    RANDOM_SEED, GA_WF_TRAIN_YEARS, GA_WF_TEST_DAYS,
    GA_WF_STEP_DAYS, ONE_YEAR_DAYS,
)
from deap import base, creator, tools

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Speed params ──
POP_SIZE = 20
NGEN = 6
CX_PB = 0.70
MUT_PB = 0.40
TOURN = 3

print("=" * 80)
print("  V3 vs V4 COMPARISON — Real Brazilian Equity Data")
print("=" * 80)

# ── Reuse ga_run.run() for data loading + payload building ──
# We'll intercept the payload building stage and then run our own GA

# Monkey-patch GA params for speed
ga_run.GA_POP_SIZE = POP_SIZE
ga_run.GA_NGEN = NGEN
ga_run.GA_WF_SPLITS = 3

print("\n[1/4] Loading data and building payloads (reusing ga_run pipeline)...")
t0 = time.time()

# Execute the data loading portion of run() manually
df = ga_run.load_full_history_all_cols(ga_run.HISTORY_CSV_PATH)

if ga_run.MERGE_ETL_FEATURES:
    try:
        consolidated_tickers = set(df[ga_run.TICKER_COL].unique())
        etl_df = ga_run.load_etl_features_long(
            ga_run.ETL_PARQUET_PATH,
            ticker_filter=lambda t: t in consolidated_tickers,
        )
        if len(etl_df) > 0:
            df[ga_run.DATE_COL] = pd.to_datetime(df[ga_run.DATE_COL])
            etl_df[ga_run.DATE_COL] = pd.to_datetime(etl_df[ga_run.DATE_COL])
            n_before = len(df.columns)
            df = df.merge(etl_df, on=[ga_run.DATE_COL, ga_run.TICKER_COL], how="left")
            print(f"   [ETL MERGE] +{len(df.columns) - n_before} ETL features")
    except Exception as e:
        print(f"   [ETL MERGE] Skipped: {e}")

exclude = {ga_run.DATE_COL, ga_run.TICKER_COL, ga_run.OPEN_COL, ga_run.HIGH_COL,
           ga_run.LOW_COL, ga_run.CLOSE_COL, "sma200", "sma200_slope", "atr"}
base_feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

tickers = df[ga_run.TICKER_COL].dropna().unique().tolist()
if ga_run.ALLOWED_SUFFIXES:
    tickers = [t for t in tickers if any(str(t).endswith(s) for s in ga_run.ALLOWED_SUFFIXES)]

print(f"   Data: {len(df):,} rows, {len(df.columns)} cols, {len(tickers)} tickers")

# Build payloads with hybrid feature selection (same as run())
from scipy.stats import spearmanr, pointbiserialr
from sklearn.feature_selection import mutual_info_classif

ticker_payloads = {}
for tkr in tickers:
    g = df[df[ga_run.TICKER_COL] == tkr].copy()
    if len(g) < ga_run.MIN_ROWS_TICKER:
        continue
    g = g.sort_values(ga_run.DATE_COL).reset_index(drop=True)

    c = g[ga_run.CLOSE_COL].to_numpy(np.float64)
    atr = g["atr"].to_numpy(np.float64)
    n = len(c)

    # Forward return for feature selection
    ret_fwd = np.empty_like(c)
    ret_fwd[:-ga_run.FWD_H] = (c[ga_run.FWD_H:] / c[:-ga_run.FWD_H]) - 1.0
    ret_fwd[-ga_run.FWD_H:] = np.nan
    atr_pct = atr / np.maximum(c, 1e-12)
    thr = np.maximum(ga_run.TARGET_RET_THRESHOLD, ga_run.TARGET_ATR_MULT * atr_pct)
    y_event = (ret_fwd > thr).astype(float)
    y_event[~np.isfinite(ret_fwd)] = np.nan
    y_event[~np.isfinite(thr)] = np.nan

    # Simple Spearman feature selection (faster than hybrid for comparison)
    feat_cols_valid = [fc for fc in base_feat_cols if fc in g.columns]
    scored = []
    for fc in feat_cols_valid:
        arr = pd.to_numeric(g[fc], errors="coerce").to_numpy(np.float64)
        mask = np.isfinite(arr) & np.isfinite(y_event)
        if mask.sum() < 30:
            continue
        r, p = spearmanr(arr[mask], y_event[mask])
        if np.isfinite(r) and abs(r) > 0.01:
            scored.append((fc, abs(r)))

    scored.sort(key=lambda x: x[1], reverse=True)
    feat_cols = [s[0] for s in scored[:ga_run.MAX_FEATURES]]

    if len(feat_cols) < 5:
        continue

    directed_cols, long_votes, short_votes = ga_run.build_direct_feature_signal(g, feat_cols)
    dates = pd.to_datetime(g[ga_run.DATE_COL], errors="coerce").to_numpy()
    o = g[ga_run.OPEN_COL].to_numpy(np.float64)
    h = g[ga_run.HIGH_COL].to_numpy(np.float64)
    l = g[ga_run.LOW_COL].to_numpy(np.float64)
    ma = g["sma200"].to_numpy(np.float64) if "sma200" in g.columns else ga_run.rolling_mean_np(c, 200, 100)
    vol = g["volume"].to_numpy(np.float64) if "volume" in g.columns else np.full(n, np.nan)

    valid_mask = np.isfinite(c) & (c > 0)
    ticker_payloads[str(tkr)] = {
        "open": o, "high": h, "low": l, "close": c, "atr": atr,
        "volume": vol,
        "score_matrix": directed_cols, "dates": dates, "ma": ma,
        "feat_cols": feat_cols, "valid_mask": valid_mask,
        "long_votes": long_votes, "short_votes": short_votes,
    }

print(f"   Built payloads for {len(ticker_payloads)} tickers in {time.time()-t0:.1f}s")

# ── Walk-forward splits ──
print("\n[2/4] Building walk-forward splits...")
from ga_run import build_temporal_windows

all_dates = sorted(set(d for p in ticker_payloads.values() for d in p["dates"]))
min_date = pd.Timestamp(all_dates[0])
max_date = pd.Timestamp(all_dates[-1])
windows = build_temporal_windows(min_date, max_date, train_years=3, test_months=6, step_months=6)
# Use last 2 windows for speed
windows = windows[-2:] if len(windows) >= 2 else windows
print(f"   {len(windows)} WF windows: {[(str(w[0].date()), str(w[3].date())) for w in windows]}")

payloads_by_window = precompute_global_payloads(windows, ticker_payloads)

# ── GA ──
print(f"\n[3/4] Running GA (pop={POP_SIZE}, gen={NGEN})...")
t0 = time.time()

if not hasattr(creator, "FitnessMax_cmp"):
    creator.create("FitnessMax_cmp", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual_cmp"):
    creator.create("Individual_cmp", list, fitness=creator.FitnessMax_cmp)

tb = base.Toolbox()

def init_ind():
    g = [random.uniform(s[1], s[2]) for s in GLOBAL_PARAM_SPECS]
    return creator.Individual_cmp(sanitize_global_genome(g))

tb.register("individual", init_ind)
tb.register("population", tools.initRepeat, list, tb.individual)
tb.register("evaluate", lambda ind: (evaluate_global_walkforward(sanitize_global_genome(list(ind)), payloads_by_window)[0],))
tb.register("mate", tools.cxBlend, alpha=0.3)
tb.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
tb.register("select", tools.selTournament, tournsize=TOURN)

pop = tb.population(n=POP_SIZE)
hof = tools.HallOfFame(3)

for gen in range(NGEN):
    fits = list(map(tb.evaluate, pop))
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit
    hof.update(pop)
    fv = [f[0] for f in fits]
    print(f"   Gen {gen+1}/{NGEN}: best={max(fv):.4f}  avg={np.mean(fv):.4f}")

    offspring = tb.select(pop, len(pop))
    offspring = list(map(tb.clone, offspring))
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CX_PB:
            tb.mate(c1, c2)
            del c1.fitness.values, c2.fitness.values
    for m in offspring:
        if random.random() < MUT_PB:
            tb.mutate(m)
            del m.fitness.values
    for ind in offspring:
        for i, s in enumerate(GLOBAL_PARAM_SPECS):
            ind[i] = float(np.clip(ind[i], s[1], s[2]))
        ind[:] = sanitize_global_genome(list(ind))
    pop = offspring

best_v4 = list(hof[0])
print(f"   GA done in {time.time()-t0:.1f}s")

# ── Evaluate V4 vs V3 ──
print("\n[4/4] Comparing V3 vs V4...")

def collect_stats(genome):
    gp = decode_global_params(sanitize_global_genome(list(genome)))
    stats = []
    for tr_pay, te_pay in payloads_by_window:
        for tkr, payload in te_pay.items():
            st = backtest_stats_global_intraday(
                payload["open"], payload["high"], payload["low"], payload["close"],
                payload["score_matrix"], payload["atr"], gp, precomputed=payload,
            )
            st["buy_hold_return"] = buyhold_capped(payload["close"])
            st["excess_return"] = st["total_return"] - st["buy_hold_return"]
            st["ticker"] = tkr
            stats.append(st)
    return stats

# V4: best genome from GA
stats_v4 = collect_stats(best_v4)

# V3: same genome but v4 genes OFF
best_v3 = list(best_v4)
best_v3[26] = 0.0  # volume_confirm_mode = off
best_v3[27] = 0.0  # momentum_confirm_days = off
best_v3[28] = 0.0  # entry_score_threshold = off
best_v3[29] = 0.0  # regime_threshold = off
stats_v3 = collect_stats(best_v3)

# ── Report ──
def summarize(stats, label):
    with_trades = [s for s in stats if s["n_trades"] > 0]
    if not with_trades:
        print(f"\n  {label}: No trades"); return {}
    r = np.array([s["total_return"] for s in with_trades])
    ex = np.array([s["excess_return"] for s in with_trades])
    sh = np.array([s["sharpe"] for s in with_trades])
    wr = np.array([s["win_rate"] for s in with_trades])
    md = np.array([s["mdd"] for s in with_trades])
    nt = np.array([s["n_trades"] for s in with_trades])
    fit = global_fitness_from_stats(stats)
    d = dict(
        fitness=fit, mean_return=r.mean(), mean_excess=ex.mean(),
        pct_excess_pos=(ex>0).mean(), mean_sharpe=sh.mean(),
        mean_win_rate=wr.mean(), median_win_rate=np.median(wr),
        mean_mdd=md.mean(), median_mdd=np.median(md),
        mean_trades=nt.mean(), tickers_with_trades=len(with_trades),
    )
    print(f"\n  {label}:")
    for k, v in d.items():
        print(f"    {k:<24s} {v:>12.4f}")
    return d

print("\n" + "=" * 80)
print("  RESULTS: V3 (old) vs V4 (new)")
print("=" * 80)

d3 = summarize(stats_v3, "V3 (v4 genes OFF)")
d4 = summarize(stats_v4, "V4 (v4 genes ON)")

if d3 and d4:
    print(f"\n  {'Metric':<24s} {'V3':>10s} {'V4':>10s} {'Delta':>10s} {'Win':>6s}")
    print("  " + "-" * 65)
    for k in d3:
        v3, v4 = d3[k], d4[k]
        delta = v4 - v3
        if "mdd" in k:
            w = "V4" if v4 > v3 else ("V3" if v3 > v4 else "-")
        elif "trades" in k or "tickers" in k:
            w = "-"
        else:
            w = "V4" if v4 > v3 else ("V3" if v3 > v4 else "-")
        print(f"  {k:<24s} {v3:>10.4f} {v4:>10.4f} {delta:>+10.4f} {w:>6s}")

gp = decode_global_params(sanitize_global_genome(best_v4))
print(f"\n  V4 GA-Optimized Gene Values:")
print(f"    volume_confirm_mode:   {gp.volume_confirm_mode}")
print(f"    momentum_confirm_days: {gp.momentum_confirm_days}")
print(f"    entry_score_threshold: {gp.entry_score_threshold:.4f}")
print(f"    regime_threshold:      {gp.regime_threshold:.4f}")

if d3 and d4:
    print("\n  KEY TAKEAWAYS:")
    for k in ["fitness", "mean_win_rate", "mean_sharpe", "mean_mdd", "mean_excess"]:
        v3, v4 = d3.get(k, 0), d4.get(k, 0)
        if "mdd" in k:
            if v4 > v3: print(f"    + {k}: improved from {v3:.4f} to {v4:.4f}")
        else:
            if v4 > v3: print(f"    + {k}: improved from {v3:.4f} to {v4:.4f}")
            elif v3 > v4: print(f"    - {k}: decreased from {v3:.4f} to {v4:.4f}")

print("\n" + "=" * 80)
