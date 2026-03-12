# -*- coding: utf-8 -*-
"""
GA + Walk-Forward ML (OOS) + Intraday (OHLC) Backtest  â€” v2 (fixed outputs)
============================================================================

Fixes vs previous version:
1) APPLY "best buy" / "best sell" now uses **next-day OHLC** (i+1) as intended.
   - Signal is computed at end of day i (close), and the suggested order is for day i+1.
   - Columns:
        signal_eod          : signal decided at close(i)
        next_day_filled     : whether the limit would be filled on day i+1
        best_buy_value      : filled price (NaN if not filled)
        best_sell_value     : filled price (NaN if not filled)
        entry_ref_price     : best_* if filled else open(i+1) (optional reference)
        stop/take levels are computed from entry_ref_price.

2) "Score & signal in Excel not working" (all holds / all best_buy == close):
   - We compute suggested entry for next day and keep fallback values for rows without next-day data.

3) Too many tickers with 0 trades (TEret=0):
   - GA fitness penalizes strategies with very low trades/exposure (prevents "do nothing" winning).
   - GA search ranges for enter_abs are made more permissive (lower thresholds).

4) Cleaner & richer metrics:
   - WF: AUC mean/std, ACC mean, PR-AUC mean, logloss, brier.
   - Trading: return/mdd/sharpe/trades/exposure/win_rate/avg_trade for GA and TEST.
   - Period print per ticker: train/test date ranges.

Expected columns in HISTORY_CSV:
- Date, ticker, open, high, low, close
- plus numeric feature columns.

Outputs:
- CSV: apply_last_{APPLY_DAYS}d__H{FWD_H}.csv
- XLSX: summary_latest + apply_last_{APPLY_DAYS}d

NOTE
- This is research/backtest code. Not financial advice.

Author: ChatGPT (generated)
"""

import sys
import types

# Mock google.colab to avoid errors in Codespaces
if 'google.colab' not in sys.modules:
    mock_colab = types.ModuleType('google.colab')
    mock_drive = types.ModuleType('google.colab.drive')
    mock_drive.mount = lambda *args, **kwargs: print("[INFO] google.colab drive.mount ignorado no Codespaces")
    mock_colab.drive = mock_drive
    sys.modules['google.colab'] = mock_colab
    sys.modules['google.colab.drive'] = mock_drive

from google.colab import drive
drive.mount("/content/drive")

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import os
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
from scipy.stats import spearmanr
from numba import njit


# ==============================================================================
# 0) CONFIG
# ==============================================================================
HISTORY_CSV_PATH = "./history_consolidated.parquet"
OUTPUT_DIR       = "./"

DATE_COL   = "Date"
TICKER_COL = "ticker"

OPEN_COL  = "open"
HIGH_COL  = "high"
LOW_COL   = "low"
CLOSE_COL = "close"

ONLY_SA    = True
LONG_ONLY  = True

APPLY_DAYS = 5
FWD_H = 5
TARGET_RET_THRESHOLD = 0.015  # classify only meaningful +1.5% moves
TARGET_ATR_MULT = 0.75        # optional ATR-scaled threshold (effective threshold = max(ret, ATR*mult))

# Regime filter (MA200)
MA_WINDOW = 200
USE_MA_SLOPE_FILTER = False
MA_SLOPE_LOOKBACK = 10  # shorter lookback for more responsive regime detection
MA_SLOPE_EPS = 0.0
REQUIRE_MA_FOR_ENTRY = False
REQUIRE_MA_FOR_SELL_MA = False

# Z-score (normalizes score_full)
EV_CLIP = 5.0
EV_EMA_SPAN = 3

# Probability-to-direction scaling (higher = sharper separation near 0.5)
PROB_DIRECTION_SCALE = 4.0

# ATR
ATR_WINDOW = 14
ATR_MIN_PERIODS = 14
ATR_EPS = 1e-12

# GA ranges
ATR_MULT_RANGE = (1.5, 3.5)
RR_MULT_RANGE  = (1.5, 4.0)  # wider RR to pursue larger trend-following payoffs

# Friction
COST_BPS     = 12.0
SLIPPAGE_BPS = 12.0
COST_PER_TRADE_PCT = 0.0020  # fixed round-trip friction per completed trade

MIN_PRICE     = 0.01
CAP_DAILY_RET = 0.30
CAP_TRADE_RET = 3.00

ONE_YEAR_DAYS = 252
GA_WF_TRAIN_YEARS = 3
GA_WF_TEST_DAYS = 126
GA_WF_STEP_DAYS = 126
LAMBDA_MDD_1Y = 0.70
MAX_EXPOSURE_1Y = 0.70

# GA hyperparams (reduce for speed)
RANDOM_SEED = 42
GA_POP_SIZE = 100
GA_NGEN     = 40
GA_CX_PB    = 0.70
GA_MUT_PB   = 0.40
GA_TOURN    = 3
EARLY_STOP  = 10
GA_WF_SPLITS = 3
GA_MUT_SIGMA_START = 0.14
GA_MUT_SIGMA_END = 0.04
GA_GENE_MUT_PB_START = 0.25
GA_GENE_MUT_PB_END = 0.08
GA_HOF_SIZE = 5
GA_MIN_TRADES_PER_FOLD = 15
GA_OVERTRADING_TRADES_PER_FOLD = 150
GA_FOLD_STABILITY_PENALTY = 2.8
GA_MIN_TRADES_FOR_SIGNIFICANCE = 15

# ML (walk-forward)
WF_SPLITS = 5
ML_RECENCY_HALF_LIFE = 252
ML_RET_CAP = 0.30          # cap fwd return before ATR-normalization
ML_MIN_TRAIN = 260
SCORE_LOOKBACK = 504

# Calibrate score_ev using realized forward-return feedback (per ticker)
USE_RETURN_FEEDBACK_CALIBRATION = True
RETURN_FEEDBACK_BLEND = 0.70
RETURN_FEEDBACK_BLEND_MAX = 0.10
RETURN_FEEDBACK_MIN_ROWS = 300
RETURN_FEEDBACK_BINS = 10
RETURN_FEEDBACK_MIN_IC = 0.04
RETURN_FEEDBACK_TARGET_IC = 0.08
RETURN_FEEDBACK_BIN_SHRINK = 80.0

# Feature selection
MIN_ROWS_TICKER = 350  # enough for MA200 min_periods + some buffer
MIN_FEAT_NONNA_FRAC = 0.60
MIN_FEAT_STD = 1e-12
MIN_VALID_SAMPLES_FOR_CORRELATION = 30
MAX_FEATURES = 40  # use best 40 features (per-ticker Spearman ranked); more signal with less noise dilution
FEAT_DEDUP_CORR_THRESH = 0.80  # max pairwise |r| allowed between selected features (greedy dedup)

# Intraday entry (limit) based on signal strength
ENTRY_DISCOUNT_RANGE = (0.0, 0.4)
FAST_PERIOD_RANGE = (10, 30)
SLOW_PERIOD_RANGE = (40, 100)
SCORE_CROSS_MIN_ABS = 0.05
ENTRY_SCORE_TRIGGER_ABS = 0.005
ML_STRONG_SCORE_ABS = 0.08

# Avoid "do nothing" strategies
GA_MIN_TRADES = 25
GA_TARGET_TRADES = 40
GA_MIN_EXPOSURE = 0.05
GA_TRADE_BONUS_PER = 0.015
MAX_TRADES_PER_YEAR = 60
OVERTRADING_PENALTY_PER_TRADE = 0.03
GA_MIN_WF_AUC_TO_RUN = 0.53
GA_MIN_WF_AP_TO_RUN = 0.52
GA_MIN_WF_QUALITY_TO_RUN = 0.30
GA_INTERNAL_EXTRA_TRADE_COST_BPS = 30.0

# Prints
PRINT_EVERY = 1
PRINT_FOLD_DETAILS = False
PRINT_TOP_N = 12
PRINT_TAIL_DETAILS = False
PRINT_ML_METRICS = False
TIME_STOP_BARS = 10
ENTRY_VOL_LOOKBACK = 60
ENTRY_ATR_MAX_MULT = 1.8
TEST_ONLY_PREFIX_C_TICKERS = False
TEST_TICKER_PREFIX = "C"
EVAL_ONLY_TICKER = ""
GA_VERBOSE_PER_GENERATION = False
GA_VERBOSE_MAX_TRADES_PER_GEN = 120
GA_PLOT_BEST_GENERATION_TRADES = False
USE_MONTE_CARLO_REALITY_CHECK = True
MC_SHUFFLES_MAX = 16
MC_SHUFFLES_BLOCK = 4
MC_PVALUE_MAX = 0.20
GA_SELECT_ROBUST_FROM_HOF = True
GA_RUN_MC_EVERY_WINDOW = False
# On Windows, ProcessPoolExecutor with large payload initargs causes deadlocks.
# Force single-threaded evaluation on Windows; use multiprocessing on Linux/Mac.
import platform as _platform
GA_EVAL_WORKERS = 1 if _platform.system() == "Windows" else max(1, min(8, (os.cpu_count() or 2)))
GA_TWO_STAGE = True
RUN_MODE = "load"  # "train" always retrains (seeding from checkpoint), "load" skips GA entirely
SLOW_STEP_PRINT_SEC = 2.0  # print only timings above this per-step threshold
GA_STAGE1_TOP_N = 6
GA_STAGE2_PADDING_RATIO = 0.60
GA_STAGE2_MIN_SPAN_RATIO = 0.18

# ── Fast / Full mode ────────────────────────────────────────────────────────
# Set FAST_MODE = True for quick validation; False for full production run.
FAST_MODE = False   # ← ALL FAST_MODE OBJECTIVES PASSED → now running FULL production mode

if FAST_MODE:
    # Small population, moderate gens, 4 uniformly-sampled windows from full history
    # (uniform sampling covers crises: 2008, 2015, 2020, plus recent)
    # Target: ~20-30 min total on single-thread Windows
    GA_STAGE1_POP_SIZE = 20
    GA_STAGE1_NGEN     = 20
    GA_STAGE2_POP_SIZE = 30
    GA_STAGE2_NGEN     = 15
    GA_MAX_WINDOWS     = 4   # 4 uniformly-sampled windows: covers crisis periods
    GA_MAX_WINDOWS_STAGE2 = 5
else:
    # FULL production mode: all 78 tickers, larger population, more generations
    # Calibrated for single-thread Windows: gen 1 ≈ 15min, gen 2+ ≈ 5-6min with 8 windows
    # 20 gens × 5.5min ≈ 2.1 hours total (feasible during day)
    GA_STAGE1_POP_SIZE = 30
    GA_STAGE1_NGEN     = 20   # 20 gens ≈ 2.1 hours with 78 tickers × 8 windows
    GA_STAGE2_POP_SIZE = 60
    GA_STAGE2_NGEN     = 15
    GA_MAX_WINDOWS     = 8    # 8 uniformly-sampled windows — better crisis coverage than 6
    GA_MAX_WINDOWS_STAGE2 = 8

# ==============================================================================
# 1) DEAP SETUP
# ==============================================================================
def setup_global_deap():
    if not hasattr(creator, "FitnessMax_PT"):
        creator.create("FitnessMax_PT", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual_PT"):
        creator.create("Individual_PT", list, fitness=creator.FitnessMax_PT)

setup_global_deap()

# ==============================================================================
# 2) DATA CLASSES
# ==============================================================================
@dataclass
class Params:
    fast_period: float
    slow_period: float
    atr_mult: float
    rr_mult: float
    entry_discount: float
    rule_cols: Tuple[int, ...] = ()
    rule_ops: Tuple[int, ...] = ()
    rule_thrs: Tuple[float, ...] = ()

    @property
    def has_rule_genome(self) -> bool:
        return len(self.rule_cols) > 0 and len(self.rule_cols) == len(self.rule_ops) == len(self.rule_thrs)

def sanitize_params(p: Params) -> Params:
    fast_period = int(np.clip(round(float(p.fast_period)), FAST_PERIOD_RANGE[0], FAST_PERIOD_RANGE[1]))
    slow_period = int(np.clip(round(float(p.slow_period)), SLOW_PERIOD_RANGE[0], SLOW_PERIOD_RANGE[1]))
    if slow_period <= fast_period:
        slow_period = min(SLOW_PERIOD_RANGE[1], fast_period + 1)
        if slow_period <= fast_period:
            fast_period = max(FAST_PERIOD_RANGE[0], slow_period - 1)
    atr_mult = round(float(np.clip(p.atr_mult, ATR_MULT_RANGE[0], ATR_MULT_RANGE[1])) / 0.25) * 0.25
    rr_mult  = round(float(np.clip(p.rr_mult,  RR_MULT_RANGE[0],  RR_MULT_RANGE[1])) / 0.25) * 0.25
    entry_discount = round(float(np.clip(p.entry_discount, ENTRY_DISCOUNT_RANGE[0], ENTRY_DISCOUNT_RANGE[1])) / 0.1) * 0.1
    return Params(float(fast_period), float(slow_period), atr_mult, rr_mult, entry_discount)


# ==============================================================================
# 2.1) GLOBAL GA (20 parÃ¢metros)
# ==============================================================================

GLOBAL_PARAM_SPECS = [
    ("vote_threshold_long", 0.15, 0.55, 0.05, False),  # tighter range (was 0.20-0.60)
    ("vote_threshold_short", 0.10, 0.55, 0.05, False),
    ("z_threshold", 0.15, 0.80, 0.05, False),
    ("signal_ema_span", 2.0, 12.0, 1.0, True),
    ("entry_confirmation_days", 1.0, 3.0, 1.0, True),
    ("score_percentile_trigger", 0.35, 0.80, 0.05, False),
    ("stop_atr_mult", 1.0, 4.0, 0.25, False),
    ("stop_tighten_after_bars", 3.0, 15.0, 1.0, True),
    ("stop_tighten_factor", 0.40, 0.85, 0.05, False),
    ("max_loss_per_trade_pct", 0.02, 0.15, 0.01, False),
    ("reward_risk_ratio", 1.0, 5.0, 0.25, False),
    ("partial_take_pct", 0.0, 0.60, 0.10, False),
    ("partial_take_level", 0.5, 1.5, 0.25, False),
    ("time_stop_bars", 5.0, 25.0, 1.0, True),
    ("entry_discount_atr_frac", 0.0, 0.5, 0.05, False),
    ("volatility_filter_percentile", 0.0, 0.40, 0.05, False),
    ("score_strength_scaling", 0.0, 1.0, 0.1, False),
    ("ma_filter_period", 100.0, 300.0, 50.0, True),
    ("ma_filter_mode", 0.0, 2.0, 1.0, True),
    ("consecutive_loss_cooldown", 5.0, 20.0, 1.0, True),  # lowered min from 8→5 for more flexibility
    ("equity_drawdown_stop_pct", 0.08, 0.30, 0.02, False),  # tighter range (was 0.10-0.40), step 0.02
    # ── NEW genes for v3 ──
    ("vol_regime_mode", 0.0, 2.0, 1.0, True),  # 0=off, 1=conservative(widen stops in high vol), 2=aggressive(skip high vol)
    ("partial_take_pct_2", 0.0, 0.40, 0.10, False),  # 2nd partial take (0=disabled)
    ("partial_take_level_2", 1.0, 3.0, 0.25, False),  # 2nd partial at higher RR multiple
    ("min_signal_strength", 0.0, 0.40, 0.05, False),  # minimum abs(score_ev)/score95 to enter
]


@dataclass
class GlobalParams:
    vote_threshold_long: float
    vote_threshold_short: float
    z_threshold: float
    signal_ema_span: int
    entry_confirmation_days: int
    score_percentile_trigger: float
    stop_atr_mult: float
    stop_tighten_after_bars: int
    stop_tighten_factor: float
    max_loss_per_trade_pct: float
    reward_risk_ratio: float
    partial_take_pct: float
    partial_take_level: float
    time_stop_bars: int
    entry_discount_atr_frac: float
    volatility_filter_percentile: float
    score_strength_scaling: float
    ma_filter_period: int
    ma_filter_mode: int
    consecutive_loss_cooldown: int
    equity_drawdown_stop_pct: float
    # ── NEW v3 genes ──
    vol_regime_mode: int  # 0=off, 1=conservative(widen stops high vol), 2=skip high vol
    partial_take_pct_2: float  # 2nd partial take percentage
    partial_take_level_2: float  # 2nd partial take at this RR multiple
    min_signal_strength: float  # min |score_ev|/score95 to enter trade


def sanitize_global_genome(genome: List[float]) -> List[float]:
    out = []
    for g, (_, lo, hi, step, is_int) in zip(genome, GLOBAL_PARAM_SPECS):
        v = float(np.clip(float(g), lo, hi))
        v = round(v / step) * step
        if is_int:
            v = int(round(v))
        out.append(v)
    return out


def decode_global_params(genome: List[float]) -> GlobalParams:
    gg = sanitize_global_genome(genome)
    return GlobalParams(**{spec[0]: gg[i] for i, spec in enumerate(GLOBAL_PARAM_SPECS)})


def rolling_mean_np(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    window = max(1, int(window))
    min_periods = max(1, int(min_periods))
    csum = np.zeros(n + 1, dtype=np.float64)
    ccount = np.zeros(n + 1, dtype=np.int32)
    finite = np.isfinite(arr)
    csum[1:] = np.cumsum(np.where(finite, arr, 0.0))
    ccount[1:] = np.cumsum(finite.astype(np.int32))
    for i in range(n):
        j0 = max(0, i - window + 1)
        cnt = ccount[i + 1] - ccount[j0]
        if cnt >= min_periods:
            out[i] = (csum[i + 1] - csum[j0]) / float(cnt)
    return out


def ewm_mean_np(x: np.ndarray, span: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    span = max(1, int(span))
    alpha = 2.0 / (span + 1.0)
    prev = np.nan
    for i in range(n):
        xi = arr[i]
        if not np.isfinite(xi):
            out[i] = prev
            continue
        if np.isfinite(prev):
            prev = alpha * xi + (1.0 - alpha) * prev
        else:
            prev = xi
        out[i] = prev
    return out


def rolling_percentile_rank(arr: np.ndarray, window: int = 252) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    valid = np.isfinite(x)
    n = len(x)
    if n == 0:
        return out

    from bisect import bisect_right, insort

    sorted_vals = []
    for i in range(n):
        xi = x[i]
        if np.isfinite(xi):
            insort(sorted_vals, float(xi))
        j_rm = i - window
        if j_rm >= 0 and valid[j_rm]:
            xrm = float(x[j_rm])
            k = bisect_right(sorted_vals, xrm) - 1
            if k >= 0:
                sorted_vals.pop(k)
        if len(sorted_vals) >= 20 and np.isfinite(xi):
            out[i] = bisect_right(sorted_vals, float(xi)) / float(len(sorted_vals))
    return out


def rolling_quantile_trigger(arr: np.ndarray, q: float, window: int = SCORE_LOOKBACK) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    valid = np.isfinite(x)
    n = len(x)
    if n == 0:
        return out

    from bisect import bisect_left, insort

    sorted_vals = []
    for i in range(n):
        xi = x[i]
        if np.isfinite(xi):
            insort(sorted_vals, float(xi))
        j_rm = i - window
        if j_rm >= 0 and valid[j_rm]:
            xrm = float(x[j_rm])
            k = bisect_left(sorted_vals, xrm)
            if k < len(sorted_vals):
                sorted_vals.pop(k)
        if len(sorted_vals) >= 20 and np.isfinite(xi):
            idx = int(np.clip(math.ceil(q * len(sorted_vals)) - 1, 0, len(sorted_vals) - 1))
            out[i] = sorted_vals[idx]
    return out


def precompute_global_payloads(
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    ticker_payloads: Dict[str, Dict[str, Any]],
) -> List[Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]]:
    """Build GA evaluation payloads from per-ticker payloads (consistent with Phase 3).

    Uses the same score_matrix (per-ticker Spearman-ranked & direction-directed z-scores)
    that Phase 3 applies, eliminating any train/apply feature mismatch.
    """
    payloads_by_window = []

    for tr_start, tr_end, te_start, te_end in windows:
        tr_start_np = np.datetime64(pd.Timestamp(tr_start))
        tr_end_np   = np.datetime64(pd.Timestamp(tr_end))
        te_start_np = np.datetime64(pd.Timestamp(te_start))
        te_end_np   = np.datetime64(pd.Timestamp(te_end))

        def _slice_payload(tkr_payload, start_np, end_np):
            dates = tkr_payload["dates"]
            mask = (dates >= start_np) & (dates <= end_np)
            if mask.sum() < 30:
                return None
            sliced = {}
            for k, v in tkr_payload.items():
                if isinstance(v, np.ndarray) and len(v) == len(dates):
                    sliced[k] = v[mask]
                else:
                    sliced[k] = v  # scalars, lists (feat_cols)
            # Pre-compute vol_rank on the slice
            c_sl = sliced["close"]
            atr_sl = sliced["atr"]
            sliced["vol_rank"] = rolling_percentile_rank(atr_sl / np.maximum(c_sl, ATR_EPS), 252)
            return sliced

        tr_payloads, te_payloads = {}, {}
        for tkr, payload in ticker_payloads.items():
            sl_tr = _slice_payload(payload, tr_start_np, tr_end_np)
            sl_te = _slice_payload(payload, te_start_np, te_end_np)
            if sl_tr is not None:
                tr_payloads[tkr] = sl_tr
            if sl_te is not None:
                te_payloads[tkr] = sl_te

        payloads_by_window.append((tr_payloads, te_payloads))

    return payloads_by_window

def backtest_stats_global_intraday(o, h, l, c, score_matrix, atr, gp: GlobalParams, precomputed: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    atr = np.asarray(atr, dtype=np.float64)
    x = np.asarray(score_matrix, dtype=np.float64)
    n = len(c)
    if n < 5 or x.ndim != 2 or x.shape[0] != n:
        return {"total_return": 0.0, "mdd": 0.0, "sharpe": 0.0, "n_trades": 0.0, "win_rate": 0.0, "avg_trade": 0.0}

    feat_n = max(1, x.shape[1])
    ma = rolling_mean_np(c, int(gp.ma_filter_period), max(20, int(gp.ma_filter_period // 2)))
    if precomputed is not None and ("vol_rank" in precomputed):
        vol_rank = np.asarray(precomputed["vol_rank"], dtype=np.float64)
    else:
        vol_rank = rolling_percentile_rank(atr / np.maximum(c, ATR_EPS), 252)

    votes_long = (x > gp.z_threshold).sum(axis=1) / feat_n
    votes_short = (x < -gp.z_threshold).sum(axis=1) / feat_n
    score_raw = votes_long - votes_short
    score_ev = ewm_mean_np(score_raw, int(gp.signal_ema_span))
    score_pctl = rolling_quantile_trigger(score_ev, float(gp.score_percentile_trigger), max(63, SCORE_LOOKBACK))

    score95 = np.nanpercentile(np.abs(score_ev), 95) if np.isfinite(np.nanmax(np.abs(score_ev))) else 1.0
    score95 = max(score95, ATR_EPS)


    equity = 1.0
    peak = 1.0
    mdd = 0.0
    trade_rets = []
    pos = 0
    entry_px = np.nan
    bars = 0
    partial_taken = False
    partial_taken_2 = False
    consec_long = 0
    consec_short = 0
    consec_stops = 0
    cooldown = 0
    exposure_acc = 0.0

    for i in range(1, n):
        if cooldown > 0:
            cooldown -= 1

        if abs(pos) > 0:
            exposure_acc += float(min(1.0, abs(pos)))
        if pos != 0:
            bars += 1
            # ── Dynamic stop tightening ──
            tighten = 1.0
            if bars >= gp.stop_tighten_after_bars:
                tighten = gp.stop_tighten_factor
            elif bars >= 3:
                # Early P&L: tighten if underwater after 3+ bars
                early_pnl = (c[i - 1] / max(entry_px, ATR_EPS) - 1.0) * np.sign(pos)
                if early_pnl < -0.002:
                    tighten = max(gp.stop_tighten_factor, 0.85)
            # Vol-regime stop adjustment (mode 1: widen stops in high vol)
            vol_stop_adj = 1.0
            if gp.vol_regime_mode == 1 and np.isfinite(vol_rank[i - 1]):
                if vol_rank[i - 1] > 0.75:
                    vol_stop_adj = 1.15
                elif vol_rank[i - 1] < 0.25:
                    vol_stop_adj = 0.90
            stop_mult = gp.stop_atr_mult * tighten * vol_stop_adj
            stop_abs = max(ATR_EPS, stop_mult * max(atr[i], ATR_EPS))
            take_abs = gp.reward_risk_ratio * stop_abs
            hard_loss = abs(o[i] / max(entry_px, ATR_EPS) - 1.0)

            if hard_loss > gp.max_loss_per_trade_pct:
                ret = (o[i] / entry_px - 1.0) * pos - 0.0005
                equity *= (1.0 + ret)
                trade_rets.append(ret)
                consec_stops += 1 if ret < 0 else 0
                if ret > 0:
                    consec_stops = 0
                pos = 0
                continue

            fav = ((h[i] - entry_px) if pos > 0 else (entry_px - l[i]))
            adv = ((entry_px - l[i]) if pos > 0 else (h[i] - entry_px))

            if gp.partial_take_pct > 0 and (not partial_taken) and fav >= gp.partial_take_level * stop_abs:
                part_ret = gp.partial_take_pct * gp.partial_take_level * stop_abs / max(entry_px, ATR_EPS)
                equity *= (1.0 + part_ret - 0.0003)
                pos = pos * (1.0 - gp.partial_take_pct)
                partial_taken = True

            # 2nd partial take at higher level
            if gp.partial_take_pct_2 > 0 and partial_taken and (not partial_taken_2) and fav >= gp.partial_take_level_2 * stop_abs:
                part_ret2 = gp.partial_take_pct_2 * gp.partial_take_level_2 * stop_abs / max(entry_px, ATR_EPS)
                equity *= (1.0 + part_ret2 - 0.0003)
                pos = pos * (1.0 - gp.partial_take_pct_2)
                partial_taken_2 = True

            stop_hit = adv >= stop_abs
            take_hit = fav >= take_abs
            time_stop = (bars >= gp.time_stop_bars) and (fav < 0.5 * stop_abs)

            if stop_hit or take_hit or time_stop:
                exit_px = c[i]
                if stop_hit:
                    ideal_exit = entry_px - np.sign(pos) * stop_abs
                    if pos > 0:
                        exit_px = min(ideal_exit, o[i]) if o[i] < ideal_exit else ideal_exit
                    else:
                        exit_px = max(ideal_exit, o[i]) if o[i] > ideal_exit else ideal_exit
                elif take_hit:
                    ideal_exit = entry_px + np.sign(pos) * take_abs
                    if pos > 0:
                        exit_px = max(ideal_exit, o[i]) if o[i] > ideal_exit else ideal_exit
                    else:
                        exit_px = min(ideal_exit, o[i]) if o[i] < ideal_exit else ideal_exit
                
                cost = 0.0003 if (partial_taken or partial_taken_2) else 0.0005
                ret = (exit_px / entry_px - 1.0) * pos - cost
                equity *= (1.0 + ret)
                trade_rets.append(ret)
                if stop_hit and ret < 0:
                    consec_stops += 1
                elif ret > 0:
                    consec_stops = 0
                pos = 0
                bars = 0
                partial_taken = False
                partial_taken_2 = False
                if gp.consecutive_loss_cooldown > 0 and consec_stops >= 2:
                    cooldown = gp.consecutive_loss_cooldown
                continue

        if pos == 0 and cooldown == 0 and np.isfinite(score_ev[i - 1]) and np.isfinite(score_pctl[i - 1]):
            if gp.volatility_filter_percentile > 0 and np.isfinite(vol_rank[i - 1]) and vol_rank[i - 1] < gp.volatility_filter_percentile:
                continue
            # Vol regime mode 2: skip entries in very high volatility periods
            if gp.vol_regime_mode == 2 and np.isfinite(vol_rank[i - 1]) and vol_rank[i - 1] > 0.85:
                continue
            # Liquidity filter (ATR proxy)
            if atr[i-1] < 0.01:
                continue
            # Min signal strength filter
            if gp.min_signal_strength > 0:
                if abs(score_ev[i - 1]) / score95 < gp.min_signal_strength:
                    continue

            vl = votes_long[i - 1]
            vs = votes_short[i - 1]
            long_raw = (vl >= gp.vote_threshold_long) and (score_ev[i - 1] >= score_pctl[i - 1])
            short_raw = (vs >= gp.vote_threshold_short) and (-score_ev[i - 1] >= score_pctl[i - 1])

            if long_raw:
                consec_long += 1
            else:
                consec_long = 0
            if short_raw:
                consec_short += 1
            else:
                consec_short = 0

            long_ok = consec_long >= gp.entry_confirmation_days
            short_ok = consec_short >= gp.entry_confirmation_days

            if gp.ma_filter_mode == 1 and np.isfinite(ma[i - 1]):
                if c[i - 1] < ma[i - 1]:
                    long_ok = long_ok and (score_ev[i - 1] * 0.5 >= score_pctl[i - 1])
                if c[i - 1] > ma[i - 1]:
                    short_ok = short_ok and (-score_ev[i - 1] * 0.5 >= score_pctl[i - 1])
            elif gp.ma_filter_mode == 2 and np.isfinite(ma[i - 1]):
                if c[i - 1] < ma[i - 1]:
                    long_ok = False
                if c[i - 1] > ma[i - 1]:
                    short_ok = False

            side = 1 if long_ok else (-1 if (short_ok and not LONG_ONLY) else 0)
            if side != 0 and np.isfinite(o[i]) and np.isfinite(atr[i]):
                strength = float(np.clip(abs(score_ev[i - 1]) / score95, 0.0, 1.0))
                discount = gp.entry_discount_atr_frac * (1.0 - gp.score_strength_scaling * strength)
                limit_px = o[i] - side * discount * atr[i]
                fill = (l[i] <= limit_px <= h[i])
                if fill and limit_px > 0:
                    pos = side
                    entry_px = float(limit_px)
                    bars = 0
                    partial_taken = False
                    partial_taken_2 = False
                elif np.isfinite(o[i]) and o[i] > 0:
                    pos = side
                    entry_px = float(o[i])
                    bars = 0
                    partial_taken = False
                    partial_taken_2 = False

        peak = max(peak, equity)
        mdd = min(mdd, (equity / max(peak, ATR_EPS)) - 1.0)
        # Equity drawdown circuit-breaker (progressive re-entry: 20 bars, was 30)
        if gp.equity_drawdown_stop_pct > 0 and (equity / max(peak, ATR_EPS)) - 1.0 < -gp.equity_drawdown_stop_pct:
            cooldown = max(cooldown, 20)  # ~1 month pause after DD exceeds threshold

    exposure = float(exposure_acc / max(1, n - 1))
    if len(trade_rets) == 0:
        return {"total_return": float(equity - 1.0), "mdd": float(mdd), "sharpe": 0.0, "n_trades": 0.0, "win_rate": 0.0, "avg_trade": 0.0, "exposure": exposure}
    tr = np.asarray(trade_rets, dtype=np.float64)
    # Annualized Sharpe: use trades_per_year, not raw trade count
    n_years = max(n / 252.0, 0.1)
    trades_per_year = len(tr) / n_years
    ann_factor = np.sqrt(min(trades_per_year, 252))
    ann_sharpe = float(np.mean(tr) / (np.std(tr) + 1e-12) * ann_factor)
    return {
        "total_return": float(equity - 1.0),
        "mdd": float(mdd),
        "sharpe": ann_sharpe,
        "n_trades": float(len(tr)),
        "win_rate": float((tr > 0).mean()),
        "avg_trade": float(np.mean(tr)),
        "trade_std": float(np.std(tr)),
        "exposure": exposure,
    }


def global_fitness_from_stats(per_ticker_stats: List[Dict[str, float]]) -> float:
    """
    Multi-objective fitness optimised for:
      - High win rate (prefer few quality trades over many losing ones)
      - Low max drawdown (MDD must stay small)
      - Beat buy-and-hold on excess return
      - Statistical volume (enough trades for significance)
    """
    if len(per_ticker_stats) == 0:
        return -1e9
    sharpe    = np.array([s.get("sharpe", 0.0)       for s in per_ticker_stats], dtype=np.float64)
    ret       = np.array([s.get("total_return", 0.0)  for s in per_ticker_stats], dtype=np.float64)
    excess    = np.array([s.get("excess_return", s.get("total_return", 0.0)) for s in per_ticker_stats], dtype=np.float64)
    mdd_vals  = np.array([s.get("mdd", 0.0)           for s in per_ticker_stats], dtype=np.float64)
    ntr       = np.array([s.get("n_trades", 0.0)      for s in per_ticker_stats], dtype=np.float64)
    exposure  = np.array([s.get("exposure", 0.0)      for s in per_ticker_stats], dtype=np.float64)
    win_rates = np.array([s.get("win_rate", 0.0)      for s in per_ticker_stats], dtype=np.float64)

    med_sharpe        = float(np.median(sharpe))
    mean_sharpe       = float(np.mean(sharpe))
    mean_ret          = float(np.mean(ret))
    med_ret           = float(np.median(ret))
    mean_excess       = float(np.mean(excess))
    med_excess        = float(np.median(excess))
    pct_positive      = float((ret > 0).mean())
    pct_excess_pos    = float((excess > 0).mean())
    med_trades        = float(np.median(ntr))
    mean_trades       = float(np.mean(ntr))
    mean_exposure     = float(np.mean(exposure))
    mean_win_rate     = float(np.mean(win_rates))
    med_win_rate      = float(np.median(win_rates))
    mean_mdd          = float(np.mean(mdd_vals))   # negative
    median_mdd        = float(np.median(mdd_vals)) # negative

    # ── Activity bonus/penalty ────────────────────────────────────────────
    # Target: 15-80 trades per year per ticker. Penalise extremes.
    # Tighter target range (was 100) to reduce churn and cost drag.
    trade_bonus = 0.0
    if med_trades < 8:
        trade_bonus -= (8 - med_trades) * 0.10   # hard penalty for near-zero trades
    elif med_trades < 15:
        trade_bonus -= (15 - med_trades) * 0.04
    if med_trades > 80:
        trade_bonus -= (med_trades - 80) * 0.012  # tighter target (was 100 × 0.008)
    if med_trades > 200:
        trade_bonus -= (med_trades - 200) * 0.020  # heavy penalty above 200 (was 250)
    if med_trades > 400:
        trade_bonus -= (med_trades - 400) * 0.025  # catastrophic above 400/yr
    if mean_exposure < 0.40:
        trade_bonus -= (0.40 - mean_exposure) * 2.0

    # ── MDD penalty (rebalanced v3) ───────────────────────────────────────
    # Target: MDD ≤ 20%. Slightly softer slopes to allow higher-return strategies,
    # but tighter absolute target.
    mdd_penalty = 0.0
    abs_mdd = abs(median_mdd)
    if abs_mdd > 0.08:
        mdd_penalty += (abs_mdd - 0.08) * 5.0    # gentle above 8%
    if abs_mdd > 0.15:
        mdd_penalty += (abs_mdd - 0.15) * 12.0   # moderate above 15%
    if abs_mdd > 0.20:
        mdd_penalty += (abs_mdd - 0.20) * 40.0   # target cliff at 20% (was 22%)
    if abs_mdd > 0.25:
        mdd_penalty += (abs_mdd - 0.25) * 50.0
    if abs_mdd > 0.30:
        mdd_penalty += (abs_mdd - 0.30) * 45.0
    if abs_mdd > 0.40:
        mdd_penalty += (abs_mdd - 0.40) * 60.0

    # Tail-risk penalty: worst-quartile MDD
    mdd_p25 = float(np.percentile(mdd_vals, 25))
    abs_mdd_tail = abs(mdd_p25)
    if abs_mdd_tail > 0.25:
        mdd_penalty += (abs_mdd_tail - 0.25) * 10.0  # tighter tail-risk target (was 30%)
    if abs_mdd_tail > 0.35:
        mdd_penalty += (abs_mdd_tail - 0.35) * 20.0
    if abs_mdd_tail > 0.45:
        mdd_penalty += (abs_mdd_tail - 0.45) * 30.0

    # ── Under-performance penalty ─────────────────────────────────────────
    underperf_penalty = 0.0
    if mean_excess < 0:
        underperf_penalty += 10.0 * abs(mean_excess)
    if pct_excess_pos < 0.50:
        underperf_penalty += (0.50 - pct_excess_pos) * 2.5  # was 2.0
    if mean_ret < 0.015:
        underperf_penalty += (0.015 - mean_ret) * 25.0
    if mean_ret < 0.0:
        underperf_penalty += abs(mean_ret) * 30.0

    # ── Win-rate bonus (v3: more aggressive tiers) ────────────────────────
    win_rate_bonus = 0.0
    # Sub-50% hard penalty
    if mean_win_rate < 0.50:
        win_rate_bonus -= (0.50 - mean_win_rate) * 10.0  # was 8.0
    # Tiered bonuses starting at 52%
    if mean_win_rate > 0.52:
        win_rate_bonus += (mean_win_rate - 0.52) * 6.0   # was 5.0
    if mean_win_rate > 0.56:
        win_rate_bonus += (mean_win_rate - 0.56) * 5.0   # new tier
    if mean_win_rate > 0.60:
        win_rate_bonus += (mean_win_rate - 0.60) * 6.0   # big bonus above 60%
    # Median win rate
    if med_win_rate > 0.55:
        win_rate_bonus += (med_win_rate - 0.55) * 4.0    # was 3.0
    if med_win_rate > 0.62:
        win_rate_bonus += (med_win_rate - 0.62) * 5.0    # was 3.0
    if med_win_rate > 0.68:
        win_rate_bonus += (med_win_rate - 0.68) * 6.0    # premium tier

    # ── Consistency bonus: reward stable per-ticker performance ───────────
    consistency_bonus = 0.0
    if len(win_rates) > 5:
        wr_std = float(np.std(win_rates))
        if wr_std < 0.08:
            consistency_bonus += 0.3   # very consistent across tickers
        elif wr_std > 0.15:
            consistency_bonus -= (wr_std - 0.15) * 2.0   # penalise inconsistency
    ret_std = float(np.std(ret))
    if ret_std < 0.10:
        consistency_bonus += 0.2

    # ── Main fitness (v3) ─────────────────────────────────────────────────
    fitness = (
        1.8 * np.clip(mean_excess, -1.0, 5.0) +   # beat B&H: highest weight (was 1.6)
        1.3 * np.clip(med_excess,  -1.0, 5.0) +   # was 1.2
        1.4 * pct_excess_pos +                      # % tickers beating B&H (was 1.2)
        0.5 * np.clip(mean_ret,    -1.0, 5.0) +
        0.3 * np.clip(med_ret,     -1.0, 5.0) +
        0.5 * np.clip(mean_sharpe, -2.0, 3.0) +
        0.4 * np.clip(med_sharpe,  -2.0, 3.0) +
        0.3 * pct_positive +
        1.1 * mean_win_rate +                       # win rate main weight (was 0.9)
        0.8 * med_win_rate +                        # median win rate (was 0.6)
        win_rate_bonus +
        consistency_bonus +
        trade_bonus -
        mdd_penalty -
        underperf_penalty
    )
    return float(fitness)


def evaluate_global_walkforward(genome: List[float], payloads_by_window: List[Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]]) -> Tuple[float, float, float]:
    # Returns (fitness, sharpe_train, sharpe_val)
    train_fits = []
    val_fits = []
    # Returns (fitness, sharpe_train, sharpe_val)
    gp = decode_global_params(genome)
    
    for tr_payloads, te_payloads in payloads_by_window:
        if tr_payloads and te_payloads:
            # Evaluate on train
            tr_stats = []
            for _, payload in tr_payloads.items():
                st = backtest_stats_global_intraday(
                    payload["open"], payload["high"], payload["low"], payload["close"],
                    payload["score_matrix"], payload["atr"], gp, precomputed=payload,
                )
                st["buy_hold_return"] = buyhold_capped(payload["close"])
                st["excess_return"] = st["total_return"] - st["buy_hold_return"]
                tr_stats.append(st)
            
            # Evaluate on test (OOS)
            te_stats = []
            for _, payload in te_payloads.items():
                st = backtest_stats_global_intraday(
                    payload["open"], payload["high"], payload["low"], payload["close"],
                    payload["score_matrix"], payload["atr"], gp, precomputed=payload,
                )
                st["buy_hold_return"] = buyhold_capped(payload["close"])
                st["excess_return"] = st["total_return"] - st["buy_hold_return"]
                te_stats.append(st)
                
            # Train fitness
            tr_fit = global_fitness_from_stats(tr_stats)
            # Test fitness (what we optimize for)
            te_fit = global_fitness_from_stats(te_stats)
            
            train_fits.append(tr_fit)
            val_fits.append(te_fit)
            
    if not val_fits:
        return -1e9, 0.0, 0.0
        
    mean_tr = float(np.mean(train_fits))
    mean_te = float(np.mean(val_fits))
    
    # Combine train and test: primarily OOS, small penalty for overfitting
    overfit_penalty = max(0.0, mean_tr - mean_te) * 0.3
    fitness = mean_te - overfit_penalty
    
    return fitness, mean_tr, mean_te

# Global variable for multiprocessing to avoid pickling issues
_GLOBAL_PAYLOADS_BY_WINDOW = None
_GLOBAL_PAYLOAD_FILE = None   # path to temp pickle; workers load from here

def _init_worker_file(fpath):
    """Initializer for ProcessPoolExecutor workers on Windows (spawn mode).
    Loads the payloads from a temp pickle file instead of receiving via pipe
    (avoids deadlocks when initargs are large, ~100 MB, on Windows spawn).
    """
    import pickle as _pickle
    global _GLOBAL_PAYLOADS_BY_WINDOW
    with open(fpath, "rb") as _f:
        _GLOBAL_PAYLOADS_BY_WINDOW = _pickle.load(_f)

def _eval_ind_global(ind):
    global _GLOBAL_PAYLOADS_BY_WINDOW
    res = evaluate_global_walkforward(list(ind), _GLOBAL_PAYLOADS_BY_WINDOW)
    return res

def run_global_ga_20params(
    ticker_payloads: Dict[str, Dict[str, Any]],
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    pop_size: int = 100,
    ngen: int = 40,
    ga_max_windows: int = 4,
    seed_genome: Optional[List[float]] = None,
):
    """Run the GA using per-ticker payloads directly (consistent with Phase 3).

    Parameters
    ----------
    ticker_payloads : dict returned by Phase 1 (contains per-ticker score_matrix,
                      Spearman-selected & direction-directed z-scores).
    seed_genome     : previous best genome to warm-start the population; works even
                      if the feature set or model has changed (genome = trading params,
                      not feature weights).
    """
    _n_params = len(GLOBAL_PARAM_SPECS)
    _ind_cls_name = f"Individual_Global{_n_params}"
    if not hasattr(creator, _ind_cls_name):
        creator.create(_ind_cls_name, list, fitness=creator.FitnessMax_PT)
    Individual_Global = getattr(creator, _ind_cls_name)

    ga_t0 = time.perf_counter()
    # Select windows for GA evaluation.
    # BOTH FAST and FULL modes use uniform sampling to cover all historical periods
    # (including crises like 2008, 2015-16, 2020, not just the most recent bull market).
    # Using the last N windows only creates genomes that overfit to recent conditions
    # and fail to generalise across all 36 Phase-3 windows.
    if ga_max_windows is not None and int(ga_max_windows) > 0 and len(windows) > 0:
        nw = len(windows)
        k = min(int(ga_max_windows), nw)
        if nw > k:
            # Uniformly sample k indices from [0, nw) to cover full history in all modes
            _step = nw / k
            _indices = [int(round(i * _step)) for i in range(k)]
            _indices = [min(max(0, idx), nw - 1) for idx in _indices]
            windows_ga = [windows[i] for i in _indices]
        else:
            windows_ga = windows
    else:
        windows_ga = windows

    global _GLOBAL_PAYLOADS_BY_WINDOW
    # In FAST_MODE, subsample tickers for fitness evaluation to reduce eval time.
    # A fixed seed ensures consistent comparisons across generations.
    # Full ticker set is still used in Phase 3 (apply) regardless.
    FAST_EVAL_N_TICKERS = 20   # num tickers for GA fitness evaluation in fast mode
    if FAST_MODE and len(ticker_payloads) > FAST_EVAL_N_TICKERS:
        import random as _rng
        _rng_state = _rng.getstate()
        _rng.seed(42)
        _sampled_keys = _rng.sample(list(ticker_payloads.keys()), FAST_EVAL_N_TICKERS)
        _rng.setstate(_rng_state)
        eval_ticker_payloads = {k: ticker_payloads[k] for k in _sampled_keys}
        print(f"[GLOBAL_GA] FAST_MODE: sampled {FAST_EVAL_N_TICKERS}/{len(ticker_payloads)} tickers for GA eval")
    else:
        eval_ticker_payloads = ticker_payloads
    _GLOBAL_PAYLOADS_BY_WINDOW = precompute_global_payloads(windows_ga, eval_ticker_payloads)

    # Report median feature count across tickers
    feat_counts = [len(p.get("feat_cols", [])) for p in ticker_payloads.values()]
    med_feats = int(np.median(feat_counts)) if feat_counts else 0

    num_cpus_available = multiprocessing.cpu_count()
    num_cpus = int(max(1, min(GA_EVAL_WORKERS, num_cpus_available)))
    print(f"[GLOBAL_GA] config pop={int(pop_size)} ngen={int(ngen)} windows_total={len(windows)} windows_ga={len(windows_ga)} median_feats={med_feats}")
    print(f"[GLOBAL_GA] CPUs available={num_cpus_available} | CPUs used={num_cpus}")
    if seed_genome is not None:
        print(f"[GLOBAL_GA] Warm-starting from previous checkpoint genome (len={len(seed_genome)})")

    toolbox = base.Toolbox()
    for i, (_, lo, hi, _, _) in enumerate(GLOBAL_PARAM_SPECS):
        toolbox.register(f"attr_g{i}", random.uniform, float(lo), float(hi))

    toolbox.register("individual", tools.initCycle, Individual_Global, tuple(getattr(toolbox, f"attr_g{i}") for i in range(_n_params)), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=GA_TOURN)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[s[1] for s in GLOBAL_PARAM_SPECS], up=[s[2] for s in GLOBAL_PARAM_SPECS], eta=20.0)

    def _mut(ind, sigma=0.08, gene_pb=0.20):
        for j in range(len(ind)):
            if random.random() < gene_pb:
                ind[j] = float(ind[j]) + random.gauss(0.0, sigma * (GLOBAL_PARAM_SPECS[j][2] - GLOBAL_PARAM_SPECS[j][1]))
                ind[j] = float(np.clip(ind[j], GLOBAL_PARAM_SPECS[j][1], GLOBAL_PARAM_SPECS[j][2]))
        return (ind,)

    toolbox.register("mutate", _mut)
    
    # Cache de avaliaÃ§Ãµes
    eval_cache = {}

    pop = toolbox.population(n=pop_size)

    # ── Warm-start: seed population with previous best genome ────────────
    # Even if the model or features changed, the GA trading parameters are
    # model-agnostic, so we can always continue from the last known optimum.
    if seed_genome is not None and len(seed_genome) == len(GLOBAL_PARAM_SPECS):
        sanitized_seed = sanitize_global_genome(seed_genome)
        # Seed first individual as exact copy
        pop[0][:] = sanitized_seed
        if hasattr(pop[0], 'fitness'):
            del pop[0].fitness.values
        # Seed next few as small perturbations of the seed
        n_seeds = min(max(3, pop_size // 8), len(pop) - 1)
        for j in range(1, n_seeds + 1):
            varied = [
                float(np.clip(
                    g + random.gauss(0.0, 0.06 * (hi - lo)),
                    lo, hi
                ))
                for g, (_, lo, hi, _, _) in zip(sanitized_seed, GLOBAL_PARAM_SPECS)
            ]
            pop[j][:] = sanitize_global_genome(varied)
            if hasattr(pop[j], 'fitness'):
                del pop[j].fitness.values
        print(f"[GLOBAL_GA] Seeded {n_seeds + 1} individuals from checkpoint.")

    hof = tools.HallOfFame(5) # Top 5
    best_fit_seen = -1e18
    gens_without_improvement = 0
    diversity = 100.0

    # ── Executor setup ────────────────────────────────────────────────────
    # When num_cpus == 1 (enforced on Windows to avoid spawn-mode deadlocks)
    # use a simple single-threaded executor (no pickle, no pipe). On
    # Linux/Mac with fork-based multiprocessing, use ProcessPoolExecutor.
    _use_multiproc = (num_cpus > 1)
    _payload_tmp_path = None

    if _use_multiproc:
        import pickle as _pickle
        import tempfile as _tempfile
        _payload_tmp = _tempfile.NamedTemporaryFile(
            suffix="_ga_payloads.pkl", delete=False
        )
        _payload_tmp_path = _payload_tmp.name
        with _payload_tmp:
            _pickle.dump(_GLOBAL_PAYLOADS_BY_WINDOW, _payload_tmp, protocol=4)
        print(f"[GLOBAL_GA] payloads saved to {_payload_tmp_path} for workers")
        from concurrent.futures import ProcessPoolExecutor as _PPE
        executor = _PPE(
            max_workers=num_cpus,
            initializer=_init_worker_file,
            initargs=(_payload_tmp_path,),
        )
    else:
        # Single-threaded: use a dummy executor backed by the builtin map
        from concurrent.futures import ThreadPoolExecutor as _TPE
        executor = _TPE(max_workers=1)
        print(f"[GLOBAL_GA] single-threaded evaluation (Windows/1-CPU mode)")

    for gen in range(1, int(ngen) + 1):
        gen_t0 = time.perf_counter()
        invalid = [ind for ind in pop if not hasattr(ind, 'fitness') or not ind.fitness.valid]

        inds_to_eval = []
        for ind in invalid:
            key = tuple(ind)
            if key in eval_cache:
                res = eval_cache[key]
                ind.fitness.values = (res[0],)
                ind.sharpe_train = res[1]
                ind.sharpe_val = res[2]
            else:
                inds_to_eval.append(ind)
                
        if inds_to_eval:
            chunksize = max(1, len(inds_to_eval) // (num_cpus * 4))
            results = list(executor.map(_eval_ind_global, inds_to_eval, chunksize=chunksize))
            for ind, res in zip(inds_to_eval, results):
                key = tuple(ind)
                eval_cache[key] = res
                ind.fitness.values = (res[0],)
                ind.sharpe_train = res[1]
                ind.sharpe_val = res[2]

        hof.update(pop)

        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for i in range(1, len(offspring), 2):
            if random.random() < GA_CX_PB:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        # Adaptive mutation based on diversity
        current_mut_pb = GA_MUT_PB
        if diversity < 20.0:
            current_mut_pb = min(0.9, GA_MUT_PB * 2.0)
        elif diversity < 40.0:
            current_mut_pb = min(0.7, GA_MUT_PB * 1.5)

        for i in range(len(offspring)):
            if random.random() < current_mut_pb:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # Evaluate offspring
        invalid_off = [ind for ind in offspring if not hasattr(ind, 'fitness') or not ind.fitness.valid]
        inds_to_eval_off = []
        for ind in invalid_off:
            key = tuple(ind)
            if key in eval_cache:
                res = eval_cache[key]
                ind.fitness.values = (res[0],)
                ind.sharpe_train = res[1]
                ind.sharpe_val = res[2]
            else:
                inds_to_eval_off.append(ind)
                
        if inds_to_eval_off:
            chunksize_off = max(1, len(inds_to_eval_off) // (num_cpus * 4))
            results_off = list(executor.map(_eval_ind_global, inds_to_eval_off, chunksize=chunksize_off))
            for ind, res in zip(inds_to_eval_off, results_off):
                key = tuple(ind)
                eval_cache[key] = res
                ind.fitness.values = (res[0],)
                ind.sharpe_train = res[1]
                ind.sharpe_val = res[2]

        hof.update(offspring)

        # Elitism: replace worst offspring with best from hof
        offspring.sort(key=lambda x: x.fitness.values[0], reverse=True)
        for j in range(len(hof)):
            offspring[-(j+1)] = toolbox.clone(hof[j])
        offspring.sort(key=lambda x: x.fitness.values[0], reverse=True)
        pop[:] = offspring

        # Metrics for print
        fits = [ind.fitness.values[0] for ind in pop]
        avg_fit = np.mean(fits)
        std_fit = np.std(fits)
        min_fit = np.min(fits)
        max_fit = np.max(fits)
        median_fit = np.median(fits)

        unique_inds = len(set(tuple(ind) for ind in pop))
        diversity = unique_inds / len(pop) * 100

        best_ind = hof[0]
        current_best_fit = best_ind.fitness.values[0]

        if current_best_fit > best_fit_seen:
            best_fit_seen = current_best_fit
            gens_without_improvement = 0
        else:
            gens_without_improvement += 1
        gen_dt = time.perf_counter() - gen_t0

        print(f"[GA] gen {gen:03d}/{int(ngen):03d} | best={best_fit_seen:.3f} (gen_max={max_fit:.3f}) avg={avg_fit:.3f} med={median_fit:.3f} min={min_fit:.3f} std={std_fit:.3f} | "
              f"sharpe_tr={getattr(best_ind, 'sharpe_train', 0.0):.2f} sharpe_val={getattr(best_ind, 'sharpe_val', 0.0):.2f} | "
              f"div={diversity:.1f}% | mut_pb={current_mut_pb:.2f} | eval={len(invalid_off)} | time={gen_dt:.1f}s")

        if gens_without_improvement >= 15:
            print(f"[GA] Early stopping at generation {gen} (no improvement in 15 gens)")
            break
            
    executor.shutdown()

    # Clean up temp payload file
    if _payload_tmp_path is not None:
        try:
            import os as _os
            _os.unlink(_payload_tmp_path)
        except Exception:
            pass

    print("\n[GA] Top 5 cromossomos:")
    for i, ind in enumerate(hof):
        print(f"  #{i+1}: fitness={ind.fitness.values[0]:.3f} sharpe_tr={getattr(ind, 'sharpe_train', 0.0):.2f} sharpe_val={getattr(ind, 'sharpe_val', 0.0):.2f}")

    best = list(hof[0]) if len(hof) else [s[1] for s in GLOBAL_PARAM_SPECS]
    total_dt = time.perf_counter() - ga_t0
    print(f"[GLOBAL_GA] finished in {total_dt/60.0:.1f}m | best_fit={(hof[0].fitness.values[0] if len(hof) else -1e9):.5f}")
    return decode_global_params(best), sanitize_global_genome(best), (hof[0].fitness.values[0] if len(hof) else -1e9)

# 3) HELPERS
# ==============================================================================
def _parse_dates_smart(s: pd.Series) -> pd.Series:
    ss = s.astype(str)
    frac_dash = ss.str.contains("-", regex=False).mean()
    if frac_dash > 0.5:
        return pd.to_datetime(ss, errors="coerce", dayfirst=False)
    return pd.to_datetime(ss, errors="coerce", dayfirst=True)

def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -50, 50))
    return float(1.0 / (1.0 + math.exp(-x)))

def add_sma200(df: pd.DataFrame) -> pd.DataFrame:
    df["sma200"] = df.groupby(TICKER_COL, sort=False)[CLOSE_COL].transform(
        lambda s: s.rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()
    )
    if USE_MA_SLOPE_FILTER:
        df["sma200_slope"] = df.groupby(TICKER_COL, sort=False)["sma200"].transform(
            lambda x: (x - x.shift(MA_SLOPE_LOOKBACK)) / float(MA_SLOPE_LOOKBACK)
        )
    else:
        df["sma200_slope"] = np.nan
    return df

def add_atr_ohlc_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    ATR fast: compute TR with vector ops + groupby rolling mean.
    TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    """
    h = df[HIGH_COL].astype(float)
    l = df[LOW_COL].astype(float)
    c = df[CLOSE_COL].astype(float)
    pc = df.groupby(TICKER_COL, sort=False)[CLOSE_COL].shift(1).astype(float)

    tr1 = (h - l).abs()
    tr2 = (h - pc).abs()
    tr3 = (l - pc).abs()

    tr = np.nanmax(np.vstack([tr1.to_numpy(), tr2.to_numpy(), tr3.to_numpy()]), axis=0)
    tr = pd.Series(tr, index=df.index)

    df["atr"] = tr.groupby(df[TICKER_COL], sort=False).transform(
        lambda s: s.rolling(ATR_WINDOW, min_periods=ATR_MIN_PERIODS).mean()
    )
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add robust technical indicators for ML pipeline.

    Features (enhanced set):
    - Distance from Moving Average (20-day and 50-day)
    - ROC (Rate of Change, 5-day and 10-day)
    - Volatility (20-day rolling std)
    - Relative Volume (volume vs 20-day average)
    - Stochastic Oscillator (14-day %K and %D)
    - RSI (14-day)
    - MACD histogram
    - Bollinger Band %B
    - CCI (Commodity Channel Index, 20-day)
    - ADX proxy (directional strength)
    - Williams %R
    - Price momentum (rate of change of ROC)
    """
    grouped = df.groupby(TICKER_COL, sort=False)

    # Distance from 20-day Moving Average
    df['dist_ma20'] = grouped[CLOSE_COL].transform(
        lambda x: (lambda ma20: (x - ma20) / ma20)(x.rolling(20, min_periods=1).mean())
    )

    # Distance from 50-day Moving Average
    df['dist_ma50'] = grouped[CLOSE_COL].transform(
        lambda x: (lambda ma50: (x - ma50) / ma50)(x.rolling(50, min_periods=1).mean())
    )

    # ROC (Rate of Change) - 5 day (short-term momentum)
    df['roc_5'] = grouped[CLOSE_COL].transform(
        lambda x: (x - x.shift(5)) / x.shift(5).replace(0, np.nan)
    )

    # ROC (Rate of Change) - 10 day
    df['roc_10'] = grouped[CLOSE_COL].transform(
        lambda x: (x - x.shift(10)) / x.shift(10).replace(0, np.nan)
    )

    # Volatility - 20-day rolling standard deviation of returns
    df['volatility_20'] = grouped[CLOSE_COL].transform(
        lambda x: x.pct_change().rolling(20, min_periods=1).std()
    )

    # Relative Volume - volume relative to 20-day average
    if 'volume' in df.columns:
        df['rel_volume'] = grouped['volume'].transform(
            lambda x: (lambda avg: x / avg.replace(0, np.nan))(x.rolling(20, min_periods=1).mean())
        )
    else:
        df['rel_volume'] = np.nan

    # Stochastic Oscillator (14-day %K)
    high_14 = grouped[HIGH_COL].transform(lambda x: x.rolling(14, min_periods=1).max())
    low_14 = grouped[LOW_COL].transform(lambda x: x.rolling(14, min_periods=1).min())
    denom = (high_14 - low_14).replace(0, np.nan)
    df['stochastic_k'] = 100 * (df[CLOSE_COL] - low_14) / denom

    # Stochastic %D (3-day SMA of %K)
    df['stochastic_d'] = grouped['stochastic_k'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # RSI (14-day)
    def _rsi_transform(x):
        delta = x.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    df['rsi_14'] = grouped[CLOSE_COL].transform(_rsi_transform)

    # RSI acceleration (today RSI vs 5 days ago)
    df['rsi_accel_5'] = grouped['rsi_14'].transform(lambda x: x - x.shift(5))

    # MACD histogram (12-26-9)
    ema12 = grouped[CLOSE_COL].transform(lambda x: x.ewm(span=12, adjust=False, min_periods=1).mean())
    ema26 = grouped[CLOSE_COL].transform(lambda x: x.ewm(span=26, adjust=False, min_periods=1).mean())
    macd_line = ema12 - ema26
    signal_line = macd_line.groupby(df[TICKER_COL], sort=False).transform(
        lambda x: x.ewm(span=9, adjust=False, min_periods=1).mean()
    )
    df['macd_hist'] = (macd_line - signal_line) / df[CLOSE_COL].replace(0, np.nan)

    # Bollinger Band %B (20-day, 2 std)
    bb_ma = grouped[CLOSE_COL].transform(lambda x: x.rolling(20, min_periods=1).mean())
    bb_std = grouped[CLOSE_COL].transform(lambda x: x.rolling(20, min_periods=1).std())
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)
    df['bb_pctb'] = (df[CLOSE_COL] - bb_lower) / bb_width

    # CCI (Commodity Channel Index, 20-day)
    tp = (df[HIGH_COL] + df[LOW_COL] + df[CLOSE_COL]) / 3.0
    tp_ma = tp.groupby(df[TICKER_COL], sort=False).transform(lambda x: x.rolling(20, min_periods=1).mean())
    tp_md = tp.groupby(df[TICKER_COL], sort=False).transform(lambda x: x.rolling(20, min_periods=1).apply(lambda w: np.mean(np.abs(w - w.mean())), raw=True))
    df['cci_20'] = (tp - tp_ma) / (0.015 * tp_md.replace(0, np.nan))

    # ADX proxy: absolute directional movement normalized by ATR
    plus_dm = (df[HIGH_COL] - df[HIGH_COL].groupby(df[TICKER_COL], sort=False).shift(1)).clip(lower=0)
    minus_dm = (df[LOW_COL].groupby(df[TICKER_COL], sort=False).shift(1) - df[LOW_COL]).clip(lower=0)
    plus_di = plus_dm.groupby(df[TICKER_COL], sort=False).transform(lambda x: x.rolling(14, min_periods=1).mean())
    minus_di = minus_dm.groupby(df[TICKER_COL], sort=False).transform(lambda x: x.rolling(14, min_periods=1).mean())
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    df['adx_proxy'] = (plus_di - minus_di).abs() / di_sum

    # Williams %R (14-day)
    df['williams_r'] = -100 * (high_14 - df[CLOSE_COL]) / denom

    # Price momentum (ROC of ROC: acceleration)
    df['momentum_accel'] = grouped['roc_10'].transform(
        lambda x: x - x.shift(5)
    )

    # Volatilidade relativa (ATR / Close)
    df['vol_rel_atr'] = df['atr'] / df[CLOSE_COL].replace(0, np.nan)

    # DistÃ¢ncia da mÃ©dia longa (Close vs SMA200)
    if 'sma200' in df.columns:
        df['dist_sma200'] = (df[CLOSE_COL] - df['sma200']) / df['sma200'].replace(0, np.nan)
    else:
        df['dist_sma200'] = np.nan

    # PadrÃ£o de volume (volume / mÃ©dia de 20 dias)
    df['volume_pattern_20'] = df['rel_volume']

    # Regime de mercado proxy (inclinaÃ§Ã£o da SMA200)
    if 'sma200_slope' in df.columns:
        df['regime_sma200_slope'] = df['sma200_slope']
        df['regime_bull'] = (df['sma200_slope'] > 0).astype(float)
    else:
        df['regime_sma200_slope'] = np.nan
        df['regime_bull'] = np.nan

    # ── NEW v3: Volatility regime features ──
    # Rolling vol percentile: high = volatile regime, low = calm regime
    vol_20 = grouped[CLOSE_COL].transform(
        lambda x: x.pct_change().rolling(20, min_periods=5).std()
    )
    df['vol_regime_pctile'] = vol_20.groupby(df[TICKER_COL], sort=False).transform(
        lambda x: x.rolling(252, min_periods=60).rank(pct=True)
    )
    # Volatility expansion/contraction: current vol vs 60-day avg vol
    vol_60_avg = vol_20.groupby(df[TICKER_COL], sort=False).transform(
        lambda x: x.rolling(60, min_periods=20).mean()
    )
    df['vol_expansion'] = (vol_20 / vol_60_avg.replace(0, np.nan)).clip(0.2, 5.0) - 1.0

    # ── NEW v3: Trend consistency (% of last 10 closes above MA50) ──
    ma50 = grouped[CLOSE_COL].transform(lambda x: x.rolling(50, min_periods=10).mean())
    above_ma50 = (df[CLOSE_COL] > ma50).astype(float)
    df['trend_consistency_10'] = above_ma50.groupby(df[TICKER_COL], sort=False).transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )

    # ── NEW v3: Bollinger Band width (vol proxy) ──
    df['bb_width'] = (bb_upper - bb_lower) / bb_ma.replace(0, np.nan)

    # ── NEW v3: Mean reversion intensity (distance from BB mid as fraction of width) ──
    df['mean_rev_intensity'] = ((df[CLOSE_COL] - bb_ma) / (bb_width.replace(0, np.nan) * bb_ma + 1e-9)).clip(-3, 3)

    # ── NEW v3: Intraday range efficiency (close position in day's range) ──
    day_range = (df[HIGH_COL] - df[LOW_COL]).replace(0, np.nan)
    df['range_efficiency'] = (df[CLOSE_COL] - df[LOW_COL]) / day_range

    return df

def regime_ok(sig: int, price: float, sma200: float, sma_slope: float, ml_score: float = np.nan) -> bool:
    strong_buy = np.isfinite(ml_score) and (float(ml_score) >= ML_STRONG_SCORE_ABS)
    decent_signal = np.isfinite(ml_score) and (abs(float(ml_score)) >= ENTRY_SCORE_TRIGGER_ABS)

    # If SMA200 is not available, allow entry for decent ML signals
    if sig > 0:
        if (sma200 is None) or (not np.isfinite(sma200)):
            return decent_signal
        if (price < sma200) and (not strong_buy) and (not decent_signal):
            return False

    # MA rule: relaxed â€” decent signals can override MA filter
    if sig > 0:
        if REQUIRE_MA_FOR_ENTRY:
            if (sma200 is None) or (not np.isfinite(sma200)):
                ok_ma = decent_signal
            else:
                ok_ma = (price >= sma200) or strong_buy or decent_signal
        else:
            ok_ma = True
    else:
        if REQUIRE_MA_FOR_SELL_MA and REQUIRE_MA_FOR_ENTRY:
            if (sma200 is None) or (not np.isfinite(sma200)):
                ok_ma = decent_signal
            else:
                ok_ma = (price < sma200) or decent_signal
        else:
            ok_ma = True

    # slope rule: relaxed â€” decent signals bypass slope filter
    if USE_MA_SLOPE_FILTER:
        if (sma_slope is None) or (not np.isfinite(sma_slope)):
            ok_sl = decent_signal
        elif sig > 0 and (strong_buy or decent_signal):
            ok_sl = True
        else:
            ok_sl = (sma_slope > MA_SLOPE_EPS) if sig > 0 else (sma_slope < -MA_SLOPE_EPS)
    else:
        ok_sl = True

    return bool(ok_ma and ok_sl)

def buyhold_capped(close: np.ndarray) -> float:
    n = len(close)
    if n < 2:
        return 0.0
    log_eq = 0.0
    for i in range(1, n):
        pr0, pr1 = float(close[i-1]), float(close[i])
        if pr0 <= MIN_PRICE or pr1 <= MIN_PRICE:
            continue
        daily = (pr1/pr0) - 1.0
        daily = float(np.clip(daily, -CAP_DAILY_RET, CAP_DAILY_RET))
        log_eq += math.log1p(daily)
    return float(math.exp(log_eq) - 1.0)


def _finite_or_default(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float(default)
    return x if math.isfinite(x) else float(default)


def fitness_return_1y(stats_1y: Dict[str, float]) -> float:
    ret = float(stats_1y["total_return"])
    mdd = float(stats_1y["mdd"])
    ntr = float(stats_1y["n_trades"])
    exp = float(stats_1y.get("exposure", 0.0))

    if ntr < GA_MIN_TRADES:
        return -1e9
    if exp < GA_MIN_EXPOSURE:
        return -1e9

    score = ret
    if mdd < -0.05:
        score -= LAMBDA_MDD_1Y * abs(mdd)
    if exp > MAX_EXPOSURE_1Y:
        score -= 0.5 * (exp - MAX_EXPOSURE_1Y)

    if ntr < GA_TARGET_TRADES:
        score -= (GA_TARGET_TRADES - ntr) * 0.005
    else:
        score += min(ntr - GA_TARGET_TRADES, 30) * GA_TRADE_BONUS_PER

    if ntr > MAX_TRADES_PER_YEAR:
        score -= (ntr - MAX_TRADES_PER_YEAR) * OVERTRADING_PENALTY_PER_TRADE

    return float(score)


def score_0_100_from_ev(
    score_ev: float,
    recent_scores_ev: np.ndarray,
    quality: float,
) -> float:
    """0 = venda forte, 50 = neutro, 100 = compra forte."""
    if not np.isfinite(score_ev):
        return 50.0
    recent = recent_scores_ev[np.isfinite(recent_scores_ev)] if recent_scores_ev is not None else np.array([], dtype=np.float64)
    if len(recent) < 5:
        # Sem historico suficiente, mapear via sigmoid suave
        raw = float(np.tanh(score_ev * 2.0))  # [-1, 1]
        return float(np.clip(50.0 + raw * 50.0, 0.0, 100.0))
    # Directional percentile: score_ev positivo = compra, negativo = venda
    pct_rank = float(np.mean(recent <= score_ev))
    # Mapear [0,1] -> [0,100] mantendo direcionalidade
    return float(np.clip(pct_rank * 100.0, 0.0, 100.0))

def compute_quality_factor(test_sharpe: float, test_return: float, trades_1y: float) -> float:
    q_sh = _sigmoid((float(test_sharpe) - 0.10) / 0.30) if np.isfinite(test_sharpe) else 0.5
    q_ret = _sigmoid(float(test_return) / 0.15) if np.isfinite(test_return) else 0.5
    q_tr = _sigmoid((float(trades_1y) - 8.0) / 4.0) if np.isfinite(trades_1y) else 0.3
    return float(0.50 * q_sh + 0.30 * q_ret + 0.20 * q_tr)


def build_direct_feature_signal(frame: pd.DataFrame, feat_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-column directed z-scores (no single-feature model)."""
    feat = frame[feat_cols].apply(pd.to_numeric, errors="coerce").copy()
    roll_mean = feat.rolling(SCORE_LOOKBACK, min_periods=60).mean()
    roll_std = feat.rolling(SCORE_LOOKBACK, min_periods=60).std(ddof=0).replace(0.0, np.nan)
    z = (feat - roll_mean) / roll_std
    z = z.replace([np.inf, -np.inf], np.nan).clip(-4.0, 4.0)

    # Bearish tokens: features where HIGH value = WORSE (use negative sign → high z-score = sell signal)
    bearish_tokens = ("risk", "down", "dd", "sell", "err_buy", "pe", "price_to_book",
                      "fund_dy_bearish", "fund_pe_vs_hist", "vol_regime_pctile",
                      "vol_expansion", "mean_rev_intensity")
    # Bullish override: features that contain bearish sub-strings but are INVERTED (high = bullish)
    bullish_override_tokens = ("earnings_yield", "book_yield", "xs_inv", "dy_xs",
                               "dy_vs_hist", "value_composite", "fund_pe_xs_inv",
                               "trend_consistency", "range_efficiency")
    signs = []
    for c in feat_cols:
        lc = str(c).lower()
        if any(tok in lc for tok in bullish_override_tokens):
            sign = 1.0  # explicitly bullish even if bearish sub-string present
        elif any(tok in lc for tok in bearish_tokens):
            sign = -1.0
        else:
            sign = 1.0
        signs.append(sign)
    sign_arr = np.asarray(signs, dtype=np.float64)

    z_np = z.to_numpy(np.float64)
    directed = z_np * sign_arr
    valid_counts = np.isfinite(directed).sum(axis=1)
    long_votes = np.nanmean((directed > 0.35).astype(np.float64), axis=1)
    short_votes = np.nanmean((directed < -0.35).astype(np.float64), axis=1)
    long_votes = np.where(valid_counts > 0, long_votes, np.nan)
    short_votes = np.where(valid_counts > 0, short_votes, np.nan)
    return directed.astype(np.float64), long_votes.astype(np.float64), short_votes.astype(np.float64)

def load_full_history_all_cols(path: str) -> pd.DataFrame:
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={TICKER_COL:"string"}, low_memory=False)
    df[TICKER_COL] = df[TICKER_COL].astype("string").str.strip()
    df[DATE_COL] = _parse_dates_smart(df[DATE_COL])

    for col in [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[DATE_COL, TICKER_COL, CLOSE_COL])
    df = df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)

    if ONLY_SA:
        df = df[df[TICKER_COL].str.endswith(".SA")].copy()

    df = add_sma200(df)
    df = add_atr_ohlc_fast(df)
    df = add_technical_features(df)

    # ── Derived fundamental features (computed on-the-fly, no ETL re-run needed) ──
    # These add alternative economic signal flavors so the GA can choose
    # among multiple hypotheses about dividend yield, PE, P/B direction.

    # 1. Earnings yield = 1/PE (positive signal: high yield = cheap stock = bullish)
    #    Works even with snapshot PE because cross-sectional ranks are valid
    if "trailing_pe" in df.columns:
        pe_pos = df["trailing_pe"].where(df["trailing_pe"] > 0.5, np.nan)  # only meaningful PE
        df["fund_earnings_yield"] = (1.0 / pe_pos).clip(0, 0.50)  # cap at 50% earnings yield

        # Cross-sectional rank of earnings yield per date (high rank = cheap vs market = bullish)
        df["fund_pe_xs_inv"] = df.groupby(DATE_COL)["fund_earnings_yield"].rank(
            pct=True, ascending=True, na_option="keep"
        )  # high rank = high earnings yield = cheap = bullish (sign=+1)

        # Ticker-relative PE: deviation from own rolling 2yr median (sign=-1: above hist = expensive)
        df["fund_pe_vs_hist"] = df.groupby(TICKER_COL)["trailing_pe"].transform(
            lambda x: (x - x.rolling(504, min_periods=60).median()) /
                      (x.rolling(504, min_periods=60).std().replace(0, np.nan) + 1e-9)
        ).clip(-3, 3)  # bearish when above own history → sign controlled by "pe" in bearish_tokens

    # 2. Alternative dividend yield signals (two economic hypotheses)
    if "dividend_yield" in df.columns:
        # Hypothesis A (income): high DY = more income = bullish → default sign=+1
        # (already in parquet as 'dividend_yield', which has no bearish token → sign=+1)

        # Hypothesis B (value trap): high DY = distress signal = bearish
        df["fund_dy_bearish"] = df["dividend_yield"].clip(0, 1)  # raw: bearish token 'dy_bearish'

        # Hypothesis C: DY relative to ticker's own 2yr rolling median (above = unusually generous)
        df["fund_dy_vs_hist"] = df.groupby(TICKER_COL)["dividend_yield"].transform(
            lambda x: (x - x.rolling(504, min_periods=60).median()) /
                      (x.rolling(504, min_periods=60).std().replace(0, np.nan) + 1e-9)
        ).clip(-3, 3)  # sign=+1 (above historical DY = bullish, more income than usual)

        # Cross-sectional DY rank: high rank = high yield relative to market
        df["fund_dy_xs"] = df.groupby(DATE_COL)["dividend_yield"].rank(
            pct=True, ascending=True, na_option="keep"
        )  # sign=+1 (high DY rank = high income = bullish in income hypothesis)

    # 3. Alternative P/B signals
    if "price_to_book" in df.columns:
        pb_pos = df["price_to_book"].where(df["price_to_book"] > 0.01, np.nan)
        # Book yield = 1/P/B (low P/B = cheap = bullish, avoid "price_to_book" bearish token)
        df["fund_book_yield"] = (1.0 / pb_pos).clip(0, 5.0)  # sign=+1 (high book yield = cheap)

        # Cross-sectional P/B rank inverted: low P/B tickers rank higher = cheaper
        df["fund_pb_xs_inv"] = 1.0 - df.groupby(DATE_COL)["price_to_book"].rank(
            pct=True, ascending=True, na_option="keep"
        )  # sign=+1 (high score = low P/B = cheap = bullish)

    # 4. Composite value score (cross-sectional, weights earnings_yield + book_yield + dy)
    fund_components = [c for c in ["fund_earnings_yield", "fund_book_yield", "fund_dy_xs"]
                       if c in df.columns]
    if len(fund_components) >= 2:
        # Cross-sectional normalize each component per date, then average
        normed = []
        for fc in fund_components:
            ranked = df.groupby(DATE_COL)[fc].rank(pct=True, ascending=True, na_option="keep")
            normed.append(ranked)
        df["fund_value_composite"] = pd.concat(normed, axis=1).mean(axis=1)
        # sign=+1: higher composite rank = cheaper stock = bullish

    return df


def build_temporal_windows(min_date: pd.Timestamp, max_date: pd.Timestamp, train_years: int = 3, test_months: int = 6, step_months: int = 6) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    windows = []
    cur_train_start = pd.Timestamp(min_date).normalize()
    max_date = pd.Timestamp(max_date).normalize()
    while True:
        train_end = cur_train_start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_end > max_date:
            break
        windows.append((cur_train_start, train_end, test_start, test_end))
        cur_train_start = cur_train_start + pd.DateOffset(months=step_months)
    return windows


def generate_signal_global(payload: Dict[str, Any], gp: GlobalParams, day_index: int) -> str:
    x = np.asarray(payload["score_matrix"], dtype=np.float64)
    c = np.asarray(payload["close"], dtype=np.float64)
    atr = np.asarray(payload["atr"], dtype=np.float64)
    if day_index <= 0 or day_index >= len(c) or x.ndim != 2:
        return "hold"

    feat_n = max(1, x.shape[1])
    votes_long = (x > gp.z_threshold).sum(axis=1) / feat_n
    votes_short = (x < -gp.z_threshold).sum(axis=1) / feat_n
    score_raw = votes_long - votes_short
    score_ev = pd.Series(score_raw).ewm(span=int(gp.signal_ema_span), adjust=False).mean().to_numpy()

    ma = pd.Series(c).rolling(int(gp.ma_filter_period), min_periods=max(20, int(gp.ma_filter_period // 2))).mean().to_numpy()
    vol_rel = atr / np.maximum(c, ATR_EPS)
    vol_rank = rolling_percentile_rank(vol_rel, 252)

    consec_long = 0
    consec_short = 0
    for i in range(1, day_index + 1):
        lookback = max(63, SCORE_LOOKBACK)
        w = score_ev[max(0, i - lookback + 1):i + 1]
        if len(w) < 20:
            pctl = 0.0
        else:
            pctl = float(np.nanpercentile(w, gp.score_percentile_trigger * 100))

        vl = votes_long[i - 1]
        vs = votes_short[i - 1]
        long_raw = (vl >= gp.vote_threshold_long) and (score_ev[i - 1] >= pctl)
        short_raw = (vs >= gp.vote_threshold_short) and (-score_ev[i - 1] >= pctl)

        if long_raw:
            consec_long += 1
        else:
            consec_long = 0
        if short_raw:
            consec_short += 1
        else:
            consec_short = 0

    long_ok = consec_long >= gp.entry_confirmation_days
    short_ok = consec_short >= gp.entry_confirmation_days

    if gp.volatility_filter_percentile > 0 and np.isfinite(vol_rank[day_index - 1]) and vol_rank[day_index - 1] < gp.volatility_filter_percentile:
        long_ok = False
        short_ok = False

    if gp.ma_filter_mode == 1 and np.isfinite(ma[day_index - 1]):
        if c[day_index - 1] < ma[day_index - 1]:
            long_ok = False
        if c[day_index - 1] > ma[day_index - 1]:
            short_ok = False
    elif gp.ma_filter_mode == 2 and np.isfinite(ma[day_index - 1]):
        if c[day_index - 1] < ma[day_index - 1]:
            long_ok = False
        if c[day_index - 1] > ma[day_index - 1]:
            short_ok = False

    if long_ok:
        return "buy"
    if (not LONG_ONLY) and short_ok:
        return "sell"
    return "hold"

# 4) MAIN
# ==============================================================================
def run():
    print("=== GA + Walk-Forward ML (OOS) + Intraday Backtest (v2) ===")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df = load_full_history_all_cols(HISTORY_CSV_PATH)
    exclude = {DATE_COL, TICKER_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, "sma200", "sma200_slope", "atr"}
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    base_feat_cols = num_cols[:MAX_FEATURES]

    tickers = df[TICKER_COL].dropna().unique().tolist()
    total_tickers = len(tickers)
    if EVAL_ONLY_TICKER:
        t_req = str(EVAL_ONLY_TICKER).strip().upper()
        tickers = [t for t in tickers if str(t).upper() in {t_req, f"{t_req}.SA"}]
    elif TEST_ONLY_PREFIX_C_TICKERS:
        tickers = [t for t in tickers if str(t).startswith(TEST_TICKER_PREFIX)]

    print(f"[FEATS] numeric candidates: {len(base_feat_cols)} (using up to {MAX_FEATURES})")
    print(f"[DATA] {len(df)} rows, {len(tickers)} tickers (out of {total_tickers})")
    print(f"[SYSTEM] CPUs available={multiprocessing.cpu_count()} | configured_workers={GA_EVAL_WORKERS}")

    out_global_ckpt = os.path.join(OUTPUT_DIR, "global_ga_checkpoint.json")
    out_xlsx = os.path.join(OUTPUT_DIR, f"summary_latest.xlsx")
    out_apply_csv = os.path.join(OUTPUT_DIR, f"apply_last_{APPLY_DAYS}d__H{FWD_H}.csv")

    # FASE 1: Preparar payloads
    prep_t0 = time.perf_counter()
    ticker_payloads: Dict[str, Dict[str, Any]] = {}
    reasons: Dict[str, int] = {}

    for ix_tkr, tkr in enumerate(tickers, 1):
        g = df[df[TICKER_COL] == tkr].copy()
        if len(g) < MIN_ROWS_TICKER:
            reasons["few_rows"] = reasons.get("few_rows", 0) + 1
            continue

        g = g.sort_values(DATE_COL).reset_index(drop=True)
        c_temp = g[CLOSE_COL].to_numpy(np.float64)
        atr_temp = g["atr"].to_numpy(np.float64)
        ret_fwd_temp = np.empty_like(c_temp)
        ret_fwd_temp[:-FWD_H] = (c_temp[FWD_H:] / c_temp[:-FWD_H]) - 1.0
        ret_fwd_temp[-FWD_H:] = np.nan
        atr_pct_temp = atr_temp / np.maximum(c_temp, 1e-12)
        thr_temp = np.maximum(TARGET_RET_THRESHOLD, TARGET_ATR_MULT * atr_pct_temp)
        y_event_temp = (ret_fwd_temp > thr_temp).astype(float)
        y_event_temp[~np.isfinite(ret_fwd_temp)] = np.nan
        y_event_temp[~np.isfinite(thr_temp)] = np.nan

        feat_correlations: List[tuple] = []
        for ccol in base_feat_cols:
            s_col = pd.to_numeric(g[ccol], errors="coerce")
            if float(s_col.notna().mean()) < MIN_FEAT_NONNA_FRAC:
                continue
            if float(s_col.std(skipna=True)) <= MIN_FEAT_STD:
                continue
            try:
                valid_idx_temp = np.isfinite(s_col.to_numpy()) & np.isfinite(y_event_temp)
                if valid_idx_temp.sum() > MIN_VALID_SAMPLES_FOR_CORRELATION:
                    corr, _ = spearmanr(s_col.to_numpy()[valid_idx_temp], y_event_temp[valid_idx_temp])
                    if np.isfinite(corr):
                        feat_correlations.append((ccol, abs(corr)))
            except ValueError:
                pass

        feat_correlations.sort(key=lambda x: x[1], reverse=True)
        # Greedy dedup: keep the highest-r feature first; skip any later feature whose
        # pairwise |Spearman r| with ANY already-selected feature exceeds FEAT_DEDUP_CORR_THRESH.
        # This removes quasi-identical signals (e.g. EV_buy variants r=1.00) without
        # discarding genuinely independent lower-ranked features.
        _sel_cols: List[str] = []
        _sel_arrays: List[np.ndarray] = []
        _sel_corrs: List[tuple] = []
        for _ccol, _abs_r in feat_correlations:
            _arr = pd.to_numeric(g[_ccol], errors="coerce").to_numpy(np.float64)
            _redundant = False
            for _prev_arr in _sel_arrays:
                _valid = np.isfinite(_arr) & np.isfinite(_prev_arr)
                if _valid.sum() > MIN_VALID_SAMPLES_FOR_CORRELATION:
                    _r_pair, _ = spearmanr(_arr[_valid], _prev_arr[_valid])
                    if np.isfinite(_r_pair) and abs(_r_pair) > FEAT_DEDUP_CORR_THRESH:
                        _redundant = True
                        break
            if not _redundant:
                _sel_cols.append(_ccol)
                _sel_arrays.append(_arr)
                _sel_corrs.append((_ccol, _abs_r))
            if len(_sel_cols) >= MAX_FEATURES:
                break
        feat_cols = _sel_cols
        if len(feat_cols) < 5:
            reasons["few_feats"] = reasons.get("few_feats", 0) + 1
            continue

        directed_cols, long_votes, short_votes = build_direct_feature_signal(g, feat_cols)
        dates = pd.to_datetime(g[DATE_COL], errors="coerce").to_numpy()
        o = g[OPEN_COL].to_numpy(np.float64)
        h = g[HIGH_COL].to_numpy(np.float64)
        l = g[LOW_COL].to_numpy(np.float64)
        c = g[CLOSE_COL].to_numpy(np.float64)
        atr = g["atr"].to_numpy(np.float64)
        ma = g["sma200"].to_numpy(np.float64)

        valid_cols = np.isfinite(directed_cols).any(axis=1)
        valid_mask = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & np.isfinite(atr) & valid_cols
        if valid_mask.sum() < ML_MIN_TRAIN:
            reasons["few_valid"] = reasons.get("few_valid", 0) + 1
            continue

        ticker_payloads[str(tkr)] = {
            "open": o, "high": h, "low": l, "close": c, "atr": atr,
            "score_matrix": directed_cols, "dates": dates, "ma": ma,
            "feat_cols": feat_cols, "valid_mask": valid_mask,
            "long_votes": long_votes, "short_votes": short_votes,
            "feat_corr": _sel_corrs,  # list of (col, abs_spearman_r) for selected (deduped) features
        }
        if (ix_tkr % max(1, PRINT_EVERY)) == 0 or ix_tkr == len(tickers):
            dt = time.perf_counter() - prep_t0
            print(f"[PHASE1] prep {ix_tkr}/{len(tickers)} | payloads={len(ticker_payloads)} | elapsed={dt:.1f}s")

    prep_dt = time.perf_counter() - prep_t0
    print(f"[PHASE1] done payloads={len(ticker_payloads)} in {prep_dt:.1f}s")

    if not ticker_payloads:
        print("Nenhum ticker preparado.")
        return

    # FASE 2: GA global (uma vez) com checkpoint
    # ── Always load the previous best genome for warm-starting ─────────
    # (works even if features or model architecture changed, since the
    #  genome encodes only trading strategy parameters, not feature weights)
    run_mode = str(RUN_MODE).strip().lower()
    global_params = None
    global_genome = None
    global_fit = float("nan")
    prev_genome_for_seed: Optional[List[float]] = None

    if os.path.exists(out_global_ckpt):
        try:
            ck = json.load(open(out_global_ckpt, "r", encoding="utf-8"))
            prev_genome = ck.get("genome", None)
            n_specs = len(GLOBAL_PARAM_SPECS)
            if prev_genome:
                # Handle genome length mismatch: pad with defaults for new genes
                if len(prev_genome) < n_specs:
                    for idx in range(len(prev_genome), n_specs):
                        _, lo, hi, step, is_int = GLOBAL_PARAM_SPECS[idx]
                        default_val = (lo + hi) / 2.0
                        default_val = round(default_val / step) * step
                        if is_int:
                            default_val = int(round(default_val))
                        prev_genome.append(default_val)
                    print(f"[GLOBAL_GA] padded checkpoint genome from {len(ck.get('genome',[]))} to {n_specs} genes (new genes added)")
                elif len(prev_genome) > n_specs:
                    prev_genome = prev_genome[:n_specs]
                    print(f"[GLOBAL_GA] truncated checkpoint genome to {n_specs} genes")
                if len(prev_genome) == n_specs:
                    prev_genome_for_seed = prev_genome
                    prev_fit = float(ck.get("fitness", float("nan")))
                    if run_mode == "load":
                        global_genome = prev_genome
                        global_params = decode_global_params(global_genome)
                        global_fit = prev_fit
                        print(f"[GLOBAL_GA] loaded from checkpoint (load mode) fit={global_fit:.4f}")
                    else:
                        print(f"[GLOBAL_GA] checkpoint found (fit={prev_fit:.4f}) — will warm-start GA from it")
        except Exception as e:
            print(f"[WARN] global checkpoint load failed: {e}")

    if global_params is None:
        dmin = pd.to_datetime(df[DATE_COL].min())
        dmax = pd.to_datetime(df[DATE_COL].max())
        windows = build_temporal_windows(dmin, dmax, train_years=3, test_months=6, step_months=6)
        if not windows:
            windows = [(dmin, dmax - pd.Timedelta(days=180), dmax - pd.Timedelta(days=179), dmax)]
        mode_label = "FAST" if FAST_MODE else "FULL"
        print(f"[GLOBAL_GA] start ({mode_label} mode, windows={len(windows)}, tickers={len(ticker_payloads)})")
        global_params, global_genome, global_fit = run_global_ga_20params(
            ticker_payloads=ticker_payloads,
            windows=windows,
            pop_size=GA_STAGE1_POP_SIZE,
            ngen=GA_STAGE1_NGEN,
            ga_max_windows=GA_MAX_WINDOWS,
            seed_genome=prev_genome_for_seed,  # warm-start from previous run
        )
        try:
            with open(out_global_ckpt, "w", encoding="utf-8") as f:
                json.dump({"fitness": float(global_fit), "genome": list(global_genome)}, f, ensure_ascii=False, indent=2)
            print(f"[GLOBAL_GA] checkpoint saved fit={global_fit:.4f}")
        except Exception as e:
            print(f"[WARN] global checkpoint save failed: {e}")
        print(f"[GLOBAL_GA] done fit={global_fit:.4f} mode={mode_label}")

    # FASE 3: apply rÃ¡pido sem GA por ticker
    results_summary: List[Dict[str, Any]] = []
    results_apply: List[Dict[str, Any]] = []

    for ix_tkr, (tkr, payload) in enumerate(ticker_payloads.items(), 1):
        c = payload["close"]
        dates = payload["dates"]
        n = len(c)
        if n < 252:
            continue

        # Backtest full history with global params
        st = backtest_stats_global_intraday(
            payload["open"], payload["high"], payload["low"], payload["close"],
            payload["score_matrix"], payload["atr"], global_params, precomputed=payload
        )

        # Generate apply signals (vectorized pre-compute, loop only for signals)
        x = np.asarray(payload["score_matrix"], dtype=np.float64)
        feat_n = max(1, x.shape[1])
        votes_long = (x > global_params.z_threshold).sum(axis=1) / feat_n
        votes_short = (x < -global_params.z_threshold).sum(axis=1) / feat_n
        score_raw = votes_long - votes_short
        score_ev = ewm_mean_np(score_raw, int(global_params.signal_ema_span))
        
        quality = compute_quality_factor(st["sharpe"], st["total_return"], st["n_trades"] / max(n / 252, 0.1))
        score95 = np.nanpercentile(np.abs(score_ev), 95) if np.isfinite(np.nanmax(np.abs(score_ev))) else 1.0
        score95 = max(score95, ATR_EPS)
        lookback = max(63, SCORE_LOOKBACK)
        
        # Backtest consistency metrics for confidence
        bt_sharpe = st.get("sharpe", 0.0)
        bt_win_rate = st.get("win_rate", 0.0)
        bt_n_trades = st.get("n_trades", 0.0)
        n_years = max(n / 252.0, 0.1)
        trades_per_year = bt_n_trades / n_years
        
        for i in range(max(0, n - APPLY_DAYS), n):
            sig = generate_signal_global(payload, global_params, i)
            
            recent_scores = score_ev[max(0, i - lookback + 1):i + 1]
            score_100 = score_0_100_from_ev(score_ev[i], recent_scores, quality)

            atr_val = max(payload["atr"][i], ATR_EPS)
            close_val = c[i]
            
            strength = float(np.clip(abs(score_ev[i]) / score95, 0.0, 1.0))
            discount = global_params.entry_discount_atr_frac * (1.0 - global_params.score_strength_scaling * strength)
            
            # Sempre calcular best_buy e best_sell como referencia
            best_buy = round(close_val - discount * atr_val, 4)
            best_sell = round(close_val + discount * atr_val, 4)
            
            # Entry ref: usar best_buy se sinal compra, best_sell se venda, close se hold
            if sig == "buy":
                entry_ref = best_buy
            elif sig == "sell":
                entry_ref = best_sell
            else:
                entry_ref = close_val
            
            # Stop loss e take profit baseados no ATR e params do GA
            stop_atr = global_params.stop_atr_mult * atr_val
            take_atr = global_params.reward_risk_ratio * stop_atr
            
            if sig == "buy":
                stop_loss = round(entry_ref - stop_atr, 4)
                take_profit = round(entry_ref + take_atr, 4)
            elif sig == "sell":
                stop_loss = round(entry_ref + stop_atr, 4)
                take_profit = round(entry_ref - take_atr, 4)
            else:
                # Para hold, mostrar stop/take do lado mais provavel
                if score_100 >= 50:
                    stop_loss = round(close_val - stop_atr, 4)
                    take_profit = round(close_val + take_atr, 4)
                else:
                    stop_loss = round(close_val + stop_atr, 4)
                    take_profit = round(close_val - take_atr, 4)
            
            # Confidence (0-100): combina qualidade do backtest, forca do sinal e consistencia
            # q_backtest: sharpe e win_rate do backtest
            q_bt = float(np.clip(_sigmoid((bt_sharpe - 0.1) / 0.3) * 0.5 + bt_win_rate * 0.5, 0.0, 1.0))
            # q_signal: forca direcional do sinal atual
            q_sig = strength
            # q_trades: confianca aumenta com mais trades historicos
            q_tr = float(np.clip(trades_per_year / 20.0, 0.0, 1.0))
            # q_agreement: concordancia entre features (votes)
            vl = votes_long[i] if i < len(votes_long) else 0.0
            vs = votes_short[i] if i < len(votes_short) else 0.0
            q_agree = float(np.clip(max(vl, vs) * 2.0, 0.0, 1.0))
            
            confidence = float(np.clip(
                (0.30 * q_bt + 0.25 * q_sig + 0.15 * q_tr + 0.30 * q_agree) * 100.0,
                0.0, 100.0
            ))
            
            results_apply.append({
                "Date": pd.to_datetime(dates[i]).strftime("%Y-%m-%d"),
                "ticker": tkr,
                "close": close_val,
                "signal_eod": sig,
                "score_100": round(score_100, 1),
                "confidence": round(confidence, 1),
                "best_buy_value": best_buy,
                "best_sell_value": best_sell,
                "entry_ref_price": round(entry_ref, 4),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            })

        results_summary.append({
            "ticker": tkr,
            "test_return": st["total_return"],
            "test_mdd": st["mdd"],
            "test_sharpe": st["sharpe"],
            "test_trades": st["n_trades"],
            "test_win_rate": st["win_rate"],
            "test_avg_trade": st["avg_trade"],
            "buy_hold_return": buyhold_capped(c),
        })

    df_sum = pd.DataFrame(results_summary)
    df_app = pd.DataFrame(results_apply)

    if not df_sum.empty:
        df_sum = df_sum.sort_values("test_sharpe", ascending=False)

        # -- Objective validation summary ----------------------------------
        pct_ret_pos    = (df_sum["test_return"] > 0).mean() * 100
        pct_sharpe_pos = (df_sum["test_sharpe"] > 0).mean() * 100
        mean_ret       = df_sum["test_return"].mean()
        med_ret        = df_sum["test_return"].median()
        mean_bh        = df_sum["buy_hold_return"].mean()
        win_rate_mean  = df_sum["test_win_rate"].mean()
        win_rate_med   = df_sum["test_win_rate"].median()
        median_mdd     = df_sum["test_mdd"].median()
        mean_mdd       = df_sum["test_mdd"].mean()
        mean_trades    = df_sum["test_trades"].mean()
        mode_label     = "FAST" if FAST_MODE else "FULL"
        beats_bh_pct   = (df_sum["test_return"] > df_sum["buy_hold_return"]).mean() * 100

        sep = "=" * 60
        print("\n" + sep)
        print(f"=== OBJETIVOS ({mode_label} MODE) ===")
        print(sep)
        print(f"  win_rate_mean   : {win_rate_mean:.3f}  (median: {win_rate_med:.3f})  >> objetivo: ALTO")
        print(f"  median_mdd      : {median_mdd:.3f}  (mean: {mean_mdd:.3f})     >> objetivo: BAIXO")
        print(f"  beat buy&hold % : {beats_bh_pct:.1f}%  test_return({mean_ret:.4f}) vs buy_hold({mean_bh:.4f})")
        print(f"  mean trades/tkr : {mean_trades:.1f}")
        print(f"  % ret_pos       : {pct_ret_pos:.1f}%  | % sharpe_pos: {pct_sharpe_pos:.1f}%")
        print(f"  GA fitness      : {global_fit:.4f}")
        print(sep)
        obj_win_rate_ok = win_rate_mean >= 0.52
        obj_mdd_ok      = median_mdd >= -0.25
        obj_bh_ok       = mean_ret > mean_bh
        obj_trades_ok   = mean_trades >= 10
        results_str = lambda ok: "PASS" if ok else "FAIL"
        print(f"  win_rate_mean>=0.52  : {results_str(obj_win_rate_ok)} ({win_rate_mean:.3f})")
        print(f"  median_mdd>=-25%    : {results_str(obj_mdd_ok)} ({median_mdd:.3f})")
        print(f"  beats buy&hold      : {results_str(obj_bh_ok)} ({mean_ret:.4f} vs {mean_bh:.4f})")
        print(f"  trades>=10/ticker   : {results_str(obj_trades_ok)} ({mean_trades:.1f})")
        all_pass = obj_win_rate_ok and obj_mdd_ok and obj_bh_ok and obj_trades_ok
        print(f"  TODOS OBJETIVOS     : {'ALL PASS' if all_pass else 'FAIL -- re-avaliar'}")
        print(sep)

        # ── Feature importance per ticker ─────────────────────────────────
        # Derived purely from data already in ticker_payloads (no logic change).
        # Metrics:
        #   spearman_abs_r   – |Spearman r| with forward-return event (feature selection criterion)
        #   mean_abs_zscore  – mean |directed z-score| across all history (signal magnitude)
        #   pct_days_active  – % days where |z| > 0.35 (feature fires a vote)
        _feat_imp_rows: List[Dict] = []
        for _tkr, _pl in ticker_payloads.items():
            _corrs: List[tuple] = _pl.get("feat_corr", [])   # (col, abs_r) sorted desc
            _sm: np.ndarray = _pl["score_matrix"]             # (N_days, K_selected)
            for _rank0, (_feat, _abs_r) in enumerate(_corrs[:MAX_FEATURES]):
                _col_z = _sm[:, _rank0]
                _valid_z = _col_z[np.isfinite(_col_z)]
                _mean_abs_z = float(np.mean(np.abs(_valid_z))) if len(_valid_z) > 0 else float("nan")
                _pct_active = float(np.mean(np.abs(_valid_z) > 0.35)) * 100.0 if len(_valid_z) > 0 else float("nan")
                _feat_imp_rows.append({
                    "ticker":           _tkr,
                    "rank":             _rank0 + 1,
                    "feature":          _feat,
                    "spearman_abs_r":   round(float(_abs_r), 4),
                    "mean_abs_zscore":  round(_mean_abs_z, 4),
                    "pct_days_active_%": round(_pct_active, 1),
                })
        df_feat_imp = pd.DataFrame(_feat_imp_rows) if _feat_imp_rows else pd.DataFrame()
        # ─────────────────────────────────────────────────────────────────

        try:
            with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
                df_sum.to_excel(writer, sheet_name="Summary", index=False)
                if not df_app.empty:
                    df_app.to_excel(writer, sheet_name="Apply", index=False)
                if not df_feat_imp.empty:
                    df_feat_imp.to_excel(writer, sheet_name="Feature_Importance", index=False)
        except Exception as e:
            print(f"[WARN] xlsxwriter failed, trying openpyxl: {e}")
            with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
                df_sum.to_excel(writer, sheet_name="Summary", index=False)
                if not df_app.empty:
                    df_app.to_excel(writer, sheet_name="Apply", index=False)
                if not df_feat_imp.empty:
                    df_feat_imp.to_excel(writer, sheet_name="Feature_Importance", index=False)

    if not df_app.empty:
        df_app.to_csv(out_apply_csv, index=False)

    print(f"Saved: {out_xlsx} | {out_apply_csv}")


if __name__ == "__main__":
    run()
