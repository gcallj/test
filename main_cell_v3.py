
# -*- coding: utf-8 -*-
"""
GA + Walk-Forward ML (OOS) + Intraday (OHLC) Backtest  — v2 (fixed outputs)
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

try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass  # running outside Colab

try:
    import xlsxwriter  # noqa: F401
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "xlsxwriter", "-q"])
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import os
import json
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import spearmanr
from numba import njit


# ==============================================================================
# 0) CONFIG
# ==============================================================================
HISTORY_CSV_PATH = "/content/drive/MyDrive/history_consolidated.csv"
OUTPUT_DIR       = "/content/drive/MyDrive/"

DATE_COL   = "Date"
TICKER_COL = "ticker"

OPEN_COL  = "open"
HIGH_COL  = "high"
LOW_COL   = "low"
CLOSE_COL = "close"

ONLY_SA    = True
LONG_ONLY  = False

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
# Realistic B3 costs for liquid stocks: ~30bps round-trip
COST_BPS     = 8.0
SLIPPAGE_BPS = 8.0
COST_PER_TRADE_PCT = 0.0010  # fixed round-trip friction per completed trade

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
MAX_FEATURES = 20

# Intraday entry (limit) based on signal strength
ENTRY_DISCOUNT_RANGE = (0.0, 0.4)
FAST_PERIOD_RANGE = (10, 30)
SLOW_PERIOD_RANGE = (40, 100)
SCORE_CROSS_MIN_ABS = 0.05
ENTRY_SCORE_TRIGGER_ABS = 0.005
ML_STRONG_SCORE_ABS = 0.08

# Avoid "do nothing" strategies — tuned for cost-aware backtesting
GA_MIN_TRADES = 12
GA_TARGET_TRADES = 20
GA_MIN_EXPOSURE = 0.03
GA_TRADE_BONUS_PER = 0.010
MAX_TRADES_PER_YEAR = 30  # tighter cap: 0.68% cost/trade means >30 trades/yr is expensive
OVERTRADING_PENALTY_PER_TRADE = 0.08  # aggressive penalty to discourage overtrading
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
GA_EVAL_WORKERS = max(1, (os.cpu_count() or 2) - 1)
GA_TWO_STAGE = True
RUN_MODE = "train"  # "load" resume from checkpoints, "train" starts from zero
SLOW_STEP_PRINT_SEC = 2.0  # print only timings above this per-step threshold
GA_STAGE1_POP_SIZE = 64
GA_STAGE1_NGEN = 15
GA_STAGE1_TOP_N = 6
GA_STAGE2_POP_SIZE = 120
GA_STAGE2_NGEN = 30
GA_STAGE2_PADDING_RATIO = 0.60
GA_STAGE2_MIN_SPAN_RATIO = 0.18

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
# 2.1) GLOBAL GA (20 parâmetros)
# ==============================================================================

GLOBAL_PARAM_SPECS = [
    ("vote_threshold_long", 0.10, 0.60, 0.05, False),
    ("vote_threshold_short", 0.10, 0.60, 0.05, False),
    ("z_threshold", 0.15, 0.80, 0.05, False),
    ("signal_ema_span", 2.0, 12.0, 1.0, True),
    ("entry_confirmation_days", 1.0, 4.0, 1.0, True),
    ("score_percentile_trigger", 0.40, 0.85, 0.05, False),
    ("stop_atr_mult", 1.0, 4.0, 0.25, False),
    ("stop_tighten_after_bars", 3.0, 15.0, 1.0, True),
    ("stop_tighten_factor", 0.40, 0.85, 0.05, False),
    ("max_loss_per_trade_pct", 0.02, 0.12, 0.01, False),
    ("reward_risk_ratio", 1.0, 5.0, 0.25, False),
    ("partial_take_pct", 0.0, 0.60, 0.10, False),
    ("partial_take_level", 0.5, 1.5, 0.25, False),
    ("time_stop_bars", 5.0, 25.0, 1.0, True),
    ("entry_discount_atr_frac", 0.0, 0.5, 0.05, False),
    ("volatility_filter_percentile", 0.0, 0.40, 0.05, False),
    ("score_strength_scaling", 0.0, 1.0, 0.1, False),
    ("ma_filter_period", 100.0, 300.0, 50.0, True),
    ("ma_filter_mode", 0.0, 2.0, 1.0, True),
    ("consecutive_loss_cooldown", 0.0, 5.0, 1.0, True),
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


def precompute_global_payloads(windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]], full_df: pd.DataFrame, feature_cols: List[str]) -> List[Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]]:
    payloads_by_window = []
    for tr_start, tr_end, te_start, te_end in windows:
        df_tr = full_df[(full_df[DATE_COL] >= tr_start) & (full_df[DATE_COL] <= tr_end)]
        df_te = full_df[(full_df[DATE_COL] >= te_start) & (full_df[DATE_COL] <= te_end)]
        
        def _build_payload(df_sub):
            payloads = {}
            if not df_sub.empty:
                for tk, g in df_sub.groupby(TICKER_COL, sort=False):
                    gg = g.sort_values(DATE_COL)
                    c = gg[CLOSE_COL].to_numpy(np.float64)
                    atr = gg["atr"].to_numpy(np.float64)
                    payloads[tk] = {
                        "open": gg[OPEN_COL].to_numpy(np.float64),
                        "high": gg[HIGH_COL].to_numpy(np.float64),
                        "low": gg[LOW_COL].to_numpy(np.float64),
                        "close": c,
                        "atr": atr,
                        "score_matrix": gg[feature_cols].to_numpy(np.float64),
                        "vol_rank": rolling_percentile_rank(atr / np.maximum(c, ATR_EPS), 252),
                    }
            return payloads
            
        payloads_by_window.append((_build_payload(df_tr), _build_payload(df_te)))
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
        return {"total_return": 0.0, "mdd": 0.0, "sharpe": 0.0, "sortino": 0.0, "n_trades": 0.0, "win_rate": 0.0, "avg_trade": 0.0, "exposure": 0.0, "n_days": 0.0}

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
    pos_days = 0
    partial_taken = False
    consec_long = 0
    consec_short = 0
    consec_stops = 0
    cooldown = 0

    for i in range(1, n):
        if cooldown > 0:
            cooldown -= 1

        if pos != 0:
            bars += 1
            pos_days += 1
            stop_mult = gp.stop_atr_mult * (gp.stop_tighten_factor if bars >= gp.stop_tighten_after_bars else 1.0)
            stop_abs = max(ATR_EPS, stop_mult * max(atr[i], ATR_EPS))
            take_abs = gp.reward_risk_ratio * stop_abs
            hard_loss = abs(o[i] / max(entry_px, ATR_EPS) - 1.0)

            if hard_loss > gp.max_loss_per_trade_pct:
                ret_raw = (o[i] / entry_px - 1.0) * pos
                ret = ((1.0 + ret_raw) * ((1.0 - (COST_BPS + SLIPPAGE_BPS) / 10000.0) ** 2) - 1.0) - COST_PER_TRADE_PCT
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
                equity *= (1.0 + part_ret)
                partial_taken = True

            stop_hit = adv >= stop_abs
            take_hit = fav >= take_abs
            time_stop = (bars >= gp.time_stop_bars) and (fav < 0.5 * stop_abs)

            if stop_hit or take_hit or time_stop:
                exit_px = c[i]
                if stop_hit:
                    exit_px = entry_px - pos * stop_abs
                elif take_hit:
                    exit_px = entry_px + pos * take_abs
                ret_raw = (exit_px / entry_px - 1.0) * pos
                ret = ((1.0 + ret_raw) * ((1.0 - (COST_BPS + SLIPPAGE_BPS) / 10000.0) ** 2) - 1.0) - COST_PER_TRADE_PCT
                equity *= (1.0 + ret)
                trade_rets.append(ret)
                if stop_hit and ret < 0:
                    consec_stops += 1
                elif ret > 0:
                    consec_stops = 0
                pos = 0
                bars = 0
                partial_taken = False
                if gp.consecutive_loss_cooldown > 0 and consec_stops >= 2:
                    cooldown = gp.consecutive_loss_cooldown
                continue

        if pos == 0 and cooldown == 0 and np.isfinite(score_ev[i - 1]) and np.isfinite(score_pctl[i - 1]):
            if gp.volatility_filter_percentile > 0 and np.isfinite(vol_rank[i - 1]) and vol_rank[i - 1] < gp.volatility_filter_percentile:
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

            side = 1 if long_ok else (-1 if short_ok else 0)
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

        peak = max(peak, equity)
        mdd = min(mdd, (equity / max(peak, ATR_EPS)) - 1.0)

    if len(trade_rets) == 0:
        return {"total_return": float(equity - 1.0), "mdd": float(mdd), "sharpe": 0.0, "sortino": 0.0, "n_trades": 0.0, "win_rate": 0.0, "avg_trade": 0.0, "exposure": float(pos_days / max(1.0, float(n - 1)))}
    tr = np.asarray(trade_rets, dtype=np.float64)
    n_days_total = max(1.0, float(n - 1))
    avg_holding = n_days_total / max(1.0, float(len(tr)))
    annualization = float(np.sqrt(ONE_YEAR_DAYS / max(1.0, avg_holding)))
    sharpe_ann = float(np.mean(tr) / (np.std(tr) + 1e-12) * annualization)
    downside_tr = np.minimum(tr, 0.0)
    sortino_ann = float(np.mean(tr) / (np.std(downside_tr) + 1e-12) * annualization)
    return {
        "total_return": float(equity - 1.0),
        "mdd": float(mdd),
        "sharpe": sharpe_ann,
        "sortino": sortino_ann,
        "n_trades": float(len(tr)),
        "win_rate": float((tr > 0).mean()),
        "avg_trade": float(np.mean(tr)),
        "trade_std": float(np.std(tr)),
        "exposure": float(pos_days / max(1.0, n_days_total)),
        "n_days": n_days_total,
    }


def global_fitness_from_stats(per_ticker_stats: List[Dict[str, float]]) -> float:
    if len(per_ticker_stats) == 0:
        return -1e9
    sharpe = np.array([s.get("sharpe", 0.0) for s in per_ticker_stats], dtype=np.float64)
    ret = np.array([s.get("total_return", 0.0) for s in per_ticker_stats], dtype=np.float64)
    ntr = np.array([s.get("n_trades", 0.0) for s in per_ticker_stats], dtype=np.float64)
    med_sharpe = float(np.median(sharpe))
    k = max(1, int(np.ceil(0.10 * len(ret))))
    worst_10 = float(np.mean(np.sort(ret)[:k]))
    med_trades = float(np.median(ntr))
    cross_std = float(np.std(ret))
    if med_trades < 4:
        return -1e9
    return float(0.40 * med_sharpe + 0.25 * worst_10 - 0.20 * cross_std + 0.15 * np.log(max(med_trades, 1.0)))


def evaluate_global_genome(genome: List[float], ticker_payloads: Dict[str, Dict[str, np.ndarray]]) -> float:
    gp = decode_global_params(genome)
    stats = []
    for _, payload in ticker_payloads.items():
        st = backtest_stats_global_intraday(
            payload["open"], payload["high"], payload["low"], payload["close"],
            payload["score_matrix"], payload["atr"], gp, precomputed=payload,
        )
        stats.append(st)
    return global_fitness_from_stats(stats)


def evaluate_global_walkforward(genome: List[float], payloads_by_window: List[Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]]) -> Tuple[float, float, float]:
    # Returns (fitness, sharpe_train, sharpe_oos)
    train_vals = []
    oos_vals = []
    for tr_payloads, te_payloads in payloads_by_window:
        if tr_payloads and te_payloads:
            train_vals.append(evaluate_global_genome(genome, tr_payloads))
            oos_vals.append(evaluate_global_genome(genome, te_payloads))
            
    if not oos_vals:
        return -1e9, -1e9, -1e9
        
    mean_train = float(np.mean(train_vals))
    mean_oos = float(np.mean(oos_vals))
    
    # Penalize difference between train and test
    alpha = 0.5
    fitness = mean_oos - alpha * abs(mean_train - mean_oos)
    
    return fitness, mean_train, mean_oos


def run_global_ga_20params(full_df: pd.DataFrame, feature_cols: List[str], windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]], pop_size: int = 200, ngen: int = 60, ga_max_windows: int = 4):
    if not hasattr(creator, "Individual_Global20"):
        creator.create("Individual_Global20", list, fitness=creator.FitnessMax_PT)

    ga_t0 = time.perf_counter()
    windows_ga = windows[-int(ga_max_windows):] if (ga_max_windows is not None and int(ga_max_windows) > 0) else windows
    payloads_by_window = precompute_global_payloads(windows_ga, full_df, feature_cols)
    print(f"[GLOBAL_GA] config pop={int(pop_size)} ngen={int(ngen)} windows_total={len(windows)} windows_ga={len(windows_ga)} feats={len(feature_cols)}")

    toolbox = base.Toolbox()
    for i, (_, lo, hi, _, _) in enumerate(GLOBAL_PARAM_SPECS):
        toolbox.register(f"attr_g{i}", random.uniform, float(lo), float(hi))

    toolbox.register("individual", tools.initCycle, creator.Individual_Global20, tuple(getattr(toolbox, f"attr_g{i}") for i in range(20)), n=1)
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
    def _eval_for_deap(ind):
        fit, _, _ = evaluate_global_walkforward(list(ind), payloads_by_window)
        return (float(fit),)
    toolbox.register("evaluate", _eval_for_deap)

    GA_MAX_RUNTIME_SEC = 6 * 3600  # 6-hour runtime guard
    GA_CHECKPOINT_EVERY = 5  # save checkpoint every N generations

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(GA_HOF_SIZE)
    best_fit_seen = -1e18
    no_improve_count = 0

    for gen in range(1, int(ngen) + 1):
        # Runtime guard (Fase 5.2)
        wall_elapsed = time.perf_counter() - ga_t0
        if wall_elapsed > GA_MAX_RUNTIME_SEC:
            print(f"[GLOBAL_GA] runtime guard: stopping at gen {gen} after {wall_elapsed/3600.0:.1f}h")
            break

        gen_t0 = time.perf_counter()

        # Adaptive mutation (Fase 5.3): interpolate sigma/gene_pb across generations
        progress = float(gen - 1) / max(1.0, float(ngen - 1))
        cur_sigma = GA_MUT_SIGMA_START + (GA_MUT_SIGMA_END - GA_MUT_SIGMA_START) * progress
        cur_gene_pb = GA_GENE_MUT_PB_START + (GA_GENE_MUT_PB_END - GA_GENE_MUT_PB_START) * progress
        toolbox.register("mutate", _mut, sigma=cur_sigma, gene_pb=cur_gene_pb)

        invalid = [ind for ind in pop if not ind.fitness.valid]
        eval_t0 = time.perf_counter()
        inv_total = len(invalid)
        for inv_i, ind in enumerate(invalid, 1):
            ind.fitness.values = toolbox.evaluate(ind)
            if inv_total > 0 and ((inv_i == 1) or (inv_i == inv_total) or (inv_i % max(1, inv_total // 10) == 0)):
                inv_elapsed = time.perf_counter() - eval_t0
                inv_eta = (inv_elapsed / inv_i) * (inv_total - inv_i)
                print(f"[GLOBAL_GA] gen {gen:03d} eval {inv_i:4d}/{inv_total:4d} | elapsed={inv_elapsed:.1f}s | eta={inv_eta:.1f}s")
        hof.update(pop)

        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for i in range(1, len(offspring), 2):
            if random.random() < GA_CX_PB:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values
        for i in range(len(offspring)):
            if random.random() < GA_MUT_PB:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        invalid_off = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_off:
            ind.fitness.values = toolbox.evaluate(ind)
        hof.update(offspring)
        pop[:] = offspring

        prev_best = best_fit_seen
        if len(hof):
            best_fit_seen = max(best_fit_seen, float(hof[0].fitness.values[0]))
        gen_dt = time.perf_counter() - gen_t0

        # Early stopping check
        if best_fit_seen <= prev_best:
            no_improve_count += 1
        else:
            no_improve_count = 0
        if no_improve_count >= EARLY_STOP:
            print(f"[GLOBAL_GA] early stop at gen {gen} (no improvement for {EARLY_STOP} gens)")
            break

        if (gen == 1) or (gen == int(ngen)) or (gen % max(1, int(ngen // 10)) == 0):
            elapsed = time.perf_counter() - ga_t0
            eta = (elapsed / gen) * (int(ngen) - gen)
            print(f"[GLOBAL_GA] gen {gen:03d}/{int(ngen):03d} | best={best_fit_seen:.5f} | sigma={cur_sigma:.3f} | gen_time={gen_dt:.2f}s | elapsed={elapsed/60.0:.1f}m | eta={eta/60.0:.1f}m")

        # Per-generation checkpoint (Fase 5.1)
        if gen % GA_CHECKPOINT_EVERY == 0 and len(hof):
            try:
                ck_path = f"{OUTPUT_DIR}global_params_20__H{FWD_H}__v2.json"
                with open(ck_path, "w", encoding="utf-8") as f:
                    json.dump({"fitness": float(hof[0].fitness.values[0]), "genome": list(hof[0]), "generation": gen}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    best = list(hof[0]) if len(hof) else [s[1] for s in GLOBAL_PARAM_SPECS]
    total_dt = time.perf_counter() - ga_t0
    print(f"[GLOBAL_GA] finished in {total_dt/60.0:.1f}m | best_fit={(hof[0].fitness.values[0] if len(hof) else -1e9):.5f}")
    return decode_global_params(best), sanitize_global_genome(best), (hof[0].fitness.values[0] if len(hof) else -1e9)


# ==============================================================================
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

    # Distância da média longa (Close vs SMA200)
    if 'sma200' in df.columns:
        df['dist_sma200'] = (df[CLOSE_COL] - df['sma200']) / df['sma200'].replace(0, np.nan)
    else:
        df['dist_sma200'] = np.nan

    # Padrão de volume (volume / média de 20 dias)
    df['volume_pattern_20'] = df['rel_volume']

    # Regime de mercado proxy (inclinação da SMA200)
    if 'sma200_slope' in df.columns:
        df['regime_sma200_slope'] = df['sma200_slope']
        df['regime_bull'] = (df['sma200_slope'] > 0).astype(float)
    else:
        df['regime_sma200_slope'] = np.nan
        df['regime_bull'] = np.nan

    # --- Microstructure features (Fase 4.1) ---
    hl_range = (df[HIGH_COL] - df[LOW_COL]).replace(0, np.nan)

    # Intraday range normalizado (proxy for realized volatility)
    df['intraday_range'] = hl_range / df[CLOSE_COL].replace(0, np.nan)

    # Gap ratio (abertura vs fechamento anterior)
    prev_close = grouped[CLOSE_COL].shift(1)
    df['gap_ratio'] = (df[OPEN_COL] - prev_close) / prev_close.replace(0, np.nan)

    # Close position within range (0=low, 1=high; higher = bullish bar)
    df['close_position'] = ((df[CLOSE_COL] - df[LOW_COL]) / hl_range).clip(0, 1)

    # Upper/lower shadow ratio (candlestick microstructure)
    body_top = df[[CLOSE_COL, OPEN_COL]].max(axis=1)
    body_bot = df[[CLOSE_COL, OPEN_COL]].min(axis=1)
    df['upper_shadow'] = (df[HIGH_COL] - body_top) / hl_range
    df['lower_shadow'] = (body_bot - df[LOW_COL]) / hl_range

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

    # MA rule: relaxed — decent signals can override MA filter
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

    # slope rule: relaxed — decent signals bypass slope filter
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
    mdd_abs = abs(float(stats_1y["mdd"]))
    expo = float(stats_1y["exposure"])
    n_tr = float(stats_1y["n_trades"])
    sh = float(stats_1y.get("sharpe", 0.0))
    so = float(stats_1y.get("sortino", 0.0))
    wr = _finite_or_default(stats_1y.get("win_rate", 0.0), 0.0)
    n_days = float(stats_1y.get("n_days", ONE_YEAR_DAYS))

    min_trades_hard = max(8.0, 0.03 * max(n_days, 1.0))
    if n_tr < min_trades_hard:
        return -1e9

    # Cost-aware fitness: prioritize Sharpe/Sortino, heavily penalize MDD
    f = (0.60 * ret) - (1.50 * mdd_abs) + (2.50 * sh) + (1.50 * so)

    # Hard MDD penalty: any MDD > 20% is severely punished
    if mdd_abs > 0.25:
        f -= 10.0 * (mdd_abs - 0.25)
    elif mdd_abs > 0.20:
        f -= 5.0 * (mdd_abs - 0.20)

    years = max(1e-9, n_days / ONE_YEAR_DAYS)
    trades_per_year = n_tr / years

    # Low trade penalty: need enough trades for statistical significance
    if trades_per_year < 12.0:
        f -= (12.0 - trades_per_year) * 2.0

    # Penalizar overexposure / underexposure
    if expo > MAX_EXPOSURE_1Y:
        f -= (expo - MAX_EXPOSURE_1Y) * 2.5
    if expo < GA_MIN_EXPOSURE:
        f -= (GA_MIN_EXPOSURE - expo) * 1.5

    # Aggressive overtrading penalty (each excess trade costs ~0.68%)
    if trades_per_year > MAX_TRADES_PER_YEAR:
        f -= OVERTRADING_PENALTY_PER_TRADE * (trades_per_year - MAX_TRADES_PER_YEAR)

    # Bônus/penalidades secundárias de robustez
    f += max(0.0, wr - 0.5) * 0.20
    if (n_tr >= 20.0) and (wr < 0.45):
        f -= (0.45 - wr) * 0.50

    avg_tr = _finite_or_default(stats_1y.get("avg_trade", 0.0), 0.0)
    avg_tr_clip = float(np.clip(avg_tr, -0.03, 0.03))
    f += avg_tr_clip * 0.40

    return float(f)


def monte_carlo_consistency_metric(stats: Dict[str, float]) -> float:
    """Soft consistency metric for MC test (no hard trade-count floor)."""
    ret = float(stats.get("total_return", 0.0))
    mdd_abs = abs(float(stats.get("mdd", 0.0)))
    sh = float(stats.get("sharpe", 0.0))
    so = float(stats.get("sortino", 0.0))
    wr = float(stats.get("win_rate", 0.0))
    tr = float(stats.get("n_trades", 0.0))
    expo = float(stats.get("exposure", 0.0))
    return float((0.55 * ret) - (0.55 * mdd_abs) + (1.20 * sh) + (0.90 * so) + (0.20 * max(0.0, wr - 0.5)) + (0.01 * min(tr, 60.0)) - (0.50 * max(0.0, expo - MAX_EXPOSURE_1Y)))

def build_rule_entry_mask(feature_matrix: np.ndarray, p: Params) -> np.ndarray:
    """
    Lógica de Votação (Soft Voting Miner):
    O GA define 5 regras e o sinal dispara quando a maioria (>=3) é satisfeita.
    """
    fm = np.asarray(feature_matrix, dtype=np.float32)
    if fm.ndim != 2 or (not p.has_rule_genome):
        return np.zeros(len(fm), dtype=bool)

    votes = np.zeros(fm.shape[0], dtype=np.int8)
    for col_idx, op_code, thr in zip(p.rule_cols, p.rule_ops, p.rule_thrs):
        if col_idx < 0 or col_idx >= fm.shape[1]:
            continue
        col = fm[:, int(col_idx)]
        cond = (col > thr) if int(op_code) == 0 else (col < thr)
        votes += (np.isfinite(col) & cond).astype(np.int8)

    return votes >= 3


@njit
def simulate_intraday_from_mask_numba(o, h, l, c, atr, entry_mask, stop_mult, take_mult, long_only):
    n = len(c)
    if n < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    eq = 1.0
    eq_peak = 1.0
    max_dd = 0.0
    pos = 0
    entry = 0.0
    stop_abs = 0.0
    take_abs = 0.0
    pos_days = 0
    trades = 0
    wins = 0
    trade_sum = 0.0
    longs = 0
    shorts = 0
    trade_sum_long = 0.0
    trade_sum_short = 0.0

    for i in range(1, n):
        exited_this_bar = False

        # Check exits FIRST (before daily MTM) to avoid double-counting
        if pos != 0:
            oi = o[i]
            hi = h[i]
            li = l[i]
            exit_px = np.nan
            tr = 0.0
            if pos > 0:
                stop_px = entry - stop_abs
                take_px = entry + take_abs
                if oi <= stop_px:
                    exit_px = oi
                elif oi >= take_px:
                    exit_px = oi
                elif li <= stop_px:
                    exit_px = stop_px
                elif hi >= take_px:
                    exit_px = take_px
                if np.isfinite(exit_px):
                    tr = exit_px / max(entry, 1e-12) - 1.0
                    if tr > CAP_TRADE_RET:
                        tr = CAP_TRADE_RET
                    elif tr < -CAP_TRADE_RET:
                        tr = -CAP_TRADE_RET
            else:
                stop_px = entry + stop_abs
                take_px = entry - take_abs
                if oi >= stop_px:
                    exit_px = oi
                elif oi <= take_px:
                    exit_px = oi
                elif hi >= stop_px:
                    exit_px = stop_px
                elif li <= take_px:
                    exit_px = take_px
                if np.isfinite(exit_px):
                    tr = entry / max(exit_px, 1e-12) - 1.0
                    if tr > CAP_TRADE_RET:
                        tr = CAP_TRADE_RET
                    elif tr < -CAP_TRADE_RET:
                        tr = -CAP_TRADE_RET

            if np.isfinite(exit_px):
                net_tr = ((1.0 + tr) * ((1.0 - (COST_BPS + SLIPPAGE_BPS)/10000.0)**2) - 1.0) - COST_PER_TRADE_PCT
                eq *= (1.0 + net_tr)
                trades += 1
                trade_sum += net_tr
                if net_tr > 0.0:
                    wins += 1
                if pos > 0:
                    longs += 1
                    trade_sum_long += net_tr
                else:
                    shorts += 1
                    trade_sum_short += net_tr
                pos = 0
                entry = 0.0
                exited_this_bar = True

        # Apply daily MTM only if still in position (no exit this bar)
        if (not exited_this_bar) and pos != 0:
            c0 = c[i-1]
            c1 = c[i]
            if c0 > MIN_PRICE and c1 > MIN_PRICE:
                daily = c1 / c0 - 1.0
                if daily > CAP_DAILY_RET:
                    daily = CAP_DAILY_RET
                elif daily < -CAP_DAILY_RET:
                    daily = -CAP_DAILY_RET
                eq *= (1.0 + pos * daily)
                pos_days += 1

        if eq > eq_peak:
            eq_peak = eq
        dd = (eq / eq_peak) - 1.0
        if dd < max_dd:
            max_dd = dd

        if pos == 0 and entry_mask[i]:
            atr_i = atr[i]
            if np.isfinite(atr_i) and atr_i > ATR_EPS:
                pos = 1
                entry = o[i]
                stop_abs = stop_mult * atr_i
                take_abs = take_mult * stop_abs

    total_ret = eq - 1.0
    expo = pos_days / max(1.0, float(n - 1))
    win_rate = (wins / trades) if trades > 0 else 0.0
    avg_trade = (trade_sum / trades) if trades > 0 else 0.0
    ret_long = (trade_sum_long / longs) if longs > 0 else 0.0
    ret_short = (trade_sum_short / shorts) if shorts > 0 else 0.0
    return total_ret, max_dd, float(trades), expo, float(win_rate), float(avg_trade), float(longs), float(shorts), float(ret_long), float(ret_short)



def ema_np(x: np.ndarray, span: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if len(arr) == 0:
        return out
    alpha = 2.0 / (float(max(1, span)) + 1.0)
    prev = np.nan
    for i, v in enumerate(arr):
        if not np.isfinite(v):
            out[i] = prev if np.isfinite(prev) else np.nan
            continue
        if not np.isfinite(prev):
            prev = float(v)
        else:
            prev = alpha * float(v) + (1.0 - alpha) * prev
        out[i] = prev
    return out

def ga_vote_strengths_from_row(d_row: np.ndarray, p: Params) -> Tuple[float, float, float]:
    """Return (buy_strength, sell_strength, net). Supports legacy vote mode and rule genome mode."""
    if d_row is None:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(d_row, dtype=np.float64)
    if arr.ndim != 1 or (np.isfinite(arr).sum() == 0):
        return float("nan"), float("nan"), float("nan")

    if p.has_rule_genome:
        sat = 0.0
        n_rules = len(p.rule_cols)
        for col_idx, op_code, thr in zip(p.rule_cols, p.rule_ops, p.rule_thrs):
            if int(col_idx) < 0 or int(col_idx) >= len(arr) or (not np.isfinite(arr[int(col_idx)])):
                continue
            cond = (arr[int(col_idx)] > float(thr)) if int(op_code) == 0 else (arr[int(col_idx)] < float(thr))
            sat += 1.0 if cond else 0.0
        buy_strength = sat / max(1.0, float(n_rules))
        sell_strength = 1.0 - buy_strength
        return float(np.clip(buy_strength, 0.0, 1.0)), float(np.clip(sell_strength, 0.0, 1.0)), float(2.0 * buy_strength - 1.0)

    vlong = float(np.nanmean((arr > 0.35).astype(np.float64)))
    vshort = float(np.nanmean((arr < -0.35).astype(np.float64)))
    enter_vote_frac = float(np.clip(float(p.fast_period) / max(float(FAST_PERIOD_RANGE[1]), 1.0), 0.15, 0.95))
    buy_strength = float(np.clip((vlong - enter_vote_frac) / max(1e-9, (1.0 - enter_vote_frac)), 0.0, 1.0))
    sell_strength = float(np.clip((vshort - enter_vote_frac) / max(1e-9, (1.0 - enter_vote_frac)), 0.0, 1.0))
    net = float(vlong - vshort)
    return buy_strength, sell_strength, net


def score_0_100_from_ga_votes(buy_strength: float, sell_strength: float, quality: float) -> float:
    if (not np.isfinite(buy_strength)) or (not np.isfinite(sell_strength)):
        return 50.0
    net = float(np.clip(buy_strength - sell_strength, -1.0, 1.0))
    q = float(np.clip(quality, 0.0, 1.0))
    centered = 50.0 + (50.0 * net)
    return float(np.clip(50.0 + (centered - 50.0) * (0.35 + 0.65 * q), 0.0, 100.0))


def make_signal_eod(score_ev_series: np.ndarray, i: int, p: Params, close_eod: float, sma200_eod: float, slope_eod: float, precomputed_fast_ema: np.ndarray = None) -> str:
    """
    Signal decided at end of day i (to be acted on day i+1) using EMA crossover.
    Pass precomputed_fast_ema to avoid O(n^2) recomputation.
    """
    if (i <= 0) or (i >= len(score_ev_series)):
        return "hold"
    ev_i = float(score_ev_series[i]) if np.isfinite(score_ev_series[i]) else np.nan
    if (not np.isfinite(ev_i)) or (not np.isfinite(close_eod)):
        return "hold"

    if precomputed_fast_ema is not None:
        fast = precomputed_fast_ema
    else:
        fast = ema_np(score_ev_series[:i+1], int(p.fast_period))
    if i >= len(fast) or (not np.isfinite(fast[i])) or (not np.isfinite(fast[i-1])):
        return "hold"
    entry_up = (fast[i-1] <= ENTRY_SCORE_TRIGGER_ABS) and (fast[i] > ENTRY_SCORE_TRIGGER_ABS)
    entry_dn = (fast[i-1] >= -ENTRY_SCORE_TRIGGER_ABS) and (fast[i] < -ENTRY_SCORE_TRIGGER_ABS)

    if entry_up and (ev_i > ENTRY_SCORE_TRIGGER_ABS) and regime_ok(+1, close_eod, sma200_eod, slope_eod, ev_i):
        return "buy"
    if (not LONG_ONLY) and entry_dn and (ev_i < -ENTRY_SCORE_TRIGGER_ABS) and regime_ok(-1, close_eod, sma200_eod, slope_eod, ev_i):
        return "sell"
    return "hold"



def make_signal_eod_global(score_matrix: np.ndarray, score_ev_series: np.ndarray, atr_series: np.ndarray, close_series: np.ndarray, i: int, gp: GlobalParams) -> str:
    if (i <= 0) or (i >= len(score_ev_series)):
        return "hold"
    x = np.asarray(score_matrix, dtype=np.float64)
    if x.ndim != 2 or i >= x.shape[0]:
        return "hold"
    feat_n = max(1, x.shape[1])
    row = x[i]
    vote_long = float(np.sum(row > gp.z_threshold)) / feat_n
    vote_short = float(np.sum(row < -gp.z_threshold)) / feat_n
    score_ev_i = float(score_ev_series[i]) if np.isfinite(score_ev_series[i]) else np.nan
    if not np.isfinite(score_ev_i):
        return "hold"

    lookback = max(63, SCORE_LOOKBACK)
    recent = np.asarray(score_ev_series[max(0, i - lookback + 1):i + 1], dtype=np.float64)
    recent = recent[np.isfinite(recent)]
    if len(recent) < 20:
        return "hold"
    score_thr = float(np.quantile(recent, gp.score_percentile_trigger))

    long_ok = (vote_long >= gp.vote_threshold_long) and (score_ev_i >= score_thr)
    short_ok = (vote_short >= gp.vote_threshold_short) and (-score_ev_i >= score_thr)

    if gp.ma_filter_mode in (1, 2):
        ma = pd.Series(np.asarray(close_series, dtype=np.float64)).rolling(int(gp.ma_filter_period), min_periods=max(20, int(gp.ma_filter_period // 2))).mean().to_numpy()
        ma_i = ma[i]
        close_i = float(close_series[i]) if np.isfinite(close_series[i]) else np.nan
        if np.isfinite(ma_i) and np.isfinite(close_i):
            if gp.ma_filter_mode == 1:
                if close_i < ma_i:
                    long_ok = long_ok and (score_ev_i * 0.5 >= score_thr)
                if close_i > ma_i:
                    short_ok = short_ok and (-score_ev_i * 0.5 >= score_thr)
            elif gp.ma_filter_mode == 2:
                if close_i < ma_i:
                    long_ok = False
                if close_i > ma_i:
                    short_ok = False

    if long_ok:
        return "buy"
    if (not LONG_ONLY) and short_ok:
        return "sell"
    return "hold"

def score_0_100_from_ev(
    score_ev: float,
    recent_scores_ev: np.ndarray,
    quality: float,
) -> float:
    if (not np.isfinite(score_ev)):
        return 50.0
    recent = recent_scores_ev[np.isfinite(recent_scores_ev)] if recent_scores_ev is not None else np.array([], dtype=np.float64)
    if len(recent) == 0:
        pct_rank = 0.5
    else:
        pct_rank = float(np.mean(recent <= score_ev))
    base_score = 1.0 + 98.0 * pct_rank
    tilt_factor = 0.35 + 0.65 * float(np.clip(quality, 0.0, 1.0))
    score = 50.0 + tilt_factor * (base_score - 50.0)
    return float(np.clip(score, 0.0, 100.0))

def compute_quality_factor(test_sharpe: float, test_return: float, trades_1y: float) -> float:
    q_sh = _sigmoid((float(test_sharpe) - 0.10) / 0.30) if np.isfinite(test_sharpe) else 0.5
    q_ret = _sigmoid(float(test_return) / 0.15) if np.isfinite(test_return) else 0.5
    q_tr = _sigmoid((float(trades_1y) - 8.0) / 4.0) if np.isfinite(trades_1y) else 0.3
    return float(np.clip(0.45 * q_sh + 0.35 * q_ret + 0.20 * q_tr, 0.0, 1.0))

def compute_wf_quality(wf_auc_mean: float, wf_ap_mean: float, wf_auc_std: float) -> float:
    q_auc = np.clip((float(wf_auc_mean) - 0.50) / 0.18, 0.0, 1.0) if np.isfinite(wf_auc_mean) else 0.0
    q_ap = np.clip((float(wf_ap_mean) - 0.50) / 0.20, 0.0, 1.0) if np.isfinite(wf_ap_mean) else 0.0
    q_stab = 1.0 - np.clip(float(wf_auc_std) / 0.12, 0.0, 1.0) if np.isfinite(wf_auc_std) else 0.0
    return float(np.clip(0.45 * q_auc + 0.35 * q_ap + 0.20 * q_stab, 0.0, 1.0))

def adjust_params_by_wf_quality(p: Params, wf_quality: float) -> Params:
    p = sanitize_params(p)
    q = float(np.clip(wf_quality, 0.0, 1.0))
    # weaker models ask better prices and slightly smoother crossover windows
    entry_discount = p.entry_discount * (1.0 + (1.0 - q) * 0.60)
    fast_period = int(round(p.fast_period + (1.0 - q) * 1.5))
    slow_period = int(round(p.slow_period + (1.0 - q) * 4.0))
    return sanitize_params(Params(float(fast_period), float(slow_period), p.atr_mult, p.rr_mult, entry_discount))

def compute_model_entry_price(o1: float, atr1: float, score_ev_eod: float, enter_abs: float, side: int, entry_discount: float) -> float:
    if (not np.isfinite(o1)) or (not np.isfinite(atr1)) or atr1 <= ATR_EPS:
        return float("nan")
    if (not np.isfinite(score_ev_eod)) or (not np.isfinite(enter_abs)) or enter_abs <= 0:
        return float("nan")

    strength_ratio = abs(float(score_ev_eod)) / float(enter_abs)
    strength_ratio = max(strength_ratio, 0.5)
    inv_strength = float(np.clip(1.0 / strength_ratio, 0.3, 2.0))
    discount_atr = float(entry_discount * inv_strength * atr1)
    return float(o1 - discount_atr) if side > 0 else float(o1 + discount_atr)

def compute_levels_from_atr(entry_price: float, atr_val: float, p: Params):
    """
    Returns: stop_abs, take_abs, stop_pct, take_pct, buy_entry, buy_stop, buy_take, sell_entry, sell_stop, sell_take
    """
    if (not np.isfinite(entry_price)) or (entry_price <= 0) or (not np.isfinite(atr_val)) or (atr_val <= ATR_EPS):
        return (np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan)

    p = sanitize_params(p)
    stop_abs = float(p.atr_mult * atr_val)
    take_abs = float(p.rr_mult  * stop_abs)

    stop_pct = float(stop_abs / max(entry_price, 1e-12))
    take_pct = float(take_abs / max(entry_price, 1e-12))

    buy_entry = float(entry_price)
    buy_stop  = float(entry_price - stop_abs)
    buy_take  = float(entry_price + take_abs)

    sell_entry = float(entry_price)
    sell_stop  = float(entry_price + stop_abs)
    sell_take  = float(entry_price - take_abs)

    return (stop_abs, take_abs, stop_pct, take_pct,
            buy_entry, buy_stop, buy_take,
            sell_entry, sell_stop, sell_take)

def nextday_limit_fill(o1: float, h1: float, l1: float, atr1: float, score_ev_eod: float, enter_abs: float, side: int, entry_discount: float) -> Tuple[bool, float, float]:
    if (not np.isfinite(o1)) or (not np.isfinite(h1)) or (not np.isfinite(l1)) or (not np.isfinite(atr1)) or atr1 <= ATR_EPS:
        return (False, np.nan, np.nan)

    limit = compute_model_entry_price(o1, atr1, score_ev_eod, enter_abs, side, entry_discount)
    if not np.isfinite(limit):
        return (False, np.nan, np.nan)

    if side > 0:
        if float(o1) <= limit:
            return (True, float(o1), float(limit))
        if float(l1) <= limit <= float(h1):
            return (True, float(limit), float(limit))
        return (False, np.nan, float(limit))
    else:
        if float(o1) >= limit:
            return (True, float(o1), float(limit))
        if float(l1) <= limit <= float(h1):
            return (True, float(limit), float(limit))
        return (False, np.nan, float(limit))

def make_recency_weights(n: int, half_life: int) -> np.ndarray:
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    lam = math.log(2.0) / max(1.0, float(half_life))
    idx = np.arange(n, dtype=np.float64)
    w = np.exp(-lam * ((n - 1) - idx))
    w = w / max(1e-12, float(np.mean(w)))
    return w.astype(np.float64)

def probability_to_direction_sign(p: np.ndarray) -> np.ndarray:
    # compress low-confidence probs around 0.5 and keep high-confidence tails
    x = np.asarray(p, dtype=np.float64) - 0.5
    sign = np.tanh(PROB_DIRECTION_SCALE * x)
    return np.clip(sign, -1.0, 1.0)

def _safe_pr_auc(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true_bin, y_prob))
    except Exception:
        return float("nan")


# ==============================================================================
# 4) ML (Walk-Forward OOS probabilities)
# ==============================================================================

PURGE_GAP_BARS = max(FWD_H, 10)  # purge gap to prevent label leakage between folds

def purged_time_series_split(n_samples: int, n_splits: int, purge_gap: int = PURGE_GAP_BARS):
    """Time-series split with purge gap between train and test to prevent label leakage."""
    fold_size = n_samples // (n_splits + 1)
    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        test_start = train_end + purge_gap
        test_end = test_start + fold_size
        if test_end > n_samples:
            break
        yield np.arange(train_end), np.arange(test_start, test_end)

def get_clean_walk_forward_predictions(
    X: np.ndarray,
    y_bin: np.ndarray,
    y_mag: np.ndarray,
    w: np.ndarray,
    dates: np.ndarray,
    ticker: str,
    n_splits: int = WF_SPLITS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:

    oos_prob = np.full(len(y_bin), np.nan, dtype=np.float64)
    oos_mag = np.full(len(y_bin), np.nan, dtype=np.float64)

    aucs, accs, briers, loglosses, ap_scores = [], [], [], [], []
    fold_ranges = []

    for fold_i, (train_idx, test_idx) in enumerate(purged_time_series_split(len(X), n_splits, PURGE_GAP_BARS), start=1):
        X_tr, y_tr = X[train_idx], y_bin[train_idx]
        X_te, y_te = X[test_idx], y_bin[test_idx]
        mag_tr = y_mag[train_idx]
        w_tr = w[train_idx] if w is not None else None

        # Class weight balancing (Fase 4.3) — upweight minority class
        cw = compute_sample_weight("balanced", y_tr)
        sw_tr = (w_tr * cw) if w_tr is not None else cw

        # Model 1: HistGradientBoosting (primary)
        clf_hgb = HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=400,
            max_depth=4,
            min_samples_leaf=25,
            max_leaf_nodes=31,
            l2_regularization=2.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=RANDOM_SEED,
        )
        # Model 2: ExtraTrees (diversity, Fase 4.2)
        clf_et = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        reg = HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_iter=400,
            max_depth=4,
            min_samples_leaf=25,
            max_leaf_nodes=31,
            l2_regularization=2.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=RANDOM_SEED,
        )
        clf_hgb.fit(X_tr, y_tr, sample_weight=sw_tr)
        clf_et.fit(X_tr, y_tr, sample_weight=sw_tr)
        reg.fit(X_tr, mag_tr, sample_weight=w_tr)

        # Ensemble blend: 60% HGB + 40% ExtraTrees
        p_hgb = clf_hgb.predict_proba(X_te)[:, 1].astype(np.float64)
        p_et = clf_et.predict_proba(X_te)[:, 1].astype(np.float64)
        p_te = 0.60 * p_hgb + 0.40 * p_et
        m_te = np.clip(reg.predict(X_te).astype(np.float64), 0.0, None)
        oos_prob[test_idx] = p_te
        oos_mag[test_idx] = m_te

        train_start = pd.to_datetime(dates[train_idx[0]]).date()
        train_end = pd.to_datetime(dates[train_idx[-1]]).date()
        test_start = pd.to_datetime(dates[test_idx[0]]).date()
        test_end = pd.to_datetime(dates[test_idx[-1]]).date()
        fold_msg = f"[{ticker}] Fold {fold_i}/{n_splits}: TRAIN ({len(train_idx)} rows) | TEST ({len(test_idx)} rows)"
        if PRINT_FOLD_DETAILS:
            print("  " + fold_msg)
        fold_ranges.append(f"{train_start}|{train_end}|{test_start}|{test_end}")

        try:
            auc = roc_auc_score(y_te, p_te)
        except Exception:
            auc = 0.5
        yhat = (p_te >= 0.5).astype(int)
        acc = accuracy_score(y_te, yhat)
        try:
            ll = log_loss(y_te, np.clip(p_te, 1e-6, 1-1e-6))
        except Exception:
            ll = float("nan")
        try:
            br = brier_score_loss(y_te, p_te)
        except Exception:
            br = float("nan")
        ap = _safe_pr_auc(y_te, p_te)

        aucs.append(float(auc)); accs.append(float(acc)); loglosses.append(float(ll)); briers.append(float(br)); ap_scores.append(float(ap) if np.isfinite(ap) else float("nan"))

    direction_sign = probability_to_direction_sign(oos_prob)
    oos_ev = direction_sign * oos_mag

    metrics = {
        "wf_auc_mean": float(np.nanmean(aucs)) if len(aucs) else float("nan"),
        "wf_auc_std":  float(np.nanstd(aucs)) if len(aucs) else float("nan"),
        "wf_acc_mean": float(np.nanmean(accs)) if len(accs) else float("nan"),
        "wf_logloss":  float(np.nanmean(loglosses)) if len(loglosses) else float("nan"),
        "wf_brier":    float(np.nanmean(briers)) if len(briers) else float("nan"),
        "wf_ap_mean":  float(np.nanmean(ap_scores)) if len(ap_scores) else float("nan"),
        "wf_n_folds":  int(len(aucs)),
        "wf_fold_ranges": ';'.join(fold_ranges),
    }
    return oos_ev, oos_prob, oos_mag, metrics


def predict_tail_rows(
    X_train: np.ndarray,
    y_bin_train: np.ndarray,
    y_mag_train: np.ndarray,
    w_train: np.ndarray,
    X_tail: np.ndarray,
    ticker: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_tail = len(X_tail)
    if n_tail == 0 or len(X_train) < ML_MIN_TRAIN:
        return (np.full(n_tail, np.nan), np.full(n_tail, np.nan), np.full(n_tail, np.nan))

    if PRINT_TAIL_DETAILS:
        print(f"  [{ticker}] Final model: training on {len(X_train)} rows, predicting {n_tail} tail rows")

    cw = compute_sample_weight("balanced", y_bin_train)
    sw = (w_train * cw) if w_train is not None else cw

    clf_hgb = HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=400, max_depth=4,
        min_samples_leaf=25, max_leaf_nodes=31,
        l2_regularization=2.0, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=15,
        random_state=RANDOM_SEED,
    )
    clf_et = ExtraTreesClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        max_features="sqrt", random_state=RANDOM_SEED, n_jobs=1,
    )
    clf_hgb.fit(X_train, y_bin_train, sample_weight=sw)
    clf_et.fit(X_train, y_bin_train, sample_weight=sw)
    p_hgb = clf_hgb.predict_proba(X_tail)[:, 1].astype(np.float64)
    p_et = clf_et.predict_proba(X_tail)[:, 1].astype(np.float64)
    p_tail = 0.60 * p_hgb + 0.40 * p_et

    reg = HistGradientBoostingRegressor(
        learning_rate=0.03, max_iter=400, max_depth=4,
        min_samples_leaf=25, max_leaf_nodes=31,
        l2_regularization=2.0, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=15,
        random_state=RANDOM_SEED,
    )
    reg.fit(X_train, y_mag_train, sample_weight=w_train)
    mag_tail = np.clip(reg.predict(X_tail), 0.0, None)

    direction_sign = probability_to_direction_sign(p_tail)
    ev_tail = direction_sign * mag_tail
    return ev_tail, p_tail, mag_tail


# ==============================================================================
# 5) BACKTEST (Intraday OHLC-aware)
# ==============================================================================


def backtest_stats_only_intraday(
    o, h, l, c, score_ev, sma200, sma_slope, atr, p: Params,
    return_trades: bool = False,
    dates: np.ndarray = None,
) -> Dict[str, float]:
    """
    Trade enters on day i using EMA-crossover signal computed from score_ev up to i-1.
    If return_trades=True, includes detailed trade logs in key "trades".
    """
    p = sanitize_params(p)
    n = len(c)
    if n < 3:
        base = {"total_return":0.0,"mdd":0.0,"sharpe":0.0,"n_trades":0.0,"win_rate":0.0,"avg_trade":0.0,"exposure":0.0}
        if return_trades:
            base["trades"] = []
        return base

    score_arr = np.asarray(score_ev, dtype=np.float64)
    use_column_thresholds = (score_arr.ndim == 2)

    if use_column_thresholds and p.has_rule_genome:
        feature_mat = np.asarray(score_ev, dtype=np.float32)
        entry_mask = build_rule_entry_mask(feature_mat, p)
        ret, mdd, n_trades, expo, wr, avg_tr, n_longs, n_shorts, ret_long, ret_short = simulate_intraday_from_mask_numba(
            np.asarray(o, dtype=np.float64), np.asarray(h, dtype=np.float64), np.asarray(l, dtype=np.float64),
            np.asarray(c, dtype=np.float64), np.asarray(atr, dtype=np.float64), entry_mask,
            float(p.atr_mult), float(p.rr_mult), bool(LONG_ONLY)
        )
        # Sharpe from Numba returns: use return/mdd as proxy since we don't have per-trade details
        sharpe = float(ret / (abs(mdd) + 1e-9))
        base = {
            "total_return": float(ret), "mdd": float(mdd), "sharpe": sharpe, "sortino": 0.0,
            "n_trades": float(n_trades), "n_longs": float(n_longs), "n_shorts": float(n_shorts), "win_rate": float(wr), "avg_trade": float(avg_tr), "ret_long": float(ret_long), "ret_short": float(ret_short), "exposure": float(expo),
            "max_fav_pct": float("nan"), "dd_duration": float("nan"), "regime_filtered_pct": 0.0, "n_days": float(n),
        }
        if return_trades:
            base["trades"] = []
        return base

    if use_column_thresholds:
        directed_cols = score_arr
        score_fast = None
        score_slow = None
        enter_vote_frac = float(np.clip(float(p.fast_period) / max(float(FAST_PERIOD_RANGE[1]), 1.0), 0.15, 0.95))
        exit_vote_frac = float(np.clip(float(p.slow_period) / max(float(SLOW_PERIOD_RANGE[1]), 1.0), 0.10, 0.95))
        valid_counts = np.isfinite(directed_cols).sum(axis=1)
        vote_long = np.where(valid_counts > 0, np.nanmean((directed_cols > 0.35).astype(np.float64), axis=1), np.nan)
        vote_short = np.where(valid_counts > 0, np.nanmean((directed_cols < -0.35).astype(np.float64), axis=1), np.nan)
        ev = vote_long - vote_short
    else:
        directed_cols = None
        score_fast = ema_np(score_arr, int(p.fast_period))
        score_slow = ema_np(score_arr, int(p.slow_period))
        valid_counts = None
        vote_long = None
        vote_short = None
        ev = None

    cost_leg = (1.0 - (COST_BPS + SLIPPAGE_BPS)/10000.0)
    log_cost = math.log(max(cost_leg, 1e-12))

    log_eq   = np.zeros(n, dtype=np.float64)
    log_rets = np.zeros(n, dtype=np.float64)

    pos = 0
    entry_price = 0.0
    stop_abs = 0.0
    take_abs = 0.0

    n_trades = 0
    n_wins = 0
    trade_sum = 0.0
    trade_sum_long = 0.0
    trade_sum_short = 0.0
    n_longs = 0
    n_shorts = 0
    trade_rets = []
    trade_max_favs = []
    pos_days = 0
    bars_in_pos = 0
    cur_max_fav = 0.0
    regime_checks_n = 0
    regime_filtered_n = 0

    trade_logs: List[Dict[str, float]] = []
    cur_trade = None

    def _date_at(idx: int) -> str:
        if dates is None:
            return str(idx)
        try:
            return str(pd.to_datetime(dates[idx]).date())
        except Exception:
            return str(idx)

    for i in range(1, n):
        o1 = float(o[i]); h1 = float(h[i]); l1 = float(l[i]); c1 = float(c[i])
        c0 = float(c[i-1])

        if c0 <= MIN_PRICE or c1 <= MIN_PRICE:
            log_eq[i] = log_eq[i-1]
            continue

        daily = (c1/c0) - 1.0
        daily = float(np.clip(daily, -CAP_DAILY_RET, CAP_DAILY_RET))
        step_ret = (pos * daily) if pos != 0 else 0.0
        log_eq[i] = log_eq[i-1] + math.log1p(step_ret)
        log_rets[i] = log_eq[i] - log_eq[i-1]
        if pos != 0:
            pos_days += 1
            bars_in_pos += 1

        if use_column_thresholds:
            vote_prev_long = float(vote_long[i-1]) if np.isfinite(vote_long[i-1]) else np.nan
            vote_prev_short = float(vote_short[i-1]) if np.isfinite(vote_short[i-1]) else np.nan
            vote_now_long = float(vote_long[i]) if np.isfinite(vote_long[i]) else np.nan
            vote_now_short = float(vote_short[i]) if np.isfinite(vote_short[i]) else np.nan
            f0 = float(ev[i-1]) if np.isfinite(ev[i-1]) else np.nan
            s0 = 0.0
            f1 = float(ev[i]) if np.isfinite(ev[i]) else np.nan
            s1 = 0.0
            cross_dn_now = np.isfinite(vote_now_short) and np.isfinite(vote_now_long) and (vote_now_short >= exit_vote_frac) and (vote_now_short > vote_now_long)
            cross_up_now = np.isfinite(vote_now_long) and np.isfinite(vote_now_short) and (vote_now_long >= exit_vote_frac) and (vote_now_long > vote_now_short)
        else:
            f0 = float(score_fast[i-1]) if np.isfinite(score_fast[i-1]) else np.nan
            s0 = float(score_slow[i-1]) if np.isfinite(score_slow[i-1]) else np.nan
            f1 = float(score_fast[i]) if np.isfinite(score_fast[i]) else np.nan
            s1 = float(score_slow[i]) if np.isfinite(score_slow[i]) else np.nan
            cross_dn_now = np.isfinite(f0) and np.isfinite(s0) and np.isfinite(f1) and np.isfinite(s1) and (f0 >= s0) and (f1 < s1)
            cross_up_now = np.isfinite(f0) and np.isfinite(s0) and np.isfinite(f1) and np.isfinite(s1) and (f0 <= s0) and (f1 > s1)

        if pos > 0:
            cur_max_fav = max(cur_max_fav, float((h1 / max(entry_price, 1e-12)) - 1.0), float((o1 / max(entry_price, 1e-12)) - 1.0))
        elif pos < 0:
            cur_max_fav = max(cur_max_fav, float((entry_price / max(l1, 1e-12)) - 1.0), float((entry_price / max(o1, 1e-12)) - 1.0))

        if pos != 0:
            exited = False
            exit_px = np.nan
            exit_reason = ""

            if pos > 0:
                stop_px = entry_price - stop_abs
                take_px = entry_price + take_abs

                if cross_dn_now:
                    exited, exit_px, exit_reason = True, c1, "cross_dn"
                elif (bars_in_pos >= TIME_STOP_BARS) and (c1 <= entry_price):
                    atr_i = float(atr[i]) if np.isfinite(atr[i]) else np.nan
                    fs, fill_s, _ = nextday_limit_fill(o1, h1, l1, atr_i, -SCORE_CROSS_MIN_ABS, SCORE_CROSS_MIN_ABS, -1, p.entry_discount)
                    exit_candidate = float(fill_s) if fs else c1
                    exited, exit_px, exit_reason = True, exit_candidate, "time_stop"
                elif o1 <= stop_px:
                    exited, exit_px, exit_reason = True, o1, "gap_stop"
                elif o1 >= take_px:
                    exited, exit_px, exit_reason = True, o1, "gap_take"
                else:
                    hit_stop = (l1 <= stop_px)
                    hit_take = (h1 >= take_px)
                    if hit_stop and hit_take:
                        exited, exit_px, exit_reason = True, stop_px, "stop_and_take_same_bar"
                    elif hit_stop:
                        exited, exit_px, exit_reason = True, stop_px, "stop_hit"
                    elif hit_take:
                        exited, exit_px, exit_reason = True, take_px, "take_hit"

                if not exited and REQUIRE_MA_FOR_ENTRY:
                    ma_ = float(sma200[i])
                    if np.isfinite(ma_) and (c1 < ma_):
                        exited, exit_px, exit_reason = True, c1, "ma_filter_exit"

                if exited:
                    trade_ret = (exit_px / max(entry_price, 1e-12)) - 1.0
                    trade_ret = float(np.clip(trade_ret, -CAP_TRADE_RET, CAP_TRADE_RET))
                    log_eq[i] += log_cost
                    log_rets[i] = log_eq[i] - log_eq[i-1]
                    net_trade = (1.0 + trade_ret) * (cost_leg**2) - 1.0
                    net_trade -= COST_PER_TRADE_PCT
                    n_trades += 1
                    n_longs += 1
                    if net_trade > 0:
                        n_wins += 1
                    trade_sum += net_trade
                    trade_sum_long += net_trade
                    trade_rets.append(float(net_trade))
                    trade_max_favs.append(float(max(0.0, cur_max_fav)))
                    if return_trades and cur_trade is not None:
                        cur_trade.update({
                            "exit_idx": int(i),
                            "exit_date": _date_at(i),
                            "exit_price": float(exit_px),
                            "exit_reason": exit_reason,
                            "gross_ret": float(trade_ret),
                            "net_ret": float(net_trade),
                            "bars_held": int(bars_in_pos),
                            "max_fav_pct": float(max(0.0, cur_max_fav)),
                        })
                        trade_logs.append(cur_trade)
                        cur_trade = None
                    pos = 0
                    entry_price = 0.0
                    stop_abs = 0.0
                    take_abs = 0.0
                    bars_in_pos = 0
                    cur_max_fav = 0.0
                    continue
            else:
                stop_px = entry_price + stop_abs
                take_px = entry_price - take_abs

                if cross_up_now:
                    exited, exit_px, exit_reason = True, c1, "cross_up"
                elif (bars_in_pos >= TIME_STOP_BARS) and (c1 >= entry_price):
                    atr_i = float(atr[i]) if np.isfinite(atr[i]) else np.nan
                    fb, fill_b, _ = nextday_limit_fill(o1, h1, l1, atr_i, SCORE_CROSS_MIN_ABS, SCORE_CROSS_MIN_ABS, +1, p.entry_discount)
                    exit_candidate = float(fill_b) if fb else c1
                    exited, exit_px, exit_reason = True, exit_candidate, "time_stop"
                elif o1 >= stop_px:
                    exited, exit_px, exit_reason = True, o1, "gap_stop"
                elif o1 <= take_px:
                    exited, exit_px, exit_reason = True, o1, "gap_take"
                else:
                    hit_stop = (h1 >= stop_px)
                    hit_take = (l1 <= take_px)
                    if hit_stop and hit_take:
                        exited, exit_px, exit_reason = True, stop_px, "stop_and_take_same_bar"
                    elif hit_stop:
                        exited, exit_px, exit_reason = True, stop_px, "stop_hit"
                    elif hit_take:
                        exited, exit_px, exit_reason = True, take_px, "take_hit"

                if not exited and REQUIRE_MA_FOR_ENTRY:
                    ma_ = float(sma200[i])
                    if np.isfinite(ma_) and (c1 > ma_):
                        exited, exit_px, exit_reason = True, c1, "ma_filter_exit"

                if exited:
                    trade_ret = (entry_price / max(exit_px, 1e-12)) - 1.0
                    trade_ret = float(np.clip(trade_ret, -CAP_TRADE_RET, CAP_TRADE_RET))
                    log_eq[i] += log_cost
                    log_rets[i] = log_eq[i] - log_eq[i-1]
                    net_trade = (1.0 + trade_ret) * (cost_leg**2) - 1.0
                    net_trade -= COST_PER_TRADE_PCT
                    n_trades += 1
                    n_shorts += 1
                    if net_trade > 0:
                        n_wins += 1
                    trade_sum += net_trade
                    trade_sum_short += net_trade
                    trade_rets.append(float(net_trade))
                    trade_max_favs.append(float(max(0.0, cur_max_fav)))
                    if return_trades and cur_trade is not None:
                        cur_trade.update({
                            "exit_idx": int(i),
                            "exit_date": _date_at(i),
                            "exit_price": float(exit_px),
                            "exit_reason": exit_reason,
                            "gross_ret": float(trade_ret),
                            "net_ret": float(net_trade),
                            "bars_held": int(bars_in_pos),
                            "max_fav_pct": float(max(0.0, cur_max_fav)),
                        })
                        trade_logs.append(cur_trade)
                        cur_trade = None
                    pos = 0
                    entry_price = 0.0
                    stop_abs = 0.0
                    take_abs = 0.0
                    bars_in_pos = 0
                    cur_max_fav = 0.0
                    continue

        if pos == 0 and i >= 2:
            sig = 0
            entry_trigger = ""
            if use_column_thresholds:
                vlong_prev = float(vote_long[i-1]) if np.isfinite(vote_long[i-1]) else np.nan
                vshort_prev = float(vote_short[i-1]) if np.isfinite(vote_short[i-1]) else np.nan
                vlong_prev2 = float(vote_long[i-2]) if np.isfinite(vote_long[i-2]) else np.nan
                vshort_prev2 = float(vote_short[i-2]) if np.isfinite(vote_short[i-2]) else np.nan
                ev_prev = float(ev[i-1]) if np.isfinite(ev[i-1]) else np.nan
                f_prev2 = ev_prev if not np.isfinite(vlong_prev2) else (vlong_prev2 - vshort_prev2)
                f_prev = ev_prev
                if np.isfinite(vlong_prev) and np.isfinite(vshort_prev):
                    if (vlong_prev >= enter_vote_frac) and (vlong_prev > vshort_prev):
                        sig = 1
                        entry_trigger = "cols_long_threshold"
                    elif (not LONG_ONLY) and (vshort_prev >= enter_vote_frac) and (vshort_prev > vlong_prev):
                        sig = -1
                        entry_trigger = "cols_short_threshold"
            else:
                ev_prev = float(score_arr[i-1]) if np.isfinite(score_arr[i-1]) else np.nan
                f_prev2 = float(score_fast[i-2]) if np.isfinite(score_fast[i-2]) else np.nan
                f_prev = float(score_fast[i-1]) if np.isfinite(score_fast[i-1]) else np.nan

                if np.isfinite(ev_prev) and np.isfinite(f_prev2) and np.isfinite(f_prev):
                    entry_up_prev = (f_prev2 <= ENTRY_SCORE_TRIGGER_ABS) and (f_prev > ENTRY_SCORE_TRIGGER_ABS)
                    entry_dn_prev = (f_prev2 >= -ENTRY_SCORE_TRIGGER_ABS) and (f_prev < -ENTRY_SCORE_TRIGGER_ABS)
                    if entry_up_prev and (ev_prev > ENTRY_SCORE_TRIGGER_ABS):
                        sig = 1
                        entry_trigger = "entry_cross_up"
                    elif (not LONG_ONLY) and entry_dn_prev and (ev_prev < -ENTRY_SCORE_TRIGGER_ABS):
                        sig = -1
                        entry_trigger = "entry_cross_down"

            if sig != 0:
                c_prev = float(c[i-1])
                ma_prev = float(sma200[i-1]) if np.isfinite(sma200[i-1]) else np.nan
                sl_prev = float(sma_slope[i-1]) if np.isfinite(sma_slope[i-1]) else np.nan
                regime_checks_n += 1
                if not regime_ok(sig, c_prev, ma_prev, sl_prev, ev_prev):
                    regime_filtered_n += 1
                    continue

                atr_i = float(atr[i]) if np.isfinite(atr[i]) else np.nan
                if (not np.isfinite(atr_i)) or atr_i <= ATR_EPS:
                    continue

                filled, fill_px, _ = nextday_limit_fill(o1, h1, l1, atr_i, ev_prev, ENTRY_SCORE_TRIGGER_ABS, sig, p.entry_discount)
                if not filled:
                    continue

                pos = sig
                entry_price = float(fill_px)
                stop_abs = float(p.atr_mult * atr_i)
                take_abs = float(p.rr_mult * stop_abs)
                bars_in_pos = 0
                cur_max_fav = 0.0
                if return_trades:
                    cur_trade = {
                        "side": "LONG" if sig > 0 else "SHORT",
                        "entry_idx": int(i),
                        "entry_date": _date_at(i),
                        "entry_price": float(entry_price),
                        "entry_trigger": entry_trigger,
                        "entry_ev_prev": float(ev_prev),
                        "entry_score_fast_prev2": float(f_prev2),
                        "entry_score_fast_prev": float(f_prev),
                        "atr": float(atr_i),
                        "stop_abs": float(stop_abs),
                        "take_abs": float(take_abs),
                        "entry_discount": float(p.entry_discount),
                    }
                log_eq[i] += log_cost
                log_rets[i] = log_eq[i] - log_eq[i-1]

    exposure = float(pos_days / max(n-1, 1))
    total_return = float(math.exp(log_eq[-1] - log_eq[0]) - 1.0)

    peak = np.maximum.accumulate(log_eq)
    dd = np.exp(log_eq - peak) - 1.0
    mdd = float(np.min(dd))

    mu = float(np.nanmean(log_rets))
    sd = float(np.nanstd(log_rets, ddof=1))
    sharpe = (mu / sd) * math.sqrt(252.0) if (sd > 1e-9) else 0.0
    downside = log_rets[log_rets < 0.0]
    downside_sd = float(np.nanstd(downside, ddof=1)) if len(downside) > 1 else 0.01
    sortino = (mu / max(downside_sd, 1e-9)) * math.sqrt(252.0)

    win_rate = (n_wins / n_trades) if n_trades > 0 else 0.0
    avg_trade = (trade_sum / n_trades) if n_trades > 0 else 0.0
    trade_std = float(np.nanstd(np.asarray(trade_rets, dtype=np.float64), ddof=1)) if len(trade_rets) > 1 else float("nan")
    max_fav_pct = float(np.mean(trade_max_favs)) if len(trade_max_favs) > 0 else 0.0

    peak_log = np.maximum.accumulate(log_eq)
    underwater = (log_eq < (peak_log - 1e-12)).astype(np.int32)
    cur_dd_dur = 0
    max_dd_dur = 0
    for flag in underwater:
        if flag:
            cur_dd_dur += 1
            if cur_dd_dur > max_dd_dur:
                max_dd_dur = cur_dd_dur
        else:
            cur_dd_dur = 0

    regime_filtered_pct = (float(regime_filtered_n) / float(regime_checks_n)) if regime_checks_n > 0 else 0.0

    ret_long = (trade_sum_long / n_longs) if n_longs > 0 else 0.0
    ret_short = (trade_sum_short / n_shorts) if n_shorts > 0 else 0.0

    out = {
        "total_return": total_return,
        "mdd": mdd,
        "sharpe": sharpe,
        "sortino": float(sortino),
        "n_trades": float(n_trades),
        "n_longs": float(n_longs),
        "n_shorts": float(n_shorts),
        "ret_long": float(ret_long),
        "ret_short": float(ret_short),
        "win_rate": float(win_rate),
        "avg_trade": float(avg_trade),
        "trade_std": float(trade_std) if np.isfinite(trade_std) else float("nan"),
        "exposure": float(exposure),
        "dd_duration": float(max_dd_dur),
        "regime_filtered_n": float(regime_filtered_n),
        "regime_checks_n": float(regime_checks_n),
        "regime_filtered_pct": float(regime_filtered_pct),
        "max_fav_pct": float(max_fav_pct),
        "n_days": float(n),
    }
    if return_trades:
        out["trades"] = trade_logs
    return out

def infer_ga_ranges(score_z: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x = np.abs(score_z[np.isfinite(score_z)])
    if len(x) < 200:
        return (0.15, 1.80), (0.04, 0.80)

    q30 = float(np.quantile(x, 0.30))
    q80 = float(np.quantile(x, 0.80))
    q92 = float(np.quantile(x, 0.92))

    enter_lo = max(0.08, q30 * 0.85)
    enter_hi = max(enter_lo * 2.5, q92 * 1.2, 0.60)

    exit_lo = 0.40
    exit_hi = max(exit_lo, min(0.55, enter_lo * 0.95))
    return (enter_lo, enter_hi), (exit_lo, exit_hi)



def plot_best_generation_trades(close_arr: np.ndarray, dates_arr: np.ndarray, trades: List[Dict[str, float]], ticker: str, gen: int):
    if (trades is None) or (len(trades) == 0):
        return
    try:
        x = np.arange(len(close_arr))
        plt.figure(figsize=(14, 5))
        plt.plot(x, close_arr, color="steelblue", lw=1.6, label="Close")

        for tr in trades:
            ei = tr.get("entry_idx", None)
            xi = tr.get("exit_idx", None)
            if (ei is None) or (xi is None):
                continue
            side = str(tr.get("side", "")).upper()
            col = "green" if side == "LONG" else "red"
            mk_entry = "^" if side == "LONG" else "v"
            mk_exit = "x"
            if 0 <= int(ei) < len(close_arr):
                plt.scatter(int(ei), float(close_arr[int(ei)]), color=col, marker=mk_entry, s=60)
            if 0 <= int(xi) < len(close_arr):
                plt.scatter(int(xi), float(close_arr[int(xi)]), color=col, marker=mk_exit, s=55)
            if 0 <= int(ei) < len(close_arr) and 0 <= int(xi) < len(close_arr):
                plt.plot([int(ei), int(xi)], [float(close_arr[int(ei)]), float(close_arr[int(xi)])], color=col, alpha=0.25)

        if dates_arr is not None and len(dates_arr) == len(close_arr):
            ticks = np.linspace(0, len(close_arr)-1, min(8, len(close_arr))).astype(int)
            labels = [str(pd.to_datetime(dates_arr[t]).date()) for t in ticks]
            plt.xticks(ticks, labels, rotation=25, ha="right")

        plt.title(f"[GA BEST GEN] {ticker} | gen={gen:02d} | entradas/saídas")
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color="steelblue", lw=1.6, label="Close"),
            Line2D([0], [0], marker="^", color="green", linestyle="", markersize=8, label="Entrada LONG"),
            Line2D([0], [0], marker="v", color="red", linestyle="", markersize=8, label="Entrada SHORT"),
            Line2D([0], [0], marker="x", color="black", linestyle="", markersize=8, label="Saída"),
        ]
        plt.legend(handles=legend_handles, loc="best")
        plt.ylabel("Preço")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[WARN] plot_best_generation_trades falhou: {e}")


def select_robust_param_from_hof(hof, fitness_fn) -> Tuple[Params, Dict[str, float]]:
    """Select parameter set from HOF by local stability, not only peak fitness."""
    if (hof is None) or (len(hof) == 0):
        return None, {"robust_score": float("nan"), "peak_fit": float("nan"), "stability_std": float("nan")}

    cand_scores = []
    for ind in hof:
        p0 = sanitize_params(Params(*ind))
        peak_fit = float(fitness_fn(p0))
        neigh = [peak_fit]
        deltas = [
            (-1, 0, 0.0, 0.0, 0.0), (1, 0, 0.0, 0.0, 0.0),
            (0, -2, 0.0, 0.0, 0.0), (0, 2, 0.0, 0.0, 0.0),
            (0, 0, -0.15, 0.0, 0.0), (0, 0, 0.15, 0.0, 0.0),
            (0, 0, 0.0, -0.20, 0.0), (0, 0, 0.0, 0.20, 0.0),
            (0, 0, 0.0, 0.0, -0.05), (0, 0, 0.0, 0.0, 0.05),
        ]
        for df, ds, da, dr, dd in deltas:
            pn = sanitize_params(Params(
                float(p0.fast_period + df), float(p0.slow_period + ds),
                float(p0.atr_mult + da), float(p0.rr_mult + dr), float(p0.entry_discount + dd)
            ))
            neigh.append(float(fitness_fn(pn)))

        arr = np.asarray(neigh, dtype=np.float64)
        med = float(np.nanmedian(arr)) if len(arr) else float("nan")
        mn = float(np.nanmin(arr)) if len(arr) else float("nan")
        sd = float(np.nanstd(arr)) if len(arr) else float("nan")
        robust_score = (0.45 * med) + (0.40 * mn) + (0.15 * peak_fit) - (0.20 * sd if np.isfinite(sd) else 0.0)
        cand_scores.append((robust_score, p0, {"robust_score": robust_score, "peak_fit": peak_fit, "stability_std": sd}))

    cand_scores.sort(key=lambda x: x[0], reverse=True)
    return cand_scores[0][1], cand_scores[0][2]


def ga_optimize_strategy_only(o, h, l, c, score_ev, ma, sl, atr, train_idx: np.ndarray, train_dates: np.ndarray = None, ticker: str = "", visual_close: np.ndarray = None, run_mc_check: bool = True):
    o_tr = o[train_idx]; h_tr = h[train_idx]; l_tr = l[train_idx]; c_tr = c[train_idx]
    z_tr = score_ev[train_idx]
    feature_mode = (np.asarray(score_ev).ndim == 2)
    ma_tr = ma[train_idx]; sl_tr = sl[train_idx]; atr_tr = atr[train_idx]
    d_tr = train_dates[train_idx] if train_dates is not None else None
    c_vis_tr = visual_close[train_idx] if visual_close is not None else c_tr
    verbose_gen = bool(GA_VERBOSE_PER_GENERATION and train_dates is not None)

    if np.isfinite(z_tr).sum() < 120:
        return None

    if feature_mode:
        n_features = int(np.asarray(score_ev).shape[1])
        n_rules = 5

        def decode_miner_ind(ind):
            cols=[]; ops=[]; thrs=[]
            for r in range(n_rules):
                base = r * 3
                cols.append(int(np.clip(round(ind[base]), 0, n_features - 1)))
                ops.append(int(np.clip(round(ind[base + 1]), 0, 1)))
                thrs.append(float(ind[base + 2]))
            return Params(20.0, 60.0, float(ind[-2]), float(ind[-1]), 0.0, tuple(cols), tuple(ops), tuple(thrs))

        def fit_ind(ind):
            p_ = decode_miner_ind(ind)
            tr = backtest_stats_only_intraday(o_tr, h_tr, l_tr, c_tr, z_tr, ma_tr, sl_tr, atr_tr, p_)
            return (fitness_return_1y(tr),)

        toolbox = base.Toolbox()
        for r in range(n_rules):
            toolbox.register(f"attr_col_{r}", random.randint, 0, max(0, n_features - 1))
            toolbox.register(f"attr_op_{r}", random.randint, 0, 1)
            toolbox.register(f"attr_thr_{r}", random.uniform, -1.5, 1.5)
        toolbox.register("attr_atr", random.uniform, float(ATR_MULT_RANGE[0]), float(ATR_MULT_RANGE[1]))
        toolbox.register("attr_rr", random.uniform, float(RR_MULT_RANGE[0]), float(RR_MULT_RANGE[1]))
        attrs=[]
        for r in range(n_rules):
            attrs.extend([getattr(toolbox, f"attr_col_{r}"), getattr(toolbox, f"attr_op_{r}"), getattr(toolbox, f"attr_thr_{r}")])
        attrs.extend([toolbox.attr_atr, toolbox.attr_rr])
        toolbox.register("individual", tools.initCycle, creator.Individual_PT, tuple(attrs), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", tools.selTournament, tournsize=GA_TOURN)
        toolbox.register("mate", tools.cxTwoPoint)

        def mutate_miner(ind):
            for r in range(n_rules):
                b = r * 3
                if random.random() < 0.25:
                    ind[b] = random.randint(0, max(0, n_features - 1))
                if random.random() < 0.15:
                    ind[b+1] = random.randint(0, 1)
                if random.random() < 0.30:
                    ind[b+2] += random.gauss(0, 0.2)
            if random.random() < 0.25:
                ind[-2] += random.gauss(0, 0.15)
            if random.random() < 0.25:
                ind[-1] += random.gauss(0, 0.2)
            ind[-2] = float(np.clip(ind[-2], ATR_MULT_RANGE[0], ATR_MULT_RANGE[1]))
            ind[-1] = float(np.clip(ind[-1], RR_MULT_RANGE[0], RR_MULT_RANGE[1]))
            return (ind,)

        toolbox.register("mutate", mutate_miner)
        toolbox.register("evaluate", fit_ind)

        pop = toolbox.population(n=GA_STAGE1_POP_SIZE)
        hof = tools.HallOfFame(1)
        for _ in range(GA_STAGE1_NGEN):
            invalid = [ind for ind in pop if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)
            hof.update(pop)
            offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
            for i in range(1, len(offspring), 2):
                if random.random() < GA_CX_PB:
                    toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values, offspring[i].fitness.values
            for i in range(len(offspring)):
                if random.random() < GA_MUT_PB:
                    toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
            pop[:] = offspring

        best_p = decode_miner_ind(hof[0])
        stats_train = backtest_stats_only_intraday(o_tr, h_tr, l_tr, c_tr, z_tr, ma_tr, sl_tr, atr_tr, best_p)
        fit1y = fitness_return_1y(stats_train)
        bh1y = float((c_tr[-1] / max(c_tr[0], 1e-12)) - 1.0)
        return best_p, stats_train, fit1y, bh1y

    n_tr = len(c_tr)
    n_splits_ga = min(GA_WF_SPLITS, max(2, n_tr // 160))

    ga_fold_indices = []
    if n_tr >= 240:
        tscv_ga = TimeSeriesSplit(n_splits=n_splits_ga)
        for tr_idx, te_idx in tscv_ga.split(c_tr):
            embargo = int(max(1, FWD_H))
            te_idx = te_idx[embargo:] if len(te_idx) > embargo else np.asarray([], dtype=np.int64)
            if len(te_idx) >= 80 and len(tr_idx) >= 120:
                ga_fold_indices.append(te_idx)

    fitness_cache = {}

    def fitness_internal(p_: Params) -> float:
        p_ = sanitize_params(p_)
        key = (int(round(p_.fast_period)), int(round(p_.slow_period)), round(p_.atr_mult, 3), round(p_.rr_mult, 3), round(p_.entry_discount, 3))
        if key in fitness_cache:
            return fitness_cache[key]

        def _score_fold(st: Dict[str, float]) -> float:
            n_trades = float(st.get("n_trades", 0.0))
            n_days = float(st.get("n_days", np.nan))
            dyn_min_trades = float(max(4.0, min(float(GA_MIN_TRADES_FOR_SIGNIFICANCE), 0.03 * max(n_days, 1.0)))) if np.isfinite(n_days) else float(GA_MIN_TRADES_FOR_SIGNIFICANCE)
            if n_trades < max(4.0, 0.03 * max(n_days, 1.0)):
                return -1e9

            ret = float(st.get("total_return", 0.0))
            mdd = float(st.get("mdd", 0.0))
            avg_trade = float(st.get("avg_trade", 0.0))
            trade_std = float(st.get("trade_std", np.nan))

            if (not np.isfinite(trade_std)) or trade_std <= 1e-9:
                sqn = -0.5
            else:
                sqn = (math.sqrt(max(n_trades, 1.0)) * avg_trade) / (trade_std + 1e-9)

            trade_participation = float(np.clip(n_trades / max(dyn_min_trades, 1.0), 0.0, 1.5))
            low_trade_penalty = -8.0 * max(0.0, 1.0 - trade_participation)
            neg_ret_penalty = -8.0 * max(0.0, -ret)
            deep_mdd_penalty = -4.0 * max(0.0, abs(mdd) - 0.20)

            score = (1.10 * ret) + (0.85 * sqn)
            score += min(n_trades, 80.0) * 0.02
            score += low_trade_penalty + neg_ret_penalty + deep_mdd_penalty

            regime_filtered_pct = float(st.get("regime_filtered_pct", 0.0))
            if regime_filtered_pct > 0.50:
                score *= 0.85

            if n_trades > GA_OVERTRADING_TRADES_PER_FOLD:
                score *= 0.80

            return float(score)

        if n_tr < 240:
            st = backtest_stats_only_intraday(o_tr, h_tr, l_tr, c_tr, z_tr, ma_tr, sl_tr, atr_tr, p_)
            st = dict(st)
            st["total_return"] = float(st.get("total_return", 0.0) - (GA_INTERNAL_EXTRA_TRADE_COST_BPS / 10000.0) * float(st.get("n_trades", 0.0)))
            val = _score_fold(st)
            fitness_cache[key] = val
            return val

        scores_folds = []
        for te_idx in ga_fold_indices:
            st = backtest_stats_only_intraday(
                o_tr[te_idx], h_tr[te_idx], l_tr[te_idx], c_tr[te_idx], z_tr[te_idx],
                ma_tr[te_idx], sl_tr[te_idx], atr_tr[te_idx], p_
            )
            st = dict(st)
            st["total_return"] = float(st.get("total_return", 0.0) - (GA_INTERNAL_EXTRA_TRADE_COST_BPS / 10000.0) * float(st.get("n_trades", 0.0)))
            scores_folds.append(_score_fold(st))

        if len(scores_folds) == 0:
            fitness_cache[key] = -1e9
            return -1e9

        median_score = float(np.median(scores_folds))
        worst_score = float(np.min(scores_folds))
        stability_penalty = float(np.std(scores_folds))
        robust_score = 0.6 * median_score + 0.4 * worst_score
        val = float(robust_score - (GA_FOLD_STABILITY_PENALTY * stability_penalty))
        fitness_cache[key] = val
        return val


    full_ranges = [
        FAST_PERIOD_RANGE,
        SLOW_PERIOD_RANGE,
        ATR_MULT_RANGE,
        RR_MULT_RANGE,
        ENTRY_DISCOUNT_RANGE,
    ]

    def _build_toolbox(ranges):
        toolbox = base.Toolbox()
        toolbox.register("attr_fast", random.randint, int(ranges[0][0]), int(ranges[0][1]))
        toolbox.register("attr_slow", random.randint, int(ranges[1][0]), int(ranges[1][1]))
        toolbox.register("attr_atr", random.uniform, float(ranges[2][0]), float(ranges[2][1]))
        toolbox.register("attr_rr", random.uniform, float(ranges[3][0]), float(ranges[3][1]))
        toolbox.register("attr_entry_discount", random.uniform, float(ranges[4][0]), float(ranges[4][1]))

        toolbox.register("individual", tools.initCycle, creator.Individual_PT,
                         (toolbox.attr_fast, toolbox.attr_slow, toolbox.attr_atr, toolbox.attr_rr, toolbox.attr_entry_discount), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", tools.selTournament, tournsize=GA_TOURN)
        toolbox.register("mate", tools.cxSimulatedBinary, eta=20.0)

        def _clip_ind(ind):
            ind[0] = float(int(np.clip(round(ind[0]), int(ranges[0][0]), int(ranges[0][1]))))
            ind[1] = float(int(np.clip(round(ind[1]), int(ranges[1][0]), int(ranges[1][1]))))
            if ind[1] <= ind[0]:
                ind[1] = float(min(int(ranges[1][1]), int(ind[0]) + 1))
                if ind[1] <= ind[0]:
                    ind[0] = float(max(int(ranges[0][0]), int(ind[1]) - 1))
            ind[2] = float(np.clip(ind[2], float(ranges[2][0]), float(ranges[2][1])))
            ind[3] = float(np.clip(ind[3], float(ranges[3][0]), float(ranges[3][1])))
            ind[4] = float(np.clip(ind[4], float(ranges[4][0]), float(ranges[4][1])))
            return ind

        def mutate_gaussian(ind, sigma=0.10, gene_pb=0.20):
            for j in range(len(ind)):
                if random.random() < gene_pb:
                    ind[j] = float(ind[j]) + random.gauss(0.0, sigma)
            _clip_ind(ind)
            return (ind,)

        toolbox.register("mutate", mutate_gaussian)
        toolbox.register("evaluate", lambda ind: (fitness_internal(Params(*ind)),))
        return toolbox, _clip_ind

    def _evaluate_invalid(toolbox, individuals):
        invalid = [ind for ind in individuals if not ind.fitness.valid]
        if not invalid:
            return
        workers = int(max(1, GA_EVAL_WORKERS))
        if workers <= 1 or len(invalid) < 4:
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)
            return
        with ThreadPoolExecutor(max_workers=workers) as ex:
            vals = list(ex.map(toolbox.evaluate, invalid))
        for ind, fit_v in zip(invalid, vals):
            ind.fitness.values = fit_v

    def _run_ga(ranges, pop_size, ngen, seed_params=None):
        toolbox, clip_ind = _build_toolbox(ranges)
        pop = toolbox.population(n=pop_size)

        if seed_params:
            for idx, p_seed in enumerate(seed_params[:len(pop)]):
                pop[idx][:] = [float(p_seed.fast_period), float(p_seed.slow_period), float(p_seed.atr_mult), float(p_seed.rr_mult), float(p_seed.entry_discount)]
                clip_ind(pop[idx])
                if pop[idx].fitness.valid:
                    del pop[idx].fitness.values

        hof = tools.HallOfFame(GA_HOF_SIZE)
        _evaluate_invalid(toolbox, pop)
        hof.update(pop)

        best_fit = hof[0].fitness.values[0]
        no_improve = 0
        best_gen_snapshot = None

        for _gen in range(1, int(ngen) + 1):
            frac = (_gen - 1) / max(1, (int(ngen) - 1))
            sigma = GA_MUT_SIGMA_START + (GA_MUT_SIGMA_END - GA_MUT_SIGMA_START) * frac
            gene_pb = GA_GENE_MUT_PB_START + (GA_GENE_MUT_PB_END - GA_GENE_MUT_PB_START) * frac

            offspring = [toolbox.clone(ind) for ind in toolbox.select(pop, len(pop))]

            for i in range(1, len(offspring), 2):
                if random.random() < GA_CX_PB:
                    toolbox.mate(offspring[i-1], offspring[i])
                    clip_ind(offspring[i-1]); clip_ind(offspring[i])
                    del offspring[i-1].fitness.values, offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < GA_MUT_PB:
                    toolbox.mutate(offspring[i], sigma=sigma, gene_pb=gene_pb)
                    del offspring[i].fitness.values

            _evaluate_invalid(toolbox, offspring)
            pop[:] = offspring
            hof.update(pop)

            cur = hof[0].fitness.values[0]
            if cur > best_fit + 1e-9:
                best_fit = cur
                no_improve = 0
            else:
                no_improve += 1

            if verbose_gen:
                p_gen = sanitize_params(Params(*hof[0]))
                st_gen = backtest_stats_only_intraday(o_tr, h_tr, l_tr, c_tr, z_tr, ma_tr, sl_tr, atr_tr, p_gen, return_trades=True, dates=d_tr)
                if (best_gen_snapshot is None) or (cur > float(best_gen_snapshot.get("fit", -1e18)) + 1e-12):
                    best_gen_snapshot = {"gen": int(_gen), "fit": float(cur), "params": p_gen, "stats": st_gen}

            if no_improve >= EARLY_STOP:
                break

        return pop, hof, best_gen_snapshot

    def _individual_to_params(ind):
        return sanitize_params(Params(*[float(v) for v in ind]))

    def _param_distance_norm(p1: Params, p2: Params) -> float:
        v1 = [p1.fast_period, p1.slow_period, p1.atr_mult, p1.rr_mult, p1.entry_discount]
        v2 = [p2.fast_period, p2.slow_period, p2.atr_mult, p2.rr_mult, p2.entry_discount]
        s = 0.0
        for j, (a, b) in enumerate(zip(v1, v2)):
            low, high = full_ranges[j]
            span = max(float(high) - float(low), 1e-9)
            s += ((float(a) - float(b)) / span) ** 2
        return float(math.sqrt(s / len(v1)))

    def _pick_diverse_top(candidates, top_n, min_dist=0.22):
        selected = []
        adaptive = float(min_dist)
        for _, p in candidates:
            if not selected:
                selected.append(p)
            else:
                dmin = min(_param_distance_norm(p, q) for q in selected)
                if dmin >= adaptive:
                    selected.append(p)
            if len(selected) >= top_n:
                break
        while len(selected) < top_n and adaptive > 0.02:
            adaptive *= 0.7
            for _, p in candidates:
                if p in selected:
                    continue
                dmin = min(_param_distance_norm(p, q) for q in selected) if selected else 1e9
                if dmin >= adaptive:
                    selected.append(p)
                if len(selected) >= top_n:
                    break
        return selected

    def _build_stage2_ranges(top_params):
        span_floor_ratio = float(max(0.05, min(0.50, GA_STAGE2_MIN_SPAN_RATIO)))
        ranges = []
        for j, (global_low, global_high) in enumerate(full_ranges):
            vals = [
                [p.fast_period, p.slow_period, p.atr_mult, p.rr_mult, p.entry_discount][j]
                for p in top_params
            ]
            center_low = float(np.min(vals))
            center_high = float(np.max(vals))
            global_span = float(global_high) - float(global_low)
            local_span = max(center_high - center_low, global_span * span_floor_ratio)
            pad = local_span * float(max(0.05, GA_STAGE2_PADDING_RATIO))
            low = max(float(global_low), center_low - pad)
            high = min(float(global_high), center_high + pad)
            if j in (0, 1):
                low = float(int(math.floor(low)))
                high = float(int(math.ceil(high)))
                if high <= low:
                    high = min(float(global_high), low + 1.0)
            ranges.append((low, high))
        if ranges[1][1] <= ranges[0][0]:
            ranges[1] = (ranges[1][0], max(ranges[1][1], ranges[0][0] + 1.0))
        return ranges

    best_gen_snapshot = None
    if bool(GA_TWO_STAGE):
        pop1, hof1, snap1 = _run_ga(full_ranges, int(GA_STAGE1_POP_SIZE), int(GA_STAGE1_NGEN))
        cand_map = {}
        for ind in list(hof1) + list(pop1):
            p_ = _individual_to_params(ind)
            key = (p_.fast_period, p_.slow_period, round(p_.atr_mult, 4), round(p_.rr_mult, 4), round(p_.entry_discount, 4))
            fit = float(ind.fitness.values[0]) if ind.fitness.valid else float(fitness_internal(p_))
            if (key not in cand_map) or (fit > cand_map[key][0]):
                cand_map[key] = (fit, p_)
        ranked = sorted(cand_map.values(), key=lambda x: x[0], reverse=True)
        top_regions = _pick_diverse_top(ranked, top_n=int(max(2, GA_STAGE1_TOP_N)))
        if not top_regions and len(ranked):
            top_regions = [ranked[0][1]]

        stage2_ranges = _build_stage2_ranges(top_regions) if top_regions else full_ranges
        seed_params = list(top_regions)
        if snap1 is not None:
            seed_params.append(snap1["params"])

        pop, hof, snap2 = _run_ga(stage2_ranges, int(GA_STAGE2_POP_SIZE), int(GA_STAGE2_NGEN), seed_params=seed_params)
        best_gen_snapshot = snap2 if snap2 is not None else snap1
    else:
        pop, hof, best_gen_snapshot = _run_ga(full_ranges, int(GA_POP_SIZE), int(GA_NGEN))

    if verbose_gen and (best_gen_snapshot is not None):
        p_gen = best_gen_snapshot["params"]
        st_gen = best_gen_snapshot["stats"]
        gfit = float(best_gen_snapshot["fit"])
        gnum = int(best_gen_snapshot["gen"])
        n_long = int(sum(1 for t in st_gen.get("trades", []) if t.get("side") == "LONG"))
        n_short = int(sum(1 for t in st_gen.get("trades", []) if t.get("side") == "SHORT"))
        print(
            f"[GA BEST GEN {ticker}] gen={gnum:02d} fit={gfit:.4f} best=(fast={int(p_gen.fast_period)}, slow={int(p_gen.slow_period)}, atr={p_gen.atr_mult:.3f}, rr={p_gen.rr_mult:.3f}, disc={p_gen.entry_discount:.3f}) "
            f"ret={100.0*float(st_gen.get('total_return', np.nan)):.2f}% mdd={100.0*float(st_gen.get('mdd', np.nan)):.2f}% sh={float(st_gen.get('sharpe', np.nan)):.2f} "
            f"trades={int(st_gen.get('n_trades', 0))} (compra={n_long}, venda={n_short})"
        )
        trades = st_gen.get("trades", [])
        if GA_PLOT_BEST_GENERATION_TRADES:
            plot_best_generation_trades(c_vis_tr, d_tr, trades, ticker, gnum)

    if GA_SELECT_ROBUST_FROM_HOF:
        best_p, robust_meta = select_robust_param_from_hof(hof, fitness_internal)
        if best_p is None:
            best_p = sanitize_params(Params(*hof[0]))
            robust_meta = {"robust_score": float("nan"), "peak_fit": float("nan"), "stability_std": float("nan")}
    else:
        best_p = sanitize_params(Params(*hof[0]))
        robust_meta = {"robust_score": float("nan"), "peak_fit": float(hof[0].fitness.values[0]), "stability_std": float("nan")}

    start = max(0, len(c_tr) - ONE_YEAR_DAYS)
    stats_train = backtest_stats_only_intraday(
        o_tr[start:], h_tr[start:], l_tr[start:], c_tr[start:], z_tr[start:],
        ma_tr[start:], sl_tr[start:], atr_tr[start:], best_p
    )
    fit1y = fitness_return_1y(stats_train)

    mc_pvalue = float("nan")
    mc_real_metric = monte_carlo_consistency_metric(stats_train)
    mc_rand_mean = float("nan")
    mc_rand_std = float("nan")
    mc_shuffles_used = 0
    if run_mc_check and USE_MONTE_CARLO_REALITY_CHECK and (len(c_tr[start:]) > 120):
        rnd = np.random.default_rng(RANDOM_SEED)
        rand_metrics = []
        base_slice = z_tr[start:]
        mc_block = max(1, int(MC_SHUFFLES_BLOCK))
        mc_limit = max(mc_block, int(MC_SHUFFLES_MAX))
        z95 = 1.959963984540054

        while mc_shuffles_used < mc_limit:
            this_block = min(mc_block, mc_limit - mc_shuffles_used)
            for _ in range(this_block):
                perm = rnd.permutation(len(base_slice))
                z_rand = base_slice[perm]
                st_rand = backtest_stats_only_intraday(
                    o_tr[start:], h_tr[start:], l_tr[start:], c_tr[start:], z_rand,
                    ma_tr[start:], sl_tr[start:], atr_tr[start:], best_p
                )
                rand_metrics.append(float(monte_carlo_consistency_metric(st_rand)))
            mc_shuffles_used += this_block

            arr = np.asarray(rand_metrics, dtype=np.float64)
            if len(arr):
                mc_rand_mean = float(np.nanmean(arr))
                mc_rand_std = float(np.nanstd(arr))
                exceed = float(np.sum(arr >= mc_real_metric))
                mc_pvalue = float((exceed + 1.0) / (len(arr) + 1.0))

                p_hat = exceed / float(len(arr))
                ci_half = z95 * math.sqrt(max(0.0, p_hat * (1.0 - p_hat) / float(len(arr))))
                pvalue_ci_low = max(0.0, p_hat - ci_half)
                pvalue_ci_high = min(1.0, p_hat + ci_half)

                if (pvalue_ci_high < MC_PVALUE_MAX) or (pvalue_ci_low > MC_PVALUE_MAX):
                    break

        if np.isfinite(mc_pvalue) and (mc_pvalue > MC_PVALUE_MAX):
            fit1y -= float((mc_pvalue - MC_PVALUE_MAX) * 10.0)

    if verbose_gen:
        print(
            f"[GA ROBUST PICK {ticker}] robust={robust_meta.get('robust_score', np.nan):.4f} peak={robust_meta.get('peak_fit', np.nan):.4f} "
            f"std={robust_meta.get('stability_std', np.nan):.4f} mc_p={mc_pvalue:.3f} mc_real={mc_real_metric:.4f} "
            f"mc_rand={mc_rand_mean:.4f}±{mc_rand_std:.4f} shuffles_used={mc_shuffles_used}"
        )

    bh1y  = buyhold_capped(c_tr[start:])

    return best_p, stats_train, fit1y, bh1y


def aggregate_window_stats(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not stats_list:
        return {
            "total_return": float("nan"), "mdd": float("nan"), "sharpe": float("nan"),
            "sortino": float("nan"), "n_trades": float("nan"), "win_rate": float("nan"),
            "avg_trade": float("nan"), "trade_std": float("nan"), "exposure": float("nan"),
            "dd_duration": float("nan"), "regime_filtered_pct": float("nan"), "max_fav_pct": float("nan"),
        }

    keys = set()
    for st in stats_list:
        if isinstance(st, dict):
            keys.update(st.keys())

    out = {}
    for k in keys:
        vals = [float(st.get(k, np.nan)) for st in stats_list if isinstance(st, dict)]
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[k] = float(np.mean(vals)) if len(vals) else float("nan")

    return out


def build_ga_walkforward_windows(has_oos_idx: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    windows = []
    n = len(has_oos_idx)
    tr_len = GA_WF_TRAIN_YEARS * ONE_YEAR_DAYS
    te_len = GA_WF_TEST_DAYS
    step = GA_WF_STEP_DAYS
    if n < (tr_len + te_len):
        return windows
    end = tr_len
    while (end + te_len) <= n:
        tr_idx = has_oos_idx[end - tr_len:end]
        te_idx = has_oos_idx[end:end + te_len]
        windows.append((tr_idx, te_idx))
        end += step
    return windows


# ==============================================================================
# 7) LOAD
# ==============================================================================
def calibrate_ev_with_realized_feedback(
    score_ev: np.ndarray,
    y_atr_norm: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    out = score_ev.copy()
    meta = {"ic": float("nan"), "n_calib": 0.0, "bins_used": 0.0, "blend_used": 0.0}

    if (not USE_RETURN_FEEDBACK_CALIBRATION) or (RETURN_FEEDBACK_BLEND >= 1.0):
        return out, meta

    calib_mask = np.isfinite(score_ev) & np.isfinite(y_atr_norm) & valid_mask
    n_calib = int(np.sum(calib_mask))
    meta["n_calib"] = float(n_calib)
    if n_calib < RETURN_FEEDBACK_MIN_ROWS:
        return out, meta

    ev_cal = score_ev[calib_mask]
    y_cal = y_atr_norm[calib_mask]

    try:
        ic = float(spearmanr(ev_cal, y_cal).statistic)
    except Exception:
        ic = float("nan")
    if np.isfinite(ic):
        meta["ic"] = ic
    if (not np.isfinite(ic)) or (ic < RETURN_FEEDBACK_MIN_IC):
        return out, meta

    ic_strength = float(np.clip((ic - RETURN_FEEDBACK_MIN_IC) / max(1e-12, RETURN_FEEDBACK_TARGET_IC - RETURN_FEEDBACK_MIN_IC), 0.0, 1.0))
    blend_used = float(np.clip(RETURN_FEEDBACK_BLEND_MAX * ic_strength, 0.0, RETURN_FEEDBACK_BLEND_MAX))
    meta["blend_used"] = blend_used
    if blend_used <= 0.0:
        return out, meta

    try:
        bins = pd.qcut(ev_cal, q=RETURN_FEEDBACK_BINS, labels=False, duplicates="drop")
    except Exception:
        return out, meta

    bins = np.asarray(bins, dtype=float)
    if int(np.isfinite(bins).sum()) < RETURN_FEEDBACK_MIN_ROWS:
        return out, meta
    bins = bins.astype(int)
    n_bins = int(np.max(bins)) + 1
    if n_bins < 3:
        return out, meta
    meta["bins_used"] = float(n_bins)

    means = np.full(n_bins, np.nan, dtype=np.float64)
    global_mean = float(np.nanmean(y_cal)) if np.isfinite(np.nanmean(y_cal)) else 0.0
    for b in range(n_bins):
        m = (bins == b)
        if np.any(m):
            n_b = int(np.sum(m))
            raw_b = float(np.nanmean(y_cal[m]))
            shrink = float(n_b / (n_b + RETURN_FEEDBACK_BIN_SHRINK))
            means[b] = shrink * raw_b + (1.0 - shrink) * global_mean

    # enforce monotonic mapping to reduce overfitting noise (higher EV bin => no lower mapped return)
    means = np.maximum.accumulate(means)

    ev_all_mask = np.isfinite(score_ev)
    ev_all = score_ev[ev_all_mask]
    if len(ev_all) < 10:
        return out, meta

    ranks = pd.Series(ev_all).rank(method="average", pct=True).to_numpy()
    ids = np.minimum((ranks * n_bins).astype(int), n_bins - 1)
    mapped = means[ids]

    std_m = float(np.nanstd(mapped))
    std_e = float(np.nanstd(ev_all))
    if (not np.isfinite(std_m)) or (std_m <= 1e-12) or (not np.isfinite(std_e)) or (std_e <= 1e-12):
        return out, meta

    mapped = mapped * (std_e / std_m)
    out[ev_all_mask] = ((1.0 - blend_used) * ev_all) + (blend_used * mapped)
    return out, meta




def build_direct_feature_signal(frame: pd.DataFrame, feat_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-column directed z-scores (no single-feature model)."""
    feat = frame[feat_cols].apply(pd.to_numeric, errors="coerce").copy()
    roll_mean = feat.rolling(SCORE_LOOKBACK, min_periods=60).mean()
    roll_std = feat.rolling(SCORE_LOOKBACK, min_periods=60).std(ddof=0).replace(0.0, np.nan)
    z = (feat - roll_mean) / roll_std
    z = z.replace([np.inf, -np.inf], np.nan).clip(-4.0, 4.0)

    bearish_tokens = {"risk", "down", "dd", "sell", "err_buy", "pe", "price_to_book"}
    signs = []
    for c in feat_cols:
        # Use word-boundary matching to avoid false positives (e.g., "speed" matching "pe")
        parts = set(str(c).lower().replace("-", "_").split("_"))
        sign = -1.0 if bool(parts & bearish_tokens) else 1.0
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
    # Auto-detect format: Parquet is 5-10x faster than CSV
    if path.endswith(".parquet") or path.endswith(".pq"):
        df = pd.read_parquet(path)
        df[TICKER_COL] = df[TICKER_COL].astype("string").str.strip()
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    else:
        df = pd.read_csv(path, dtype={TICKER_COL:"string"}, low_memory=False)
        df[TICKER_COL] = df[TICKER_COL].astype("string").str.strip()
        df[DATE_COL] = _parse_dates_smart(df[DATE_COL])

    for col in [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[DATE_COL, TICKER_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL])
    df = df[(df[CLOSE_COL] > MIN_PRICE) & (df[OPEN_COL] > MIN_PRICE) & (df[HIGH_COL] > MIN_PRICE) & (df[LOW_COL] > MIN_PRICE)]
    if ONLY_SA:
        df = df[df[TICKER_COL].str.endswith(".SA", na=False)]

    df = df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    df = add_sma200(df)
    df = add_atr_ohlc_fast(df)
    df = add_technical_features(df)
    return df


# ==============================================================================
# 8) RUN
# ==============================================================================
def _pct(x):
    return f"{x*100:7.1f}%" if np.isfinite(x) else "    nan "


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
        w = w[np.isfinite(w)]
        if len(w) < 20:
            continue
        score_thr = float(np.quantile(w, gp.score_percentile_trigger))

        if gp.volatility_filter_percentile > 0 and np.isfinite(vol_rank[i]) and vol_rank[i] < gp.volatility_filter_percentile:
            consec_long = 0
            consec_short = 0
            continue

        long_raw = (votes_long[i] >= gp.vote_threshold_long) and (score_ev[i] >= score_thr)
        short_raw = (votes_short[i] >= gp.vote_threshold_short) and (-score_ev[i] >= score_thr)

        if gp.ma_filter_mode == 1 and np.isfinite(ma[i]):
            if c[i] < ma[i]:
                long_raw = long_raw and (score_ev[i] * 0.5 >= score_thr)
            if c[i] > ma[i]:
                short_raw = short_raw and (-score_ev[i] * 0.5 >= score_thr)
        elif gp.ma_filter_mode == 2 and np.isfinite(ma[i]):
            if c[i] < ma[i]:
                long_raw = False
            if c[i] > ma[i]:
                short_raw = False

        consec_long = consec_long + 1 if long_raw else 0
        consec_short = consec_short + 1 if short_raw else 0

        if i == day_index:
            if consec_long >= gp.entry_confirmation_days:
                return "buy"
            if (not LONG_ONLY) and consec_short >= gp.entry_confirmation_days:
                return "sell"
    return "hold"

def _flt(x, w=6, p=3):
    return f"{x:{w}.{p}f}" if np.isfinite(x) else f"{'nan':>{w}}"

def run():
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
    print(f"[TICKERS] total: {len(tickers)}")

    out_xlsx = f"{OUTPUT_DIR}apply_PER_TICKER_WFGA_intraday__H{FWD_H}__APPLY{APPLY_DAYS}D__v2.xlsx"
    out_apply_csv = f"{OUTPUT_DIR}apply_last_{APPLY_DAYS}d__H{FWD_H}__v2.csv"
    out_summary_ckpt = f"{OUTPUT_DIR}summary_last_{APPLY_DAYS}d__H{FWD_H}__v2.partial.csv"
    out_global_ckpt = f"{OUTPUT_DIR}global_params_20__H{FWD_H}__v2.json"

    grouped = {t: g.sort_values(DATE_COL).copy() for t, g in df.groupby(TICKER_COL, sort=False)}

    # FASE 1: preparação única por ticker
    ticker_payloads: Dict[str, Dict[str, Any]] = {}
    reasons: Dict[str, int] = {}
    prep_t0 = time.perf_counter()
    for ix_tkr, tkr in enumerate(tickers, 1):
        g = grouped[tkr]
        if len(g) < MIN_ROWS_TICKER:
            reasons["short_len"] = reasons.get("short_len", 0) + 1
            continue

        # Compute train/test split FIRST to avoid look-ahead bias in feature selection
        directed_cols_tmp, _, _ = build_direct_feature_signal(g, base_feat_cols) if len(base_feat_cols) > 0 else (np.zeros((len(g), 0)), np.zeros(len(g)), np.zeros(len(g)))
        dates = pd.to_datetime(g[DATE_COL], errors="coerce").to_numpy()
        o = g[OPEN_COL].to_numpy(np.float64)
        h = g[HIGH_COL].to_numpy(np.float64)
        l_arr = g[LOW_COL].to_numpy(np.float64)
        c = g[CLOSE_COL].to_numpy(np.float64)
        atr = g["atr"].to_numpy(np.float64)
        ma = g["sma200"].to_numpy(np.float64)
        valid_cols_check = np.isfinite(directed_cols_tmp).any(axis=1) if directed_cols_tmp.shape[1] > 0 else np.ones(len(g), dtype=bool)
        valid_mask = np.isfinite(o) & np.isfinite(h) & np.isfinite(l_arr) & np.isfinite(c) & np.isfinite(atr) & valid_cols_check
        has_oos_idx = np.where(valid_mask)[0]
        if len(has_oos_idx) < 40:
            reasons["few_valid"] = reasons.get("few_valid", 0) + 1
            continue
        split_point = int(len(has_oos_idx) * 0.80)
        train_mask = np.zeros(len(g), dtype=bool)
        train_mask[has_oos_idx[:split_point]] = True

        # Feature selection using ONLY training data (prevents look-ahead bias)
        c_temp = g[CLOSE_COL].to_numpy(np.float64)
        atr_temp = g["atr"].to_numpy(np.float64)
        ret_fwd_temp = (np.roll(c_temp, -FWD_H) - c_temp) / np.maximum(c_temp, 1e-12)
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
                s_vals = s_col.to_numpy()
                # Use only training data for feature selection
                valid_idx_temp = train_mask & np.isfinite(s_vals) & np.isfinite(y_event_temp)
                if valid_idx_temp.sum() > MIN_VALID_SAMPLES_FOR_CORRELATION:
                    corr, _ = spearmanr(s_vals[valid_idx_temp], y_event_temp[valid_idx_temp])
                    if np.isfinite(corr):
                        feat_correlations.append((ccol, abs(corr)))
            except ValueError:
                pass

        feat_correlations.sort(key=lambda x: x[1], reverse=True)
        feat_cols = [col for col, _ in feat_correlations[:MAX_FEATURES]]
        if len(feat_cols) < 5:
            reasons["few_feats"] = reasons.get("few_feats", 0) + 1
            continue

        directed_cols, long_votes, short_votes = build_direct_feature_signal(g, feat_cols)
        l = l_arr  # rename for consistency with rest of code

        # Recompute valid_mask with final directed_cols (selected features only)
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
    run_mode = str(RUN_MODE).strip().lower()
    global_params = None
    global_genome = None
    global_fit = float("nan")
    if run_mode == "load" and os.path.exists(out_global_ckpt):
        try:
            ck = json.load(open(out_global_ckpt, "r", encoding="utf-8"))
            global_genome = ck.get("genome", None)
            if global_genome is not None:
                global_params = decode_global_params(global_genome)
                global_fit = float(ck.get("fitness", np.nan))
                print(f"[GLOBAL_GA] loaded from checkpoint fit={global_fit:.4f}")
        except Exception as e:
            print(f"[WARN] global checkpoint load failed: {e}")

    if global_params is None:
        dmin = pd.to_datetime(df[DATE_COL].min())
        dmax = pd.to_datetime(df[DATE_COL].max())
        windows = build_temporal_windows(dmin, dmax, train_years=3, test_months=6, step_months=6)
        if not windows:
            windows = [(dmin, dmax - pd.Timedelta(days=180), dmax - pd.Timedelta(days=179), dmax)]
        print(f"[GLOBAL_GA] start (windows={len(windows)}, tickers={len(ticker_payloads)})")
        global_params, global_genome, global_fit = run_global_ga_20params(
            full_df=df,
            feature_cols=base_feat_cols,
            windows=windows,
            pop_size=GA_POP_SIZE,
            ngen=GA_NGEN,
        )
        try:
            with open(out_global_ckpt, "w", encoding="utf-8") as f:
                json.dump({"fitness": float(global_fit), "genome": list(global_genome)}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] global checkpoint save failed: {e}")
        print(f"[GLOBAL_GA] done fit={global_fit:.4f}")

    # FASE 3: apply rápido sem GA por ticker
    results_summary: List[Dict[str, Any]] = []
    results_apply: List[Dict[str, Any]] = []

    phase3_t0 = time.perf_counter()
    for idx, (tkr, payload) in enumerate(ticker_payloads.items(), 1):
        tkr_t0 = time.perf_counter()
        o = payload["open"]; h = payload["high"]; l = payload["low"]; c = payload["close"]; atr = payload["atr"]
        x = payload["score_matrix"]; dates = payload["dates"]
        valid_mask = payload["valid_mask"]

        has_oos_idx = np.where(valid_mask)[0]
        if len(has_oos_idx) < 40:
            continue
        split_point = int(len(has_oos_idx) * 0.80)
        tr_idx = has_oos_idx[:split_point]
        te_idx = has_oos_idx[split_point:]
        if len(te_idx) == 0:
            continue

        stats_ga_train = backtest_stats_global_intraday(o[tr_idx], h[tr_idx], l[tr_idx], c[tr_idx], x[tr_idx], atr[tr_idx], global_params)
        te_stats = backtest_stats_global_intraday(o[te_idx], h[te_idx], l[te_idx], c[te_idx], x[te_idx], atr[te_idx], global_params)
        fit_ga = float(global_fit)
        bh_ga = float((c[tr_idx[-1]] / max(c[tr_idx[0]], 1e-12)) - 1.0) if len(tr_idx) > 1 else 0.0

        score_raw = (payload["long_votes"] - payload["short_votes"]).astype(np.float64)
        score_ev_apply = pd.Series(score_raw).ewm(span=int(global_params.signal_ema_span), adjust=False).mean().to_numpy()

        start_apply = max(0, len(c) - APPLY_DAYS)
        for i in range(start_apply, len(c)):
            date_i = pd.to_datetime(dates[i])
            close_i = float(c[i]) if np.isfinite(c[i]) else np.nan
            atr_i = float(atr[i]) if np.isfinite(atr[i]) else np.nan
            ev_i = float(score_ev_apply[i]) if np.isfinite(score_ev_apply[i]) else np.nan
            signal_eod = generate_signal_global(payload, global_params, i)
            final_signal = signal_eod

            best_p_adj = Params(20.0, 60.0, float(global_params.stop_atr_mult), float(global_params.reward_risk_ratio), float(global_params.entry_discount_atr_frac))
            best_buy_value = np.nan; best_sell_value = np.nan; entry_ref_price = close_i; limit_price = np.nan
            next_day_filled=False; next_day_filled_buy=False; next_day_filled_sell=False
            projected_buy_limit=np.nan; projected_sell_limit=np.nan
            if (i+1) < len(c):
                o1,h1,l1 = float(o[i+1]), float(h[i+1]), float(l[i+1])
                atr1 = float(atr[i+1]) if np.isfinite(atr[i+1]) else np.nan
                projected_buy_limit = compute_model_entry_price(o1, atr1, ev_i, SCORE_CROSS_MIN_ABS, +1, best_p_adj.entry_discount)
                projected_sell_limit = compute_model_entry_price(o1, atr1, ev_i, SCORE_CROSS_MIN_ABS, -1, best_p_adj.entry_discount)
                fb, fill_b, lim_b = nextday_limit_fill(o1, h1, l1, atr1, ev_i, SCORE_CROSS_MIN_ABS, +1, best_p_adj.entry_discount)
                fs, fill_s, lim_s = nextday_limit_fill(o1, h1, l1, atr1, ev_i, SCORE_CROSS_MIN_ABS, -1, best_p_adj.entry_discount)
                next_day_filled_buy = bool(fb); next_day_filled_sell = bool(fs)
                best_buy_value = float(fill_b) if fb else (float(lim_b) if np.isfinite(lim_b) else np.nan)
                best_sell_value = float(fill_s) if fs else (float(lim_s) if np.isfinite(lim_s) else np.nan)
                if final_signal == "buy":
                    next_day_filled = bool(fb); limit_price = float(lim_b) if np.isfinite(lim_b) else np.nan; entry_ref_price = float(fill_b) if fb else o1
                elif final_signal == "sell":
                    next_day_filled = bool(fs); limit_price = float(lim_s) if np.isfinite(lim_s) else np.nan; entry_ref_price = float(fill_s) if fs else o1
                else:
                    entry_ref_price = o1
            levels = compute_levels_from_atr(entry_ref_price, atr_i, best_p_adj)
            (stop_abs, take_abs, stop_pct, take_pct,
             buy_entry, buy_stop, buy_take,
             sell_entry, sell_stop, sell_take) = levels

            results_apply.append({
                "Date": date_i, "ticker": tkr, "close": close_i, "atr": atr_i,
                "signal_eod": signal_eod, "signal": final_signal,
                "score_0_100": float(score_0_100_from_ev(ev_i, score_ev_apply[max(0, i-SCORE_LOOKBACK):i+1], 0.5)),
                "score_ev": ev_i,
                "ga_mode": "global_20",
                "gp_stop_atr_mult": float(global_params.stop_atr_mult),
                "gp_reward_risk_ratio": float(global_params.reward_risk_ratio),
                "gp_time_stop_bars": float(global_params.time_stop_bars),
                "ga_return_1y": float(stats_ga_train.get("total_return", np.nan)),
                "ga_mdd_1y": float(stats_ga_train.get("mdd", np.nan)),
                "ga_sharpe_1y": float(stats_ga_train.get("sharpe", np.nan)),
                "ga_trades_1y": float(stats_ga_train.get("n_trades", np.nan)),
                "ga_exposure_1y": float(stats_ga_train.get("exposure", np.nan)),
                "buyhold_return_1y": float(bh_ga),
                "test_return": float(te_stats.get("total_return", np.nan)),
                "test_mdd": float(te_stats.get("mdd", np.nan)),
                "test_sharpe": float(te_stats.get("sharpe", np.nan)),
                "test_n_trades": float(te_stats.get("n_trades", np.nan)),
                "test_win_rate": float(te_stats.get("win_rate", np.nan)),
                "test_avg_trade": float(te_stats.get("avg_trade", np.nan)),
                "next_day_filled": bool(next_day_filled), "next_day_filled_buy": bool(next_day_filled_buy), "next_day_filled_sell": bool(next_day_filled_sell),
                "limit_price_next_day": limit_price, "best_buy_value": best_buy_value, "best_sell_value": best_sell_value,
                "entry_ref_price": entry_ref_price,
                "stop_abs": float(stop_abs), "take_abs": float(take_abs),
                "stop_pct": float(stop_pct), "take_pct": float(take_pct),
                "buy_entry": float(buy_entry), "buy_stop": float(buy_stop), "buy_take": float(buy_take),
                "sell_entry": float(sell_entry), "sell_stop": float(sell_stop), "sell_take": float(sell_take),
                "train_start": pd.to_datetime(dates[tr_idx[0]]).date(), "train_end": pd.to_datetime(dates[tr_idx[-1]]).date(),
                "test_start": pd.to_datetime(dates[te_idx[0]]).date(), "test_end": pd.to_datetime(dates[te_idx[-1]]).date(),
            })

        latest_i = int(has_oos_idx[-1])
        latest_ev = float(score_ev_apply[latest_i]) if np.isfinite(score_ev_apply[latest_i]) else np.nan
        latest_score100 = score_0_100_from_ev(latest_ev, score_ev_apply[max(0, latest_i-SCORE_LOOKBACK):latest_i+1], 0.5)
        results_summary.append({
            "ticker": tkr,
            "feat_count_used": int(len(payload["feat_cols"])),
            "ga_mode": "global_20",
            "gp_vote_threshold_long": float(global_params.vote_threshold_long),
            "gp_vote_threshold_short": float(global_params.vote_threshold_short),
            "gp_z_threshold": float(global_params.z_threshold),
            "gp_signal_ema_span": float(global_params.signal_ema_span),
            "gp_entry_confirmation_days": float(global_params.entry_confirmation_days),
            "gp_score_percentile_trigger": float(global_params.score_percentile_trigger),
            "gp_stop_atr_mult": float(global_params.stop_atr_mult),
            "gp_reward_risk_ratio": float(global_params.reward_risk_ratio),
            "gp_time_stop_bars": float(global_params.time_stop_bars),
            "ga_return_1y": float(stats_ga_train.get("total_return", np.nan)),
            "ga_mdd_1y": float(stats_ga_train.get("mdd", np.nan)),
            "ga_sharpe_1y": float(stats_ga_train.get("sharpe", np.nan)),
            "ga_trades_1y": float(stats_ga_train.get("n_trades", np.nan)),
            "ga_exposure_1y": float(stats_ga_train.get("exposure", np.nan)),
            "ga_fitness_1y": float(fit_ga),
            "buyhold_return_1y": float(bh_ga),
            "test_return": float(te_stats.get("total_return", np.nan)),
            "test_mdd": float(te_stats.get("mdd", np.nan)),
            "test_sharpe": float(te_stats.get("sharpe", np.nan)),
            "test_n_trades": float(te_stats.get("n_trades", np.nan)),
            "test_win_rate": float(te_stats.get("win_rate", np.nan)),
            "test_avg_trade": float(te_stats.get("avg_trade", np.nan)),
            "latest_date": pd.to_datetime(dates[latest_i]),
            "latest_close": float(c[latest_i]),
            "latest_atr": float(atr[latest_i]) if np.isfinite(atr[latest_i]) else np.nan,
            "latest_score_ev": latest_ev,
            "signal": generate_signal_global(payload, global_params, latest_i),
            "score_0_100": float(latest_score100),
            "train_start": pd.to_datetime(dates[tr_idx[0]]).date(), "train_end": pd.to_datetime(dates[tr_idx[-1]]).date(),
            "test_start": pd.to_datetime(dates[te_idx[0]]).date(), "test_end": pd.to_datetime(dates[te_idx[-1]]).date(),
        })

        tkr_dt = time.perf_counter() - tkr_t0
        if idx % PRINT_EVERY == 0 or idx == len(ticker_payloads):
            elapsed3 = time.perf_counter() - phase3_t0
            eta3 = (elapsed3 / idx) * (len(ticker_payloads) - idx)
            print(f"[PHASE3] {idx:4d}/{len(ticker_payloads):4d} | {tkr:<10} | tr={int(stats_ga_train.get('n_trades',0)):4d} te={int(te_stats.get('n_trades',0)):4d} | tkr={tkr_dt:.2f}s | elapsed={elapsed3/60.0:.1f}m | eta={eta3/60.0:.1f}m")

    phase3_dt = time.perf_counter() - phase3_t0
    print(f"[PHASE3] done tickers={len(results_summary)} in {phase3_dt/60.0:.1f}m")

    if not results_summary:
        print("Nenhum ticker gerou resultado.")
        return

    summary_df = pd.DataFrame(results_summary).copy()
    summary_df["latest_date"] = pd.to_datetime(summary_df["latest_date"], errors="coerce")
    summary_df = summary_df.sort_values(["score_0_100", "ga_fitness_1y"], ascending=[False, False])

    apply_df = pd.DataFrame(results_apply).copy()
    apply_df["Date"] = pd.to_datetime(apply_df["Date"], errors="coerce")
    apply_df = apply_df.sort_values(["ticker", "Date"], ascending=[True, False])
    apply_df.to_csv(out_apply_csv, index=False, encoding="utf-8")
    summary_df.to_csv(out_summary_ckpt, index=False, encoding="utf-8")

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary_latest")
        apply_df.to_excel(writer, index=False, sheet_name=f"apply_last_{APPLY_DAYS}d")

    print(f"Saved: {out_xlsx} | {out_apply_csv}")


if __name__ == "__main__":
    run()































