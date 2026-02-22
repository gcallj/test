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
from concurrent.futures import ThreadPoolExecutor
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
MAX_FEATURES = 8

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
    ("score_percentile_trigger", 0.55, 0.90, 0.05, False),
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
    consec_long = 0
    consec_short = 0
    consec_stops = 0
    cooldown = 0

    for i in range(1, n):
        if cooldown > 0:
            cooldown -= 1

        if pos != 0:
            bars += 1
            stop_mult = gp.stop_atr_mult * (gp.stop_tighten_factor if bars >= gp.stop_tighten_after_bars else 1.0)
            stop_abs = max(ATR_EPS, stop_mult * max(atr[i], ATR_EPS))
            take_abs = gp.reward_risk_ratio * stop_abs
            hard_loss = abs(o[i] / max(entry_px, ATR_EPS) - 1.0)

            if hard_loss > gp.max_loss_per_trade_pct:
                ret = (o[i] / entry_px - 1.0) * pos - 0.0020
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
                equity *= (1.0 + part_ret - 0.0010)
                pos = pos * (1.0 - gp.partial_take_pct)
                partial_taken = True

            stop_hit = adv >= stop_abs
            take_hit = fav >= take_abs
            time_stop = (bars >= gp.time_stop_bars) and (fav < 0.5 * stop_abs)

            if stop_hit or take_hit or time_stop:
                exit_px = c[i]
                if stop_hit:
                    exit_px = entry_px - np.sign(pos) * stop_abs
                elif take_hit:
                    exit_px = entry_px + np.sign(pos) * take_abs
                
                cost = 0.0010 if partial_taken else 0.0020
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
                if gp.consecutive_loss_cooldown > 0 and consec_stops >= 2:
                    cooldown = gp.consecutive_loss_cooldown
                continue

        if pos == 0 and cooldown == 0 and np.isfinite(score_ev[i - 1]) and np.isfinite(score_pctl[i - 1]):
            if gp.volatility_filter_percentile > 0 and np.isfinite(vol_rank[i - 1]) and vol_rank[i - 1] < gp.volatility_filter_percentile:
                continue
            
            # Filtro de liquidez/volume (ex: não operar se volume relativo for muito baixo)
            # Assumindo que temos acesso ao volume ou podemos usar o ATR como proxy de liquidez
            if atr[i-1] < 0.01: # Filtro basico de liquidez
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
        return {"total_return": float(equity - 1.0), "mdd": float(mdd), "sharpe": 0.0, "n_trades": 0.0, "win_rate": 0.0, "avg_trade": 0.0}
    tr = np.asarray(trade_rets, dtype=np.float64)
    return {
        "total_return": float(equity - 1.0),
        "mdd": float(mdd),
        "sharpe": float(np.mean(tr) / (np.std(tr) + 1e-12) * np.sqrt(len(tr))),
        "n_trades": float(len(tr)),
        "win_rate": float((tr > 0).mean()),
        "avg_trade": float(np.mean(tr)),
        "trade_std": float(np.std(tr)),
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


def evaluate_global_walkforward(genome: List[float], payloads_by_window: List[Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]]) -> Tuple[float, float, float]:
    # Returns (fitness, sharpe_train, sharpe_val)
    train_sharpes = []
    val_sharpes = []
    
    for tr_payloads, te_payloads in payloads_by_window:
        if tr_payloads and te_payloads:
            # Evaluate on train
            tr_stats = []
            gp = decode_global_params(genome)
            for _, payload in tr_payloads.items():
                st = backtest_stats_global_intraday(
                    payload["open"], payload["high"], payload["low"], payload["close"],
                    payload["score_matrix"], payload["atr"], gp, precomputed=payload,
                )
                tr_stats.append(st)
            
            # Evaluate on test
            te_stats = []
            for _, payload in te_payloads.items():
                st = backtest_stats_global_intraday(
                    payload["open"], payload["high"], payload["low"], payload["close"],
                    payload["score_matrix"], payload["atr"], gp, precomputed=payload,
                )
                te_stats.append(st)
                
            # Compute metrics
            active_tr = [s for s in tr_stats if s.get("n_trades", 0) > 0]
            active_te = [s for s in te_stats if s.get("n_trades", 0) > 0]
            
            tr_sharpe = np.median([s.get("sharpe", 0.0) for s in active_tr]) if active_tr else 0.0
            te_sharpe = np.median([s.get("sharpe", 0.0) for s in active_te]) if active_te else 0.0
            
            # Filtros de robustez no treino
            mdds = [s.get("mdd", 0.0) for s in active_tr]
            penalty = 0.0
            if mdds and min(mdds) < -0.50:
                penalty += abs(min(mdds) + 0.50) * 5.0
                
            # Filtros de robustez na validacao
            te_win_rate = np.mean([s.get("win_rate", 0.0) for s in active_te]) if active_te else 0.0
            te_avg_trade = np.mean([s.get("avg_trade", 0.0) for s in active_te]) if active_te else 0.0
            te_trades = sum([s.get("n_trades", 0.0) for s in te_stats])
            
            if te_win_rate <= 0.45:
                penalty += (0.45 - te_win_rate) * 5.0
            if te_avg_trade <= 0.002:
                penalty += (0.002 - te_avg_trade) * 500.0
            if te_trades < 30:
                penalty += (30 - te_trades) * 0.05
                
            train_sharpes.append(tr_sharpe)
            val_sharpes.append(te_sharpe - penalty)
            
    if not val_sharpes:
        return -1e9, 0.0, 0.0
        
    mean_tr_sharpe = float(np.mean(train_sharpes))
    mean_te_sharpe = float(np.mean(val_sharpes))
    
    # Fitness = sharpe_oos - alpha * abs(sharpe_train - sharpe_oos)
    alpha = 0.5
    fitness = mean_te_sharpe - alpha * abs(mean_tr_sharpe - mean_te_sharpe)
    
    return fitness, mean_tr_sharpe, mean_te_sharpe

# Global variable for multiprocessing to avoid pickling issues
_GLOBAL_PAYLOADS_BY_WINDOW = None

def _eval_ind_global(ind):
    global _GLOBAL_PAYLOADS_BY_WINDOW
    res = evaluate_global_walkforward(list(ind), _GLOBAL_PAYLOADS_BY_WINDOW)
    return res

def run_global_ga_20params(full_df: pd.DataFrame, feature_cols: List[str], windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]], pop_size: int = 100, ngen: int = 40, ga_max_windows: int = 4):
    if not hasattr(creator, "Individual_Global20"):
        creator.create("Individual_Global20", list, fitness=creator.FitnessMax_PT)

    ga_t0 = time.perf_counter()
    windows_ga = windows[-int(ga_max_windows):] if (ga_max_windows is not None and int(ga_max_windows) > 0) else windows
    
    global _GLOBAL_PAYLOADS_BY_WINDOW
    _GLOBAL_PAYLOADS_BY_WINDOW = precompute_global_payloads(windows_ga, full_df, feature_cols)
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
    
    import multiprocessing
    
    # Cache de avaliações
    eval_cache = {}
    
    def eval_ind_with_cache(ind):
        tup = tuple(ind)
        if tup in eval_cache:
            return eval_cache[tup]
        return _eval_ind_global(ind)

    toolbox.register("evaluate", eval_ind_with_cache)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(5) # Top 5
    best_fit_seen = -1e18
    gens_without_improvement = 0
    diversity = 100.0
    
    pool = multiprocessing.Pool(processes=max(2, multiprocessing.cpu_count() - 1))
    
    for gen in range(1, int(ngen) + 1):
        gen_t0 = time.perf_counter()
        invalid = [ind for ind in pop if not hasattr(ind, 'fitness') or not ind.fitness.valid]
        
        # Paralelizar com multiprocessing
        if invalid:
            results = pool.map(_eval_ind_global, invalid)
            for ind, res in zip(invalid, results):
                ind.fitness.values = (res[0],)
                ind.sharpe_train = res[1]
                ind.sharpe_val = res[2]
                eval_cache[tuple(ind)] = res
                
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

        invalid_off = [ind for ind in offspring if not hasattr(ind, 'fitness') or not ind.fitness.valid]
        if invalid_off:
            results = pool.map(_eval_ind_global, invalid_off)
            for ind, res in zip(invalid_off, results):
                ind.fitness.values = (res[0],)
                ind.sharpe_train = res[1]
                ind.sharpe_val = res[2]
                eval_cache[tuple(ind)] = res
                
        hof.update(offspring)
        
        # Elitism: replace worst offspring with best from hof
        offspring.sort(key=lambda x: x.fitness.values[0], reverse=True)
        for i in range(len(hof)):
            offspring[-(i+1)] = toolbox.clone(hof[i])
            
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
            print(f"[GA] Early stopping at generation {gen} (no improvement in 10 gens)")
            break
            
    pool.close()
    pool.join()
    
    print("\n[GA] Top 5 cromossomos:")
    for i, ind in enumerate(hof):
        print(f"  #{i+1}: fitness={ind.fitness.values[0]:.3f} sharpe_tr={getattr(ind, 'sharpe_train', 0.0):.2f} sharpe_val={getattr(ind, 'sharpe_val', 0.0):.2f}")

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
    if (not np.isfinite(score_ev)):
        return 50.0
    recent = recent_scores_ev[np.isfinite(recent_scores_ev)] if recent_scores_ev is not None else np.array([], dtype=np.float64)
    if len(recent) == 0:
        pct_rank = 0.5
    else:
        pct_rank = float(np.mean(recent <= score_ev))
    
    # Usar percentil rank direto (0-100)
    score = pct_rank * 100.0
    return float(np.clip(score, 0.0, 100.0))

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

    bearish_tokens = ("risk", "down", "dd", "sell", "err_buy", "pe", "price_to_book")
    signs = []
    for c in feat_cols:
        lc = str(c).lower()
        sign = -1.0 if any(tok in lc for tok in bearish_tokens) else 1.0
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


# ==============================================================================
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
        feat_cols = [col for col, _ in feat_correlations[:MAX_FEATURES]]
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
            pop_size=40,
            ngen=20,
            ga_max_windows=2
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

        # Generate apply signals for last APPLY_DAYS
        for i in range(max(0, n - APPLY_DAYS), n):
            sig = generate_signal_global(payload, global_params, i)
            
            # Calculate score_ev for the day
            x = np.asarray(payload["score_matrix"], dtype=np.float64)
            feat_n = max(1, x.shape[1])
            votes_long = (x > global_params.z_threshold).sum(axis=1) / feat_n
            votes_short = (x < -global_params.z_threshold).sum(axis=1) / feat_n
            score_raw = votes_long - votes_short
            score_ev = pd.Series(score_raw).ewm(span=int(global_params.signal_ema_span), adjust=False).mean().to_numpy()
            
            # Calculate quality factor
            quality = compute_quality_factor(st["sharpe"], st["total_return"], st["n_trades"] / (n / 252))
            
            # Calculate 0-100 score
            lookback = max(63, SCORE_LOOKBACK)
            recent_scores = score_ev[max(0, i - lookback + 1):i + 1]
            score_100 = score_0_100_from_ev(score_ev[i], recent_scores, quality)

            # Calculate next day entry levels
            atr_val = payload["atr"][i]
            close_val = c[i]
            
            # Calculate strength
            score95 = np.nanpercentile(np.abs(score_ev), 95) if np.isfinite(np.nanmax(np.abs(score_ev))) else 1.0
            score95 = max(score95, ATR_EPS)
            strength = float(np.clip(abs(score_ev[i]) / score95, 0.0, 1.0))
            
            discount = global_params.entry_discount_atr_frac * (1.0 - global_params.score_strength_scaling * strength)
            
            best_buy = close_val - discount * atr_val
            best_sell = close_val + discount * atr_val
            
            # Check if filled next day
            next_day_filled = False
            entry_ref_price = np.nan
            if i + 1 < n:
                next_o = payload["open"][i+1]
                next_h = payload["high"][i+1]
                next_l = payload["low"][i+1]
                
                if sig == "buy":
                    limit_px = next_o - discount * payload["atr"][i+1]
                    next_day_filled = (next_l <= limit_px <= next_h)
                    entry_ref_price = limit_px if next_day_filled else next_o
                elif sig == "sell":
                    limit_px = next_o + discount * payload["atr"][i+1]
                    next_day_filled = (next_l <= limit_px <= next_h)
                    entry_ref_price = limit_px if next_day_filled else next_o
            
            results_apply.append({
                "Date": pd.to_datetime(dates[i]).strftime("%Y-%m-%d"),
                "ticker": tkr,
                "close": close_val,
                "signal_eod": sig,
                "score_100": score_100,
                "best_buy_value": best_buy if sig == "buy" else np.nan,
                "best_sell_value": best_sell if sig == "sell" else np.nan,
                "next_day_filled": next_day_filled,
                "entry_ref_price": entry_ref_price
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
        
        # Resumo Pós-Phase3
        pct_ret_pos = (df_sum["test_return"] > 0).mean() * 100
        pct_sharpe_pos = (df_sum["test_sharpe"] > 0).mean() * 100
        mean_ret = df_sum["test_return"].mean()
        med_ret = df_sum["test_return"].median()
        mean_bh = df_sum["buy_hold_return"].mean()
        
        print("\n=== RESUMO PÓS-PHASE3 ===")
        print(f"% tickers com test_return > 0: {pct_ret_pos:.1f}%")
        print(f"% tickers com test_sharpe > 0: {pct_sharpe_pos:.1f}%")
        print(f"Média test_return: {mean_ret:.4f} | Mediana: {med_ret:.4f}")
        print(f"Comparação: média test_return ({mean_ret:.4f}) vs média buy_hold ({mean_bh:.4f})")

        try:
            with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
                df_sum.to_excel(writer, sheet_name="Summary", index=False)
                if not df_app.empty:
                    df_app.to_excel(writer, sheet_name="Apply", index=False)
        except Exception as e:
            print(f"[WARN] xlsxwriter failed, trying openpyxl: {e}")
            with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
                df_sum.to_excel(writer, sheet_name="Summary", index=False)
                if not df_app.empty:
                    df_app.to_excel(writer, sheet_name="Apply", index=False)

    if not df_app.empty:
        df_app.to_csv(out_apply_csv, index=False)

    print(f"Saved: {out_xlsx} | {out_apply_csv}")


if __name__ == "__main__":
    run()
