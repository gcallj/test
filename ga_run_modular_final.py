# -*- coding: utf-8 -*-
"""
ga_run.py — Two-stage global GA with ProcessPoolExecutor + memmap
=================================================================
Replaces run_global_ga_20params from main_cell_v3.py.
Workers open PayloadStore in read-only mode (zero-copy memmap).
Checkpoint per generation; RuntimeGuard adjusts ngen/workers in-flight.

NOTE: This module includes copies of backtest_stats_global_intraday and its
helper functions from main_cell_v3.py so that worker processes can evaluate
genomes without importing the Colab-specific main module.
"""

import json
import math
import os
import pickle
import random
import time
from bisect import bisect_left, bisect_right, insort
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from payload_store import PayloadStore, WindowPlan
from auto_tune import AutoTuneResult, RuntimeGuard

# ---------------------------------------------------------------------------
# Constants (mirrored from main_cell_v3.py to keep module self-contained)
# ---------------------------------------------------------------------------
ATR_EPS = 1e-12
SCORE_LOOKBACK = 504

GA_CX_PB = 0.70
GA_MUT_PB = 0.40
GA_TOURN = 3
GA_MUT_SIGMA_START = 0.14
GA_MUT_SIGMA_END = 0.04
GA_GENE_MUT_PB_START = 0.25
GA_GENE_MUT_PB_END = 0.08
GA_HOF_SIZE = 10

GA_STAGE1_TOP_N = 6
GA_STAGE2_PADDING_RATIO = 0.60
GA_STAGE2_MIN_SPAN_RATIO = 0.18

# Friction / cost constants (mirrored from ga_run.py config)
LONG_ONLY = True
MIN_VOL_FIN_DAILY = 200_000
BROKERAGE_PER_ORDER = 7.00
DEFAULT_POSITION_SIZE = 5000.0
B3_FEES_PCT = 0.000325 * 2
SLIPPAGE_BPS = 10.0
IR_SWING_PCT = 0.15
COST_PER_TRADE_PCT = (2 * BROKERAGE_PER_ORDER / DEFAULT_POSITION_SIZE) + B3_FEES_PCT + (2 * SLIPPAGE_BPS / 10000)
MIN_PRICE = 0.01
CAP_DAILY_RET = 0.30
CAP_TRADE_RET = 3.00
GA_ALPHA_FOCUS = os.environ.get("GA_ALPHA_FOCUS", "off").lower() == "on"

GLOBAL_PARAM_SPECS = [
    ("vote_threshold_long", 0.15, 0.55, 0.05, False),
    ("vote_threshold_short", 0.10, 0.55, 0.05, False),
    ("z_threshold", 0.15, 0.80, 0.05, False),
    ("signal_ema_span", 2.0, 12.0, 1.0, True),
    ("entry_confirmation_days", 1.0, 3.0, 1.0, True),
    ("score_percentile_trigger", 0.35, 0.80, 0.05, False),
    ("stop_atr_mult", 1.0, 2.5, 0.25, False),
    ("stop_tighten_after_bars", 3.0, 15.0, 1.0, True),
    ("stop_tighten_factor", 0.40, 0.85, 0.05, False),
    ("max_loss_per_trade_pct", 0.02, 0.10, 0.01, False),
    ("reward_risk_ratio", 1.0, 5.0, 0.25, False),
    ("partial_take_pct", 0.0, 0.60, 0.10, False),
    ("partial_take_level", 0.5, 1.5, 0.25, False),
    ("time_stop_bars", 5.0, 25.0, 1.0, True),
    ("entry_discount_atr_frac", 0.0, 0.5, 0.05, False),
    ("volatility_filter_percentile", 0.0, 0.40, 0.05, False),
    ("score_strength_scaling", 0.0, 1.0, 0.1, False),
    ("ma_filter_period", 100.0, 300.0, 50.0, True),
    ("ma_filter_mode", 0.0, 2.0, 1.0, True),
    ("consecutive_loss_cooldown", 5.0, 20.0, 1.0, True),
    ("equity_drawdown_stop_pct", 0.08, 0.22, 0.02, False),
    ("vol_regime_mode", 0.0, 2.0, 1.0, True),
    ("partial_take_pct_2", 0.0, 0.40, 0.10, False),
    ("partial_take_level_2", 1.0, 3.0, 0.25, False),
    ("min_signal_strength", 0.0, 0.40, 0.05, False),
    ("trailing_stop_mode", 0.0, 2.0, 1.0, True),
    ("volume_confirm_mode", 0.0, 2.0, 1.0, True),
    ("momentum_confirm_days", 0.0, 5.0, 1.0, True),
    ("entry_score_threshold", 0.0, 0.50, 0.05, False),
    ("regime_threshold", 0.20, 0.70, 0.05, False),
]


# ---------------------------------------------------------------------------
# GlobalParams dataclass (mirrored from main_cell_v3.py)
# ---------------------------------------------------------------------------
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
    # -- NEW v3 genes --
    vol_regime_mode: int  # 0=off, 1=conservative(widen stops high vol), 2=skip high vol
    partial_take_pct_2: float  # 2nd partial take percentage
    partial_take_level_2: float  # 2nd partial take at this RR multiple
    min_signal_strength: float  # min |score_ev|/score95 to enter trade
    trailing_stop_mode: int  # 0=off, 1=breakeven after 1x stop, 2=trail at 50% after 2x
    # -- NEW v4 genes (signal precision + regime) --
    volume_confirm_mode: int  # 0=off, 1=vol>MA20, 2=vol>MA50
    momentum_confirm_days: int  # 0=off, 1-5=require positive return over N days
    entry_score_threshold: float  # min score_ev/score95 to enter
    regime_threshold: float  # min regime_score to allow long entries


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


# ---------------------------------------------------------------------------
# Utility functions (mirrored from main_cell_v3.py for worker processes)
# ---------------------------------------------------------------------------
def _rolling_mean_np(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
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


def _ewm_mean_np(x: np.ndarray, span: int) -> np.ndarray:
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


def _rolling_percentile_rank(arr: np.ndarray, window: int = 252) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    valid = np.isfinite(x)
    n = len(x)
    if n == 0:
        return out
    sorted_vals: list = []
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


def _rolling_quantile_trigger(arr: np.ndarray, q: float, window: int = SCORE_LOOKBACK) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    valid = np.isfinite(x)
    n = len(x)
    if n == 0:
        return out
    sorted_vals: list = []
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


# ---------------------------------------------------------------------------
# New helper functions from main ga_run.py
# ---------------------------------------------------------------------------
def compute_regime_score(c: np.ndarray, ma200: np.ndarray, vol_rank: np.ndarray) -> np.ndarray:
    """Score de 0 (bear) a 1 (bull) para cada barra.
    Combina: tendencia longa (close>MA200), tendencia media (MA50),
    slope da MA200, e volatilidade baixa."""
    n = len(c)
    regime = np.full(n, 0.5, dtype=np.float64)

    # Tendencia longa: close > MA200
    trend_long = np.where(np.isfinite(ma200) & (c > ma200), 0.30, 0.0)

    # Tendencia media: close > MA50 (rolling)
    ma50 = _rolling_mean_np(c, 50, 10)
    trend_med = np.where(np.isfinite(ma50) & (c > ma50), 0.20, 0.0)

    # Slope da MA200 (positiva = uptrend)
    ma200_shift = np.roll(ma200, 20)
    ma200_shift[:20] = np.nan
    ma200_slope = np.where(
        np.isfinite(ma200) & np.isfinite(ma200_shift) & (ma200 > 1e-6),
        (ma200 - ma200_shift) / np.maximum(ma200, 1e-6),
        0.0
    )
    slope_score = np.clip((ma200_slope + 0.05) * 5.0, 0.0, 0.25)

    # Volatilidade baixa = favoravel
    vol_score = np.where(np.isfinite(vol_rank), np.clip((1.0 - vol_rank) * 0.15, 0.0, 0.15), 0.075)

    # RSI zone (nao oversold nem overbought) -- approximate via price momentum
    # Use 14-day ROC as RSI proxy to avoid recomputing
    roc14 = np.empty(n, dtype=np.float64)
    roc14[:14] = 0.0
    roc14[14:] = (c[14:] / np.maximum(c[:-14], 1e-8)) - 1.0
    # Map: moderate momentum [-5%, +10%] -> bonus, extremes -> 0
    rsi_proxy = np.where((roc14 > -0.05) & (roc14 < 0.10), 0.10, 0.0)

    regime = np.clip(trend_long + trend_med + slope_score + vol_score + rsi_proxy, 0.0, 1.0)
    return regime


def compute_weighted_votes(x: np.ndarray, z_threshold: float, feat_n: int):
    """Votacao ponderada por magnitude: combina contagem binaria com peso continuo."""
    binary_long = (x > z_threshold).sum(axis=1) / feat_n
    binary_short = (x < -z_threshold).sum(axis=1) / feat_n

    # Peso continuo: magnitude do excesso sobre threshold
    z_excess_long = np.maximum(x - z_threshold, 0.0)
    z_excess_short = np.maximum(-x - z_threshold, 0.0)
    weighted_long = np.nanmean(z_excess_long, axis=1)
    weighted_short = np.nanmean(z_excess_short, axis=1)

    # Mix 50% binario + 50% ponderado (normalizado)
    votes_long = 0.5 * binary_long + 0.5 * np.clip(weighted_long / 1.5, 0.0, 1.0)
    votes_short = 0.5 * binary_short + 0.5 * np.clip(weighted_short / 1.5, 0.0, 1.0)

    return votes_long, votes_short


def _safe_metric_ratio(numerator: float, denominator: float, cap: float = 10.0) -> float:
    num = float(numerator) if np.isfinite(numerator) else 0.0
    den = float(denominator) if np.isfinite(denominator) else 0.0
    if abs(den) < 1e-12:
        if num > 0:
            return float(cap)
        if num < 0:
            return float(-cap)
        return 0.0
    return float(np.clip(num / den, -cap, cap))


def _safe_positive_ratio(numerator: float, denominator: float, cap: float = 10.0) -> float:
    num = float(numerator) if np.isfinite(numerator) else 0.0
    den = float(denominator) if np.isfinite(denominator) else 0.0
    if num <= 0:
        return 0.0
    if abs(den) < 1e-12:
        return float(cap)
    return float(np.clip(num / den, 0.0, cap))


def _compute_cagr(total_return: float, n_bars: int) -> float:
    n_years = max(float(n_bars) / 252.0, 1.0 / 252.0)
    equity = 1.0 + float(total_return)
    if not np.isfinite(equity):
        return 0.0
    if equity <= 0:
        return -1.0
    return float(np.clip(equity ** (1.0 / n_years) - 1.0, -1.0, 10.0))


def _extract_alpha_ann(stat: Dict[str, float]) -> float:
    alpha_ann = stat.get("alpha_ann", None)
    if alpha_ann is not None and np.isfinite(alpha_ann):
        return float(alpha_ann)
    if "buy_hold_cagr" in stat:
        return float(stat.get("cagr", 0.0) - stat.get("buy_hold_cagr", 0.0))
    return float(stat.get("excess_return", stat.get("total_return", 0.0)))


def _attach_benchmark_stats(stats: Dict[str, float], close: np.ndarray) -> Dict[str, float]:
    buy_hold_return = buyhold_capped(close)
    buy_hold_cagr = _compute_cagr(buy_hold_return, len(close))
    stats["buy_hold_return"] = float(buy_hold_return)
    stats["buy_hold_cagr"] = float(buy_hold_cagr)
    stats["alpha_ann"] = float(stats.get("cagr", _compute_cagr(stats.get("total_return", 0.0), len(close))) - buy_hold_cagr)
    return stats


def _finalize_backtest_stats(
    trade_rets: List[float],
    trade_exit_types: List[str],
    trade_hold_bars: List[float],
    exposure: float,
    total_return: float,
    mdd: float,
    n_bars: int,
    max_drawdown_duration: int,
) -> Dict[str, float]:
    base_stats = {
        "total_return": float(total_return),
        "mdd": float(mdd),
        "sharpe": 0.0,
        "sortino": 0.0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "payoff_ratio": 0.0,
        "max_drawdown_duration": float(max_drawdown_duration),
        "cagr": _compute_cagr(total_return, n_bars),
        "n_trades": 0.0,
        "win_rate": 0.0,
        "win_rate_target": 0.0,
        "win_rate_target_robust": 0.0,
        "pct_exit_take": 0.0,
        "pct_exit_stop": 0.0,
        "pct_exit_time": 0.0,
        "avg_trade": 0.0,
        "trade_std": 0.0,
        "avg_hold_bars": 0.0,
        "median_hold_bars": 0.0,
        "max_hold_bars": 0.0,
        "exposure": float(exposure),
        "pct_gt15": 0.0,
        "pct_10_15": 0.0,
        "pct_5_10": 0.0,
        "pct_2_5": 0.0,
        "pct_0_2": 0.0,
        "pct_n2_0": 0.0,
        "pct_n5_n2": 0.0,
        "pct_n10_n5": 0.0,
        "pct_lt_n10": 0.0,
        "dist_quality": 0.0,
        "big_wins_pct": 0.0,
        "big_losses_pct": 0.0,
    }
    if len(trade_rets) == 0:
        return base_stats

    tr = np.asarray(trade_rets, dtype=np.float64)
    exit_arr = np.asarray(trade_exit_types)
    hold_arr = np.asarray(trade_hold_bars, dtype=np.float64) if len(trade_hold_bars) > 0 else np.zeros(len(tr), dtype=np.float64)
    if hold_arr.shape[0] != tr.shape[0]:
        hold_arr = np.resize(hold_arr, tr.shape[0])
    n_years = max(float(n_bars) / 252.0, 0.1)
    trades_per_year = len(tr) / n_years
    ann_factor = np.sqrt(min(trades_per_year, 252.0))
    mean_tr = float(np.mean(tr))
    std_tr = float(np.std(tr))
    downside = np.minimum(tr, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) > 0 else 0.0

    pct_exit_take = float((exit_arr == "take").mean()) if len(exit_arr) > 0 else 0.0
    pct_exit_stop = float(((exit_arr == "stop") | (exit_arr == "hard_loss")).mean()) if len(exit_arr) > 0 else 0.0
    pct_exit_time = float((exit_arr == "time").mean()) if len(exit_arr) > 0 else 0.0
    take_mask = (exit_arr == "take")
    time_pos_mask = (exit_arr == "time") & (tr > 0)
    wr_target_robust = float((take_mask | time_pos_mask).mean()) if len(exit_arr) > 0 else 0.0

    wins = tr[tr > 0]
    losses = tr[tr < 0]
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = abs(float(np.mean(losses))) if len(losses) > 0 else 0.0
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
    win_rate = float((tr > 0).mean())
    expectancy = float(np.clip(win_rate * avg_win - (1.0 - win_rate) * avg_loss, -1.0, 1.0))
    ann_sharpe = float(np.clip(_safe_metric_ratio(mean_tr, std_tr, cap=10.0) * ann_factor, -10.0, 10.0))
    ann_sortino = float(np.clip(_safe_metric_ratio(mean_tr, downside_dev, cap=10.0) * ann_factor, -10.0, 10.0))
    profit_factor = _safe_positive_ratio(gross_profit, gross_loss, cap=10.0)
    payoff_ratio = _safe_positive_ratio(avg_win, avg_loss, cap=10.0)

    pct_gt15 = float((tr > 0.15).mean())
    pct_10_15 = float(((tr > 0.10) & (tr <= 0.15)).mean())
    pct_5_10 = float(((tr > 0.05) & (tr <= 0.10)).mean())
    pct_2_5 = float(((tr > 0.02) & (tr <= 0.05)).mean())
    pct_0_2 = float(((tr > 0.00) & (tr <= 0.02)).mean())
    pct_n2_0 = float(((tr > -0.02) & (tr <= 0.00)).mean())
    pct_n5_n2 = float(((tr > -0.05) & (tr <= -0.02)).mean())
    pct_n10_n5 = float(((tr > -0.10) & (tr <= -0.05)).mean())
    pct_lt_n10 = float((tr <= -0.10).mean())
    big_wins = pct_gt15 + pct_10_15 + pct_5_10
    big_losses = pct_lt_n10 + pct_n10_n5 + pct_n5_n2
    dist_quality = float(np.clip((big_wins - big_losses + 0.5), 0.0, 1.0))

    base_stats.update({
        "sharpe": ann_sharpe,
        "sortino": ann_sortino,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "payoff_ratio": payoff_ratio,
        "n_trades": float(len(tr)),
        "win_rate": win_rate,
        "win_rate_target": pct_exit_take,
        "win_rate_target_robust": wr_target_robust,
        "pct_exit_take": pct_exit_take,
        "pct_exit_stop": pct_exit_stop,
        "pct_exit_time": pct_exit_time,
        "avg_trade": mean_tr,
        "trade_std": std_tr,
        "avg_hold_bars": float(np.mean(hold_arr)) if len(hold_arr) > 0 else 0.0,
        "median_hold_bars": float(np.median(hold_arr)) if len(hold_arr) > 0 else 0.0,
        "max_hold_bars": float(np.max(hold_arr)) if len(hold_arr) > 0 else 0.0,
        "pct_gt15": pct_gt15,
        "pct_10_15": pct_10_15,
        "pct_5_10": pct_5_10,
        "pct_2_5": pct_2_5,
        "pct_0_2": pct_0_2,
        "pct_n2_0": pct_n2_0,
        "pct_n5_n2": pct_n5_n2,
        "pct_n10_n5": pct_n10_n5,
        "pct_lt_n10": pct_lt_n10,
        "dist_quality": dist_quality,
        "big_wins_pct": big_wins,
        "big_losses_pct": big_losses,
    })
    return base_stats


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


# ---------------------------------------------------------------------------
# Backtest engine (from main ga_run.py — full v6 with trailing stops,
# 2nd partial, vol regime, volume/momentum confirmation, regime score,
# realistic costs)
# ---------------------------------------------------------------------------
def _backtest_stats_global_intraday(
    o, h, l, c, score_matrix, atr, gp: GlobalParams,
    precomputed: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
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
    ma = _rolling_mean_np(c, int(gp.ma_filter_period), max(20, int(gp.ma_filter_period // 2)))
    if precomputed is not None and ("vol_rank" in precomputed):
        vol_rank = np.asarray(precomputed["vol_rank"], dtype=np.float64)
    else:
        vol_rank = _rolling_percentile_rank(atr / np.maximum(c, ATR_EPS), 252)

    # Weighted voting (magnitude-aware)
    votes_long, votes_short = compute_weighted_votes(x, gp.z_threshold, feat_n)
    score_raw = votes_long - votes_short
    score_ev = _ewm_mean_np(score_raw, int(gp.signal_ema_span))
    score_pctl = _rolling_quantile_trigger(score_ev, float(gp.score_percentile_trigger), max(63, SCORE_LOOKBACK))

    score95 = np.nanpercentile(np.abs(score_ev), 95) if np.isfinite(np.nanmax(np.abs(score_ev))) else 1.0
    score95 = max(score95, ATR_EPS)

    # Regime score (multi-factor bull/bear detection)
    ma200 = _rolling_mean_np(c, 200, 50)
    regime_score = compute_regime_score(c, ma200, vol_rank)

    # Volume confirmation: precompute volume moving averages
    vol_data = precomputed.get("volume", None) if precomputed is not None else None
    vol_ma20 = None
    vol_ma50 = None
    if vol_data is not None and gp.volume_confirm_mode > 0:
        vol_arr = np.asarray(vol_data, dtype=np.float64)
        vol_ma20 = _rolling_mean_np(vol_arr, 20, 5)
        if gp.volume_confirm_mode == 2:
            vol_ma50 = _rolling_mean_np(vol_arr, 50, 10)

    # Momentum confirmation: precompute N-day returns
    mom_days = int(gp.momentum_confirm_days)
    mom_ret = np.full(n, np.nan, dtype=np.float64)
    if mom_days > 0:
        mom_ret[mom_days:] = (c[mom_days:] / np.maximum(c[:-mom_days], 1e-8)) - 1.0

    equity = 1.0
    peak = 1.0
    mdd = 0.0
    trade_rets = []
    trade_exit_types = []  # v8: track "take", "stop", "time", "hard_loss"
    trade_hold_bars = []
    pos = 0
    entry_px = np.nan
    bars = 0
    partial_taken = False
    partial_taken_2 = False
    max_fav = 0.0  # track maximum favorable excursion for trailing stop
    consec_long = 0
    consec_short = 0
    consec_stops = 0
    cooldown = 0
    exposure_acc = 0.0
    current_drawdown_duration = 0
    max_drawdown_duration = 0

    for i in range(1, n):
        if cooldown > 0:
            cooldown -= 1

        if abs(pos) > 0:
            exposure_acc += float(min(1.0, abs(pos)))
        if pos != 0:
            bars += 1
            # -- Dynamic stop tightening --
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
                ret = (o[i] / entry_px - 1.0) * pos - COST_PER_TRADE_PCT
                equity *= (1.0 + ret)
                trade_rets.append(ret)
                trade_exit_types.append("hard_loss")
                trade_hold_bars.append(float(max(1, bars)))
                consec_stops += 1 if ret < 0 else 0
                if ret > 0:
                    consec_stops = 0
                pos = 0
                max_fav = 0.0
                continue

            fav = ((h[i] - entry_px) if pos > 0 else (entry_px - l[i]))
            adv = ((entry_px - l[i]) if pos > 0 else (h[i] - entry_px))
            max_fav = max(max_fav, fav)

            # -- Trailing stop logic --
            effective_stop = stop_abs
            if gp.trailing_stop_mode >= 1 and max_fav > stop_abs:
                # Mode 1+2: move stop to breakeven after profit > 1x stop
                effective_stop = max_fav  # breakeven = 0 adverse from max
                if gp.trailing_stop_mode == 2 and max_fav > 2.0 * stop_abs:
                    # Mode 2: trail at 50% of max favorable excursion
                    effective_stop = 0.5 * max_fav

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

            # Use effective_stop for trailing, but original stop_abs for take/time
            trail_adv = max_fav - fav  # how far price pulled back from best
            stop_hit = (adv >= stop_abs) if gp.trailing_stop_mode == 0 else (trail_adv >= effective_stop if max_fav > stop_abs else adv >= stop_abs)
            take_hit = fav >= take_abs
            time_stop = (bars >= gp.time_stop_bars) and (fav < 0.5 * stop_abs)

            if stop_hit or take_hit or time_stop:
                exit_px = c[i]
                if stop_hit:
                    # For trailing stop: exit at trailed level, not original stop
                    if gp.trailing_stop_mode > 0 and max_fav > stop_abs:
                        trail_stop_px = entry_px + np.sign(pos) * (max_fav - effective_stop)
                        ideal_exit = trail_stop_px
                    else:
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

                gross_ret = (exit_px / entry_px - 1.0) * pos
                # Custos: corretagem + B3 + slippage
                cost = COST_PER_TRADE_PCT
                if partial_taken or partial_taken_2:
                    cost *= 0.7  # partial takes already paid some friction
                net_ret = gross_ret - cost
                # IR: 15% sobre lucro (swing trade, isencao mensal ignorada por conservadorismo)
                if net_ret > 0:
                    net_ret = net_ret * (1.0 - IR_SWING_PCT)
                ret = net_ret
                equity *= (1.0 + ret)
                trade_rets.append(ret)
                # v8: track exit type for win_rate_target computation
                if take_hit:
                    trade_exit_types.append("take")
                elif stop_hit:
                    trade_exit_types.append("stop")
                else:
                    trade_exit_types.append("time")
                trade_hold_bars.append(float(max(1, bars)))
                if stop_hit and ret < 0:
                    consec_stops += 1
                elif ret > 0:
                    consec_stops = 0
                pos = 0
                bars = 0
                partial_taken = False
                partial_taken_2 = False
                max_fav = 0.0
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
            # Entry score threshold filter (v4)
            if gp.entry_score_threshold > 0:
                if abs(score_ev[i - 1]) / score95 < gp.entry_score_threshold:
                    continue
            # Volume confirmation filter (v4)
            if gp.volume_confirm_mode == 1 and vol_ma20 is not None and i - 1 < len(vol_ma20):
                if np.isfinite(vol_ma20[i - 1]) and vol_data[i - 1] < vol_ma20[i - 1]:
                    continue
            elif gp.volume_confirm_mode == 2 and vol_ma50 is not None and i - 1 < len(vol_ma50):
                if np.isfinite(vol_ma50[i - 1]) and vol_data[i - 1] < vol_ma50[i - 1]:
                    continue
            # Momentum confirmation filter (v4)
            if mom_days > 0 and np.isfinite(mom_ret[i - 1]) and mom_ret[i - 1] < 0:
                continue
            # Regime filter (v4): continuous regime score replaces binary MA filter
            if gp.regime_threshold > 0 and np.isfinite(regime_score[i - 1]):
                if regime_score[i - 1] < gp.regime_threshold:
                    continue  # market regime unfavorable for longs

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
                # Support-adaptive discount: scale by distance to recent support
                base_discount = gp.entry_discount_atr_frac * (1.0 - gp.score_strength_scaling * strength)
                recent_low = np.nanmin(l[max(0, i - 10):i]) if i > 0 else l[i]
                support_dist = (o[i] - recent_low) / max(atr[i], ATR_EPS) if np.isfinite(recent_low) else 1.0
                support_factor = float(np.clip(support_dist / 3.0, 0.3, 1.5))
                discount = base_discount * support_factor
                limit_px = o[i] - side * discount * atr[i]
                fill = (l[i] <= limit_px <= h[i])
                if fill and limit_px > 0:
                    pos = side
                    entry_px = float(limit_px)
                    bars = 0
                    partial_taken = False
                    partial_taken_2 = False
                    max_fav = 0.0
                elif np.isfinite(o[i]) and o[i] > 0:
                    pos = side
                    entry_px = float(o[i])
                    bars = 0
                    partial_taken = False
                    partial_taken_2 = False
                    max_fav = 0.0

        peak = max(peak, equity)
        drawdown = (equity / max(peak, ATR_EPS)) - 1.0
        mdd = min(mdd, drawdown)
        if drawdown < -1e-12:
            current_drawdown_duration += 1
            max_drawdown_duration = max(max_drawdown_duration, current_drawdown_duration)
        else:
            current_drawdown_duration = 0
        # Equity drawdown circuit-breaker (progressive re-entry: 20 bars, was 30)
        if gp.equity_drawdown_stop_pct > 0 and drawdown < -gp.equity_drawdown_stop_pct:
            cooldown = max(cooldown, 20)  # ~1 month pause after DD exceeds threshold

    exposure = float(exposure_acc / max(1, n - 1))
    return _finalize_backtest_stats(
        trade_rets=trade_rets,
        trade_exit_types=trade_exit_types,
        trade_hold_bars=trade_hold_bars,
        exposure=exposure,
        total_return=float(equity - 1.0),
        mdd=float(mdd),
        n_bars=n,
        max_drawdown_duration=max_drawdown_duration,
    )


# ---------------------------------------------------------------------------
# Fitness aggregation (v6 from main ga_run.py — MDD hard walls, win-rate
# tiers, distribution quality bonuses, calmar bonus, consistency bonus)
# ---------------------------------------------------------------------------
def global_fitness_from_stats(per_ticker_stats: List[Dict[str, float]]) -> float:
    """
    Multi-objective fitness optimised for:
      - High win rate (prefer few quality trades over many losing ones)
      - Low max drawdown (MDD must stay small)
      - Beat buy-and-hold on annualized alpha
      - Statistical volume (enough trades for significance)
    """
    if len(per_ticker_stats) == 0:
        return -1e9
    sharpe    = np.array([s.get("sharpe", 0.0) for s in per_ticker_stats], dtype=np.float64)
    sortino   = np.array([s.get("sortino", 0.0) for s in per_ticker_stats], dtype=np.float64)
    ret       = np.array([s.get("total_return", 0.0) for s in per_ticker_stats], dtype=np.float64)
    raw_excess = np.array([s.get("excess_return", s.get("total_return", 0.0)) for s in per_ticker_stats], dtype=np.float64)
    alpha_ann_vals = np.array([_extract_alpha_ann(s) for s in per_ticker_stats], dtype=np.float64)
    expectancy_vals = np.array([s.get("expectancy", s.get("avg_trade", 0.0)) for s in per_ticker_stats], dtype=np.float64)
    payoff_vals = np.array([s.get("payoff_ratio", 0.0) for s in per_ticker_stats], dtype=np.float64)
    profit_factor_vals = np.array([s.get("profit_factor", 0.0) for s in per_ticker_stats], dtype=np.float64)
    cagr_vals = np.array([s.get("cagr", 0.0) for s in per_ticker_stats], dtype=np.float64)
    hold_avg_vals = np.array([s.get("avg_hold_bars", 0.0) for s in per_ticker_stats], dtype=np.float64)
    hold_med_vals = np.array([s.get("median_hold_bars", 0.0) for s in per_ticker_stats], dtype=np.float64)
    hold_max_vals = np.array([s.get("max_hold_bars", 0.0) for s in per_ticker_stats], dtype=np.float64)
    mdd_duration_vals = np.array([s.get("max_drawdown_duration", 0.0) for s in per_ticker_stats], dtype=np.float64)
    mdd_vals  = np.array([s.get("mdd", 0.0) for s in per_ticker_stats], dtype=np.float64)
    # Phase 1.3: Cap extreme MDD at -60% to prevent outliers (-99%) from destroying median
    mdd_vals  = np.clip(mdd_vals, -0.60, 0.0)
    ntr       = np.array([s.get("n_trades", 0.0) for s in per_ticker_stats], dtype=np.float64)
    exposure  = np.array([s.get("exposure", 0.0) for s in per_ticker_stats], dtype=np.float64)
    win_rates = np.array([s.get("win_rate", 0.0) for s in per_ticker_stats], dtype=np.float64)
    wr_targets = np.array([s.get("win_rate_target", 0.0) for s in per_ticker_stats], dtype=np.float64)
    dist_qual  = np.array([s.get("dist_quality", 0.5) for s in per_ticker_stats], dtype=np.float64)
    big_wins   = np.array([s.get("big_wins_pct", 0.0) for s in per_ticker_stats], dtype=np.float64)
    big_losses = np.array([s.get("big_losses_pct", 0.0) for s in per_ticker_stats], dtype=np.float64)

    med_sharpe        = float(np.median(sharpe))
    mean_sharpe       = float(np.mean(sharpe))
    med_sortino       = float(np.median(sortino))
    mean_sortino      = float(np.mean(sortino))
    mean_ret          = float(np.mean(ret))
    med_ret           = float(np.median(ret))
    mean_excess       = float(np.mean(raw_excess))
    med_excess        = float(np.median(raw_excess))
    mean_alpha_ann    = float(np.mean(alpha_ann_vals))
    med_alpha_ann     = float(np.median(alpha_ann_vals))
    mean_expectancy   = float(np.mean(expectancy_vals))
    med_expectancy    = float(np.median(expectancy_vals))
    mean_payoff       = float(np.mean(payoff_vals))
    med_payoff        = float(np.median(payoff_vals))
    mean_profit_factor = float(np.mean(profit_factor_vals))
    med_profit_factor = float(np.median(profit_factor_vals))
    mean_cagr         = float(np.mean(cagr_vals))
    med_cagr          = float(np.median(cagr_vals))
    mean_hold_avg     = float(np.mean(hold_avg_vals))
    med_hold_med      = float(np.median(hold_med_vals))
    p75_hold_max      = float(np.percentile(hold_max_vals, 75)) if len(hold_max_vals) > 0 else 0.0
    mean_mdd_duration = float(np.mean(mdd_duration_vals))
    med_mdd_duration  = float(np.median(mdd_duration_vals))
    pct_positive      = float((ret > 0).mean())
    pct_excess_pos    = float((raw_excess > 0).mean())
    pct_alpha_pos     = float((alpha_ann_vals > 0).mean())
    med_trades        = float(np.median(ntr))
    mean_trades       = float(np.mean(ntr))
    mean_exposure     = float(np.mean(exposure))
    mean_win_rate     = float(np.mean(win_rates))
    med_win_rate      = float(np.median(win_rates))
    mean_wr_target    = float(np.mean(wr_targets))
    med_wr_target     = float(np.median(wr_targets))
    mean_mdd          = float(np.mean(mdd_vals))   # negative
    median_mdd        = float(np.median(mdd_vals)) # negative
    alpha_pos_floor   = 0.22 if GA_ALPHA_FOCUS else 0.18

    # -- Activity bonus/penalty --------------------------------------------
    # Target: 15-80 trades per year per ticker. Penalise extremes.
    # Tighter target range (was 100) to reduce churn and cost drag.
    trade_bonus = 0.0
    if med_trades < 8:
        trade_bonus -= (8 - med_trades) * 0.10   # hard penalty for near-zero trades
    elif med_trades < 15:
        trade_bonus -= (15 - med_trades) * 0.04
    if med_trades > 80:
        trade_bonus -= (med_trades - 80) * 0.012  # tighter target (was 100 x 0.008)
    if med_trades > 200:
        trade_bonus -= (med_trades - 200) * 0.020  # heavy penalty above 200 (was 250)
    if med_trades > 400:
        trade_bonus -= (med_trades - 400) * 0.025  # catastrophic above 400/yr
    if mean_exposure < 0.40:
        trade_bonus -= (0.40 - mean_exposure) * 2.0

    # -- MDD penalty (v7 -- smooth softplus, no hard walls) ------------------
    # Softplus replaces piecewise linear to avoid discontinuities in GA landscape
    def _softplus(x, k=5.0):
        """Smooth penalty: 0 when x<0, rises steeply when x>0."""
        cx = max(min(float(k * x), 50.0), -50.0)
        return math.log1p(math.exp(cx)) / k

    mdd_penalty = 0.0
    abs_mdd = abs(median_mdd)
    # Primary: smooth ramp centered at 15%, steep above 20%
    mdd_penalty += 1.6 * _softplus((abs_mdd - 0.15) / 0.03, k=5.0)
    # Bonus for excellent control
    if abs_mdd < 0.18:
        mdd_penalty -= 4.0
    if abs_mdd < 0.12:
        mdd_penalty -= 3.0

    # Tail-risk penalty (worst-quartile): smooth ramp above 22%
    mdd_p25 = float(np.percentile(mdd_vals, 25))
    abs_mdd_tail = abs(mdd_p25)
    mdd_penalty += 0.8 * _softplus((abs_mdd_tail - 0.22) / 0.04, k=5.0)
    if abs_mdd_tail > 0.30:
        mdd_penalty += (abs_mdd_tail - 0.30) * 60.0
    if abs_mdd_tail > 0.40:
        mdd_penalty += (abs_mdd_tail - 0.40) * 100.0

    alpha_drag_penalty = 0.0
    if mean_alpha_ann < -0.06 and abs_mdd > 0.10:
        alpha_drag_penalty += (abs_mdd - 0.10) * (40.0 if GA_ALPHA_FOCUS else 20.0)
    if mean_alpha_ann < -0.08 and abs_mdd > 0.12:
        alpha_drag_penalty += (abs_mdd - 0.12) * (80.0 if GA_ALPHA_FOCUS else 40.0)
    if med_alpha_ann < -0.06 and abs_mdd > 0.11:
        alpha_drag_penalty += (abs_mdd - 0.11) * (36.0 if GA_ALPHA_FOCUS else 18.0)
    if pct_alpha_pos < alpha_pos_floor and abs_mdd > 0.10:
        alpha_drag_penalty += (alpha_pos_floor - pct_alpha_pos) * (18.0 if GA_ALPHA_FOCUS else 9.0)

    # -- Under-performance penalty -----------------------------------------
    underperf_penalty = 0.0
    alpha_mean_w = 34.0 if GA_ALPHA_FOCUS else 24.0
    alpha_med_w = 20.0 if GA_ALPHA_FOCUS else 14.0
    alpha_floor_mean = -0.02 if GA_ALPHA_FOCUS else -0.03
    alpha_floor_med = -0.015 if GA_ALPHA_FOCUS else -0.025
    if mean_alpha_ann < 0:
        underperf_penalty += alpha_mean_w * abs(mean_alpha_ann)
    if med_alpha_ann < 0:
        underperf_penalty += alpha_med_w * abs(med_alpha_ann)
    if mean_alpha_ann < alpha_floor_mean:
        underperf_penalty += (alpha_floor_mean - mean_alpha_ann) * (22.0 if GA_ALPHA_FOCUS else 10.0)
    if med_alpha_ann < alpha_floor_med:
        underperf_penalty += (alpha_floor_med - med_alpha_ann) * (16.0 if GA_ALPHA_FOCUS else 8.0)
    if pct_alpha_pos < alpha_pos_floor:
        underperf_penalty += (alpha_pos_floor - pct_alpha_pos) * (10.0 if GA_ALPHA_FOCUS else 6.0)
    if pct_alpha_pos < 0.35:
        underperf_penalty += (0.35 - pct_alpha_pos) * (7.0 if GA_ALPHA_FOCUS else 5.0)
    if mean_alpha_ann < -0.08:
        underperf_penalty += (abs(mean_alpha_ann) - 0.08) * (60.0 if GA_ALPHA_FOCUS else 30.0)
    if mean_alpha_ann < -0.09:
        underperf_penalty += (abs(mean_alpha_ann) - 0.09) * (85.0 if GA_ALPHA_FOCUS else 42.0)
    if med_alpha_ann < -0.06:
        underperf_penalty += (abs(med_alpha_ann) - 0.06) * (40.0 if GA_ALPHA_FOCUS else 20.0)
    if med_alpha_ann < -0.075:
        underperf_penalty += (abs(med_alpha_ann) - 0.075) * (55.0 if GA_ALPHA_FOCUS else 28.0)
    if mean_ret < 0.015:
        underperf_penalty += (0.015 - mean_ret) * 25.0
    if mean_ret < 0.0:
        underperf_penalty += abs(mean_ret) * 30.0

    alpha_recovery_bonus = 0.0
    if mean_alpha_ann > -0.10:
        alpha_recovery_bonus += (mean_alpha_ann + 0.10) * (24.0 if GA_ALPHA_FOCUS else 12.0)
    if mean_alpha_ann > -0.12:
        alpha_recovery_bonus += (mean_alpha_ann + 0.12) * (18.0 if GA_ALPHA_FOCUS else 10.0)
    if mean_alpha_ann > -0.09:
        alpha_recovery_bonus += (mean_alpha_ann + 0.09) * (36.0 if GA_ALPHA_FOCUS else 18.0)
    if mean_alpha_ann > -0.08:
        alpha_recovery_bonus += (mean_alpha_ann + 0.08) * (28.0 if GA_ALPHA_FOCUS else 14.0)
    if mean_alpha_ann > -0.07:
        alpha_recovery_bonus += (mean_alpha_ann + 0.07) * (52.0 if GA_ALPHA_FOCUS else 26.0)
    if mean_alpha_ann > -0.04:
        alpha_recovery_bonus += (mean_alpha_ann + 0.04) * (40.0 if GA_ALPHA_FOCUS else 20.0)
    if med_alpha_ann > -0.10:
        alpha_recovery_bonus += (med_alpha_ann + 0.10) * (10.0 if GA_ALPHA_FOCUS else 6.0)
    if med_alpha_ann > -0.08:
        alpha_recovery_bonus += (med_alpha_ann + 0.08) * (20.0 if GA_ALPHA_FOCUS else 10.0)
    if med_alpha_ann > -0.06:
        alpha_recovery_bonus += (med_alpha_ann + 0.06) * (18.0 if GA_ALPHA_FOCUS else 10.0)
    if pct_alpha_pos > 0.17:
        alpha_recovery_bonus += (pct_alpha_pos - 0.17) * (10.0 if GA_ALPHA_FOCUS else 5.0)
    if pct_alpha_pos > 0.18:
        alpha_recovery_bonus += (pct_alpha_pos - 0.18) * (8.0 if GA_ALPHA_FOCUS else 4.0)
    if pct_alpha_pos > 0.30:
        alpha_recovery_bonus += (pct_alpha_pos - 0.30) * (12.0 if GA_ALPHA_FOCUS else 6.0)

    # -- Win-rate bonus (v9: operational floor at 60%) ----------
    # Rationale: WR < 60% nao deve ser operado (aprovado pelo usuario).
    # Gradient: hard penalty abaixo de 60% + bonus agressivo acima.
    win_rate_bonus = 0.0
    # HARD floor: penalty severa abaixo de 60% (novo em v9)
    if mean_win_rate < 0.60:
        win_rate_bonus -= (0.60 - mean_win_rate) * 30.0  # 2x antes (15->30)
    if mean_win_rate < 0.55:
        win_rate_bonus -= (0.55 - mean_win_rate) * 20.0  # penalty extra abaixo de 55%
    if mean_win_rate < 0.50:
        win_rate_bonus -= (0.50 - mean_win_rate) * 40.0  # catastrofico abaixo de 50%
    # Base: linear from 60% upward (BASELINE agora e 60%, nao 55%)
    if mean_win_rate > 0.60:
        win_rate_bonus += (mean_win_rate - 0.60) * 8.0
    if mean_win_rate > 0.65:
        win_rate_bonus += (mean_win_rate - 0.65) * 9.0
    if mean_win_rate > 0.70:
        win_rate_bonus += (mean_win_rate - 0.70) * 10.0
    # Median win rate (consistency across tickers)
    if med_win_rate < 0.60:
        win_rate_bonus -= (0.60 - med_win_rate) * 10.0  # penalty ate mediana >= 60%
    if med_win_rate > 0.60:
        win_rate_bonus += (med_win_rate - 0.60) * 4.0
    if med_win_rate > 0.65:
        win_rate_bonus += (med_win_rate - 0.65) * 6.0
    # v7: Aggressive push above current baseline (68%)
    # Steep gradient only kicks in above 0.68 to incentivize pushing past current ceiling
    if med_win_rate > 0.68:
        win_rate_bonus += (med_win_rate - 0.68) * 18.0
    if med_win_rate > 0.70:
        win_rate_bonus += (med_win_rate - 0.70) * 26.0
    if mean_win_rate > 0.68:
        win_rate_bonus += (mean_win_rate - 0.68) * 12.0

    # v9: penalty por pct_sub60 (quantos tickers estao abaixo de 60%)
    # Desincentiva modelos que tem WR mean alto mas muita variacao
    pct_sub60 = float((win_rates < 0.60).mean())
    if pct_sub60 > 0.30:
        win_rate_bonus -= (pct_sub60 - 0.30) * 15.0  # forte penalty se > 30% dos tickers sub-60

    # -- Win-rate TARGET bonus (v8: reward actual alvo hits) ----------
    # Pushes GA to find genomes where trades actually hit take-profit target,
    # not just get closed by trailing stop at small profits. Baseline median
    # across 119 tickers is ~45%; max ~83%. Aggressive gradient above 50%.
    # Bonus values calibrated so that +10pp on median adds ~15 to fitness
    # (comparable weight to WR bonus 40-60x at similar thresholds).
    wr_target_bonus = 0.0
    # Sub-35% penalty (alvo raramente atingido = modelo nao sabe onde colocar alvo)
    if mean_wr_target < 0.35:
        wr_target_bonus -= (0.35 - mean_wr_target) * 25.0
    # Linear from 35% baseline upward
    if mean_wr_target > 0.35:
        wr_target_bonus += (mean_wr_target - 0.35) * 10.0
    # Tiered: aggressive rewards above 45% (current baseline median)
    if mean_wr_target > 0.45:
        wr_target_bonus += (mean_wr_target - 0.45) * 20.0
    if mean_wr_target > 0.55:
        wr_target_bonus += (mean_wr_target - 0.55) * 30.0
    if mean_wr_target > 0.65:
        wr_target_bonus += (mean_wr_target - 0.65) * 40.0
    # Median-based reward (consistencia across tickers)
    if med_wr_target > 0.40:
        wr_target_bonus += (med_wr_target - 0.40) * 8.0
    # Aggressive push above current median 45.5%
    if med_wr_target > 0.50:
        wr_target_bonus += (med_wr_target - 0.50) * 25.0
    if med_wr_target > 0.60:
        wr_target_bonus += (med_wr_target - 0.60) * 45.0
    if med_wr_target > 0.70:
        wr_target_bonus += (med_wr_target - 0.70) * 60.0

    # -- Consistency bonus: reward stable per-ticker performance -----------
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

    # -- Calmar ratio bonus (v4) ---------------------------------------------
    calmar_bonus = 0.0
    if abs(median_mdd) > 0.01:
        calmar = mean_ret / abs(median_mdd)
        calmar_bonus = 0.8 * float(np.clip(calmar, 0.0, 5.0))

    # -- Distribution quality bonus (v6: reward high-gain precision) --------
    mean_dist_qual = float(np.mean(dist_qual))
    mean_big_wins = float(np.mean(big_wins))
    mean_big_losses = float(np.mean(big_losses))
    dist_bonus = 0.0
    if mean_dist_qual > 0.55:
        dist_bonus += (mean_dist_qual - 0.55) * 5.0
    if mean_dist_qual > 0.70:
        dist_bonus += (mean_dist_qual - 0.70) * 8.0
    if mean_big_losses > 0.12:
        dist_bonus -= (mean_big_losses - 0.12) * 8.0  # stronger penalty for big losses
    # v6: bonus for high % of trades with >10% gain (precision in high-gain trades)
    mean_pct_gt10 = float(np.mean([s.get("pct_gt15", 0.0) + s.get("pct_10_15", 0.0) for s in per_ticker_stats]))
    if mean_pct_gt10 > 0.10:
        dist_bonus += (mean_pct_gt10 - 0.10) * 8.0  # reward more >10% trades
    if mean_pct_gt10 > 0.20:
        dist_bonus += (mean_pct_gt10 - 0.20) * 12.0  # strong reward

    # -- Trade quality bonus (P0 metrics) ----------------------------------
    sortino_bonus = 0.0
    if mean_sortino > 0.0:
        sortino_bonus += 0.35 * float(np.clip(mean_sortino, 0.0, 4.0))
    if med_sortino > 0.0:
        sortino_bonus += 0.25 * float(np.clip(med_sortino, 0.0, 4.0))

    profit_factor_bonus = 0.0
    if mean_profit_factor > 1.0:
        profit_factor_bonus += 0.40 * float(np.clip(mean_profit_factor - 1.0, 0.0, 2.0))
    if med_profit_factor > 1.0:
        profit_factor_bonus += 0.30 * float(np.clip(med_profit_factor - 1.0, 0.0, 2.0))
    if mean_payoff > 1.0:
        profit_factor_bonus += 0.10 * float(np.clip(mean_payoff - 1.0, 0.0, 2.0))
    if med_payoff > 1.0:
        profit_factor_bonus += 0.10 * float(np.clip(med_payoff - 1.0, 0.0, 2.0))

    expectancy_bonus = 0.0
    if mean_expectancy > 0.0:
        expectancy_bonus += 30.0 * float(np.clip(mean_expectancy, 0.0, 0.03))
    if med_expectancy > 0.0:
        expectancy_bonus += 18.0 * float(np.clip(med_expectancy, 0.0, 0.03))
    if mean_cagr > 0.0:
        expectancy_bonus += 0.10 * float(np.clip(mean_cagr, 0.0, 2.0))
    if med_cagr > 0.0:
        expectancy_bonus += 0.08 * float(np.clip(med_cagr, 0.0, 2.0))

    drawdown_duration_penalty = 0.0
    med_mdd_duration_years = med_mdd_duration / 252.0
    mean_mdd_duration_years = mean_mdd_duration / 252.0
    if med_mdd_duration_years > 0.15:
        drawdown_duration_penalty += (med_mdd_duration_years - 0.15) * 2.0
    if med_mdd_duration_years > 0.35:
        drawdown_duration_penalty += (med_mdd_duration_years - 0.35) * 4.0
    if mean_mdd_duration_years > 0.20:
        drawdown_duration_penalty += (mean_mdd_duration_years - 0.20) * 1.5

    swing_bonus = 0.0
    if mean_hold_avg < 2.0:
        swing_bonus -= (2.0 - mean_hold_avg) * 0.40
    elif mean_hold_avg <= 18.0:
        swing_bonus += min(mean_hold_avg - 2.0, 8.0) * 0.10
    elif mean_hold_avg > 30.0:
        swing_bonus -= (mean_hold_avg - 30.0) * 0.08
    if med_hold_med >= 3.0 and med_hold_med <= 15.0:
        swing_bonus += 0.90
    elif med_hold_med > 22.0:
        swing_bonus -= (med_hold_med - 22.0) * 0.07
    elif med_hold_med < 2.0:
        swing_bonus -= (2.0 - med_hold_med) * 0.20
    if p75_hold_max > 45.0:
        swing_bonus -= (p75_hold_max - 45.0) * 0.03

    quality_bonus = sortino_bonus + profit_factor_bonus + expectancy_bonus
    alpha_quality_scale = 1.0
    if mean_alpha_ann < 0.0:
        alpha_quality_scale *= 0.35 if GA_ALPHA_FOCUS else 0.45
    if mean_alpha_ann < -0.06:
        alpha_quality_scale *= 0.70
    if med_alpha_ann < -0.05:
        alpha_quality_scale *= 0.80
    if pct_alpha_pos < (0.22 if GA_ALPHA_FOCUS else 0.18):
        alpha_quality_scale *= 0.75 if GA_ALPHA_FOCUS else 0.85
    if pct_alpha_pos < 0.35:
        alpha_quality_scale *= 0.75
    quality_bonus *= alpha_quality_scale

    # -- Main fitness (v5.2 alpha-hard) ------------------------------------
    # GA_ALPHA_FOCUS: amplifica sinal de alpha para forcar GA a priorizar
    # reducao de underperformance vs B&H via alpha anualizado.
    _alpha_w_mean = 26.0 if GA_ALPHA_FOCUS else 12.0
    _alpha_w_med = 18.0 if GA_ALPHA_FOCUS else 8.0
    _alpha_pos_w = 4.5 if GA_ALPHA_FOCUS else 2.5
    _alpha_clip_lo = -0.35 if GA_ALPHA_FOCUS else -0.20
    fitness = (
        _alpha_w_mean * np.clip(mean_alpha_ann, _alpha_clip_lo, 0.50) +
        _alpha_w_med  * np.clip(med_alpha_ann,  _alpha_clip_lo, 0.50) +
        _alpha_pos_w * pct_alpha_pos +              # % tickers beating B&H on annualized alpha
        0.5 * np.clip(mean_ret,    -1.0, 5.0) +
        0.3 * np.clip(med_ret,     -1.0, 5.0) +
        0.5 * np.clip(mean_sharpe, -2.0, 3.0) +
        0.4 * np.clip(med_sharpe,  -2.0, 3.0) +
        0.3 * pct_positive +
        # Win rate: use ONLY tiered bonus (no double-counting with direct weight)
        win_rate_bonus +
        wr_target_bonus +                              # v8: reward alvo hits
        consistency_bonus +
        calmar_bonus +
        alpha_recovery_bonus +
        swing_bonus +
        quality_bonus +
        dist_bonus +                                   # v5: distribution quality (big wins vs big losses)
        trade_bonus -
        alpha_drag_penalty -
        drawdown_duration_penalty -
        mdd_penalty -
        underperf_penalty
    )
    return float(fitness)


# ---------------------------------------------------------------------------
# Genome evaluation on payloads
# ---------------------------------------------------------------------------
def _evaluate_genome_on_payloads(
    gp: GlobalParams,
    ticker_payloads: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[float, List[Dict[str, float]]]:
    """Evaluate a decoded GlobalParams against ticker payloads.
    Returns (fitness_scalar, list_of_per_ticker_stats)."""
    stats = []
    for _, payload in ticker_payloads.items():
        st = _backtest_stats_global_intraday(
            payload["open"], payload["high"], payload["low"], payload["close"],
            payload["score_matrix"], payload["atr"], gp, precomputed=payload,
        )
        _attach_benchmark_stats(st, payload["close"])
        st["excess_return"] = st["total_return"] - st["buy_hold_return"]
        stats.append(st)
    return global_fitness_from_stats(stats), stats


def _summarize_window_stats(stats: List[Dict[str, float]], fitness: float) -> Dict[str, float]:
    if not stats:
        return {
            "fitness": float(fitness),
            "ticker_count": 0,
            "win_rate": 0.0,
            "win_rate_target": 0.0,
            "mdd": 0.0,
            "excess_return": 0.0,
            "alpha_ann": 0.0,
        }
    win_rates = np.array([s.get("win_rate", 0.0) for s in stats], dtype=np.float64)
    wr_targets = np.array([s.get("win_rate_target", 0.0) for s in stats], dtype=np.float64)
    mdd_vals = np.array([s.get("mdd", 0.0) for s in stats], dtype=np.float64)
    excess_vals = np.array([s.get("excess_return", s.get("total_return", 0.0)) for s in stats], dtype=np.float64)
    alpha_vals = np.array([_extract_alpha_ann(s) for s in stats], dtype=np.float64)
    return {
        "fitness": float(fitness),
        "ticker_count": int(len(stats)),
        "win_rate": float(np.median(win_rates)) if len(win_rates) else 0.0,
        "win_rate_target": float(np.median(wr_targets)) if len(wr_targets) else 0.0,
        "mdd": float(np.median(mdd_vals)) if len(mdd_vals) else 0.0,
        "excess_return": float(np.mean(excess_vals)) if len(excess_vals) else 0.0,
        "alpha_ann": float(np.mean(alpha_vals)) if len(alpha_vals) else 0.0,
    }


def compute_walkforward_diagnostics(
    genome: List[float],
    store: PayloadStore,
    window_plans: List[WindowPlan],
    window_indices: List[int],
) -> Dict[str, Any]:
    gp = decode_global_params(genome)
    window_metrics: List[Dict[str, Any]] = []
    train_summaries: List[Dict[str, float]] = []
    oos_summaries: List[Dict[str, float]] = []

    for wi in window_indices:
        wp = window_plans[wi]
        train_payloads, test_payloads = store.get_window_payloads(wp)
        if not train_payloads or not test_payloads:
            continue
        tr_fit, tr_stats = _evaluate_genome_on_payloads(gp, train_payloads)
        te_fit, te_stats = _evaluate_genome_on_payloads(gp, test_payloads)
        tr_summary = _summarize_window_stats(tr_stats, tr_fit)
        te_summary = _summarize_window_stats(te_stats, te_fit)
        train_summaries.append(tr_summary)
        oos_summaries.append(te_summary)
        window_metrics.append({
            "window_index": int(wi),
            "train_start": str(wp.train_start) if wp.train_start is not None else None,
            "train_end": str(wp.train_end) if wp.train_end is not None else None,
            "test_start": str(wp.test_start) if wp.test_start is not None else None,
            "test_end": str(wp.test_end) if wp.test_end is not None else None,
            "train": tr_summary,
            "oos": te_summary,
        })

    def _mean_summary(items: List[Dict[str, float]]) -> Dict[str, float]:
        if not items:
            return {
                "fitness": 0.0,
                "win_rate": 0.0,
                "win_rate_target": 0.0,
                "mdd": 0.0,
                "excess_return": 0.0,
                "alpha_ann": 0.0,
            }
        return {
            "fitness": float(np.mean([item["fitness"] for item in items])),
            "win_rate": float(np.mean([item["win_rate"] for item in items])),
            "win_rate_target": float(np.mean([item["win_rate_target"] for item in items])),
            "mdd": float(np.mean([item["mdd"] for item in items])),
            "excess_return": float(np.mean([item["excess_return"] for item in items])),
            "alpha_ann": float(np.mean([item["alpha_ann"] for item in items])),
        }

    train_mean = _mean_summary(train_summaries)
    oos_mean = _mean_summary(oos_summaries)
    gap_summary = {
        "fitness_gap": float(train_mean["fitness"] - oos_mean["fitness"]),
        "win_rate_gap": float(train_mean["win_rate"] - oos_mean["win_rate"]),
        "win_rate_target_gap": float(train_mean["win_rate_target"] - oos_mean["win_rate_target"]),
        "mdd_gap": float(train_mean["mdd"] - oos_mean["mdd"]),
        "excess_return_gap": float(train_mean["excess_return"] - oos_mean["excess_return"]),
        "alpha_ann_gap": float(train_mean["alpha_ann"] - oos_mean["alpha_ann"]),
    }

    return {
        "selected_window_indices": [int(wi) for wi in window_indices],
        "window_count": int(len(window_metrics)),
        "train_mean": train_mean,
        "oos_mean": oos_mean,
        "gap_summary": gap_summary,
        "window_metrics": window_metrics,
    }


def _evaluate_walkforward(
    genome: List[float],
    store: PayloadStore,
    window_plans: List[WindowPlan],
    window_indices: List[int],
) -> Tuple[float, float, float]:
    """Evaluate genome on selected windows. Returns (fitness, mean_train, mean_oos)."""
    gp = decode_global_params(genome)
    train_vals = []
    oos_vals = []

    for wi in window_indices:
        wp = window_plans[wi]
        train_payloads, test_payloads = store.get_window_payloads(wp)
        if train_payloads and test_payloads:
            tr_fit, _ = _evaluate_genome_on_payloads(gp, train_payloads)
            te_fit, _ = _evaluate_genome_on_payloads(gp, test_payloads)
            train_vals.append(tr_fit)
            oos_vals.append(te_fit)

    if not oos_vals:
        return -1e9, -1e9, -1e9

    mean_train = float(np.mean(train_vals))
    mean_oos = float(np.mean(oos_vals))
    # Combine train and test: primarily OOS, small penalty for overfitting
    overfit_penalty = max(0.0, mean_train - mean_oos) * 0.3
    fitness = mean_oos - overfit_penalty
    return fitness, mean_train, mean_oos


# ---------------------------------------------------------------------------
# Worker process functions (module-level for Windows spawn compatibility)
# ---------------------------------------------------------------------------
_worker_store: Optional[PayloadStore] = None
_worker_window_plans: Optional[List[WindowPlan]] = None


def _worker_init(store_path: str, window_plans_bytes: bytes):
    """Called once per worker process. Opens PayloadStore read-only."""
    global _worker_store, _worker_window_plans
    _worker_store = PayloadStore(store_path, mode="r")
    _worker_window_plans = pickle.loads(window_plans_bytes)


def _evaluate_batch(args: Tuple[List[List[float]], List[int]]) -> List[Tuple[float, float, float]]:
    """
    Evaluate a batch of genomes against specified window indices.
    Called in worker process. Uses module-level _worker_store.
    """
    genome_batch, window_indices = args
    results = []
    assert _worker_store is not None
    assert _worker_window_plans is not None

    for genome in genome_batch:
        fit = _evaluate_walkforward(genome, _worker_store, _worker_window_plans, window_indices)
        results.append(fit)
    return results


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
@dataclass
class GACheckpoint:
    stage: int
    generation: int
    best_fitness: float
    best_genome: List[float]
    hall_of_fame: List[List[float]]
    hof_fitnesses: List[float]
    elapsed_seconds: float
    timestamp: str


def _save_checkpoint(ckpt: GACheckpoint, checkpoint_dir: str) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    fname = f"stage{ckpt.stage}_gen{ckpt.generation:03d}.json"
    path = os.path.join(checkpoint_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(ckpt), f, ensure_ascii=False, indent=2)
    return path


def _load_latest_checkpoint(checkpoint_dir: str) -> Optional[GACheckpoint]:
    if not os.path.isdir(checkpoint_dir):
        return None
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("stage") and f.endswith(".json")]
    if not files:
        return None
    files.sort()
    path = os.path.join(checkpoint_dir, files[-1])
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return GACheckpoint(**raw)


# ---------------------------------------------------------------------------
# DEAP setup (lazy, guarded)
# ---------------------------------------------------------------------------
_deap_ready = False


def _setup_deap():
    global _deap_ready
    if _deap_ready:
        return
    from deap import base, creator
    if not hasattr(creator, "FitnessMax_GA"):
        creator.create("FitnessMax_GA", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual_Global20"):
        creator.create("Individual_Global20", list, fitness=creator.FitnessMax_GA)
    _deap_ready = True


# ---------------------------------------------------------------------------
# Population evaluation
# ---------------------------------------------------------------------------
def _evaluate_population_parallel(
    population: list,
    window_indices: List[int],
    n_workers: int,
    store_path: str,
    window_plans_bytes: bytes,
    batch_size: int = 8,
) -> List[Tuple[float, float, float]]:
    """
    Evaluate all invalid individuals in population using ProcessPoolExecutor.
    Falls back to sequential if ProcessPool fails.
    Returns list of (fitness, mean_train, mean_oos) for each invalid individual.
    """
    invalid = [ind for ind in population if not ind.fitness.valid]
    if not invalid:
        return []

    genome_list = [list(ind) for ind in invalid]

    # Sequential path
    if n_workers <= 1:
        return _evaluate_batch_sequential(genome_list, window_indices, store_path, window_plans_bytes)

    # Build batches
    batches = []
    for i in range(0, len(genome_list), batch_size):
        batches.append((genome_list[i:i + batch_size], window_indices))

    try:
        all_results: List[Tuple[float, float, float]] = []
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(store_path, window_plans_bytes),
        ) as executor:
            futures = [executor.submit(_evaluate_batch, batch) for batch in batches]
            for fi, future in enumerate(futures):
                batch_results = future.result(timeout=3600)
                all_results.extend(batch_results)
                # Progress
                done = min(len(all_results), len(invalid))
                if fi == 0 or fi == len(futures) - 1 or (fi + 1) % max(1, len(futures) // 5) == 0:
                    print(f"    eval {done}/{len(invalid)} (batch {fi+1}/{len(futures)})")
        return all_results

    except Exception as e:
        print(f"[GA] ProcessPool failed ({e}), falling back to sequential")
        return _evaluate_batch_sequential(genome_list, window_indices, store_path, window_plans_bytes)


def _evaluate_batch_sequential(
    genome_list: List[List[float]],
    window_indices: List[int],
    store_path: str,
    window_plans_bytes: bytes,
) -> List[Tuple[float, float, float]]:
    """Sequential fallback for evaluation."""
    # Use module-level store if already initialized, otherwise open
    global _worker_store, _worker_window_plans
    if _worker_store is None or _worker_window_plans is None:
        _worker_init(store_path, window_plans_bytes)
    assert _worker_store is not None
    assert _worker_window_plans is not None

    results = []
    total = len(genome_list)
    for gi, genome in enumerate(genome_list):
        fit = _evaluate_walkforward(genome, _worker_store, _worker_window_plans, window_indices)
        results.append(fit)
        if gi == 0 or gi == total - 1 or (gi + 1) % max(1, total // 10) == 0:
            print(f"    eval {gi+1}/{total}")
    return results


# ---------------------------------------------------------------------------
# Stage 1 -> Stage 2 transfer
# ---------------------------------------------------------------------------
def _pick_diverse_top(hof, pop, n_top: int) -> List[List[float]]:
    """
    Extract top diverse genomes from HOF + population for Stage 2 seeding.
    Uses L2 distance in normalized parameter space.
    """
    # Collect all unique candidates with fitness
    seen = set()
    candidates = []
    for ind in list(hof) + list(pop):
        key = tuple(round(g, 6) for g in ind)
        if key not in seen:
            seen.add(key)
            fit = ind.fitness.values[0] if ind.fitness.valid else -1e9
            candidates.append((list(ind), fit))

    candidates.sort(key=lambda x: x[1], reverse=True)
    if len(candidates) <= n_top:
        return [c[0] for c in candidates]

    # Normalize parameter space
    ranges = np.array([(hi - lo) for _, lo, hi, _, _ in GLOBAL_PARAM_SPECS], dtype=np.float64)
    ranges = np.maximum(ranges, 1e-12)

    selected = [candidates[0][0]]  # best always included
    for _ in range(n_top - 1):
        best_dist = -1
        best_cand = None
        for genome, _ in candidates:
            if any(genome == s for s in selected):
                continue
            min_dist = float("inf")
            g_arr = np.array(genome, dtype=np.float64) / ranges
            for s in selected:
                s_arr = np.array(s, dtype=np.float64) / ranges
                d = float(np.linalg.norm(g_arr - s_arr))
                min_dist = min(min_dist, d)
            if min_dist > best_dist:
                best_dist = min_dist
                best_cand = genome
        if best_cand is not None:
            selected.append(best_cand)
        else:
            break

    return selected


def _build_stage2_ranges(seeds: List[List[float]]) -> List[Tuple[float, float]]:
    """
    Build narrower search ranges around Stage 1 seeds.
    Returns list of (lo, hi) per gene.
    """
    seeds_arr = np.array(seeds, dtype=np.float64)
    ranges = []
    for i, (name, lo_full, hi_full, step, is_int) in enumerate(GLOBAL_PARAM_SPECS):
        col = seeds_arr[:, i]
        c_min, c_max = float(col.min()), float(col.max())
        full_span = hi_full - lo_full
        pad = GA_STAGE2_PADDING_RATIO * (c_max - c_min + step)
        min_span = GA_STAGE2_MIN_SPAN_RATIO * full_span

        new_lo = max(lo_full, c_min - pad)
        new_hi = min(hi_full, c_max + pad)
        span = new_hi - new_lo
        if span < min_span:
            center = (new_lo + new_hi) / 2.0
            new_lo = max(lo_full, center - min_span / 2.0)
            new_hi = min(hi_full, center + min_span / 2.0)
        ranges.append((new_lo, new_hi))
    return ranges


# ---------------------------------------------------------------------------
# Single GA stage
# ---------------------------------------------------------------------------
def _run_stage(
    stage: int,
    pop_size: int,
    ngen: int,
    window_indices: List[int],
    n_workers: int,
    store_path: str,
    window_plans_bytes: bytes,
    checkpoint_dir: str,
    batch_size: int = 8,
    seed_genomes: Optional[List[List[float]]] = None,
    param_ranges: Optional[List[Tuple[float, float]]] = None,
    resume_from: Optional[GACheckpoint] = None,
    guard: Optional[RuntimeGuard] = None,
) -> Tuple[list, Any]:  # (population, HallOfFame)
    """
    Run one stage of the GA. Saves checkpoint after every generation.
    Returns (final_population, hall_of_fame).
    """
    _setup_deap()
    from deap import base, creator, tools

    toolbox = base.Toolbox()

    # Determine parameter ranges
    if param_ranges is None:
        ranges = [(lo, hi) for (_, lo, hi, _, _) in GLOBAL_PARAM_SPECS]
    else:
        ranges = param_ranges

    _n_params = len(GLOBAL_PARAM_SPECS)

    for i, (lo, hi) in enumerate(ranges):
        toolbox.register(f"attr_g{i}", random.uniform, float(lo), float(hi))

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual_Global20,
        tuple(getattr(toolbox, f"attr_g{i}") for i in range(_n_params)),
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "select", tools.selTournament, tournsize=GA_TOURN
    )
    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=[lo for lo, _ in ranges],
        up=[hi for _, hi in ranges],
        eta=20.0,
    )

    # Population init
    pop = toolbox.population(n=pop_size)

    # Seed with top individuals from previous stage or checkpoint
    if seed_genomes:
        for i, g in enumerate(seed_genomes[:pop_size]):
            pop[i][:] = g
            del pop[i].fitness.values

    hof = tools.HallOfFame(GA_HOF_SIZE)
    start_gen = 1
    ga_t0 = time.perf_counter()

    # Resume from checkpoint
    if resume_from is not None and resume_from.stage == stage:
        start_gen = resume_from.generation + 1
        for g, f in zip(resume_from.hall_of_fame, resume_from.hof_fitnesses):
            ind = creator.Individual_Global20(g)
            ind.fitness.values = (f,)
            hof.insert(ind)
        print(f"[GA] Stage {stage} resuming from gen {start_gen}, "
              f"best={resume_from.best_fitness:.5f}")

    current_ngen = ngen
    current_workers = n_workers

    for gen in range(start_gen, current_ngen + 1):
        gen_t0 = time.perf_counter()

        # Mutation params (adaptive decay)
        frac = (gen - 1) / max(1, current_ngen - 1)
        sigma = GA_MUT_SIGMA_START + (GA_MUT_SIGMA_END - GA_MUT_SIGMA_START) * frac
        gene_pb = GA_GENE_MUT_PB_START + (GA_GENE_MUT_PB_END - GA_GENE_MUT_PB_START) * frac

        def _mut(ind, _sigma=sigma, _gene_pb=gene_pb):
            for j in range(len(ind)):
                if random.random() < _gene_pb:
                    lo, hi = ranges[j]
                    ind[j] = float(ind[j]) + random.gauss(0.0, _sigma * (hi - lo))
                    ind[j] = float(np.clip(ind[j], lo, hi))
            return (ind,)

        toolbox.register("mutate", _mut)

        # Evaluate population
        eval_t0 = time.perf_counter()
        results = _evaluate_population_parallel(
            pop, window_indices, current_workers,
            store_path, window_plans_bytes, batch_size,
        )
        # Assign fitness
        idx = 0
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = (results[idx][0],)
                idx += 1

        hof.update(pop)

        # Selection + crossover + mutation -> offspring
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for i in range(1, len(offspring), 2):
            if random.random() < GA_CX_PB:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values
        for i in range(len(offspring)):
            if random.random() < GA_MUT_PB:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # Evaluate offspring
        results_off = _evaluate_population_parallel(
            offspring, window_indices, current_workers,
            store_path, window_plans_bytes, batch_size,
        )
        idx = 0
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = (results_off[idx][0],)
                idx += 1

        hof.update(offspring)
        pop[:] = offspring

        gen_elapsed = time.perf_counter() - gen_t0
        eval_elapsed = time.perf_counter() - eval_t0
        best_fit = float(hof[0].fitness.values[0]) if len(hof) else -1e9

        # Save checkpoint
        ckpt = GACheckpoint(
            stage=stage,
            generation=gen,
            best_fitness=best_fit,
            best_genome=list(hof[0]) if len(hof) else [s[1] for s in GLOBAL_PARAM_SPECS],
            hall_of_fame=[list(ind) for ind in hof],
            hof_fitnesses=[float(ind.fitness.values[0]) for ind in hof],
            elapsed_seconds=gen_elapsed,
            timestamp=datetime.now().isoformat(),
        )
        _save_checkpoint(ckpt, checkpoint_dir)

        # Progress
        total_elapsed = time.perf_counter() - ga_t0
        eta = (total_elapsed / max(gen - start_gen + 1, 1)) * (current_ngen - gen)
        print(f"[GA] S{stage} gen {gen:03d}/{current_ngen:03d} | "
              f"best={best_fit:.5f} | gen={gen_elapsed:.1f}s | "
              f"elapsed={total_elapsed/60:.1f}m | eta={eta/60:.1f}m | "
              f"workers={current_workers}")

        # RuntimeGuard
        if guard is not None:
            adj = guard.after_generation(gen, current_ngen, gen_elapsed)
            if adj["checkpoint_and_exit"]:
                print(f"[GA] S{stage} guard: checkpoint and exit at gen {gen}")
                break
            if adj["ngen_adjusted"] is not None:
                current_ngen = adj["ngen_adjusted"]
            if adj["reduce_workers"] is not None:
                current_workers = adj["reduce_workers"]

    return pop, hof


# ---------------------------------------------------------------------------
# Entry point: two-stage global GA
# ---------------------------------------------------------------------------
def run_global_ga_two_stage(
    store_path: str,
    window_plans: List[WindowPlan],
    feature_cols: List[str],
    tune_result: AutoTuneResult,
    checkpoint_dir: str,
    run_mode: str = "train",
    seed_genome: Optional[List[float]] = None,
) -> Tuple[GlobalParams, List[float], float]:
    """
    Two-stage global GA with ProcessPoolExecutor.

    Stage 1: Exploration with fewer windows, larger population.
    Stage 2: Refinement of top individuals with more windows.

    If seed_genome is provided, Stage 1 population is warm-started with
    the seed + small perturbations (previous best genome).

    Returns: (GlobalParams, sanitized_genome, best_fitness)
    """
    _setup_deap()

    random.seed(42)
    np.random.seed(42)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Serialize window plans once (small: just indices)
    window_plans_bytes = pickle.dumps(window_plans)

    # Select window indices for each stage — UNIFORM sampling across full history
    # (covers crises: 2008, 2015, 2020, plus recent — avoids recency bias)
    n_available = len(window_plans)
    s1_win_count = min(tune_result.stage1_window_count, n_available)
    s2_win_count = min(tune_result.stage2_window_count, n_available)

    def _uniform_sample_indices(n_total: int, k: int) -> List[int]:
        if n_total <= k:
            return list(range(n_total))
        step = n_total / k
        indices = [int(round(i * step)) for i in range(k)]
        return [min(max(0, idx), n_total - 1) for idx in indices]

    s1_indices = _uniform_sample_indices(n_available, s1_win_count)
    s2_indices = _uniform_sample_indices(n_available, s2_win_count)
    print(f"[GA] Window indices: S1={s1_indices} S2={s2_indices} (out of {n_available})")

    guard = RuntimeGuard(tune_result)

    # Check for resume
    resume_ckpt = None
    if run_mode == "train":
        resume_ckpt = _load_latest_checkpoint(checkpoint_dir)
        if resume_ckpt:
            print(f"[GA] found checkpoint: stage={resume_ckpt.stage} "
                  f"gen={resume_ckpt.generation} fit={resume_ckpt.best_fitness:.5f}")

    # Stage 1
    s1_done = (resume_ckpt is not None and resume_ckpt.stage >= 2)
    s1_resume = resume_ckpt if (resume_ckpt and resume_ckpt.stage == 1) else None

    # Build warm-start seeds from previous checkpoint genome
    s1_seed_genomes = None
    if seed_genome is not None and len(seed_genome) == len(GLOBAL_PARAM_SPECS):
        sanitized_seed = sanitize_global_genome(seed_genome)
        s1_seed_genomes = [sanitized_seed]
        # Add small perturbations of the seed
        n_perturbations = min(7, tune_result.stage1_pop // 8)
        for _ in range(n_perturbations):
            varied = []
            for g, (_, lo, hi, _, _) in zip(sanitized_seed, GLOBAL_PARAM_SPECS):
                v = float(np.clip(g + random.gauss(0.0, 0.06 * (hi - lo)), lo, hi))
                varied.append(v)
            s1_seed_genomes.append(sanitize_global_genome(varied))
        print(f"[GA] Warm-starting Stage 1 with {len(s1_seed_genomes)} seeds from previous best")

    if not s1_done:
        print(f"\n[GA] === STAGE 1: Exploration ===")
        print(f"[GA] windows={s1_win_count} pop={tune_result.stage1_pop} "
              f"ngen={tune_result.stage1_ngen} workers={tune_result.n_workers}")

        pop1, hof1 = _run_stage(
            stage=1,
            pop_size=tune_result.stage1_pop,
            ngen=tune_result.stage1_ngen,
            window_indices=s1_indices,
            n_workers=tune_result.n_workers,
            store_path=store_path,
            window_plans_bytes=window_plans_bytes,
            checkpoint_dir=checkpoint_dir,
            batch_size=tune_result.batch_size,
            seed_genomes=s1_seed_genomes,
            resume_from=s1_resume,
            guard=guard,
        )
    else:
        # Reconstruct HOF from checkpoint
        print(f"[GA] Stage 1 already done (from checkpoint)")
        pop1 = None
        from deap import tools, creator
        hof1 = tools.HallOfFame(GA_HOF_SIZE)
        assert resume_ckpt is not None
        for g, f in zip(resume_ckpt.hall_of_fame, resume_ckpt.hof_fitnesses):
            ind = creator.Individual_Global20(g)
            ind.fitness.values = (f,)
            hof1.insert(ind)

    # Stage 1 -> Stage 2 transfer
    if pop1 is not None:
        seeds = _pick_diverse_top(hof1, pop1, GA_STAGE1_TOP_N)
    else:
        seeds = [list(ind) for ind in hof1][:GA_STAGE1_TOP_N]

    best_s1 = float(hof1[0].fitness.values[0]) if len(hof1) else -1e9
    print(f"\n[GA] Stage 1 best fitness: {best_s1:.5f}")
    print(f"[GA] Transferring {len(seeds)} seeds to Stage 2")

    # Stage 2
    if tune_result.stage2_pop > 0 and tune_result.stage2_ngen > 0:
        s2_ranges = _build_stage2_ranges(seeds)
        s2_resume = resume_ckpt if (resume_ckpt and resume_ckpt.stage == 2) else None

        print(f"\n[GA] === STAGE 2: Refinement ===")
        print(f"[GA] windows={s2_win_count} pop={tune_result.stage2_pop} "
              f"ngen={tune_result.stage2_ngen} workers={tune_result.n_workers}")

        pop2, hof2 = _run_stage(
            stage=2,
            pop_size=tune_result.stage2_pop,
            ngen=tune_result.stage2_ngen,
            window_indices=s2_indices,
            n_workers=tune_result.n_workers,
            store_path=store_path,
            window_plans_bytes=window_plans_bytes,
            checkpoint_dir=checkpoint_dir,
            batch_size=tune_result.batch_size,
            seed_genomes=seeds,
            param_ranges=s2_ranges,
            resume_from=s2_resume,
            guard=guard,
        )

        final_hof = hof2
    else:
        print("[GA] Stage 2 skipped (pop=0 or ngen=0)")
        final_hof = hof1

    # Extract best
    if len(final_hof):
        best_genome = list(final_hof[0])
        best_fit = float(final_hof[0].fitness.values[0])
    else:
        best_genome = [s[1] for s in GLOBAL_PARAM_SPECS]
        best_fit = -1e9

    best_genome_clean = sanitize_global_genome(best_genome)
    best_params = decode_global_params(best_genome_clean)

    print(f"\n[GA] DONE | best_fitness={best_fit:.5f}")
    return best_params, best_genome_clean, best_fit
