"""Regime-conditional trading metrics (plan item D2).

Splits trade-level returns by the market regime at entry date (trend vs chop)
so a single-number alpha doesn't hide regime dependency. The market regime is
classified via 14-period ADX over a benchmark series (Ibovespa, by convention):
ADX above the threshold (default 25) -> "trend", otherwise "chop".

Usage:

    from analysis.regime_metrics import classify_regime_from_ohlc, breakdown_by_regime

    regime = classify_regime_from_ohlc(ibov_df["High"], ibov_df["Low"], ibov_df["Close"])
    seg = breakdown_by_regime(trade_df, regime_series=regime)

`trade_df` must contain at least an `entry_date` column and the metric columns
(`ret` or `return`, and optionally `won`, `alpha`). The returned frame has one
row per regime label, including any "missing" bucket for trades whose entry date
falls outside the regime index.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


TREND_LABEL = "trend"
CHOP_LABEL = "chop"
DEFAULT_ADX_WINDOW = 14
DEFAULT_ADX_THRESHOLD = 25.0


def _wilder_smooth(values: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing — exponential MA with alpha = 1/window."""
    return values.ewm(alpha=1.0 / float(window), adjust=False, min_periods=window).mean()


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = DEFAULT_ADX_WINDOW,
) -> pd.Series:
    """Return Wilder's ADX (0-100). NaN for the first `window` bars."""
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, close must share the same length")

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr = _wilder_smooth(tr, window)
    plus_di = 100.0 * _wilder_smooth(plus_dm, window) / atr.replace(0, np.nan)
    minus_di = 100.0 * _wilder_smooth(minus_dm, window) / atr.replace(0, np.nan)
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return _wilder_smooth(dx, window)


def classify_regime_from_ohlc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    window: int = DEFAULT_ADX_WINDOW,
    threshold: float = DEFAULT_ADX_THRESHOLD,
) -> pd.Series:
    """Return regime label per bar: `trend` if ADX > threshold, else `chop`.

    NaN bars in ADX (typically the first `window` rows) propagate as NaN.
    """
    adx = compute_adx(high, low, close, window=window)
    out = pd.Series(np.full(len(adx), None, dtype=object), index=adx.index, name="regime")
    mask_finite = adx.notna()
    out.loc[mask_finite & (adx > threshold)] = TREND_LABEL
    out.loc[mask_finite & (adx <= threshold)] = CHOP_LABEL
    return out


def classify_regime_from_close(
    close: pd.Series,
    *,
    window: int = 50,
    slope_threshold: float = 0.0,
) -> pd.Series:
    """Cheap fallback when only close is available.

    Uses the sign of a `window`-bar linear regression slope on a log-price
    series — strictly trending up/down marks `trend`, flat marks `chop`. The
    slope is computed in a vectorized fashion via the closed-form OLS for an
    arithmetic-progression x-axis. Provided as a no-OHLC fallback only.
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    log_p = np.log(close.where(close > 0))
    # Closed-form slope of y on x where x = 0..n-1:
    # b = (n * sum(x*y) - sum(x)*sum(y)) / (n * sum(x^2) - sum(x)^2)
    x = np.arange(window, dtype=float)
    sx = x.sum()
    sx2 = (x * x).sum()
    denom = window * sx2 - sx * sx

    def _slope(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        return float(window * np.dot(x, arr) - sx * arr.sum()) / denom

    slope = log_p.rolling(window).apply(_slope, raw=True)
    out = pd.Series(np.full(len(close), None, dtype=object), index=close.index, name="regime")
    mask_finite = slope.notna()
    out.loc[mask_finite & (slope.abs() > slope_threshold)] = TREND_LABEL
    out.loc[mask_finite & (slope.abs() <= slope_threshold)] = CHOP_LABEL
    return out


def _coerce_entry_dates(trade_df: pd.DataFrame, col: str) -> pd.Series:
    if col not in trade_df.columns:
        raise ValueError(f"trade_df missing required column: {col!r}")
    return pd.to_datetime(trade_df[col], errors="coerce")


def _ensure_return_column(trade_df: pd.DataFrame) -> str:
    for cand in ("ret", "return", "trade_ret", "trade_return"):
        if cand in trade_df.columns:
            return cand
    raise ValueError("trade_df must contain a return column (ret/return/trade_ret/trade_return)")


def breakdown_by_regime(
    trade_df: pd.DataFrame,
    regime_series: pd.Series,
    *,
    entry_date_col: str = "entry_date",
    alpha_col: Optional[str] = "alpha",
) -> pd.DataFrame:
    """Aggregate per-trade metrics conditioned on entry-bar regime.

    Returns a DataFrame indexed by regime label (`trend`, `chop`, and a `missing`
    bucket for trades whose entry date is not in the regime index). Columns:
    `n_trades`, `wr`, `mean_ret`, `median_ret`, `total_ret`, and `mean_alpha`
    when `alpha_col` is present in `trade_df`.
    """
    if regime_series is None or len(regime_series) == 0:
        raise ValueError("regime_series is empty")
    ret_col = _ensure_return_column(trade_df)
    entry_dates = _coerce_entry_dates(trade_df, entry_date_col)
    regime_idx = pd.to_datetime(regime_series.index, errors="coerce")
    regime_map = pd.Series(regime_series.values, index=regime_idx)

    df = trade_df.copy()
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
    df["_regime"] = entry_dates.map(regime_map).fillna("missing")

    grouped = df.groupby("_regime", sort=False)
    parts: dict = {
        "n_trades": grouped[ret_col].size(),
        "wr": grouped[ret_col].apply(lambda s: float((s > 0).mean()) if len(s) else 0.0),
        "mean_ret": grouped[ret_col].mean(),
        "median_ret": grouped[ret_col].median(),
        "total_ret": grouped[ret_col].sum(),
    }
    if alpha_col and alpha_col in df.columns:
        df[alpha_col] = pd.to_numeric(df[alpha_col], errors="coerce")
        parts["mean_alpha"] = df.groupby("_regime", sort=False)[alpha_col].mean()
    out = pd.DataFrame(parts)

    # Promote a global ALL row computed on the pooled data (not means of means).
    all_row = {
        "n_trades": int(df[ret_col].count()),
        "wr": float((df[ret_col] > 0).mean()) if len(df) else 0.0,
        "mean_ret": float(df[ret_col].mean()) if len(df) else 0.0,
        "median_ret": float(df[ret_col].median()) if len(df) else 0.0,
        "total_ret": float(df[ret_col].sum()) if len(df) else 0.0,
    }
    if alpha_col and alpha_col in df.columns:
        all_row["mean_alpha"] = float(df[alpha_col].mean()) if len(df) else 0.0
    out.loc["ALL"] = pd.Series(all_row)
    return out


def main(argv: Optional[list] = None) -> int:
    """CLI: classify a benchmark OHLC CSV and dump regime-conditioned trade stats.

    Usage:
        python analysis/regime_metrics.py --benchmark ibov.csv --trades trades.csv
            [--out report.json] [--window 14] [--threshold 25.0]
    """
    import argparse
    import json

    p = argparse.ArgumentParser(description="Regime-conditional trade metrics (D2).")
    p.add_argument("--benchmark", required=True, help="CSV with columns: Date,High,Low,Close (benchmark).")
    p.add_argument("--trades", required=True, help="CSV with at least entry_date and ret columns.")
    p.add_argument("--window", type=int, default=DEFAULT_ADX_WINDOW)
    p.add_argument("--threshold", type=float, default=DEFAULT_ADX_THRESHOLD)
    p.add_argument("--out", default=None, help="Optional JSON output path.")
    args = p.parse_args(argv)

    bench = pd.read_csv(args.benchmark)
    bench["Date"] = pd.to_datetime(bench["Date"], errors="coerce")
    bench = bench.dropna(subset=["Date"]).set_index("Date")
    regime = classify_regime_from_ohlc(
        bench["High"], bench["Low"], bench["Close"],
        window=args.window, threshold=args.threshold,
    )
    trades = pd.read_csv(args.trades)
    seg = breakdown_by_regime(trades, regime)
    report = {
        "regime_dependency_index": regime_dependency_index(seg),
        "segments": seg.reset_index().to_dict(orient="records"),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0


def regime_dependency_index(seg_df: pd.DataFrame) -> float:
    """Single-number score in [0, 1] for how much edge depends on regime.

    0 -> trend and chop perform identically (regime-agnostic edge).
    1 -> all the edge sits in one regime (other regime is at or below zero).
    """
    if TREND_LABEL not in seg_df.index or CHOP_LABEL not in seg_df.index:
        return float("nan")
    a = float(seg_df.loc[TREND_LABEL, "mean_ret"])
    b = float(seg_df.loc[CHOP_LABEL, "mean_ret"])
    denom = abs(a) + abs(b)
    if denom <= 1e-12:
        return 0.0
    return float(abs(a - b) / denom)


if __name__ == "__main__":
    raise SystemExit(main())
