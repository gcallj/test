"""
1) Add technical pattern verification columns to apply output
2) Generate candlestick charts for top-5 confidence + PETR4
   showing BUY signals with stop/take profit and SELL reasons
"""
import sys, json, warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from ga_run import (
    decode_global_params, ewm_mean_np, rolling_mean_np,
    ATR_EPS, SCORE_LOOKBACK, MERGE_ETL_FEATURES, MAX_FEATURES, LONG_ONLY,
)

# ── 1. Load data ────────────────────────────────────────────────────────────
with open("global_ga_checkpoint.json") as f:
    ckpt = json.load(f)
gp = decode_global_params(ckpt["genome"])

hc = pd.read_parquet("output/history_consolidated.parquet")
hc["Date"] = pd.to_datetime(hc["Date"])

etl = pd.read_parquet("output/data/expanded_stock_reduced.parquet")
etl_date_col = ("Date",) if ("Date",) in etl.columns else "Date"
if isinstance(etl.columns[0], tuple):
    # MultiIndex columns - need to reshape for pattern extraction
    pass

ap = pd.read_csv("apply_last_5d__H5.csv")
ap["Date"] = pd.to_datetime(ap["Date"])

# ── 2. TECHNICAL PATTERN COLUMNS ───────────────────────────────────────────
# Patterns: chan_pattern, dbl_pattern, hs_pattern, tri_pattern, wed_pattern
# Values: 1=bullish, -1=bearish, 0=none
# Also: momentum_rsi, trend_macd_diff, cross_X_Y (SMA crossovers)

PATTERN_NAMES = {
    "chan_pattern": "Channel Breakout",
    "dbl_pattern": "Double Top/Bottom",
    "hs_pattern": "Head & Shoulders",
    "tri_pattern": "Triangle Breakout",
    "wed_pattern": "Wedge Breakout",
}

INDICATOR_NAMES = {
    "momentum_rsi": "RSI",
    "momentum_stoch_rsi": "Stoch RSI",
    "momentum_wr": "Williams %R",
    "trend_macd_diff": "MACD Histogram",
    "cross_5_20": "SMA5/20 Cross",
    "cross_20_50": "SMA20/50 Cross",
    "cross_50_100": "SMA50/100 Cross",
}

def get_ticker_series(etl_df, feature_name, ticker):
    """Extract a feature series for a ticker from the multi-index ETL."""
    col = (feature_name, ticker)
    if col in etl_df.columns:
        return etl_df[col].values
    return None

def check_patterns_for_ticker(etl_df, ticker, n_rows):
    """Check if patterns happened in last 5 days (daily) and recently (weekly/monthly proxy)."""
    result = {}

    for pat, label in PATTERN_NAMES.items():
        series = get_ticker_series(etl_df, pat, ticker)
        if series is None:
            result[f"{label}_last5d"] = "N/A"
            result[f"{label}_last20d"] = "N/A"
            continue

        last5 = series[-5:] if len(series) >= 5 else series
        last20 = series[-20:] if len(series) >= 20 else series  # ~1 month

        # Last 5 days
        bull_5 = int(np.sum(last5 == 1))
        bear_5 = int(np.sum(last5 == -1))
        if bull_5 > 0 and bear_5 == 0:
            result[f"{label}_last5d"] = f"BULL({bull_5}d)"
        elif bear_5 > 0 and bull_5 == 0:
            result[f"{label}_last5d"] = f"BEAR({bear_5}d)"
        elif bull_5 > 0 and bear_5 > 0:
            result[f"{label}_last5d"] = f"MIXED(B{bull_5}/S{bear_5})"
        else:
            result[f"{label}_last5d"] = "---"

        # Last 20 days (~monthly)
        bull_20 = int(np.sum(last20 == 1))
        bear_20 = int(np.sum(last20 == -1))
        if bull_20 > 0 and bear_20 == 0:
            result[f"{label}_last20d"] = f"BULL({bull_20}d)"
        elif bear_20 > 0 and bull_20 == 0:
            result[f"{label}_last20d"] = f"BEAR({bear_20}d)"
        elif bull_20 > 0 and bear_20 > 0:
            result[f"{label}_last20d"] = f"MIXED(B{bull_20}/S{bear_20})"
        else:
            result[f"{label}_last20d"] = "---"

    # RSI status
    rsi = get_ticker_series(etl_df, "momentum_rsi", ticker)
    if rsi is not None and len(rsi) >= 5:
        rsi_last = rsi[-1]
        if np.isfinite(rsi_last):
            if rsi_last > 70:
                result["RSI_status"] = f"OVERBOUGHT({rsi_last:.0f})"
            elif rsi_last < 30:
                result["RSI_status"] = f"OVERSOLD({rsi_last:.0f})"
            else:
                result["RSI_status"] = f"NEUTRAL({rsi_last:.0f})"
        else:
            result["RSI_status"] = "N/A"
    else:
        result["RSI_status"] = "N/A"

    # SMA crossovers in last 5 days
    cross_signals = []
    for cross_feat, cross_label in [("cross_5_20", "SMA5x20"), ("cross_20_50", "SMA20x50"), ("cross_50_100", "SMA50x100")]:
        series = get_ticker_series(etl_df, cross_feat, ticker)
        if series is not None and len(series) >= 5:
            last5 = series[-5:]
            # Cross = 1 means golden cross, -1 = death cross (transition)
            golden = np.sum(last5 == 1)
            death = np.sum(last5 == -1)
            if golden > 0:
                cross_signals.append(f"{cross_label}:GOLDEN")
            elif death > 0:
                cross_signals.append(f"{cross_label}:DEATH")
    result["SMA_Cross_last5d"] = ", ".join(cross_signals) if cross_signals else "---"

    return result


# Build pattern analysis for all tickers in apply
print("Analyzing technical patterns for all tickers...")
pattern_rows = []
for _, row in ap.iterrows():
    ticker = row["ticker"]
    patterns = check_patterns_for_ticker(etl, ticker, len(etl))
    patterns["ticker"] = ticker
    pattern_rows.append(patterns)

pat_df = pd.DataFrame(pattern_rows)
ap_enriched = ap.merge(pat_df, on="ticker", how="left")

# Save enriched apply file
out_csv = "apply_with_patterns.csv"
ap_enriched.to_csv(out_csv, index=False)
print(f"Saved: {out_csv} ({ap_enriched.shape})")

# Print summary for top buys
buys = ap_enriched[ap_enriched["signal"] == "buy"].sort_values("confidence", ascending=False)
pattern_cols = [c for c in ap_enriched.columns if c not in ap.columns and c != "ticker"]
print(f"\nTop 10 BUY signals with pattern analysis:")
for _, r in buys.head(10).iterrows():
    print(f"\n  {r['ticker']} | conf={r['confidence']:.1f} | score={r['score_100']:.1f}")
    for pc in pattern_cols:
        val = r[pc]
        if val != "---" and val != "N/A":
            print(f"    {pc}: {val}")

# ── 3. CANDLESTICK CHARTS ──────────────────────────────────────────────────
# Top 5 by confidence (buy signals) + PETR4
top5 = buys.head(5)["ticker"].tolist()
chart_tickers = top5 + (["PETR4.SA"] if "PETR4.SA" not in top5 else [])
print(f"\nGenerating charts for: {chart_tickers}")


def compute_signals_for_ticker(ticker_data, gp, etl_df, ticker):
    """Compute GA signals for a full ticker history. Returns DataFrame with signals."""
    df = ticker_data.copy().sort_values("Date").reset_index(drop=True)
    n = len(df)

    c = df["close"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    lo = df["low"].values.astype(np.float64)

    # ATR
    atr_period = 14
    tr = np.maximum(h - lo, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))))
    tr[0] = h[0] - lo[0]
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr[atr_period - 1] = np.mean(tr[:atr_period])
        for i in range(atr_period, n):
            atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period
    atr = np.where(np.isfinite(atr), atr, np.nanmedian(tr) if np.any(np.isfinite(tr)) else 1.0)

    # Feature matrix from ETL (multi-index)
    exclude = {"Date", "ticker", "split", "signal", "open", "high", "low", "close", "volume", "price"}
    # Get all features for this ticker from ETL, aligned to df dates
    # ETL is wide format - need to align by date index
    etl_dates = etl_df[("Date",)].values if ("Date",) in etl_df.columns else etl_df.index.values
    try:
        etl_dates = pd.to_datetime(etl_dates)
    except:
        etl_dates = pd.to_datetime(etl_df.iloc[:, 0].values)

    # Build date->index mapping for ETL
    etl_date_map = {d: i for i, d in enumerate(etl_dates)}
    df_dates = pd.to_datetime(df["Date"].values)
    etl_idx = np.array([etl_date_map.get(d, -1) for d in df_dates])

    ticker_feats = {}
    for col in etl_df.columns:
        if isinstance(col, tuple) and col[1] == ticker and col[0] not in exclude:
            feat_name = col[0]
            full_arr = etl_df[col].values
            # Extract only the rows matching df dates
            aligned = np.full(n, np.nan)
            valid_mask = etl_idx >= 0
            aligned[valid_mask] = full_arr[etl_idx[valid_mask]]
            ticker_feats[feat_name] = aligned

    if len(ticker_feats) < 5:
        for col in df.columns:
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col]):
                ticker_feats[col] = df[col].values.astype(np.float64)

    feat_names = list(ticker_feats.keys())
    X = np.column_stack([ticker_feats[f] for f in feat_names])

    # Spearman selection
    from scipy.stats import spearmanr
    target_proxy = np.roll(c, -20) / np.maximum(c, 1e-8) - 1
    target_proxy[-20:] = np.nan
    valid = np.isfinite(target_proxy)
    scores = []
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = valid & np.isfinite(col)
        if mask.sum() > 50:
            rho, _ = spearmanr(col[mask], target_proxy[mask])
            scores.append((j, abs(rho) if np.isfinite(rho) else 0.0))
        else:
            scores.append((j, 0.0))
    scores.sort(key=lambda x: -x[1])
    top_idx = [s[0] for s in scores[:MAX_FEATURES]]
    X_sel = X[:, top_idx]

    # Z-score
    win = 252
    X_z = np.full_like(X_sel, np.nan)
    for i in range(min(win, n), n):
        block = X_sel[max(0, i - win):i + 1]
        for j in range(block.shape[1]):
            col = block[:, j]
            finite = np.isfinite(col)
            if finite.sum() > 10:
                mu = np.nanmean(col[finite])
                sd = np.nanstd(col[finite])
                if sd > 1e-12:
                    X_z[i, j] = (col[-1] - mu) / sd

    # Signal generation (single-pass)
    feat_n = max(1, X_z.shape[1])
    votes_long = np.nansum(X_z > gp.z_threshold, axis=1) / feat_n
    votes_short = np.nansum(X_z < -gp.z_threshold, axis=1) / feat_n
    score_raw = votes_long - votes_short
    score_ev = ewm_mean_np(score_raw, int(gp.signal_ema_span))
    ma = rolling_mean_np(c, int(gp.ma_filter_period), int(gp.ma_filter_period // 2))

    signals = np.zeros(n, dtype=int)  # 0=hold, 1=buy
    consec_long = 0
    lookback = max(63, SCORE_LOOKBACK)

    for i in range(1, n):
        w = score_ev[max(0, i - lookback):i]
        pctl = float(np.nanpercentile(w, gp.score_percentile_trigger * 100)) if len(w) >= 20 else 0.0
        vl = votes_long[i - 1]
        long_raw = (vl >= gp.vote_threshold_long) and (score_ev[i - 1] >= pctl)
        consec_long = consec_long + 1 if long_raw else 0
        long_ok = consec_long >= gp.entry_confirmation_days
        if gp.ma_filter_mode >= 1 and np.isfinite(ma[i - 1]):
            if c[i - 1] < ma[i - 1]:
                long_ok = False
        if long_ok:
            signals[i] = 1

    # Compute stop/take/entry + sell reasons
    stop_losses = np.full(n, np.nan)
    take_profits = np.full(n, np.nan)
    entry_prices = np.full(n, np.nan)
    sell_reasons = [""] * n

    # Track positions for sell reason
    in_position = False
    pos_entry = 0.0
    pos_stop = 0.0
    pos_take = 0.0
    pos_entry_bar = 0
    pos_high_since = 0.0

    for i in range(n):
        atr_val = max(atr[i], ATR_EPS)
        close_val = c[i]

        if signals[i] == 1:
            sev = score_ev[i] if np.isfinite(score_ev[i]) else 0.0
            recent = score_ev[max(0, i - lookback):i + 1]
            recent = recent[np.isfinite(recent)]
            score95 = np.nanpercentile(np.abs(recent), 95) if len(recent) > 10 else 1.0
            score95 = max(score95, ATR_EPS)
            strength = float(np.clip(abs(sev) / score95, 0.0, 1.0))
            discount = gp.entry_discount_atr_frac * (1.0 - gp.score_strength_scaling * strength)

            entry_ref = close_val - discount * atr_val
            stop_atr = gp.stop_atr_mult * atr_val
            take_atr = gp.reward_risk_ratio * stop_atr

            stop_losses[i] = entry_ref - stop_atr
            take_profits[i] = entry_ref + take_atr
            entry_prices[i] = entry_ref

            if not in_position:
                in_position = True
                pos_entry = entry_ref
                pos_stop = entry_ref - stop_atr
                pos_take = entry_ref + take_atr
                pos_entry_bar = i
                pos_high_since = close_val

        if in_position:
            pos_high_since = max(pos_high_since, h[i] if np.isfinite(h[i]) else close_val)
            bars_held = i - pos_entry_bar

            # Check exit conditions
            exited = False
            if lo[i] <= pos_stop:
                sell_reasons[i] = "STOP LOSS"
                exited = True
            elif h[i] >= pos_take:
                sell_reasons[i] = "TAKE PROFIT"
                exited = True
            elif bars_held >= gp.time_stop_bars:
                sell_reasons[i] = f"TIME STOP ({int(gp.time_stop_bars)}d)"
                exited = True
            elif close_val < pos_entry * (1 - gp.max_loss_per_trade_pct):
                sell_reasons[i] = f"HARD STOP ({gp.max_loss_per_trade_pct*100:.0f}%)"
                exited = True
            elif gp.trailing_stop_mode == 1 and pos_high_since >= pos_entry + (pos_entry - pos_stop):
                # Breakeven trailing: if price reached 1x stop distance in profit, move stop to breakeven
                new_stop = pos_entry
                if lo[i] <= new_stop:
                    sell_reasons[i] = "BREAKEVEN STOP"
                    exited = True
            elif gp.ma_filter_mode >= 1 and np.isfinite(ma[i]) and close_val < ma[i]:
                if signals[i] != 1:  # lost MA support while not getting new buy
                    sell_reasons[i] = f"BELOW SMA{int(gp.ma_filter_period)}"
                    exited = True

            if exited:
                in_position = False

    df["signal_ga"] = np.where(signals == 1, "buy", "hold")
    df["stop_loss"] = stop_losses
    df["take_profit"] = take_profits
    df["entry_price"] = entry_prices
    df["sell_reason"] = sell_reasons

    return df


def plot_candle_chart(df, ticker, gp, pattern_info, output_dir="charts"):
    """Plot candlestick chart with buy/sell signals."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter last year
    one_year_ago = df["Date"].max() - pd.Timedelta(days=365)
    df_year = df[df["Date"] >= one_year_ago].copy().reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(22, 11))

    dates_plot = df_year["Date"].values
    opens = df_year["open"].values
    highs = df_year["high"].values
    lows = df_year["low"].values
    closes = df_year["close"].values
    n_bars = len(df_year)

    # Candlesticks
    width = 0.6
    for i in range(n_bars):
        d = mdates.date2num(pd.Timestamp(dates_plot[i]))
        color = "#26a69a" if closes[i] >= opens[i] else "#ef5350"
        body_lo = min(opens[i], closes[i])
        body_hi = max(opens[i], closes[i])
        ax.plot([d, d], [lows[i], highs[i]], color=color, linewidth=0.7)
        ax.bar(d, body_hi - body_lo, bottom=body_lo, width=width,
               color=color, edgecolor=color, linewidth=0.4)

    # Buy signals
    buys = df_year[df_year["signal_ga"] == "buy"]
    n_buys = len(buys)
    for _, row in buys.iterrows():
        d = mdates.date2num(row["Date"])
        sl = row["stop_loss"]
        tp = row["take_profit"]
        low_v = row["low"]

        ax.plot(d, low_v * 0.99, marker="^", color="#00c853", markersize=8,
                markeredgecolor="black", markeredgewidth=0.4, zorder=5)

        line_len = 5
        if np.isfinite(sl):
            ax.plot([d, d + line_len], [sl, sl], color="#ff1744", linewidth=0.9,
                    linestyle="--", alpha=0.5, zorder=4)
        if np.isfinite(tp):
            ax.plot([d, d + line_len], [tp, tp], color="#2979ff", linewidth=0.9,
                    linestyle="--", alpha=0.5, zorder=4)

    # Sell signals (exits) with reason labels
    sells = df_year[df_year["sell_reason"] != ""]
    n_sells = len(sells)
    for _, row in sells.iterrows():
        d = mdates.date2num(row["Date"])
        high_v = row["high"]
        reason = row["sell_reason"]

        # Red triangle down at top
        ax.plot(d, high_v * 1.01, marker="v", color="#ff1744", markersize=7,
                markeredgecolor="black", markeredgewidth=0.4, zorder=5)

        # Reason text (rotate to avoid overlap)
        ax.annotate(reason, xy=(d, high_v * 1.015), fontsize=5.5,
                    color="#d50000", ha="center", va="bottom", rotation=45,
                    fontweight="bold", alpha=0.8)

    # Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, fontsize=9)
    ax.set_ylabel("Preco", fontsize=12)

    # Title with pattern info
    title = f"{ticker} - Candlestick 1 Ano | BUY + Stop/Take Profit + Motivo de Saida"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.15)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#00c853",
               markeredgecolor="black", markersize=11, label=f"BUY ({n_buys})"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#ff1744",
               markeredgecolor="black", markersize=11, label=f"EXIT ({n_sells})"),
        Line2D([0], [0], color="#ff1744", linestyle="--", linewidth=1.5, label="Stop Loss"),
        Line2D([0], [0], color="#2979ff", linestyle="--", linewidth=1.5, label="Take Profit"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.9)

    # Pattern info box
    if pattern_info:
        lines = []
        for k, v in pattern_info.items():
            if v not in ("---", "N/A", ""):
                lines.append(f"{k}: {v}")
        if lines:
            pat_text = "\n".join(lines[:8])  # max 8 lines
            ax.text(0.99, 0.97, pat_text,
                    transform=ax.transAxes, fontsize=8, va="top", ha="right",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                              alpha=0.9, edgecolor="gray"))

    # Stats box
    conf = ap[ap["ticker"] == ticker]["confidence"].values
    conf_str = f"{conf[0]:.1f}" if len(conf) > 0 else "?"
    sig = ap[ap["ticker"] == ticker]["signal"].values
    sig_str = sig[0] if len(sig) > 0 else "?"

    stats = f"Sinal atual: {sig_str.upper()} | Conf: {conf_str} | Stop: {gp.stop_atr_mult}x ATR | R:R {gp.reward_risk_ratio}:1"
    ax.text(0.5, 0.02, stats, transform=ax.transAxes, fontsize=10,
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray"))

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{ticker.replace('.','_').replace('-','_')}_candle_signals.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ── 4. Generate charts ─────────────────────────────────────────────────────
for ticker in chart_tickers:
    print(f"\n{'='*60}")
    print(f"Processing {ticker}...")

    ticker_data = hc[hc["ticker"] == ticker].copy()
    if len(ticker_data) < 300:
        print(f"  SKIP: only {len(ticker_data)} rows (need 300+)")
        continue

    # Get pattern info
    pat_row = pat_df[pat_df["ticker"] == ticker]
    pattern_info = {}
    if not pat_row.empty:
        for col in pat_row.columns:
            if col != "ticker":
                val = pat_row.iloc[0][col]
                if val not in ("---", "N/A", ""):
                    pattern_info[col] = val

    # Compute signals
    df_signals = compute_signals_for_ticker(ticker_data, gp, etl, ticker)

    # Plot
    out_path = plot_candle_chart(df_signals, ticker, gp, pattern_info)

    n_buys_year = (df_signals[df_signals["Date"] >= df_signals["Date"].max() - pd.Timedelta(days=365)]["signal_ga"] == "buy").sum()
    n_sells_year = (df_signals[df_signals["Date"] >= df_signals["Date"].max() - pd.Timedelta(days=365)]["sell_reason"] != "").sum()
    print(f"  Chart saved: {out_path}")
    print(f"  Last year: {n_buys_year} buys, {n_sells_year} exits")
    if pattern_info:
        print(f"  Active patterns: {pattern_info}")

print(f"\nDone! Charts in ./charts/")
