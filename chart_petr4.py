"""
PETR4 Candlestick chart with GA buy/hold signals, stop-loss and take-profit.
Single-pass signal generation (fast).
"""
import sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from ga_run import (
    decode_global_params,
    ewm_mean_np, rolling_mean_np, ATR_EPS, SCORE_LOOKBACK,
    MERGE_ETL_FEATURES, MAX_FEATURES, LONG_ONLY,
)

# ── 1. Load GA checkpoint ───────────────────────────────────────────────────
with open("global_ga_checkpoint.json") as f:
    ckpt = json.load(f)
genome = ckpt["genome"]
gp = decode_global_params(genome)
print(f"GA params: stop_atr={gp.stop_atr_mult}, RR={gp.reward_risk_ratio}, "
      f"vote_thr_long={gp.vote_threshold_long}, z_thr={gp.z_threshold}, "
      f"ma_period={gp.ma_filter_period}, ma_mode={gp.ma_filter_mode}")

# ── 2. Load data ────────────────────────────────────────────────────────────
hc = pd.read_parquet("output/history_consolidated.parquet")
hc["Date"] = pd.to_datetime(hc["Date"])

if MERGE_ETL_FEATURES:
    try:
        etl = pd.read_parquet("output/data/expanded_stock_reduced.parquet")
        etl["Date"] = pd.to_datetime(etl["Date"])
        etl_cols = [c for c in etl.columns if c not in hc.columns or c in ("Date", "ticker")]
        if len(etl_cols) > 2:
            hc = hc.merge(etl[etl_cols], on=["Date", "ticker"], how="left")
            print(f"[ETL MERGE] +{len(etl_cols)-2} ETL features")
    except Exception as e:
        print(f"[ETL MERGE] skip: {e}")

petr = hc[hc["ticker"] == "PETR4.SA"].copy().sort_values("Date").reset_index(drop=True)
n = len(petr)
print(f"PETR4 rows: {n}")

# ── 3. Compute features ────────────────────────────────────────────────────
dates = petr["Date"].values
o = petr["open"].values.astype(np.float64)
h = petr["high"].values.astype(np.float64)
lo = petr["low"].values.astype(np.float64)
c = petr["close"].values.astype(np.float64)

# ATR(14)
atr_period = 14
tr = np.maximum(h - lo, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))))
tr[0] = h[0] - lo[0]
atr = np.full(n, np.nan)
atr[atr_period - 1] = np.mean(tr[:atr_period])
for i in range(atr_period, n):
    atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period
atr = np.where(np.isfinite(atr), atr, np.nanmedian(tr))

# Feature matrix & Spearman selection
exclude = {"Date", "ticker", "split", "signal", "open", "high", "low", "close", "volume", "price"}
feat_cols = [c_ for c_ in petr.columns if c_ not in exclude and pd.api.types.is_numeric_dtype(petr[c_])]
X = petr[feat_cols].values.astype(np.float64)

from scipy.stats import spearmanr
target_proxy = (pd.Series(c).shift(-20) / pd.Series(c) - 1).values
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

# Z-score per rolling window
win = 252
X_z = np.full_like(X_sel, np.nan)
for i in range(win, n):
    block = X_sel[max(0, i - win):i + 1]
    for j in range(block.shape[1]):
        col = block[:, j]
        finite = np.isfinite(col)
        if finite.sum() > 10:
            mu = np.nanmean(col[finite])
            sd = np.nanstd(col[finite])
            if sd > 1e-12:
                X_z[i, j] = (col[-1] - mu) / sd

print(f"Z-scored features: {X_z.shape}")

# ── 4. Single-pass signal generation ───────────────────────────────────────
feat_n = max(1, X_z.shape[1])
votes_long = np.nansum(X_z > gp.z_threshold, axis=1) / feat_n
votes_short = np.nansum(X_z < -gp.z_threshold, axis=1) / feat_n
score_raw = votes_long - votes_short
score_ev = ewm_mean_np(score_raw, int(gp.signal_ema_span))

# MA filter
ma = rolling_mean_np(c, int(gp.ma_filter_period), int(gp.ma_filter_period // 2))

# Single-pass signal loop (mimics generate_signal_global but O(n))
signals = np.full(n, 0, dtype=int)  # 0=hold, 1=buy
consec_long = 0
lookback = max(63, SCORE_LOOKBACK)

for i in range(1, n):
    # Compute percentile threshold from recent scores
    w = score_ev[max(0, i - lookback):i]
    if len(w) >= 20:
        pctl = float(np.nanpercentile(w, gp.score_percentile_trigger * 100))
    else:
        pctl = 0.0

    vl = votes_long[i - 1]
    long_raw = (vl >= gp.vote_threshold_long) and (score_ev[i - 1] >= pctl)

    if long_raw:
        consec_long += 1
    else:
        consec_long = 0

    long_ok = consec_long >= gp.entry_confirmation_days

    # MA filter
    if gp.ma_filter_mode >= 1 and np.isfinite(ma[i - 1]):
        if c[i - 1] < ma[i - 1]:
            long_ok = False

    if long_ok:
        signals[i] = 1

# Compute stop/take profit for buy signals
stop_losses = np.full(n, np.nan)
take_profits = np.full(n, np.nan)
entry_prices = np.full(n, np.nan)

for i in range(n):
    if signals[i] != 1:
        continue
    atr_val = max(atr[i], ATR_EPS)
    close_val = c[i]

    sev = score_ev[i] if np.isfinite(score_ev[i]) else 0.0
    recent = score_ev[max(0, i - lookback):i + 1]
    recent = recent[np.isfinite(recent)]
    score95 = np.nanpercentile(np.abs(recent), 95) if len(recent) > 10 else 1.0
    score95 = max(score95, ATR_EPS)
    strength = float(np.clip(abs(sev) / score95, 0.0, 1.0))
    discount = gp.entry_discount_atr_frac * (1.0 - gp.score_strength_scaling * strength)

    entry_ref = round(close_val - discount * atr_val, 4)
    stop_atr = gp.stop_atr_mult * atr_val
    take_atr = gp.reward_risk_ratio * stop_atr
    stop_losses[i] = round(entry_ref - stop_atr, 4)
    take_profits[i] = round(entry_ref + take_atr, 4)
    entry_prices[i] = entry_ref

petr["signal_ga"] = np.where(signals == 1, "buy", "hold")
petr["stop_loss"] = stop_losses
petr["take_profit"] = take_profits
petr["entry_price"] = entry_prices

# ── 5. Filter last year ─────────────────────────────────────────────────────
one_year_ago = pd.Timestamp("2025-03-18")
df = petr[petr["Date"] >= one_year_ago].copy().reset_index(drop=True)
n_buys = (df["signal_ga"] == "buy").sum()
print(f"Last year: {len(df)} bars, {n_buys} BUY signals")

# ── 6. Plot candlestick chart ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 11))

dates_plot = df["Date"].values
opens = df["open"].values
highs = df["high"].values
lows = df["low"].values
closes = df["close"].values
n_bars = len(df)

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

# Buy signals with stop-loss and take-profit
buys = df[df["signal_ga"] == "buy"].copy()
for _, row in buys.iterrows():
    d = mdates.date2num(row["Date"])
    sl = row["stop_loss"]
    tp = row["take_profit"]
    entry = row["entry_price"]
    close_v = row["close"]
    low_v = row[("low")]

    # Green triangle at bottom
    ax.plot(d, low_v * 0.99, marker="^", color="#00c853", markersize=8,
            markeredgecolor="black", markeredgewidth=0.4, zorder=5)

    # Stop loss (red) and take profit (blue) horizontal lines
    line_len = 6
    if np.isfinite(sl):
        ax.plot([d, d + line_len], [sl, sl], color="#ff1744", linewidth=1.0,
                linestyle="--", alpha=0.6, zorder=4)

    if np.isfinite(tp):
        ax.plot([d, d + line_len], [tp, tp], color="#2979ff", linewidth=1.0,
                linestyle="--", alpha=0.6, zorder=4)

    # Entry price (green dotted)
    if np.isfinite(entry):
        ax.plot([d, d + line_len], [entry, entry], color="#00c853", linewidth=0.7,
                linestyle=":", alpha=0.4, zorder=3)

# Formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45, fontsize=9)
ax.set_ylabel("Preco (BRL)", fontsize=12)
ax.set_title("PETR4.SA - Candlestick 1 Ano | Sinais GA: BUY + Stop Loss (vermelho) + Take Profit (azul)",
             fontsize=14, fontweight="bold", pad=15)
ax.grid(True, alpha=0.15, linestyle="-")

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#00c853",
           markeredgecolor="black", markersize=12, label="BUY Signal"),
    Line2D([0], [0], color="#ff1744", linestyle="--", linewidth=2, label="Stop Loss"),
    Line2D([0], [0], color="#2979ff", linestyle="--", linewidth=2, label="Take Profit"),
    Line2D([0], [0], color="#00c853", linestyle=":", linewidth=1.5, label="Entry Price"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=11,
          framealpha=0.9, edgecolor="gray")

# Info box
info = (f"Sinais BUY: {n_buys}  |  Stop: {gp.stop_atr_mult}x ATR  |  "
        f"R:R {gp.reward_risk_ratio}:1  |  MA filter: SMA{int(gp.ma_filter_period)}")
ax.text(0.99, 0.97, info,
        transform=ax.transAxes, fontsize=10, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85, edgecolor="gray"))

plt.tight_layout()
out_path = "PETR4_candle_signals.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
