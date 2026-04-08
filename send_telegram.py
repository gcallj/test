"""Envia summary_latest.xlsx atual para o Telegram com caption completo."""
import os
import json
import numpy as np
import pandas as pd
from openpyxl import load_workbook

# Use current env token if set, otherwise fallback to the one in ga_run.py
if not os.environ.get("TELEGRAM_BOT_TOKEN"):
    from ga_run import TELEGRAM_BOT_TOKEN as _default_token
    os.environ["TELEGRAM_BOT_TOKEN"] = _default_token

from ga_run import _send_telegram

xlsx = "summary_latest.xlsx"
if not os.path.exists(xlsx):
    raise RuntimeError(f"{xlsx} not found")

# Read Apply + Summary from xlsx
wb = load_workbook(xlsx, read_only=True)

# -- Apply metrics
ws_app = wb["Apply"]
h_app = [c.value for c in next(ws_app.rows)]
rows_app = list(ws_app.iter_rows(min_row=2, values_only=True))

def col(h, name):
    return h.index(name) if name in h else -1

sig_col = col(h_app, "signal")
date_col = col(h_app, "Date")
rec_col = col(h_app, "recomendacao")
pot_col = col(h_app, "potencial_pct")
conf_col = col(h_app, "confidence")
rank_col = col(h_app, "rank")
tkr_col = col(h_app, "ticker")
alvo_col = col(h_app, "alvo")
ent_col = col(h_app, "entrada")

n_buy = sum(1 for r in rows_app if r[sig_col] == "buy")
n_sell = sum(1 for r in rows_app if r[sig_col] == "sell")
n_hold = sum(1 for r in rows_app if r[sig_col] == "hold")
n_total = len(rows_app)
latest_date = max((str(r[date_col]) for r in rows_app), default="N/A")

confs = [float(r[conf_col]) for r in rows_app if r[conf_col] is not None]
conf_med = float(np.median(confs)) if confs else 0.0

# -- Summary metrics
ws_sum = wb["Summary"]
h_sum = [c.value for c in next(ws_sum.rows)]
rows_sum = list(ws_sum.iter_rows(min_row=2, values_only=True))

ret_col_s = col(h_sum, "test_return")
bh_col_s = col(h_sum, "buy_hold_return")
wr_col_s = col(h_sum, "test_win_rate")
mdd_col_s = col(h_sum, "test_mdd")

# Build full arrays with all metrics aligned (so we can filter together)
full_data = []
for r in rows_sum:
    if r[ret_col_s] is None or r[wr_col_s] is None:
        continue
    full_data.append({
        "ret": float(r[ret_col_s]),
        "bh": float(r[bh_col_s]) if r[bh_col_s] is not None else 0.0,
        "wr": float(r[wr_col_s]),
        "mdd": float(r[mdd_col_s]) if r[mdd_col_s] is not None else 0.0,
    })

# Overall WR/MDD (todos os tickers)
all_wrs = np.array([d["wr"] for d in full_data])
all_mdds = np.array([d["mdd"] for d in full_data])
wr_med = float(np.median(all_wrs)) * 100
mdd_med = float(np.median(all_mdds)) * 100

# Filtro WR > 65% para Ret/BH
filtered = [d for d in full_data if d["wr"] > 0.65]
n_filtered = len(filtered)

# Annualized (15 years reference)
n_years = 15.0
def annualize(r):
    return (max(1.0 + r, 0.001)) ** (1.0 / n_years) - 1.0

if n_filtered > 0:
    rets = np.array([d["ret"] for d in filtered])
    bhs = np.array([d["bh"] for d in filtered])
    ret_ann = np.array([annualize(r) for r in rets])
    bh_ann = np.array([annualize(r) for r in bhs])
    alpha_ann = ret_ann - bh_ann

    ret_ann_med = float(np.median(ret_ann)) * 100
    bh_ann_med = float(np.median(bh_ann)) * 100
    alpha_ann_med = float(np.median(alpha_ann)) * 100
    pct_beat_bh = float((rets > bhs).mean()) * 100
else:
    ret_ann_med = 0
    bh_ann_med = 0
    alpha_ann_med = 0
    pct_beat_bh = 0

# Current checkpoint fitness
ck = json.load(open("global_ga_checkpoint.json"))
global_fit = float(ck.get("fitness", 0.0))

# Top BUYS por rank
buys = []
for r in rows_app:
    if r[sig_col] != "buy":
        continue
    buys.append({
        "ticker": str(r[tkr_col]),
        "rank": float(r[rank_col]) if r[rank_col] is not None else 0.0,
        "pot": str(r[pot_col]),
        "ent": float(r[ent_col]) if r[ent_col] is not None else 0.0,
        "alvo": float(r[alvo_col]) if r[alvo_col] is not None else 0.0,
    })
buys.sort(key=lambda x: -x["rank"])

top_buys = ""
if buys:
    top_buys = "\n\nTop BUY (por rank):\n" + "\n".join(
        f"  {b['ticker']} @{b['ent']:.2f} alvo @{b['alvo']:.2f} ({b['pot']})"
        for b in buys[:8]
    )

wb.close()

caption = (
    f"Trading Signals B3 - {latest_date}\n"
    f"\n"
    f"Sinais: {n_buy} BUY | {n_hold} HOLD | {n_sell} SELL\n"
    f"Tickers: {n_total}\n"
    f"\n"
    f"WR: {wr_med:.1f}% | MDD: {mdd_med:.1f}%\n"
    f"(WR>65%, n={n_filtered})\n"
    f"Ret a.a.: {ret_ann_med:+.1f}% vs B&H {bh_ann_med:+.1f}% (~{int(n_years)}a)\n"
    f"Alpha a.a.: {alpha_ann_med:+.1f}pp | Batem B&H: {pct_beat_bh:.0f}%\n"
    f"Confidence: {conf_med:.0f} | Fitness: {global_fit:.2f}"
    f"{top_buys}"
)

print("=== CAPTION ===")
print(caption)
print()
print(f"=== SENDING {xlsx} ({os.path.getsize(xlsx)} bytes) ===")
_send_telegram(xlsx, caption)
