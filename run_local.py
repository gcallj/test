#!/usr/bin/env python3
"""
Local runner for GA pipeline — overrides Colab paths and runs full training.
Usage: python run_local.py [--mode train|load] [--fast]
"""
import sys
import os
import time

# Override paths BEFORE importing main_cell_v3
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_DIR)

# Monkey-patch the config before importing
import main_cell_v3 as mc

# Local paths
mc.HISTORY_CSV_PATH = os.path.join(REPO_DIR, "history_consolidated.parquet")
mc.OUTPUT_DIR = REPO_DIR + os.sep

# Parse CLI args
mode = "train"
fast = False
for arg in sys.argv[1:]:
    if arg in ("--load", "load"):
        mode = "load"
    elif arg in ("--train", "train"):
        mode = "train"
    elif arg in ("--fast", "fast"):
        fast = True

mc.RUN_MODE = mode

if fast:
    # Quick validation mode: smaller GA, fewer generations
    mc.GA_POP_SIZE = 30
    mc.GA_NGEN = 10
    mc.GA_STAGE1_POP_SIZE = 20
    mc.GA_STAGE1_NGEN = 5
    mc.GA_STAGE2_POP_SIZE = 30
    mc.GA_STAGE2_NGEN = 8
    mc.EARLY_STOP = 5
    mc.WF_SPLITS = 3
    mc.PRINT_EVERY = 1
    print("[LOCAL] FAST mode: reduced GA pop/gen for quick validation")

print(f"[LOCAL] Mode: {mode}")
print(f"[LOCAL] Input: {mc.HISTORY_CSV_PATH}")
print(f"[LOCAL] Output: {mc.OUTPUT_DIR}")
print(f"[LOCAL] GA config: pop={mc.GA_POP_SIZE} ngen={mc.GA_NGEN}")

t0 = time.perf_counter()
mc.run()
dt = time.perf_counter() - t0
print(f"\n[LOCAL] Pipeline completed in {dt/60.0:.1f} min")

# --- Post-run metrics summary ---
import pandas as pd
import numpy as np

xlsx_path = os.path.join(mc.OUTPUT_DIR, f"apply_PER_TICKER_WFGA_intraday__H{mc.FWD_H}__APPLY{mc.APPLY_DAYS}D__v2.xlsx")
csv_path = os.path.join(mc.OUTPUT_DIR, f"apply_last_{mc.APPLY_DAYS}d__H{mc.FWD_H}__v2.csv")
summary_path = os.path.join(mc.OUTPUT_DIR, f"summary_last_{mc.APPLY_DAYS}d__H{mc.FWD_H}__v2.partial.csv")

print("\n" + "="*70)
print("  PIPELINE RESULTS SUMMARY")
print("="*70)

if os.path.exists(summary_path):
    df = pd.read_csv(summary_path)
    print(f"\nTickers processed: {len(df)}")

    # KPI calculations
    wr = df["test_win_rate"].dropna()
    rr = df["gp_reward_risk_ratio"].dropna()
    mdd = df["test_mdd"].dropna().abs()
    ret = df["test_return"].dropna()
    sharpe = df["test_sharpe"].dropna()
    score = df["score_0_100"].dropna()
    trades = df["test_n_trades"].dropna()

    active = df[df["test_n_trades"] > 0]
    print(f"Active tickers (>0 trades): {len(active)}")

    if len(active) > 0:
        print(f"\n--- KPI METRICS (active tickers) ---")
        wr_act = active["test_win_rate"].dropna()
        print(f"  Win Rate (median):    {wr_act.median()*100:.1f}%   (target: >65%)")
        print(f"  Win Rate (mean):      {wr_act.mean()*100:.1f}%")

        rr_val = active["gp_reward_risk_ratio"].dropna()
        print(f"  Risk-Reward (median): {rr_val.median():.2f}     (target: >4.0)")

        mdd_act = active["test_mdd"].dropna().abs()
        print(f"  MDD (median):         {mdd_act.median()*100:.1f}%   (target: <18%)")
        print(f"  MDD (max):            {mdd_act.max()*100:.1f}%")

        ret_act = active["test_return"].dropna()
        print(f"  Return (median):      {ret_act.median()*100:.1f}%   (target: >15%)")
        print(f"  Return (mean):        {ret_act.mean()*100:.1f}%")

        sh_act = active["test_sharpe"].dropna()
        print(f"  Sharpe (median):      {sh_act.median():.3f}")

        sc_act = active["score_0_100"].dropna()
        print(f"  Confidence (median):  {sc_act.median():.1f}     (target: >70)")

        tr_act = active["test_n_trades"].dropna()
        print(f"  Trades/ticker (med):  {tr_act.median():.0f}")

        # Buy&Hold comparison
        if "buyhold_return_1y" in active.columns:
            bh = active["buyhold_return_1y"].dropna()
            ga_ret = active["ga_return_1y"].dropna()
            if len(bh) > 0 and len(ga_ret) > 0:
                beats_bh = (ga_ret.values[:min(len(ga_ret), len(bh))] > bh.values[:min(len(ga_ret), len(bh))]).mean()
                print(f"  Beats B&H (%):        {beats_bh*100:.1f}%  (target: >60%)")

        # Signals summary
        if "signal" in active.columns:
            sigs = active["signal"].value_counts()
            print(f"\n--- SIGNALS ---")
            for s, cnt in sigs.items():
                print(f"  {s}: {cnt}")

    print(f"\n--- OUTPUT FILES ---")
    for p in [xlsx_path, csv_path, summary_path]:
        if os.path.exists(p):
            sz = os.path.getsize(p)
            print(f"  {os.path.basename(p)}: {sz/1024:.0f} KB")
        else:
            print(f"  {os.path.basename(p)}: NOT FOUND")
else:
    print("[WARN] No summary file found — pipeline may have failed")

# --- Telegram send ---
try:
    import requests
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if bot_token and chat_id and os.path.exists(xlsx_path):
        print(f"\n[TELEGRAM] Sending {os.path.basename(xlsx_path)}...")
        url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
        with open(xlsx_path, "rb") as f:
            resp = requests.post(url, data={"chat_id": chat_id, "caption": f"GA Pipeline Results - {time.strftime('%Y-%m-%d %H:%M')}"}, files={"document": f})
        if resp.status_code == 200:
            print("[TELEGRAM] Sent successfully!")
        else:
            print(f"[TELEGRAM] Failed: {resp.status_code} {resp.text[:200]}")
    elif not bot_token or not chat_id:
        print("\n[TELEGRAM] Skipped — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars")
except ImportError:
    print("\n[TELEGRAM] Skipped — requests library not installed")
except Exception as e:
    print(f"\n[TELEGRAM] Error: {e}")
