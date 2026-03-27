#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Predict Pipeline (no model retraining)
=============================================
1. Run ETL to download fresh market data
2. Forward-fill BIN/REG signals for new dates (no retraining needed)
3. Run FINAL to consolidate
4. Run GA in load mode (signals + Telegram)

Usage:
    python predict_daily.py
"""

import os
import sys
import time
import pandas as pd
import numpy as np


OUTPUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUTPUT_DATA = os.path.join(OUTPUT_BASE, "data")

ENS_PATH = os.path.join(OUTPUT_BASE, "ensemble_signals_history.parquet")
REG_PATH = os.path.join(OUTPUT_BASE, "forecast_history_wide.parquet")
ETL_PATH = os.path.join(OUTPUT_DATA, "expanded_stock_reduced.parquet")

for d in [OUTPUT_BASE, OUTPUT_DATA]:
    os.makedirs(d, exist_ok=True)


def step1_etl():
    """Download fresh market data via ETL."""
    print("\n" + "=" * 60)
    print("STEP 1: ETL (download fresh data)")
    print("=" * 60)
    t0 = time.time()
    from stock_etl import run_etl
    run_etl()
    print(f"[OK] ETL done in {time.time() - t0:.0f}s")


def get_etl_dates_tickers():
    """Get all unique dates and tickers from ETL parquet."""
    import pyarrow.parquet as pq
    import ast

    pf = pq.ParquetFile(ETL_PATH)
    all_cols = pf.schema_arrow.names

    # Parse ticker names from MultiIndex columns
    tickers = set()
    for c in all_cols:
        try:
            t = ast.literal_eval(c)
            if isinstance(t, tuple) and len(t) == 2:
                tickers.add(t[1])
        except (ValueError, SyntaxError):
            pass

    # Read dates from parquet index (DatetimeIndex stored in pandas metadata)
    # columns=[] reads zero data columns but preserves the index
    df_idx = pd.read_parquet(ETL_PATH, columns=[])
    dates = pd.to_datetime(df_idx.index)
    print(f"[ETL] index type: {type(df_idx.index)}, first: {dates[0]}, last: {dates[-1]}")

    return sorted(dates), sorted(tickers)


def step2_forward_fill_signals():
    """Forward-fill BIN ensemble and REG forecast signals to cover new ETL dates."""
    print("\n" + "=" * 60)
    print("STEP 2: Forward-fill BIN/REG signals for new dates")
    print("=" * 60)
    t0 = time.time()

    etl_dates, etl_tickers = get_etl_dates_tickers()
    etl_max = max(etl_dates)
    print(f"[ETL] dates: {min(etl_dates).date()} to {etl_max.date()}, tickers: {len(etl_tickers)}")

    # --- Forward-fill ENS (ensemble_signals_history) ---
    if os.path.exists(ENS_PATH):
        ens = pd.read_parquet(ENS_PATH)
        ens["Date"] = pd.to_datetime(ens["Date"])
        ens_max = ens["Date"].max()
        print(f"[ENS] current max date: {ens_max.date()}, rows: {len(ens)}")

        if etl_max > ens_max:
            # Get new dates that need signals
            new_dates = [d for d in etl_dates if d > ens_max]
            print(f"[ENS] forward-filling {len(new_dates)} new dates")

            # Get last known signals per ticker
            last_signals = ens[ens["Date"] == ens_max].copy()

            new_rows = []
            for new_date in new_dates:
                chunk = last_signals.copy()
                chunk["Date"] = new_date
                new_rows.append(chunk)

            if new_rows:
                ens_extended = pd.concat([ens] + new_rows, ignore_index=True)
                ens_extended.to_parquet(ENS_PATH, index=False)
                print(f"[ENS] extended to {ens_extended['Date'].max().date()} ({len(ens_extended)} rows)")
            del last_signals, new_rows
        else:
            print(f"[ENS] already up to date")
    else:
        print(f"[ENS] not found at {ENS_PATH}, skipping")

    # --- Forward-fill REG (forecast_history_wide) ---
    if os.path.exists(REG_PATH):
        reg = pd.read_parquet(REG_PATH)
        reg["Date"] = pd.to_datetime(reg["Date"])
        reg_max = reg["Date"].max()
        print(f"[REG] current max date: {reg_max.date()}, rows: {len(reg)}")

        if etl_max > reg_max:
            new_dates = [d for d in etl_dates if d > reg_max]
            print(f"[REG] forward-filling {len(new_dates)} new dates")

            last_forecasts = reg[reg["Date"] == reg_max].copy()

            new_rows = []
            for new_date in new_dates:
                chunk = last_forecasts.copy()
                chunk["Date"] = new_date
                new_rows.append(chunk)

            if new_rows:
                reg_extended = pd.concat([reg] + new_rows, ignore_index=True)
                reg_extended.to_parquet(REG_PATH, index=False)
                print(f"[REG] extended to {reg_extended['Date'].max().date()} ({len(reg_extended)} rows)")
            del last_forecasts, new_rows
        else:
            print(f"[REG] already up to date")
    else:
        print(f"[REG] not found at {REG_PATH}, skipping")

    print(f"[OK] Forward-fill done in {time.time() - t0:.0f}s")


def step3_final():
    """Run FINAL consolidation."""
    print("\n" + "=" * 60)
    print("STEP 3: FINAL (consolidate all signals)")
    print("=" * 60)
    t0 = time.time()
    from stock_final_output import run_final_output
    run_final_output()
    print(f"[OK] FINAL done in {time.time() - t0:.0f}s")


def step4_ga_load():
    """Run GA in load mode."""
    print("\n" + "=" * 60)
    print("STEP 4: GA load (generate signals + Telegram)")
    print("=" * 60)
    t0 = time.time()

    # Force load mode
    os.environ["GA_RUN_MODE"] = "load"

    from ga_run import run
    run()
    print(f"[OK] GA done in {time.time() - t0:.0f}s")


def main():
    print("=" * 60)
    print("DAILY PREDICT PIPELINE (no retraining)")
    print("=" * 60)
    t_start = time.time()

    step1_etl()
    step2_forward_fill_signals()
    step3_final()
    step4_ga_load()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE in {elapsed / 60:.1f} min")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
