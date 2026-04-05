#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Predict Pipeline
======================
1. Run ETL to download fresh market data
2. Refresh BIN predictions from saved artifacts (no label forward-fill)
3. Rebuild REG forecasts from the fresh ETL history
4. Run FINAL to consolidate
5. Run GA in load mode (signals + Telegram)

Usage:
    python predict_daily.py
"""

import os
import time


OUTPUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUTPUT_DATA = os.path.join(OUTPUT_BASE, "data")

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


def step2_bin_refresh():
    """Refresh BIN probabilities from saved artifacts when available."""
    print("\n" + "=" * 60)
    print("STEP 2: BIN refresh (load saved artifacts)")
    print("=" * 60)
    t0 = time.time()
    from stock_bin_models import run_bin_models
    # auto = load artifacts if present, fallback to retrain only when necessary
    run_bin_models(mode="auto")
    print(f"[OK] BIN done in {time.time() - t0:.0f}s")

def step3_reg_refresh():
    """Rebuild REG forecasts from the fresh ETL history."""
    print("\n" + "=" * 60)
    print("STEP 3: REG refresh (recompute forecasts)")
    print("=" * 60)
    t0 = time.time()
    from stock_reg_models import run_reg_models
    run_reg_models(mode="full")
    print(f"[OK] REG done in {time.time() - t0:.0f}s")


def step4_final():
    """Run FINAL consolidation."""
    print("\n" + "=" * 60)
    print("STEP 4: FINAL (consolidate all signals)")
    print("=" * 60)
    t0 = time.time()
    from stock_final_output import run_final_output
    run_final_output()
    print(f"[OK] FINAL done in {time.time() - t0:.0f}s")


def step5_ga_load():
    """Run GA in load mode."""
    print("\n" + "=" * 60)
    print("STEP 5: GA load (generate signals + Telegram)")
    print("=" * 60)
    t0 = time.time()

    # Force load mode
    os.environ["GA_RUN_MODE"] = "load"

    from ga_run import run
    run()
    print(f"[OK] GA done in {time.time() - t0:.0f}s")


def main():
    print("=" * 60)
    print("DAILY PREDICT PIPELINE")
    print("=" * 60)
    t_start = time.time()

    step1_etl()
    step2_bin_refresh()
    step3_reg_refresh()
    step4_final()
    step5_ga_load()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE in {elapsed / 60:.1f} min")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
