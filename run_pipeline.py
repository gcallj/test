#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Pipeline Runner
=====================
Executes all 4 pipeline steps sequentially:
  1. ETL (Copy_of_STOCK_ETL_v2)       -> downloads data, generates features/targets, saves parquet
  2. BIN (bin_Stock_modelos_individuais) -> trains/loads binary classification models (up20/dd5)
  3. REG (reg_Stock_modelos_individuais) -> baseline regression forecasts (best_entry/best_sale)
  4. FINAL (Final_stock_output)          -> consolidates all signals into final output

All outputs are saved under ./output/ directory in the codespace.

Usage:
    python run_pipeline.py              # run all steps
    python run_pipeline.py --steps 1    # run only ETL
    python run_pipeline.py --steps 1,2  # run ETL + BIN
    python run_pipeline.py --steps 3,4  # run REG + FINAL
"""

import os
import sys
import time
import argparse


# ============================================================
# Ensure output directories exist
# ============================================================
OUTPUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUTPUT_DATA = os.path.join(OUTPUT_BASE, "data")
OUTPUT_MODELS = os.path.join(OUTPUT_DATA, "models")

for d in [OUTPUT_BASE, OUTPUT_DATA, OUTPUT_MODELS]:
    os.makedirs(d, exist_ok=True)


def step1_etl():
    """Step 1: ETL -- download, feature engineering, target creation, save parquet."""
    print("\n" + "=" * 80)
    print("STEP 1/4 -- ETL (Copy_of_STOCK_ETL_v2)")
    print("=" * 80)
    from stock_etl import run_etl
    run_etl()
    print("\n[OK] ETL concluido.")
    print(f"   Outputs em: {OUTPUT_DATA}/expanded_stock*.parquet")


def step2_bin_models():
    """Step 2: Binary classification models (LGBM, HGB, RF, ET)."""
    print("\n" + "=" * 80)
    print("STEP 2/4 -- BINARY MODELS (bin_Stock_modelos_individuais)")
    print("=" * 80)
    from stock_bin_models import run_bin_models
    run_bin_models()
    print("\n[OK] Binary models concluido.")
    print(f"   Models em: {OUTPUT_MODELS}/")
    print(f"   Signals em: {OUTPUT_BASE}/ensemble_signals_history*.parquet")


def step3_reg_models():
    """Step 3: Regression baseline forecasts (best entry / best sale)."""
    print("\n" + "=" * 80)
    print("STEP 3/4 -- REGRESSION MODELS (reg_Stock_modelos_individuais)")
    print("=" * 80)
    from stock_reg_models import run_reg_models
    run_reg_models()
    print("\n[OK] Regression models concluido.")
    print(f"   Outputs em: {OUTPUT_BASE}/forecast_history_wide.parquet")


def step4_final_output():
    """Step 4: Final consolidation -- merge all signals, compute EV, fund score."""
    print("\n" + "=" * 80)
    print("STEP 4/4 -- FINAL OUTPUT (Final_stock_output)")
    print("=" * 80)
    from stock_final_output import run_final_output
    run_final_output()
    print("\n[OK] Final output concluido.")
    print(f"   Output em: {OUTPUT_BASE}/history_consolidated.parquet")


STEPS = {
    1: ("ETL",              step1_etl),
    2: ("Binary Models",    step2_bin_models),
    3: ("Regression Models", step3_reg_models),
    4: ("Final Output",     step4_final_output),
}


def main():
    parser = argparse.ArgumentParser(description="Stock Pipeline Runner")
    parser.add_argument(
        "--steps",
        type=str,
        default="1,2,3,4",
        help="Comma-separated step numbers to run (default: 1,2,3,4)",
    )
    args = parser.parse_args()

    steps_to_run = [int(s.strip()) for s in args.steps.split(",")]

    print("=" * 80)
    print("STOCK PIPELINE RUNNER")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Steps to run:     {steps_to_run}")
    print(f"Working dir:      {os.getcwd()}")
    print("=" * 80)

    t0_global = time.time()

    for step_num in steps_to_run:
        if step_num not in STEPS:
            print(f"[WARN] Step {step_num} unknown, skipping.")
            continue

        name, func = STEPS[step_num]
        t0 = time.time()
        try:
            func()
        except Exception as e:
            print(f"\n[FAIL] STEP {step_num} ({name}) FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        elapsed = time.time() - t0
        print(f"[TIME] Step {step_num} ({name}) took {elapsed:.1f}s")

    total_time = time.time() - t0_global
    print("\n" + "=" * 80)
    print(f"[OK] PIPELINE COMPLETE -- Total time: {total_time:.1f}s")
    print(f"Todos os arquivos salvos em: {OUTPUT_BASE}/")
    print("=" * 80)

    # List output files
    print("\nArquivos gerados:")
    for root, _, files in os.walk(OUTPUT_BASE):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            relpath = os.path.relpath(fpath, os.path.dirname(OUTPUT_BASE))
            print(f"  {relpath:60s} {size_mb:8.2f} MB")


if __name__ == "__main__":
    main()
