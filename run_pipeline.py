#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Pipeline Runner
=====================
Executes all 4 pipeline steps sequentially:
  1. ETL (Copy_of_STOCK_ETL_v2)       → downloads data, generates features/targets, saves parquet
  2. BIN (bin_Stock_modelos_individuais) → trains/loads binary classification models (up20/dd5)
  3. REG (reg_Stock_modelos_individuais) → baseline regression forecasts (best_entry/best_sale)
  4. FINAL (Final_stock_output)          → consolidates all signals into final output

All outputs are saved under ./output/ directory.

Modes
-----
  full (default)
      Run all 4 steps normally.  Step 2 loads saved binary models and rescores
      all history (no retraining).  Produces the most up-to-date predictions.
      Typical runtime: ~2-4 h.

  data-only
      Run only Step 1 (ETL) + Step 4 (FINAL).  Steps 2 and 3 are skipped and
      their LAST SAVED output files are reused.
      Use this for routine daily/weekly price-data refreshes when the binary and
      regression models have been updated recently enough.
      Typical runtime: ~35-45 min (ETL + final consolidation).

      NOTE: Records newer than the last Step 2/3 run will have p_up20=0 /
      regression=NaN because no new ML predictions exist for them yet.
      This matches the CURRENT behaviour (binary models are effectively zero
      due to a feature-schema mismatch — see WARN message when running step 2).

Usage
-----
    python run_pipeline.py                         # full (all 4 steps)
    python run_pipeline.py --mode data-only        # ETL + final only (~35 min)
    python run_pipeline.py --steps 1              # ETL only
    python run_pipeline.py --steps 1,2            # ETL + binary models
    python run_pipeline.py --steps 3,4            # regression + final
    python run_pipeline.py --mode data-only --steps 1,4   # explicit data-only
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


def step1_etl(mode: str = "full"):
    """Step 1: ETL — download, feature engineering, target creation, save parquet."""
    print("\n" + "=" * 80)
    print("STEP 1/4 — ETL (Copy_of_STOCK_ETL_v2)")
    print("=" * 80)
    from stock_etl import run_etl
    run_etl()
    print("\n✅ ETL concluído.")
    print(f"   Outputs em: {OUTPUT_DATA}/expanded_stock*.parquet")


def step2_bin_models(mode: str = "full"):
    """Step 2: Binary classification models (LGBM, HGB, RF, ET)."""
    print("\n" + "=" * 80)
    print("STEP 2/4 — BINARY MODELS (bin_Stock_modelos_individuais)")
    if mode == "data-only":
        print("  [mode=data-only] pular se cache existir")
    print("=" * 80)
    from stock_bin_models import run_bin_models
    run_bin_models(mode=mode)
    print("\n✅ Binary models concluído.")
    print(f"   Models em: {OUTPUT_MODELS}/")
    print(f"   Signals em: {OUTPUT_BASE}/ensemble_signals_history*.parquet")


def step3_reg_models(mode: str = "full"):
    """Step 3: Regression baseline forecasts (best entry / best sale)."""
    print("\n" + "=" * 80)
    print("STEP 3/4 — REGRESSION MODELS (reg_Stock_modelos_individuais)")
    if mode == "data-only":
        print("  [mode=data-only] pular se cache existir")
    print("=" * 80)
    from stock_reg_models import run_reg_models
    run_reg_models(mode=mode)
    print("\n✅ Regression models concluído.")
    print(f"   Outputs em: {OUTPUT_BASE}/forecast_history_wide.parquet")


def step4_final_output(mode: str = "full"):
    """Step 4: Final consolidation — merge all signals, compute EV, fund score."""
    print("\n" + "=" * 80)
    print("STEP 4/4 — FINAL OUTPUT (Final_stock_output)")
    print("=" * 80)
    from stock_final_output import run_final_output
    run_final_output()
    print("\n✅ Final output concluído.")
    print(f"   Output em: {OUTPUT_BASE}/history_consolidated.parquet")


STEPS = {
    1: ("ETL",               step1_etl),
    2: ("Binary Models",     step2_bin_models),
    3: ("Regression Models", step3_reg_models),
    4: ("Final Output",      step4_final_output),
}


def main():
    parser = argparse.ArgumentParser(
        description="Stock Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python run_pipeline.py                        # pipeline completo (~2-4h)
  python run_pipeline.py --mode data-only       # só ETL + final (~35min)
  python run_pipeline.py --steps 1              # só ETL
  python run_pipeline.py --steps 2,3,4         # pular ETL, rodar o resto
        """,
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help=(
            "Passos a executar, separados por vírgula (padrão: todos). "
            "Ex.: --steps 1,4  executa só ETL e final output."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "data-only"],
        help=(
            "'full'      – todos os passos normalmente (padrão). "
            "'data-only' – só ETL (passo 1) + consolidação final (passo 4); "
            "pula ML dos passos 2 e 3 usando os arquivos em cache."
        ),
    )
    args = parser.parse_args()

    # Resolve which steps to run
    if args.steps is not None:
        steps_to_run = [int(s.strip()) for s in args.steps.split(",")]
    elif args.mode == "data-only":
        # data-only always means: ETL + final consolidation only
        steps_to_run = [1, 4]
    else:
        steps_to_run = [1, 2, 3, 4]

    mode = args.mode

    print("=" * 80)
    print("STOCK PIPELINE RUNNER")
    print("=" * 80)
    print(f"Mode:             {mode}")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Steps to run:     {steps_to_run}")
    print(f"Working dir:      {os.getcwd()}")
    if mode == "data-only":
        print()
        print("  data-only: steps 2 e 3 serão pulados se os arquivos de cache existirem.")
        print("  Isso reutiliza as previsões ML do último run completo (~35min total).")
    print("=" * 80)

    t0_global = time.time()

    for step_num in steps_to_run:
        if step_num not in STEPS:
            print(f"⚠️  Step {step_num} desconhecido, ignorando.")
            continue

        name, func = STEPS[step_num]
        t0 = time.time()
        try:
            func(mode=mode)
        except Exception as e:
            print(f"\n❌ STEP {step_num} ({name}) FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        elapsed = time.time() - t0
        print(f"⏱️  Step {step_num} ({name}) took {elapsed:.1f}s")

    total_time = time.time() - t0_global
    print("\n" + "=" * 80)
    print(f"✅ PIPELINE COMPLETE — Total time: {total_time:.1f}s")
    print(f"📁 Todos os arquivos salvos em: {OUTPUT_BASE}/")
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
