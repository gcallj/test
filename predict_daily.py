#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Predict Pipeline (fast mode)
===================================
1. Run ETL to download fresh market data from yfinance
2. Run GA in load mode (causal features only -> signals + Telegram)

BIN/REG/FINAL steps are SKIPPED in daily mode because the GA uses only
causal OHLCV-derived features (48 allowed, 48 model outputs blocked).
For research with BIN/REG, use: python run_pipeline.py --mode full

Usage:
    python predict_daily.py              # ETL + GA (default, ~40 min)
    python predict_daily.py --full       # ETL + BIN + REG + FINAL + GA (~3.5 h)
"""

import argparse
import os
import time


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(REPO_DIR, "output")
OUTPUT_DATA = os.path.join(OUTPUT_BASE, "data")

for d in [OUTPUT_BASE, OUTPUT_DATA]:
    os.makedirs(d, exist_ok=True)


def step_etl():
    """Download fresh market data via ETL."""
    print("\n" + "=" * 60)
    print("STEP 1: ETL (download fresh data)")
    print("=" * 60)
    t0 = time.time()
    from stock_etl import run_etl
    run_etl()
    print(f"[OK] ETL done in {time.time() - t0:.0f}s")


def step_bin_refresh():
    """Refresh BIN probabilities from saved artifacts (research only)."""
    print("\n" + "=" * 60)
    print("STEP 2: BIN refresh (load saved artifacts)")
    print("=" * 60)
    t0 = time.time()
    from stock_bin_models import run_bin_models
    run_bin_models(mode="auto")
    print(f"[OK] BIN done in {time.time() - t0:.0f}s")


def step_reg_refresh():
    """Rebuild REG forecasts (research only)."""
    print("\n" + "=" * 60)
    print("STEP 3: REG refresh (recompute forecasts)")
    print("=" * 60)
    t0 = time.time()
    from stock_reg_models import run_reg_models
    run_reg_models(mode="full")
    print(f"[OK] REG done in {time.time() - t0:.0f}s")


def step_final():
    """Run FINAL consolidation (research only)."""
    print("\n" + "=" * 60)
    print("STEP 4: FINAL (consolidate all signals)")
    print("=" * 60)
    t0 = time.time()
    from stock_final_output import run_final_output
    run_final_output()
    print(f"[OK] FINAL done in {time.time() - t0:.0f}s")


def step_ga_load():
    """Run GA in load mode — generates signals from causal features."""
    print("\n" + "=" * 60)
    print("STEP GA: load mode (causal features -> signals + Telegram)")
    print("=" * 60)
    t0 = time.time()
    os.environ["GA_RUN_MODE"] = "load"
    from ga_run import run
    run()
    print(f"[OK] GA done in {time.time() - t0:.0f}s")


def step_refresh_close():
    """Patch Apply.close/Date with live yfinance closes (from PR #134).

    stock_etl.py intentionally shifts features by 1 bar to prevent leakage, but
    the shift also moves the OHLCV base columns — so Apply.close for Date=T
    ends up carrying T-1 close. This step pulls live closes from yfinance and
    overwrites just the close+Date cells so the sheet matches broker quotes.
    Other fields (entrada/alvo/stop/pivot/recomendacao) stay as the GA
    computed them — they are forward-looking levels and remain consistent.
    """
    print("\n" + "=" * 60)
    print("STEP REFRESH: refresh Apply.close / Date from yfinance")
    print("=" * 60)
    try:
        from refresh_close import refresh
        stats = refresh()
        if stats.get("enabled"):
            print(f"[OK] refresh_close: {stats.get('n_refreshed')} refreshed, "
                  f"{stats.get('n_missing')} missing, "
                  f"max_date={stats.get('max_new_date')}")
        else:
            print(f"[SKIP] refresh_close: {stats.get('reason')}")
    except Exception as e:
        print(f"[WARN] refresh_close failed (non-fatal): {e}")


def step_guide_overlay():
    """Overlay analyst-guide PDF onto summary_latest.xlsx (from PR #134).
    No-op if no PDF present in ./guides/."""
    print("\n" + "=" * 60)
    print("STEP GUIDE: apply PDF guide overlay (if any)")
    print("=" * 60)
    try:
        from guide_overlay import apply_overlay
        stats = apply_overlay()
        if stats.get("enabled"):
            print(f"[OK] guide overlay: {stats}")
        else:
            reason = stats.get("reason") or "unknown"
            print(f"[SKIP] guide overlay: {reason} "
                  f"(drop a PDF in ./guides/ or set PDF_GUIDE_PATH)")
    except Exception as e:
        print(f"[WARN] guide overlay failed (non-fatal): {e}")


def step_log_predictions():
    """C1 (overfitting prevention): append signals do summary_latest.xlsx
    em analysis/predictions_log.jsonl para criar ground truth no futuro.
    Nunca quebra o pipeline — erros viram warning."""
    print("\n" + "=" * 60)
    print("STEP LOG: append predictions to predictions_log.jsonl")
    print("=" * 60)
    try:
        from analysis.log_predictions import append_log
        n = append_log(dry_run=False)
        print(f"[OK] logged {n} predictions")
    except Exception as e:
        # Best-effort: nao queremos que um bug no logger quebre o daily pipeline
        print(f"[WARN] log_predictions falhou (nao-fatal): {e}")


def main():
    parser = argparse.ArgumentParser(description="Daily Predict Pipeline")
    parser.add_argument(
        "--full", action="store_true",
        help="Run full pipeline including BIN/REG/FINAL (research mode, ~3.5h)"
    )
    args = parser.parse_args()

    print("=" * 60)
    if args.full:
        print("DAILY PIPELINE — FULL (ETL + BIN + REG + FINAL + GA)")
    else:
        print("DAILY PIPELINE — FAST (ETL + GA causal)")
    print("=" * 60)
    t_start = time.time()

    step_etl()

    if args.full:
        step_bin_refresh()
        step_reg_refresh()
        step_final()

    step_ga_load()
    step_refresh_close()   # PR #134: fix close/Date de yfinance live
    step_guide_overlay()   # PR #134: overlay PDF analyst guide (no-op se sem PDF)
    step_log_predictions()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE in {elapsed / 60:.1f} min")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
