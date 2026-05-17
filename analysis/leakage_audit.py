#!/usr/bin/env python3
"""
Leakage Audit: scans history_consolidated.parquet for data leakage risks.

Checks:
1. Model-output columns that should NOT be GA features
2. Static fundamentals (snapshot applied retroactively)
3. Split column coverage (empty splits = no train/test boundary)
4. Feature contamination in GA feature pool

Usage:
    python analysis/leakage_audit.py --history output/history_consolidated.parquet
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_OUTPUT_PREFIXES = ("p_", "pred_", "EV_", "best_buy", "best_sell", "buy_trust", "sell_trust")
MODEL_OUTPUT_COLS = {
    "risk_return", "upside_pct", "downside_pct", "err_buy_pct", "err_sell_pct",
    "up20_bin", "dd5_bin", "signal",
}
SNAPSHOT_FUNDAMENTALS = {"dividend_yield", "trailing_pe", "price_to_book", "market_cap", "fund_score"}
OHLCV_COLS = {"Date", "ticker", "open", "high", "low", "close", "volume", "price"}


def audit_model_outputs(df):
    """Check which model-output columns exist in the dataframe."""
    issues = []
    for col in df.columns:
        if col in MODEL_OUTPUT_COLS or any(col.startswith(p) for p in MODEL_OUTPUT_PREFIXES):
            nunique = df[col].dropna().nunique()
            issues.append({"column": col, "type": "model_output", "nunique": nunique,
                           "pct_non_null": f"{df[col].notna().mean():.1%}"})
    return issues


def audit_snapshot_fundamentals(df):
    """Check if fundamentals are static (same value across all dates per ticker)."""
    issues = []
    for col in SNAPSHOT_FUNDAMENTALS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().all():
            issues.append({"column": col, "type": "all_nan", "detail": "Column exists but all NaN"})
            continue
        # Check variance per ticker
        var_per_ticker = pd.to_numeric(df[col], errors="coerce").groupby(df["ticker"]).std()
        n_zero_var = (var_per_ticker == 0).sum()
        n_tickers = len(var_per_ticker)
        pct_static = n_zero_var / max(n_tickers, 1)
        issues.append({
            "column": col, "type": "snapshot_fundamental",
            "pct_static_tickers": f"{pct_static:.1%}",
            "n_static": n_zero_var, "n_total": n_tickers,
            "severity": "CRITICAL" if pct_static > 0.90 else ("HIGH" if pct_static > 0.50 else "LOW"),
        })
    return issues


def audit_split_column(df):
    """Check split column coverage."""
    issues = []
    if "split" not in df.columns:
        issues.append({"type": "missing_split", "detail": "No split column found"})
        return issues
    split_vals = df["split"].fillna("").astype(str).str.strip()
    counts = split_vals.value_counts().to_dict()
    empty = counts.get("", 0) + counts.get("nan", 0) + counts.get("NaN", 0)
    total = len(df)
    issues.append({
        "type": "split_coverage",
        "total_rows": total,
        "empty_split_rows": empty,
        "pct_empty": f"{empty/max(total,1):.1%}",
        "split_values": {k: v for k, v in counts.items() if k not in ("", "nan", "NaN")},
    })
    return issues


def audit_causal_features(df):
    """List which columns would pass vs be blocked by the GA causal allowlist."""
    exclude = {"Date", "ticker", "open", "high", "low", "close", "sma200", "sma200_slope", "atr"}
    blocked_cols = OHLCV_COLS | MODEL_OUTPUT_COLS | SNAPSHOT_FUNDAMENTALS | {"signal", "split"}

    causal = []
    blocked = []
    for col in df.columns:
        if col in exclude or col in blocked_cols:
            blocked.append(col)
        elif any(col.startswith(p) for p in MODEL_OUTPUT_PREFIXES):
            blocked.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            causal.append(col)
    return {"causal_features": sorted(causal), "blocked_features": sorted(blocked),
            "n_causal": len(causal), "n_blocked": len(blocked)}


def main():
    parser = argparse.ArgumentParser(description="Leakage Audit for Trading Pipeline")
    parser.add_argument("--history", default="output/history_consolidated.parquet")
    parser.add_argument("--out-dir", default="output/audit")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.history}...")
    df = pd.read_parquet(args.history)
    print(f"  Shape: {df.shape}, Tickers: {df['ticker'].nunique()}")

    print("\n=== 1. MODEL OUTPUT COLUMNS ===")
    model_issues = audit_model_outputs(df)
    for iss in model_issues:
        print(f"  LEAK: {iss['column']:30s} | {iss['pct_non_null']} non-null | nunique={iss['nunique']}")
    print(f"  Total model-output columns found: {len(model_issues)}")

    print("\n=== 2. SNAPSHOT FUNDAMENTALS ===")
    snap_issues = audit_snapshot_fundamentals(df)
    for iss in snap_issues:
        if iss["type"] == "snapshot_fundamental":
            print(f"  {iss['severity']:8s} {iss['column']:20s} | {iss['pct_static_tickers']} of tickers static")
        else:
            print(f"  INFO: {iss['column']:20s} | {iss['detail']}")

    print("\n=== 3. SPLIT COLUMN ===")
    split_issues = audit_split_column(df)
    for iss in split_issues:
        if iss["type"] == "split_coverage":
            print(f"  Total rows: {iss['total_rows']}")
            print(f"  Empty split: {iss['empty_split_rows']} ({iss['pct_empty']})")
            print(f"  Named splits: {iss['split_values']}")

    print("\n=== 4. CAUSAL FEATURE AUDIT ===")
    feat_audit = audit_causal_features(df)
    print(f"  Causal (allowed):  {feat_audit['n_causal']} features")
    print(f"  Blocked:           {feat_audit['n_blocked']} columns")
    print(f"  Causal list: {feat_audit['causal_features'][:20]}{'...' if feat_audit['n_causal'] > 20 else ''}")

    # Save reports
    all_issues = model_issues + snap_issues + split_issues
    flags_df = pd.DataFrame([i for i in all_issues if isinstance(i, dict)])
    flags_path = os.path.join(args.out_dir, "leakage_flags.csv")
    flags_df.to_csv(flags_path, index=False)
    print(f"\nSaved: {flags_path}")

    # Markdown report
    md_path = os.path.join(args.out_dir, "leakage_report.md")
    with open(md_path, "w") as f:
        f.write("# Leakage Audit Report\n\n")
        f.write(f"**Data:** {args.history}\n")
        f.write(f"**Shape:** {df.shape}\n\n")
        f.write(f"## Model Outputs Found: {len(model_issues)}\n")
        for iss in model_issues:
            f.write(f"- `{iss['column']}` ({iss['pct_non_null']} non-null)\n")
        f.write(f"\n## Snapshot Fundamentals\n")
        for iss in snap_issues:
            if iss["type"] == "snapshot_fundamental":
                f.write(f"- `{iss['column']}`: {iss['severity']} ({iss['pct_static_tickers']} static)\n")
        f.write(f"\n## Causal Features: {feat_audit['n_causal']}\n")
        f.write(f"Blocked: {feat_audit['n_blocked']}\n")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
