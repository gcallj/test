#!/usr/bin/env python3
"""Curate universe: identifica tickers que sangram metricos no Run 2.

Le summary_latest.xlsx do Run 2 (genoma SE CLEAN), extrai per-ticker WR e
alpha, e classifica:
  - keep: WR >= keep_wr_min OR alpha >= keep_alpha_min
  - drop: WR < drop_wr_max AND alpha < drop_alpha_max
  - borderline: o resto

Output: markdown report com tabela ranqueada + lista de tickers a remover
do universo SE para Run 3.

Uso:
  python analysis/curate_universe.py
  python analysis/curate_universe.py --keep-wr-min 0.55 --drop-alpha-max -0.10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "summary_latest.xlsx"
OUT = Path("/tmp/curate_universe_report.md")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-wr-min", type=float, default=0.55, help="WR minimo para keep")
    ap.add_argument("--keep-alpha-min", type=float, default=-0.05, help="alpha minimo para keep")
    ap.add_argument("--drop-wr-max", type=float, default=0.50, help="WR maximo para drop")
    ap.add_argument("--drop-alpha-max", type=float, default=-0.15, help="alpha maximo para drop")
    args = ap.parse_args()

    if not SUMMARY.exists():
        print(f"[FAIL] {SUMMARY} nao existe. Rode retrain primeiro.")
        return 1

    print(f"[1/3] Carregando {SUMMARY}...")
    # Tenta varias abas comuns
    xl = pd.ExcelFile(SUMMARY)
    print(f"  Sheets: {xl.sheet_names}")
    df = None
    for sheet in xl.sheet_names:
        try:
            tmp = xl.parse(sheet)
            cols = [c.lower() for c in tmp.columns]
            if "ticker" in cols and any(c in cols for c in ["win_rate", "wr", "wr_med"]):
                df = tmp
                df.columns = [c.lower() for c in df.columns]
                print(f"  Using sheet '{sheet}' ({len(df)} rows)")
                break
        except Exception:
            continue

    if df is None:
        print("[FAIL] Nao achei sheet com colunas ticker + win_rate")
        return 2

    # Normalizar colunas
    rename = {}
    for c in list(df.columns):
        if c in ("ticker",):
            rename[c] = "ticker"
        elif c in ("win_rate", "wr", "wr_med"):
            rename[c] = "wr"
        elif c in ("alpha_ann", "mean_alpha_ann", "alpha"):
            rename[c] = "alpha"
        elif c in ("n_trades", "trades"):
            rename[c] = "n_trades"
        elif c in ("total_return", "ret"):
            rename[c] = "ret"
        elif c in ("mdd", "max_drawdown"):
            rename[c] = "mdd"
    df = df.rename(columns=rename)

    if "ticker" not in df.columns or "wr" not in df.columns:
        print(f"[FAIL] Colunas insuficientes apos rename: {df.columns.tolist()}")
        return 2

    df = df.dropna(subset=["ticker", "wr"]).copy()
    df["wr"] = pd.to_numeric(df["wr"], errors="coerce")
    if "alpha" in df.columns:
        df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    else:
        df["alpha"] = 0.0

    # Filter SE only
    df = df[df["ticker"].astype(str).str.endswith(".ST")].copy()
    if df.empty:
        print("[FAIL] Nenhum ticker .ST no summary")
        return 2

    print(f"[2/3] Classificando {len(df)} tickers .ST...")

    def classify(row):
        wr = row["wr"]
        alpha = row.get("alpha", 0.0)
        if wr >= args.keep_wr_min or (not pd.isna(alpha) and alpha >= args.keep_alpha_min):
            return "keep"
        if wr < args.drop_wr_max and (pd.isna(alpha) or alpha < args.drop_alpha_max):
            return "drop"
        return "borderline"

    df["decision"] = df.apply(classify, axis=1)
    df_sorted = df.sort_values(["wr", "alpha"], ascending=[False, False])

    keep = df[df["decision"] == "keep"]["ticker"].tolist()
    drop = df[df["decision"] == "drop"]["ticker"].tolist()
    borderline = df[df["decision"] == "borderline"]["ticker"].tolist()

    print(f"  keep: {len(keep)}, drop: {len(drop)}, borderline: {len(borderline)}")

    print("[3/3] Writing report...")
    sections = [
        "# Curate Universe Report (SE Run 2)\n",
        f"Source: `{SUMMARY.name}`",
        f"Thresholds: keep if WR>={args.keep_wr_min} OR alpha>={args.keep_alpha_min}",
        f"            drop if WR<{args.drop_wr_max} AND alpha<{args.drop_alpha_max}",
        f"\nResult: keep={len(keep)}, drop={len(drop)}, borderline={len(borderline)}\n",
        "## Ranking completo\n",
        "```",
        df_sorted[["ticker", "wr", "alpha", "decision"] + ([c for c in ("n_trades", "ret", "mdd") if c in df_sorted.columns])].to_string(index=False),
        "```\n",
        "## Tickers a manter (keep + borderline)\n",
        "```python",
        "omx_tickers_curated = [",
    ]
    for t in (keep + borderline):
        sections.append(f'    "{t}",')
    sections += [
        "]",
        "```\n",
        "## Tickers a remover\n",
    ]
    for t in drop:
        row = df[df["ticker"] == t].iloc[0]
        sections.append(f"- `{t}` (WR={row['wr']:.3f}, alpha={row.get('alpha', 0):+.3f})")

    OUT.write_text("\n".join(sections), encoding="utf-8")
    print(f"\n[OK] Report saved to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
