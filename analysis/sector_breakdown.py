"""Per-segment aggregation of per-ticker trading metrics (plan item D3).

Rolls per-ticker WR / alpha / trades / MDD up to the 11 segments defined in
`ticker_metadata.SECTOR_MAP`, plus a global ALL row. Surfaces concentration and
dead-sector signals that single-number alpha hides.

Usage as a library:

    from analysis.sector_breakdown import breakdown_by_segment, from_summary_xlsx
    df_per_ticker = from_summary_xlsx("summary_latest.xlsx")
    seg_df = breakdown_by_segment(df_per_ticker)

CLI:

    python analysis/sector_breakdown.py --summary summary_latest.xlsx
    python analysis/sector_breakdown.py --summary summary_latest.xlsx --json out.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Allow direct execution: `python analysis/sector_breakdown.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ticker_metadata import get_segment_for_tickers  # noqa: E402


METRIC_AGGS = {
    "alpha_ann_pp": "median",
    "wr": "median",
    "n_trades": "sum",
    "mdd": "median",
    "ret_ann": "median",
    "bh_ann": "median",
    "sharpe": "median",
}


_SUMMARY_RENAMES = {
    "test_win_rate": "wr",
    "test_return": "ret_total",
    "buy_hold_return": "bh_total",
    "test_mdd": "mdd",
    "test_sharpe": "sharpe",
    "win_rate": "wr",
}


def _coerce_per_ticker(rows: pd.DataFrame) -> pd.DataFrame:
    if "ticker" not in rows.columns:
        raise ValueError("Input DataFrame requires a `ticker` column")
    out = rows.copy()
    out["ticker"] = out["ticker"].astype(str).str.strip()
    for col in METRIC_AGGS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def breakdown_by_segment(
    rows: pd.DataFrame,
    segments_override: Optional[dict] = None,
) -> pd.DataFrame:
    """Aggregate per-ticker metrics by segment.

    The output has one row per segment (and an `ALL` row) indexed by segment.
    Columns are: `n_tickers`, every metric column present in `rows` aggregated
    per `METRIC_AGGS`, and `beat_bh_pct` when both `ret_ann` and `bh_ann` are
    present.
    """
    df = _coerce_per_ticker(rows)
    if segments_override is not None:
        df["segment"] = df["ticker"].map(segments_override).fillna("unknown")
    else:
        seg_map = get_segment_for_tickers(df["ticker"].unique().tolist())
        df["segment"] = df["ticker"].map(seg_map).fillna("unknown")

    metrics_present = [c for c in METRIC_AGGS if c in df.columns]
    has_beat = {"ret_ann", "bh_ann"}.issubset(df.columns)
    if has_beat:
        df["_beat_bh"] = (df["ret_ann"] > df["bh_ann"]).astype(float)

    grouped = df.groupby("segment", sort=False)
    parts: dict = {"n_tickers": grouped["ticker"].nunique()}
    for col in metrics_present:
        parts[col] = grouped[col].agg(METRIC_AGGS[col])
    if has_beat:
        parts["beat_bh_pct"] = grouped["_beat_bh"].mean() * 100.0
    out = pd.DataFrame(parts)

    all_parts: dict = {"n_tickers": df["ticker"].nunique()}
    for col in metrics_present:
        all_parts[col] = (
            df[col].median() if METRIC_AGGS[col] == "median" else df[col].sum()
        )
    if has_beat:
        all_parts["beat_bh_pct"] = df["_beat_bh"].mean() * 100.0
    out.loc["ALL"] = pd.Series(all_parts)

    sort_col = "alpha_ann_pp" if "alpha_ann_pp" in out.columns else None
    if sort_col is not None:
        # Keep ALL at the bottom regardless of value.
        seg_part = out.drop(index="ALL").sort_values(sort_col, ascending=False)
        out = pd.concat([seg_part, out.loc[["ALL"]]])
    return out


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in _SUMMARY_RENAMES.items() if k in df.columns})
    # Derive alpha_ann_pp when only ret_ann/bh_ann are present.
    if "alpha_ann_pp" not in df.columns and {"ret_ann", "bh_ann"}.issubset(df.columns):
        df["alpha_ann_pp"] = (
            pd.to_numeric(df["ret_ann"], errors="coerce")
            - pd.to_numeric(df["bh_ann"], errors="coerce")
        ) * 100.0
    return df


def from_summary_xlsx(path: str | Path, sheet: str = "Summary") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    return _normalize_columns(df)


def format_markdown(seg_df: pd.DataFrame, title: str = "Per-segment breakdown") -> str:
    """Render the breakdown DataFrame as a markdown table for log embedding."""
    df = seg_df.reset_index().rename(columns={"index": "segment"})
    # Float formatting per column
    for col in df.columns:
        if col in ("segment", "n_tickers"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "—")
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False, name=None)]
    return f"### {title}\n\n" + "\n".join([header, sep, *rows]) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(description="Per-segment trading metrics rollup (D3).")
    p.add_argument("--summary", default="summary_latest.xlsx", help="Path to summary xlsx.")
    p.add_argument("--sheet", default="Summary", help="Sheet name to read.")
    p.add_argument("--json", dest="json_out", default=None, help="Optional path to dump JSON.")
    p.add_argument("--markdown", dest="md_out", default=None, help="Optional path to dump markdown.")
    args = p.parse_args(argv)

    src = Path(args.summary)
    if not src.exists():
        print(f"[sector_breakdown] summary not found: {src}", file=sys.stderr)
        return 2
    df = from_summary_xlsx(src, sheet=args.sheet)
    seg = breakdown_by_segment(df)
    print(seg.to_string())
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(seg.reset_index().to_dict(orient="records"), indent=2, default=str),
            encoding="utf-8",
        )
    if args.md_out:
        Path(args.md_out).write_text(format_markdown(seg), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
