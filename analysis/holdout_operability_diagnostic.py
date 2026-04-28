#!/usr/bin/env python3
"""Diagnose why the operational Premium/HQ universe disappears in holdout."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.recent_operability import attach_recent_operability


DEFAULT_PREFIX = ROOT / "analysis" / "holdout_operability_diagnostic_current"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def stat_view(stat: Dict[str, Any], gr) -> Dict[str, Any]:
    wr = _safe_float(stat.get("win_rate"))
    wr_target = _safe_float(stat.get("win_rate_target"))
    alpha = _safe_float(stat.get("alpha_ann", stat.get("excess_return", 0.0)))
    mdd_abs = abs(_safe_float(stat.get("mdd")))
    trades = _safe_int(stat.get("n_trades"))
    universe = gr.classify_operational_universe_tier(wr, wr_target, alpha, mdd_abs, trades)
    expected_hold = _safe_float(stat.get("expected_hold_score", gr.expected_hold_score_from_stat(stat)))
    view = {
        "universe": universe,
        "operable": bool(universe in ("PREMIUM", "HIGH_QUAL")),
        "n_trades": trades,
        "win_rate": wr,
        "win_rate_target": wr_target,
        "alpha_ann": alpha,
        "expectancy": _safe_float(stat.get("expectancy")),
        "mdd_abs": mdd_abs,
        "median_hold_bars": _safe_float(stat.get("median_hold_bars")),
        "max_hold_bars": _safe_float(stat.get("max_hold_bars")),
        "expected_hold_score": expected_hold,
        "wr_after_3_bars": _safe_float(stat.get("wr_after_3_bars")),
        "target_wr_after_3_bars": _safe_float(stat.get("target_wr_after_3_bars")),
        "alpha_after_3_bars": _safe_float(stat.get("alpha_after_3_bars")),
        "early_take_dependency": _safe_float(stat.get("early_take_dependency"), 1.0),
        "swing_survival_3d": _safe_float(stat.get("swing_survival_3d")),
    }
    return attach_recent_operability(view)


def failure_reasons(view: Dict[str, Any]) -> list[str]:
    """Explain why a ticker is not acceptable as an OOS swing candidate."""
    reasons: list[str] = []
    if _safe_int(view.get("n_trades")) < 20:
        reasons.append("low_trades_lt_20")
    if _safe_float(view.get("win_rate")) < 0.67:
        reasons.append("wr_lt_67")
    if _safe_float(view.get("win_rate_target")) < 0.65:
        reasons.append("wr_target_lt_65")
    if _safe_float(view.get("alpha_ann")) < -0.20:
        reasons.append("alpha_lt_minus20")
    if _safe_float(view.get("expectancy")) < 0.0:
        reasons.append("expectancy_negative")
    if _safe_float(view.get("mdd_abs"), 1.0) > 0.16:
        reasons.append("mdd_gt_16")
    if _safe_float(view.get("median_hold_bars")) < 3.0:
        reasons.append("hold_lt_3")
    if _safe_float(view.get("median_hold_bars")) > 10.0:
        reasons.append("hold_gt_10")
    if _safe_float(view.get("expected_hold_score")) < 70.0:
        reasons.append("expected_hold_lt_70")
    if _safe_float(view.get("wr_after_3_bars")) < 0.60:
        reasons.append("wr_after_3_lt_60")
    if _safe_float(view.get("swing_survival_3d")) < 0.55:
        reasons.append("survival3_lt_55")
    if _safe_float(view.get("early_take_dependency"), 1.0) > 0.45:
        reasons.append("early_take_dependency_gt_45")
    if view.get("universe") == "REGULAR":
        reasons.append("universe_regular")
    return reasons


def _slice_by_index(cache: Dict[str, Dict[str, Any]], holdout_days: int, lookback_days: int) -> tuple[dict, dict]:
    train: dict[str, dict] = {}
    holdout: dict[str, dict] = {}
    for tk, payload in cache.items():
        dates = payload.get("dates")
        if dates is None:
            continue
        n = len(dates)
        if n <= holdout_days + max(60, lookback_days):
            continue
        cutoff = n - int(holdout_days)
        hold_start = max(0, cutoff - int(lookback_days))
        train[tk] = _slice_payload(payload, 0, cutoff)
        holdout[tk] = _slice_payload(payload, hold_start, n)
    return train, holdout


def _slice_payload(payload: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    out: dict[str, Any] = {}
    n = len(payload.get("dates", []))
    for key, value in payload.items():
        if value is None:
            continue
        arr = np.asarray(value) if not isinstance(value, np.ndarray) else value
        if arr.ndim == 1 and len(arr) == n:
            out[key] = arr[start:end]
        elif arr.ndim == 2 and arr.shape[0] == n:
            out[key] = arr[start:end, :]
        else:
            out[key] = value
    return out


def _ticker_stat(genome: list[float], payload: Dict[str, Any], gr) -> Dict[str, Any]:
    gp = gr.decode_global_params(genome)
    stat = gr.backtest_stats_global_intraday(
        payload["open"],
        payload["high"],
        payload["low"],
        payload["close"],
        payload["score_matrix"],
        payload["atr"],
        gp,
        precomputed=payload,
    )
    gr._attach_benchmark_stats(stat, payload["close"])
    stat["excess_return"] = stat.get("total_return", 0.0) - stat.get("buy_hold_return", 0.0)
    return stat


def build_diagnostic_rows(genome: list[float], train_cache: dict, holdout_cache: dict, gr) -> list[dict]:
    rows: list[dict] = []
    tickers = sorted(set(train_cache) | set(holdout_cache))
    for ticker in tickers:
        train_stat = _ticker_stat(genome, train_cache[ticker], gr) if ticker in train_cache else {}
        hold_stat = _ticker_stat(genome, holdout_cache[ticker], gr) if ticker in holdout_cache else {}
        train_view = stat_view(train_stat, gr) if train_stat else {}
        hold_view = stat_view(hold_stat, gr) if hold_stat else {}
        reasons = failure_reasons(hold_view) if hold_view else ["missing_holdout_payload"]
        row = {"ticker": ticker}
        for prefix, view in (("train", train_view), ("holdout", hold_view)):
            for key, value in view.items():
                row[f"{prefix}_{key}"] = value
        row["train_operable_to_holdout_nonoperable"] = bool(
            train_view.get("operable") and not hold_view.get("operable")
        )
        row["holdout_failure_reasons"] = ";".join(reasons)
        rows.append(row)
    return rows


def summarize_rows(rows: Iterable[dict], holdout_days: int, lookback_days: int) -> dict:
    rows = list(rows)
    train_op = [r for r in rows if r.get("train_operable")]
    hold_op = [r for r in rows if r.get("holdout_operable")]
    lost = [r for r in rows if r.get("train_operable_to_holdout_nonoperable")]
    reason_counts: Counter[str] = Counter()
    for row in rows:
        for reason in str(row.get("holdout_failure_reasons", "")).split(";"):
            if reason:
                reason_counts[reason] += 1
    recent_tiers = Counter(str(r.get("holdout_recent_operability_tier", "MISSING")) for r in rows)
    recent_scores = [
        _safe_float(r.get("holdout_recent_operability_score"))
        for r in rows
        if r.get("holdout_recent_operability_score") not in (None, "")
    ]
    top_recent = sorted(
        rows,
        key=lambda r: _safe_float(r.get("holdout_recent_operability_score")),
        reverse=True,
    )[:25]
    return {
        "generated_at": _now_iso(),
        "holdout_days": int(holdout_days),
        "lookback_days": int(lookback_days),
        "n_rows": len(rows),
        "n_train_operable": len(train_op),
        "n_holdout_operable": len(hold_op),
        "n_train_operable_lost_in_holdout": len(lost),
        "holdout_universe_counts": dict(Counter(str(r.get("holdout_universe", "MISSING")) for r in rows)),
        "failure_reason_counts": dict(reason_counts.most_common()),
        "holdout_recent_tier_counts": dict(recent_tiers),
        "n_holdout_recent_ready": int(recent_tiers.get("RECENT_READY", 0)),
        "n_holdout_recent_watch": int(recent_tiers.get("RECENT_WATCH", 0)),
        "holdout_recent_score_median": float(np.median(recent_scores)) if recent_scores else 0.0,
        "holdout_recent_score_p75": float(np.percentile(recent_scores, 75)) if recent_scores else 0.0,
        "top_lost_tickers": [
            {
                "ticker": r.get("ticker"),
                "train_universe": r.get("train_universe"),
                "train_wr": r.get("train_win_rate"),
                "holdout_universe": r.get("holdout_universe"),
                "holdout_wr": r.get("holdout_win_rate"),
                "reasons": r.get("holdout_failure_reasons"),
            }
            for r in lost[:25]
        ],
        "top_recent_holdout_candidates": [
            {
                "ticker": r.get("ticker"),
                "tier": r.get("holdout_recent_operability_tier"),
                "score": r.get("holdout_recent_operability_score"),
                "n_trades": r.get("holdout_n_trades"),
                "holdout_universe": r.get("holdout_universe"),
                "holdout_wr": r.get("holdout_win_rate"),
                "holdout_wr_after_3": r.get("holdout_wr_after_3_bars"),
                "holdout_alpha": r.get("holdout_alpha_ann"),
                "holdout_hold": r.get("holdout_median_hold_bars"),
                "expected_hold": r.get("holdout_expected_hold_score"),
                "recent_blockers": r.get("holdout_recent_operability_blockers"),
                "hard_reasons": r.get("holdout_failure_reasons"),
            }
            for r in top_recent
        ],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, summary: dict) -> None:
    reasons = summary.get("failure_reason_counts", {})
    lines = [
        "# Holdout Operability Diagnostic",
        "",
        f"- Generated: `{summary.get('generated_at')}`",
        f"- Holdout days: `{summary.get('holdout_days')}`",
        f"- Lookback days for indicators: `{summary.get('lookback_days')}`",
        f"- Train operable: `{summary.get('n_train_operable')}`",
        f"- Holdout operable: `{summary.get('n_holdout_operable')}`",
        f"- Lost from train to holdout: `{summary.get('n_train_operable_lost_in_holdout')}`",
        f"- Recent ready/watch: `{summary.get('n_holdout_recent_ready')}` / `{summary.get('n_holdout_recent_watch')}`",
        f"- Recent score median/p75: `{summary.get('holdout_recent_score_median'):.2f}` / `{summary.get('holdout_recent_score_p75'):.2f}`",
        "",
        "## Holdout Failure Reasons",
        "",
    ]
    for reason, count in reasons.items():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Top Recent Holdout Candidates", ""])
    for item in summary.get("top_recent_holdout_candidates", [])[:15]:
        lines.append(
            f"- `{item.get('ticker')}` {item.get('tier')} score={item.get('score')} "
            f"trades={item.get('n_trades')} WR={item.get('holdout_wr')} "
            f"WR3={item.get('holdout_wr_after_3')} hold={item.get('holdout_hold')} "
            f"blockers=`{item.get('recent_blockers')}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(prefix: Path, holdout_days: int, lookback_days: int) -> dict:
    import ga_run as gr
    import run_local_ga_staged as staged

    _store, _windows, seed_genome, _seed_fit, _base_genome, _base_fit, _info = staged.open_existing_store()
    cache = staged.build_ticker_arrays_cache(_store)
    train_cache, holdout_cache = _slice_by_index(cache, holdout_days, lookback_days)
    rows = build_diagnostic_rows(list(seed_genome), train_cache, holdout_cache, gr)
    summary = summarize_rows(rows, holdout_days, lookback_days)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    _write_csv(csv_path, rows)
    _atomic_json(json_path, {"summary": summary, "rows": rows})
    _write_md(md_path, summary)
    print(f"[diagnostic] train_operable={summary['n_train_operable']} holdout_operable={summary['n_holdout_operable']}")
    print(f"[diagnostic] wrote {csv_path}")
    print(f"[diagnostic] wrote {json_path}")
    print(f"[diagnostic] wrote {md_path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-days", type=int, default=90)
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_PREFIX)
    args = parser.parse_args()
    run(args.output_prefix, args.holdout_days, args.lookback_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
