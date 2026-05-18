"""CLI for the paper-trade execution log (D4 phase 1).

Examples:

    # Record a signal (usually called from predict_daily.py)
    python analysis/exec_log_record.py signal --date 2026-05-15 --ticker PETR4.SA \\
        --entry 38.50 --stop 37.95 --target 39.20

    # Record a fill
    python analysis/exec_log_record.py fill --date 2026-05-15 --ticker PETR4.SA \\
        --price 38.55

    # Mark missed / cancelled
    python analysis/exec_log_record.py missed --date 2026-05-15 --ticker PETR4.SA
    python analysis/exec_log_record.py cancelled --date 2026-05-15 --ticker PETR4.SA

    # Dump aggregate state
    python analysis/exec_log_record.py summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.exec_log import (  # noqa: E402
    STATUS_CANCELLED,
    STATUS_MISSED,
    aggregate_state,
    record_fill,
    record_signal,
    record_status_change,
    summarize,
)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--date", required=True, help="YYYY-MM-DD signal date.")
    p.add_argument("--ticker", required=True, help="e.g. PETR4.SA")
    p.add_argument("--log", default=None, help="Override log path (default analysis/exec_log.jsonl).")
    p.add_argument("--source", default=None, help="Free-text source tag.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record / inspect paper-trade execution log.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sig = sub.add_parser("signal", help="Record a PENDING signal.")
    _add_common(p_sig)
    p_sig.add_argument("--entry", type=float, default=None, help="Desired entry price.")
    p_sig.add_argument("--stop", type=float, default=None, help="Desired stop price.")
    p_sig.add_argument("--target", type=float, default=None, help="Desired target price.")

    p_fill = sub.add_parser("fill", help="Record a FILLED entry, computing slippage vs the pending desired_entry.")
    _add_common(p_fill)
    p_fill.add_argument("--price", type=float, required=True, help="Actual fill price.")
    p_fill.add_argument("--time", default=None, help="ISO fill time (defaults to now-UTC).")

    p_missed = sub.add_parser("missed", help="Record a MISSED status (price not reached).")
    _add_common(p_missed)

    p_canc = sub.add_parser("cancelled", help="Record a CANCELLED status (operator cancelled).")
    _add_common(p_canc)

    p_sum = sub.add_parser("summary", help="Dump aggregate state + slippage stats as JSON.")
    p_sum.add_argument("--log", default=None, help="Override log path.")

    p_state = sub.add_parser("state", help="Print latest (date, ticker) -> entry map as JSON.")
    p_state.add_argument("--log", default=None, help="Override log path.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = args.log

    if args.cmd == "signal":
        entry = record_signal(
            args.date, args.ticker,
            desired_entry=args.entry, desired_stop=args.stop, desired_target=args.target,
            source=args.source or "manual_cli", log_path=log_path,
        )
        if entry is None:
            print(f"[exec_log] PENDING already exists for {args.date}/{args.ticker} — skipped")
        else:
            print(json.dumps(entry, indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "fill":
        entry = record_fill(
            args.date, args.ticker, args.price, fill_time=args.time,
            source=args.source or "manual_cli", log_path=log_path,
        )
        print(json.dumps(entry, indent=2, ensure_ascii=False))
        return 0

    if args.cmd in ("missed", "cancelled"):
        entry = record_status_change(
            args.date, args.ticker,
            STATUS_MISSED if args.cmd == "missed" else STATUS_CANCELLED,
            source=args.source or "manual_cli", log_path=log_path,
        )
        print(json.dumps(entry, indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "summary":
        print(json.dumps(summarize(log_path), indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "state":
        state = aggregate_state(log_path)
        # Convert tuple keys to strings for JSON.
        out = {f"{k[0]}|{k[1]}": v for k, v in state.items()}
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
