"""Paper-trade execution log (plan item D4 phase 1).

Append-only JSONL persisting desired-vs-actual fill data per emitted BUY
signal. Feeds the future `analysis/slippage_audit.py` (D4 phase 2) once enough
rows accumulate. The append-only design keeps the log auditable: a fill update
appends a new entry rather than mutating the original PENDING row.

Schema per line (one JSON object per line):

    {
        "date": "2026-05-15",          # signal/entry date
        "ticker": "PETR4.SA",
        "desired_entry": 38.50,
        "desired_stop": 37.95,
        "desired_target": 39.20,
        "fill_price": null | 38.55,
        "fill_time": null | iso_ts,
        "slippage_bps_actual": null | 13.0,
        "status": "PENDING" | "FILLED" | "MISSED" | "CANCELLED",
        "recorded_at": iso_ts,
        "source": "predict_daily" | "manual_cli" | ...
    }

The latest-status-by-(date, ticker) is materialized via `aggregate_state` —
JSONL stays append-only so older entries are preserved as audit trail.

Usage:

    from analysis.exec_log import record_signal, record_fill, aggregate_state

    record_signal(date="2026-05-15", ticker="PETR4.SA",
                  desired_entry=38.50, desired_stop=37.95, desired_target=39.20)
    record_fill(date="2026-05-15", ticker="PETR4.SA", fill_price=38.55)
    state = aggregate_state()

CLI: see `analysis/exec_log_record.py`.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


STATUS_PENDING = "PENDING"
STATUS_FILLED = "FILLED"
STATUS_MISSED = "MISSED"
STATUS_CANCELLED = "CANCELLED"
VALID_STATUSES = (STATUS_PENDING, STATUS_FILLED, STATUS_MISSED, STATUS_CANCELLED)

DEFAULT_LOG_PATH = _REPO_ROOT / "analysis" / "exec_log.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # filter NaN


def _normalize_date(value: Any) -> str:
    """Return YYYY-MM-DD; accepts string or datetime."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    s = str(value).strip()
    if len(s) >= 10:
        return s[:10]
    return s


def _path(log_path: Optional[str | Path]) -> Path:
    return Path(log_path).expanduser().resolve() if log_path is not None else DEFAULT_LOG_PATH


def iter_log(log_path: Optional[str | Path] = None) -> Iterator[dict]:
    """Yield every entry from the JSONL log. Skips unparseable lines."""
    path = _path(log_path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def _append(entry: dict, log_path: Optional[str | Path]) -> Path:
    path = _path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def _exists_with_status(date: str, ticker: str, status: str, log_path: Optional[str | Path]) -> bool:
    for row in iter_log(log_path):
        if row.get("date") == date and row.get("ticker") == ticker and row.get("status") == status:
            return True
    return False


def record_signal(
    date: Any,
    ticker: str,
    desired_entry: Optional[float] = None,
    desired_stop: Optional[float] = None,
    desired_target: Optional[float] = None,
    *,
    source: str = "predict_daily",
    log_path: Optional[str | Path] = None,
) -> Optional[dict]:
    """Append a PENDING entry for a newly emitted signal.

    Idempotent on the (date, ticker, PENDING) tuple — a duplicate call is a
    no-op and returns None.
    """
    d = _normalize_date(date)
    t = str(ticker).strip()
    if not t:
        raise ValueError("ticker is required")
    if _exists_with_status(d, t, STATUS_PENDING, log_path):
        return None
    entry = {
        "date": d,
        "ticker": t,
        "desired_entry": _safe_float(desired_entry),
        "desired_stop": _safe_float(desired_stop),
        "desired_target": _safe_float(desired_target),
        "fill_price": None,
        "fill_time": None,
        "slippage_bps_actual": None,
        "status": STATUS_PENDING,
        "recorded_at": _now_iso(),
        "source": str(source or "unknown"),
    }
    _append(entry, log_path)
    return entry


def _find_pending(date: str, ticker: str, log_path: Optional[str | Path]) -> Optional[dict]:
    for row in iter_log(log_path):
        if row.get("date") == date and row.get("ticker") == ticker and row.get("status") == STATUS_PENDING:
            return row
    return None


def _compute_slippage_bps(desired_entry: Optional[float], fill_price: Optional[float]) -> Optional[float]:
    if desired_entry is None or fill_price is None or desired_entry == 0:
        return None
    # Positive bps = filled WORSE than desired (paid more on a buy).
    return ((fill_price - desired_entry) / desired_entry) * 10_000.0


def record_fill(
    date: Any,
    ticker: str,
    fill_price: float,
    fill_time: Optional[str] = None,
    *,
    source: str = "manual_cli",
    log_path: Optional[str | Path] = None,
) -> dict:
    """Append a FILLED entry that updates the latest PENDING row's price.

    Computes `slippage_bps_actual` against the original PENDING row's
    `desired_entry`. If no PENDING row exists this still appends a FILLED
    entry (with NaN slippage) so the audit trail is not lost.
    """
    d = _normalize_date(date)
    t = str(ticker).strip()
    fp = _safe_float(fill_price)
    if fp is None:
        raise ValueError("fill_price must be a finite number")
    pending = _find_pending(d, t, log_path)
    desired_entry = pending.get("desired_entry") if pending else None
    desired_stop = pending.get("desired_stop") if pending else None
    desired_target = pending.get("desired_target") if pending else None
    entry = {
        "date": d,
        "ticker": t,
        "desired_entry": desired_entry,
        "desired_stop": desired_stop,
        "desired_target": desired_target,
        "fill_price": fp,
        "fill_time": fill_time or _now_iso(),
        "slippage_bps_actual": _compute_slippage_bps(desired_entry, fp),
        "status": STATUS_FILLED,
        "recorded_at": _now_iso(),
        "source": str(source or "unknown"),
    }
    _append(entry, log_path)
    return entry


def record_status_change(
    date: Any,
    ticker: str,
    status: str,
    *,
    source: str = "manual_cli",
    log_path: Optional[str | Path] = None,
) -> dict:
    """Append a status-change entry (MISSED / CANCELLED). Audit-only."""
    if status not in (STATUS_MISSED, STATUS_CANCELLED):
        raise ValueError(f"status must be MISSED or CANCELLED, got {status!r}")
    d = _normalize_date(date)
    t = str(ticker).strip()
    pending = _find_pending(d, t, log_path)
    entry = {
        "date": d,
        "ticker": t,
        "desired_entry": pending.get("desired_entry") if pending else None,
        "desired_stop": pending.get("desired_stop") if pending else None,
        "desired_target": pending.get("desired_target") if pending else None,
        "fill_price": None,
        "fill_time": None,
        "slippage_bps_actual": None,
        "status": status,
        "recorded_at": _now_iso(),
        "source": str(source or "unknown"),
    }
    _append(entry, log_path)
    return entry


def aggregate_state(log_path: Optional[str | Path] = None) -> dict[tuple[str, str], dict]:
    """Return the latest entry per (date, ticker), keyed by that tuple.

    Latest = highest `recorded_at`. PENDING rows that were superseded by FILLED
    are dropped from the state map (the FILLED entry replaces them).
    """
    state: dict[tuple[str, str], dict] = {}
    for row in iter_log(log_path):
        key = (str(row.get("date")), str(row.get("ticker")))
        if key not in state or row.get("recorded_at", "") >= state[key].get("recorded_at", ""):
            state[key] = row
    return state


def summarize(log_path: Optional[str | Path] = None) -> dict:
    """Quick high-level summary, useful for the CLI."""
    state = aggregate_state(log_path)
    counts = {s: 0 for s in VALID_STATUSES}
    slippages: list[float] = []
    for entry in state.values():
        status = entry.get("status")
        if status in counts:
            counts[status] += 1
        sl = entry.get("slippage_bps_actual")
        if sl is not None:
            try:
                slippages.append(float(sl))
            except (TypeError, ValueError):
                pass
    n = len(slippages)
    out: dict = {"counts_by_status": counts, "n_filled_with_slippage": n}
    if n:
        sorted_sl = sorted(slippages)
        out["slippage_p50_bps"] = float(sorted_sl[n // 2])
        out["slippage_mean_bps"] = float(sum(slippages) / n)
        out["slippage_p90_bps"] = float(sorted_sl[min(n - 1, int(0.9 * (n - 1)))])
    return out
