"""Adaptive OOS-gate floors (plan item T4).

Computes regime-aware floors for `run_local_fullmetric_sweep_oos.oos_gate`.
The static floors (WR ≥ 60%, WR_after_3 ≥ 60%, expected_hold ≥ 70) are robust
when the benchmark is healthy but become brittle when B&H itself is weak —
during a chop window a 58% WR strategy may still be the best available.

Approach: read a baseline JSON (`analysis/_baselines.json` by default)
containing a rolling B&H-derived median, and lift the floor by a buffer when
the baseline is materially below it. The result is `max(static_floor,
baseline_value + buffer)` per metric. Output is a dict of overrides ready to
splat into the `oos_gate` kwargs.

This module is **opt-in**: callers must explicitly invoke `adaptive_floors()`
and pass the result to `oos_gate`. Default behavior of `oos_gate` is unchanged.

Baseline JSON schema (all keys optional, defaults to static floors when
absent):

    {
        "asof": "2026-04-15",
        "window": "6m_rolling",
        "bh_wr_median": 0.62,
        "bh_wr_after_3_median": 0.61,
        "bh_expected_hold_score_median": 72.0
    }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_BASELINE_PATH = _REPO_ROOT / "analysis" / "_baselines.json"

# Defaults matching the current static floors in run_local_fullmetric_sweep_oos.oos_gate.
STATIC_FLOORS = {
    "min_wr_operable": 0.60,
    "min_wr_after_3": 0.60,
    "min_expected_hold_score": 70.0,
}

# Per-metric buffer added on top of the baseline. Lifts the floor by a margin
# so we don't promote a candidate that's only "barely better than B&H".
DEFAULT_BUFFERS = {
    "min_wr_operable": 0.05,            # +5pp WR above benchmark
    "min_wr_after_3": 0.05,             # +5pp WR_after_3 above benchmark
    "min_expected_hold_score": 5.0,     # +5 points above benchmark
}

# Map baseline JSON keys -> oos_gate floor names.
_BASELINE_KEY_MAP = {
    "bh_wr_median": "min_wr_operable",
    "bh_wr_after_3_median": "min_wr_after_3",
    "bh_expected_hold_score_median": "min_expected_hold_score",
}


def load_baseline(path: Optional[str | Path] = None) -> dict:
    """Read the baseline JSON. Returns empty dict if absent or malformed."""
    p = Path(path) if path is not None else DEFAULT_BASELINE_PATH
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def adaptive_floors(
    baseline: Optional[dict] = None,
    *,
    buffers: Optional[dict] = None,
    static_floors: Optional[dict] = None,
) -> dict:
    """Return a dict of floor overrides ready to pass into `oos_gate` as kwargs.

    Each floor is `max(static_floor, baseline_value + buffer)`. Metrics whose
    baseline is absent inherit the static floor (so the result is always safe
    to splat into `oos_gate(**floors)`).
    """
    if baseline is None:
        baseline = load_baseline()
    base_buf = dict(DEFAULT_BUFFERS)
    if buffers:
        base_buf.update(buffers)
    base_static = dict(STATIC_FLOORS)
    if static_floors:
        base_static.update(static_floors)

    out: dict = {}
    for baseline_key, floor_key in _BASELINE_KEY_MAP.items():
        static = float(base_static[floor_key])
        if baseline_key in baseline and baseline[baseline_key] is not None:
            try:
                bh_val = float(baseline[baseline_key])
            except (TypeError, ValueError):
                out[floor_key] = static
                continue
            buf = float(base_buf.get(floor_key, 0.0))
            out[floor_key] = max(static, bh_val + buf)
        else:
            out[floor_key] = static
    return out


def write_baseline(
    data: dict, *, path: Optional[str | Path] = None,
) -> Path:
    """Persist the baseline JSON. Atomic write."""
    p = Path(path) if path is not None else DEFAULT_BASELINE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(p)
    return p


def describe_floors(floors: dict) -> str:
    """Render the floor dict as a single-line summary for logs."""
    parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(floors.items())]
    return "adaptive_floors: " + ", ".join(parts)
