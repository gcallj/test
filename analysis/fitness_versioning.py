"""Fitness-formula versioning (plan ANTI-1).

The `best_ever_guard` in `apply_fullmetric_sweep_best.py` compares each
candidate's fitness against historical bests. Mutating the fitness formula
without versioning invalidates that history — a candidate scored by `v6.0`
can't be compared apples-to-apples against a `v5.2` best-ever record.

This module is the canonical source for:

- `FITNESS_VERSION`: the current formula version string. Bump when you touch
  the fitness formula in `ga_run.py` (lines 2326-2353, the `# Main fitness`
  block).
- `tag_payload`: stamp a checkpoint payload with the current version.
- `get_payload_version`: read the version from a checkpoint (None for legacy
  pre-versioning files).
- `is_compatible`: True when two versions are equal OR either is None (the
  legacy fallback is permissive so we don't break old workflows).
- `versioned_regression_check`: wraps `analysis.best_ever_guard.check_regression_vs_best`
  so a version mismatch returns `(True, "version_mismatch_skip:<a>->b")`
  instead of treating the candidate as a regression.

The current `# v5.2 alpha-hard` comment in ga_run.py is the v5.2 baseline;
T2 (consistency penalty) and T3 (break-even penalty) from the plan are
expected to bump this to v6.0.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Current fitness formula version. Bump on any change to the fitness formula
# in ga_run.py:2330-2370 (Main fitness block).
#
# v6.0 (current):
#   - T3 breakeven_penalty hook (GA_BREAKEVEN_W env, default 0 = inert)
#   - T2 consistency penalty hook in evaluate_global_walkforward
#     (GA_CONSISTENCY_W env, default 0 = inert)
# v5.2:
#   - alpha-hard formulation, GA_ALPHA_FOCUS weighting.
FITNESS_VERSION: str = "v6.0"

PAYLOAD_KEY: str = "fitness_version"


def get_fitness_version() -> str:
    return FITNESS_VERSION


def tag_payload(payload: dict, version: Optional[str] = None) -> dict:
    """Return a copy of `payload` with the fitness version stamped on it."""
    out = dict(payload)
    out[PAYLOAD_KEY] = str(version or FITNESS_VERSION)
    return out


def get_payload_version(payload: dict) -> Optional[str]:
    """Return the version stamp on a payload. None for legacy/un-stamped data."""
    if payload is None:
        return None
    raw = payload.get(PAYLOAD_KEY)
    return str(raw) if raw else None


def is_compatible(a: Optional[str], b: Optional[str]) -> bool:
    """True if two versions can be compared apples-to-apples.

    The permissive rules:
      - Both None  -> compatible (legacy vs legacy; no info to disagree).
      - Either None -> compatible (legacy vs versioned; user opted in to bump,
        we keep the old comparison working until the new history accrues).
      - Both set   -> compatible iff they are exactly equal.
    """
    if a is None or b is None:
        return True
    return str(a) == str(b)


VERSION_MISMATCH_REASON = "version_mismatch_skip"


def versioned_regression_check(
    cand_metrics: dict,
    best: Optional[dict],
    base_checker: Callable[[dict, Optional[dict]], Tuple[bool, str]],
    *,
    cand_version: Optional[str] = None,
    best_version: Optional[str] = None,
) -> Tuple[bool, str]:
    """Wrap `base_checker` so version mismatches do not block promotions.

    `base_checker` must have the same signature as
    `analysis.best_ever_guard.check_regression_vs_best`. When the candidate and
    best-ever versions are NOT compatible, this returns
    `(True, "version_mismatch_skip:<cand>->-<best>")` and bypasses
    base_checker entirely. Otherwise it delegates.
    """
    if not is_compatible(cand_version, best_version):
        return True, f"{VERSION_MISMATCH_REASON}:{cand_version}->{best_version}"
    return base_checker(cand_metrics, best)
