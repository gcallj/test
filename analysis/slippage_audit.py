"""Realized-vs-modeled slippage audit (plan item D4 phase 2).

Consumes the JSONL log produced by `analysis/exec_log.py` and compares the
observed `slippage_bps_actual` against the system's modeled assumption
(`ga_run.SLIPPAGE_BPS`, default 10.0 = 0.10% per side). Flags material drift so
the cost model can be revisited (which, per ANTI-1 in the plan, requires
bumping `FITNESS_VERSION`).

Output schema:

    {
        "modeled_bps": 10.0,
        "n_observations": 42,
        "slippage_p10_bps": ...,
        "slippage_p50_bps": ...,
        "slippage_p90_bps": ...,
        "slippage_mean_bps": ...,
        "slippage_std_bps": ...,
        "bias_vs_model_bps": median - modeled,
        "bias_label": "OK" | "MATERIAL" | "MISSING",
        "recommendation": "..."
    }

Usage:

    from analysis.slippage_audit import audit_slippage
    report = audit_slippage()         # reads default analysis/exec_log.jsonl
    report = audit_slippage(min_observations=10)  # stricter sample-size floor

CLI:

    python analysis/slippage_audit.py
    python analysis/slippage_audit.py --log /tmp/test.jsonl --min 20 --modeled 12.0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.exec_log import STATUS_FILLED, iter_log  # noqa: E402


# Default modeled slippage in bps (mirrors ga_run.SLIPPAGE_BPS=10.0).
DEFAULT_MODELED_BPS = 10.0

# |bias| above this threshold triggers the MATERIAL flag (default 5bps).
DEFAULT_MATERIAL_BIAS_BPS = 5.0

# Below this many observations the audit returns MISSING (no statistically
# meaningful conclusion). Plan suggested 2-4 weeks of paper-trading.
DEFAULT_MIN_OBSERVATIONS = 10


def _percentile(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def collect_fills(
    log_path: Optional[str | Path] = None,
) -> list[dict]:
    """Return all FILLED entries with a finite slippage_bps_actual."""
    rows: list[dict] = []
    for row in iter_log(log_path):
        if row.get("status") != STATUS_FILLED:
            continue
        sl = row.get("slippage_bps_actual")
        if sl is None:
            continue
        try:
            sl_f = float(sl)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(sl_f):
            continue
        rows.append(row)
    return rows


def audit_slippage(
    log_path: Optional[str | Path] = None,
    *,
    modeled_bps: float = DEFAULT_MODELED_BPS,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    material_bias_bps: float = DEFAULT_MATERIAL_BIAS_BPS,
) -> dict[str, Any]:
    """Compute slippage stats and a model-drift recommendation."""
    fills = collect_fills(log_path)
    n = len(fills)
    if n < int(min_observations):
        return {
            "modeled_bps": float(modeled_bps),
            "n_observations": n,
            "bias_label": "MISSING",
            "recommendation": (
                f"insufficient_data: {n}<{int(min_observations)}; "
                "accumulate more fills via analysis/exec_log_record.py"
            ),
        }
    arr = np.asarray([float(r["slippage_bps_actual"]) for r in fills], dtype=float)
    median = _percentile(arr, 50.0)
    bias = median - float(modeled_bps)
    label = "MATERIAL" if abs(bias) > float(material_bias_bps) else "OK"
    direction = "higher" if bias > 0 else "lower"
    if label == "MATERIAL":
        rec = (
            f"realized slippage median is {abs(bias):.1f} bps {direction} than the "
            f"modeled {modeled_bps:.1f} bps -> update SLIPPAGE_BPS to ~{median:.1f} "
            "and bump FITNESS_VERSION (ANTI-1)"
        )
    else:
        rec = (
            f"realized slippage median {median:.1f} bps is within "
            f"+/-{material_bias_bps:.1f} bps of model — no change needed"
        )
    return {
        "modeled_bps": float(modeled_bps),
        "n_observations": int(n),
        "slippage_p10_bps": _percentile(arr, 10.0),
        "slippage_p50_bps": median,
        "slippage_p90_bps": _percentile(arr, 90.0),
        "slippage_mean_bps": float(arr.mean()),
        "slippage_std_bps": float(arr.std()),
        "bias_vs_model_bps": float(bias),
        "bias_label": label,
        "recommendation": rec,
    }


def _resolve_modeled_default() -> float:
    """Try to read the live SLIPPAGE_BPS from ga_run, fall back to constant."""
    try:
        import ga_run  # type: ignore
        return float(getattr(ga_run, "SLIPPAGE_BPS", DEFAULT_MODELED_BPS))
    except Exception:
        return DEFAULT_MODELED_BPS


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Realized slippage audit (D4 phase 2).")
    p.add_argument("--log", default=None, help="Path to exec_log.jsonl (default analysis/exec_log.jsonl).")
    p.add_argument("--modeled", type=float, default=None, help="Override modeled bps (default reads ga_run.SLIPPAGE_BPS).")
    p.add_argument("--min", dest="min_obs", type=int, default=DEFAULT_MIN_OBSERVATIONS, help="Minimum observations for a valid audit.")
    p.add_argument("--bias-threshold", type=float, default=DEFAULT_MATERIAL_BIAS_BPS, help="Absolute bps for MATERIAL flag.")
    p.add_argument("--json", dest="json_out", default=None, help="Write report to JSON file.")
    args = p.parse_args(argv)

    modeled = args.modeled if args.modeled is not None else _resolve_modeled_default()
    report = audit_slippage(
        log_path=args.log,
        modeled_bps=modeled,
        min_observations=args.min_obs,
        material_bias_bps=args.bias_threshold,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
