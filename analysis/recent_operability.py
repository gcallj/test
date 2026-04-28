#!/usr/bin/env python3
"""Recent OOS operability scoring for swing-candidate triage.

This module is intentionally diagnostic. A good recent score can move a
candidate into the next investigation queue, but it must not promote a model by
itself.
"""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Iterable

import numpy as np


READY_TIER = "RECENT_READY"
WATCH_TIER = "RECENT_WATCH"
WEAK_TIER = "RECENT_WEAK"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _scale(value: Any, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clip((_safe_float(value) - float(lo)) / (float(hi) - float(lo)))


def _hold_window_score(hold_bars: Any) -> float:
    hold = _safe_float(hold_bars)
    if 3.0 <= hold <= 10.0:
        return 1.0
    if hold < 3.0:
        return _scale(hold, 1.0, 3.0)
    return 1.0 - _scale(hold, 10.0, 20.0)


def recent_operability_score(view: dict[str, Any]) -> float:
    """Return a 0-100 recent OOS triage score.

    The score rewards edges that still look tradable after three bars, have a
    plausible swing holding profile, and do not depend heavily on early takes.
    It uses softer sample-size expectations than the hard OOS promotion gate so
    short recent windows can still surface useful diagnostics.
    """
    universe = str(view.get("universe", "") or "").upper()
    universe_score = 1.0 if universe in {"PREMIUM", "HIGH_QUAL"} else 0.45
    early_dep = _safe_float(view.get("early_take_dependency"), 1.0)

    components = (
        8.0 * _scale(view.get("n_trades"), 1.0, 8.0),
        8.0 * universe_score,
        14.0 * _scale(view.get("win_rate"), 0.52, 0.75),
        8.0 * _scale(view.get("win_rate_target"), 0.48, 0.70),
        15.0 * _scale(view.get("wr_after_3_bars"), 0.45, 0.70),
        15.0 * _scale(view.get("expected_hold_score"), 50.0, 85.0),
        10.0 * _scale(view.get("swing_survival_3d"), 0.35, 0.75),
        10.0 * _hold_window_score(view.get("median_hold_bars")),
        8.0 * _scale(view.get("alpha_ann"), -0.30, 0.05),
        4.0 * _scale(view.get("expectancy"), -0.01, 0.01),
    )
    penalty = 12.0 * _scale(early_dep, 0.35, 0.75)
    return round(_clip(sum(components) - penalty, 0.0, 100.0), 2)


def recent_operability_blockers(view: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if _safe_int(view.get("n_trades")) < 3:
        blockers.append("recent_low_trades_lt_3")
    if _safe_float(view.get("median_hold_bars")) < 2.5:
        blockers.append("recent_hold_lt_2_5")
    if _safe_float(view.get("median_hold_bars")) > 12.0:
        blockers.append("recent_hold_gt_12")
    if _safe_float(view.get("expected_hold_score")) < 65.0:
        blockers.append("recent_expected_hold_lt_65")
    if _safe_float(view.get("wr_after_3_bars")) < 0.58:
        blockers.append("recent_wr_after_3_lt_58")
    if _safe_float(view.get("win_rate")) < 0.58:
        blockers.append("recent_wr_lt_58")
    if _safe_float(view.get("alpha_ann")) < -0.25:
        blockers.append("recent_alpha_lt_minus25")
    if _safe_float(view.get("expectancy")) < 0.0:
        blockers.append("recent_expectancy_negative")
    if _safe_float(view.get("swing_survival_3d")) < 0.45:
        blockers.append("recent_survival3_lt_45")
    if _safe_float(view.get("early_take_dependency"), 1.0) > 0.55:
        blockers.append("recent_early_take_dependency_gt_55")
    return blockers


def recent_operability_tier(view: dict[str, Any], score: float | None = None) -> str:
    score = recent_operability_score(view) if score is None else _safe_float(score)
    blockers = set(recent_operability_blockers(view))
    hard_blockers = {
        "recent_low_trades_lt_3",
        "recent_hold_lt_2_5",
        "recent_hold_gt_12",
        "recent_expected_hold_lt_65",
        "recent_wr_after_3_lt_58",
        "recent_alpha_lt_minus25",
    }
    if score >= 75.0 and not (blockers & hard_blockers):
        return READY_TIER
    if score >= 60.0 and _safe_int(view.get("n_trades")) >= 1:
        return WATCH_TIER
    return WEAK_TIER


def attach_recent_operability(view: dict[str, Any]) -> dict[str, Any]:
    out = dict(view)
    score = recent_operability_score(out)
    out["recent_operability_score"] = score
    out["recent_operability_tier"] = recent_operability_tier(out, score)
    out["recent_operability_blockers"] = ";".join(recent_operability_blockers(out))
    return out


def recent_summary_from_views(views: Iterable[dict[str, Any]]) -> dict[str, Any]:
    enriched = [attach_recent_operability(v) for v in views]
    scores = [_safe_float(v.get("recent_operability_score")) for v in enriched]
    tiers = Counter(str(v.get("recent_operability_tier", WEAK_TIER)) for v in enriched)
    ready = [v for v in enriched if v.get("recent_operability_tier") == READY_TIER]
    watch = [v for v in enriched if v.get("recent_operability_tier") == WATCH_TIER]

    def pct(q: float) -> float:
        if not scores:
            return 0.0
        return float(np.percentile(np.asarray(scores, dtype=float), q))

    top = sorted(enriched, key=lambda v: _safe_float(v.get("recent_operability_score")), reverse=True)[:25]
    return {
        "n_evaluated": len(enriched),
        "n_ready": len(ready),
        "n_watch": len(watch),
        "tier_counts": dict(tiers),
        "score_median": round(float(median(scores)), 2) if scores else 0.0,
        "score_p75": round(pct(75), 2),
        "score_max": round(max(scores), 2) if scores else 0.0,
        "top_recent_candidates": [
            {
                "ticker": v.get("ticker"),
                "tier": v.get("recent_operability_tier"),
                "score": v.get("recent_operability_score"),
                "n_trades": v.get("n_trades"),
                "win_rate": v.get("win_rate"),
                "wr_after_3_bars": v.get("wr_after_3_bars"),
                "alpha_ann": v.get("alpha_ann"),
                "median_hold_bars": v.get("median_hold_bars"),
                "expected_hold_score": v.get("expected_hold_score"),
                "blockers": v.get("recent_operability_blockers"),
            }
            for v in top
        ],
    }
