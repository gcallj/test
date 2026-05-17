"""Per-segment BUY signal cap (plan item T5).

Concentration cap that demotes surplus BUY signals beyond a per-segment limit,
ordered by rank (or any custom score column). Default off — must be enabled
via the `MAX_PER_SEGMENT` env var or explicit kwarg. Uses the same 11-segment
map from `ticker_metadata.SECTOR_MAP` that `analysis.sector_breakdown` rolls up.

Usage:

    from analysis.sector_cap import apply_sector_cap

    capped_df = apply_sector_cap(apply_df)             # honours env var
    capped_df = apply_sector_cap(apply_df, max_per_segment=3)

Rows demoted have their `signal` coerced to `hold` and gain
`signal_pre_sector_cap` (original signal) and `sector_cap_reason` columns.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ticker_metadata import get_segment_for_tickers  # noqa: E402


ENV_VAR = "MAX_PER_SEGMENT"
SENTINEL_DISABLED = "off"


def get_active_cap(override: Optional[int | str] = None) -> Optional[int]:
    """Resolve the active cap.

    Returns `None` when uncapped (env var absent, `0`, negative, or set to
    `off`). Returns the positive int cap otherwise. Unknown/non-numeric values
    fall back to `None` to avoid breaking the pipeline.
    """
    raw = override if override is not None else os.environ.get(ENV_VAR)
    if raw is None or raw == "":
        return None
    s = str(raw).strip().lower()
    if s in (SENTINEL_DISABLED, "none", "null"):
        return None
    try:
        value = int(s)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _assign_segments(
    df: pd.DataFrame,
    ticker_col: str,
    segments_override: Optional[dict] = None,
) -> pd.Series:
    if segments_override is not None:
        return df[ticker_col].astype(str).map(segments_override).fillna("unknown")
    tickers = df[ticker_col].astype(str).unique().tolist()
    seg_map = get_segment_for_tickers(tickers)
    return df[ticker_col].astype(str).map(seg_map).fillna("unknown")


def apply_sector_cap(
    df: pd.DataFrame,
    *,
    max_per_segment: Optional[int | str] = None,
    ticker_col: str = "ticker",
    signal_col: str = "signal",
    rank_col: str = "rank",
    segments_override: Optional[dict] = None,
    add_reason_col: bool = True,
) -> pd.DataFrame:
    """Coerce BUY -> HOLD for rows that exceed `max_per_segment` per segment.

    The kept BUYs are the top-`max_per_segment` by `rank_col` within each
    segment (descending). When `rank_col` is missing the input order is
    preserved (DataFrame.head behavior).

    Returns a copy. No-op when the cap resolves to `None` (default), when
    `signal_col` or `ticker_col` is missing, or when no row has `signal=buy`.
    """
    cap = get_active_cap(max_per_segment)
    if cap is None:
        return df.copy()
    if signal_col not in df.columns or ticker_col not in df.columns:
        return df.copy()
    is_buy = df[signal_col].astype(str).str.lower() == "buy"
    if not is_buy.any():
        return df.copy()

    out = df.copy()
    out["_segment"] = _assign_segments(out, ticker_col, segments_override)

    # Buys only: pick top-N by rank within each segment; everyone else demoted.
    buys = out[is_buy.values].copy()
    if rank_col in buys.columns:
        buys["_rank_num"] = pd.to_numeric(buys[rank_col], errors="coerce")
        buys = buys.sort_values(["_segment", "_rank_num"], ascending=[True, False], kind="stable")
    else:
        buys["_rank_num"] = 0.0
        buys = buys.sort_values("_segment", kind="stable")

    keep_idx = buys.groupby("_segment", sort=False).head(cap).index
    demote_idx = buys.index.difference(keep_idx)
    if len(demote_idx) > 0:
        if "signal_pre_sector_cap" not in out.columns:
            out["signal_pre_sector_cap"] = out[signal_col]
        out.loc[demote_idx, signal_col] = "hold"
        if add_reason_col:
            if "sector_cap_reason" not in out.columns:
                out["sector_cap_reason"] = ""
            out.loc[demote_idx, "sector_cap_reason"] = f"sector_cap_exceeded(max={cap})"

    out = out.drop(columns=["_segment"])
    return out


def cap_summary(
    df: pd.DataFrame,
    *,
    signal_col: str = "signal",
    pre_col: str = "signal_pre_sector_cap",
) -> dict:
    """Return `{n_pre_buy, n_post_buy, n_capped, pct_capped}` after `apply_sector_cap`."""
    n_post = int((df.get(signal_col, pd.Series(dtype=str)).astype(str).str.lower() == "buy").sum())
    if pre_col not in df.columns:
        return {
            "n_pre_buy": n_post,
            "n_post_buy": n_post,
            "n_capped": 0,
            "pct_capped": 0.0,
        }
    n_pre = int((df[pre_col].astype(str).str.lower() == "buy").sum())
    n_capped = max(0, n_pre - n_post)
    pct = (100.0 * n_capped / n_pre) if n_pre > 0 else 0.0
    return {
        "n_pre_buy": n_pre,
        "n_post_buy": n_post,
        "n_capped": n_capped,
        "pct_capped": pct,
    }
