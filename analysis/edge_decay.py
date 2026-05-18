"""Edge time-decay diagnostic (plan item D5).

Surfaces model staleness by comparing recent trade-edge against the prior
window. The ratio `mean_ret(last `recent` days) / mean_ret(prior `prior` days)`
under `DEFAULT_DECAY_THRESHOLD` raises a DECAY flag. The diagnostic is meant as
a **soft** signal — it ranks candidates for the next sweep and is wired into
`recent_operability.recent_operability_blockers` as a non-hard blocker.

Usage:

    from analysis.edge_decay import edge_decay_ratio, attach_edge_decay_to_view
    info = edge_decay_ratio(trade_df)
    view = attach_edge_decay_to_view(view, info)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DECAY_LABEL = "DECAY"
OK_LABEL = "OK"
MISSING_LABEL = "MISSING"
DEFAULT_RECENT_DAYS = 30
DEFAULT_PRIOR_DAYS = 90
DEFAULT_DECAY_THRESHOLD = 0.5


def _coerce_trade_log(trade_df: pd.DataFrame, entry_date_col: str, ret_col: str) -> pd.DataFrame:
    if entry_date_col not in trade_df.columns:
        raise ValueError(f"trade_df missing entry_date column: {entry_date_col!r}")
    if ret_col not in trade_df.columns:
        raise ValueError(f"trade_df missing return column: {ret_col!r}")
    out = trade_df.copy()
    out[entry_date_col] = pd.to_datetime(out[entry_date_col], errors="coerce")
    out[ret_col] = pd.to_numeric(out[ret_col], errors="coerce")
    return out.dropna(subset=[entry_date_col, ret_col])


def _daily_mean(trade_df: pd.DataFrame, entry_date_col: str, ret_col: str) -> pd.Series:
    df = _coerce_trade_log(trade_df, entry_date_col, ret_col)
    if df.empty:
        return pd.Series(dtype=float)
    return df.groupby(entry_date_col)[ret_col].mean().sort_index()


def edge_decay_ratio(
    trade_df: pd.DataFrame,
    *,
    entry_date_col: str = "entry_date",
    ret_col: str = "ret",
    recent: int = DEFAULT_RECENT_DAYS,
    prior: int = DEFAULT_PRIOR_DAYS,
    asof: Optional[Any] = None,
    threshold: float = DEFAULT_DECAY_THRESHOLD,
) -> dict[str, Any]:
    """Return a dict describing edge decay between the recent and prior windows.

    Keys:
      ratio          recent_mean / prior_mean. NaN when either window is empty
                     or unrepresentative; clipped to 0.0 when prior_mean<=0 and
                     recent_mean<=prior_mean (signalling collapse).
      recent_mean    mean daily return over the last `recent` calendar days.
      prior_mean     mean daily return over the `prior` days preceding that.
      n_recent       number of distinct trading days observed in recent window.
      n_prior        number of distinct trading days observed in prior window.
      asof           ISO-formatted reference date (max entry date if not given).
      label          OK / DECAY / MISSING.
      threshold      the threshold used for labelling.
    """
    if recent <= 0 or prior <= 0:
        raise ValueError("recent and prior must be positive integers")

    daily = _daily_mean(trade_df, entry_date_col, ret_col)
    if daily.empty:
        return {
            "ratio": float("nan"),
            "recent_mean": float("nan"),
            "prior_mean": float("nan"),
            "n_recent": 0,
            "n_prior": 0,
            "asof": None,
            "label": MISSING_LABEL,
            "threshold": float(threshold),
        }

    asof_ts = pd.to_datetime(asof) if asof is not None else daily.index.max()
    recent_start = asof_ts - pd.Timedelta(days=recent)
    prior_start = recent_start - pd.Timedelta(days=prior)
    recent_slice = daily[(daily.index > recent_start) & (daily.index <= asof_ts)]
    prior_slice = daily[(daily.index > prior_start) & (daily.index <= recent_start)]

    r_mean = float(recent_slice.mean()) if len(recent_slice) else float("nan")
    p_mean = float(prior_slice.mean()) if len(prior_slice) else float("nan")

    if not (np.isfinite(r_mean) and np.isfinite(p_mean)):
        ratio = float("nan")
        label = MISSING_LABEL
    elif p_mean <= 0.0:
        # Prior window already weak: ratio is meaningless. Mark DECAY only
        # if the recent window is also <= prior (i.e. truly collapsing).
        ratio = 0.0 if r_mean <= p_mean else float("nan")
        label = DECAY_LABEL if ratio == 0.0 else MISSING_LABEL
    else:
        ratio = r_mean / p_mean
        label = DECAY_LABEL if ratio < threshold else OK_LABEL

    return {
        "ratio": ratio,
        "recent_mean": r_mean,
        "prior_mean": p_mean,
        "n_recent": int(len(recent_slice)),
        "n_prior": int(len(prior_slice)),
        "asof": asof_ts.isoformat() if hasattr(asof_ts, "isoformat") else str(asof_ts),
        "label": label,
        "threshold": float(threshold),
    }


def attach_edge_decay_to_view(view: dict, info: dict) -> dict:
    """Return a copy of `view` with edge-decay fields added.

    Lets `recent_operability.recent_operability_blockers` pick the soft blocker
    up via `view.get("edge_decay_ratio")` / `view.get("edge_decay_label")`.
    """
    out = dict(view)
    out["edge_decay_ratio"] = info.get("ratio")
    out["edge_decay_label"] = info.get("label")
    out["edge_decay_recent_mean"] = info.get("recent_mean")
    out["edge_decay_prior_mean"] = info.get("prior_mean")
    return out
