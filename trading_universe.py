"""Trading universe filter (plan item T1).

Configurable post-emission filter that gates BUY signals by the dynamic
universe tier (PREMIUM / HIGH_QUAL / REGULAR) already computed per ticker by
`ga_run.classify_operational_universe_tier`. The default is `all` — no
filtering, so the existing pipeline behavior is preserved unless the user
explicitly opts in via the `TRADING_UNIVERSE` env var (or the API param).

Usage as a library:

    from trading_universe import filter_apply_df
    df = filter_apply_df(apply_df)  # honours TRADING_UNIVERSE env var

Usage as a CLI on summary_latest.xlsx or apply_last_*.csv:

    python trading_universe.py --in summary_latest.xlsx --out filtered.xlsx \
        --sheet Apply --universe high_qual

Rows whose universe tier fails the filter have their `signal` coerced to
`hold`. The original signal is preserved in a `signal_pre_universe` column and
the reason is logged in `universe_filter_reason`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

UNIVERSE_ALL = "all"
UNIVERSE_HIGH_QUAL = "high_qual"
UNIVERSE_PREMIUM = "premium"
DEFAULT_UNIVERSE = UNIVERSE_ALL
VALID_UNIVERSES = (UNIVERSE_ALL, UNIVERSE_HIGH_QUAL, UNIVERSE_PREMIUM)

ENV_VAR = "TRADING_UNIVERSE"

# Mapping of universe-setting -> tiers (uppercase) that pass the filter.
ALLOWED_TIERS: dict[str, frozenset[str]] = {
    UNIVERSE_ALL: frozenset({"PREMIUM", "HIGH_QUAL", "REGULAR"}),
    UNIVERSE_HIGH_QUAL: frozenset({"PREMIUM", "HIGH_QUAL"}),
    UNIVERSE_PREMIUM: frozenset({"PREMIUM"}),
}


def get_active_universe(override: Optional[str] = None) -> str:
    """Return the active universe name.

    Order of precedence: explicit `override`, then `TRADING_UNIVERSE` env var,
    then `DEFAULT_UNIVERSE`. Unknown values fall back silently to default to
    avoid breaking the pipeline on typos.
    """
    raw = override if override is not None else os.environ.get(ENV_VAR, DEFAULT_UNIVERSE)
    if raw is None:
        return DEFAULT_UNIVERSE
    value = str(raw).strip().lower()
    if value not in VALID_UNIVERSES:
        return DEFAULT_UNIVERSE
    return value


def passes_filter(tier: str, universe: Optional[str] = None) -> bool:
    """Return True if the given tier passes under the active universe setting."""
    u = get_active_universe(universe)
    allowed = ALLOWED_TIERS.get(u, ALLOWED_TIERS[DEFAULT_UNIVERSE])
    return str(tier or "").strip().upper() in allowed


def filter_apply_df(
    df: pd.DataFrame,
    *,
    universe: Optional[str] = None,
    universe_col: str = "universe",
    signal_col: str = "signal",
    add_reason_col: bool = True,
) -> pd.DataFrame:
    """Demote BUY signals whose universe tier fails the filter.

    No-op when the active universe is `all` (default). Returns a copy.
    If `universe_col` or `signal_col` is missing the frame is returned
    unmodified (the function never crashes on partial inputs).
    """
    u = get_active_universe(universe)
    if u == UNIVERSE_ALL:
        return df.copy()
    if universe_col not in df.columns or signal_col not in df.columns:
        return df.copy()

    out = df.copy()
    tier_upper = out[universe_col].astype(str).str.upper()
    allowed = ALLOWED_TIERS[u]
    is_buy = out[signal_col].astype(str).str.lower() == "buy"
    fail_mask = is_buy & ~tier_upper.isin(allowed)
    if fail_mask.any():
        if "signal_pre_universe" not in out.columns:
            out["signal_pre_universe"] = out[signal_col]
        out.loc[fail_mask, signal_col] = "hold"
        if add_reason_col:
            if "universe_filter_reason" not in out.columns:
                out["universe_filter_reason"] = ""
            out.loc[fail_mask, "universe_filter_reason"] = f"filtered_by_universe={u}"
    return out


def _read(src: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(src, sheet_name=sheet) if src.lower().endswith(".xlsx") else pd.read_csv(src)


def _write(df: pd.DataFrame, dst: str, sheet: str) -> None:
    if dst.lower().endswith(".xlsx"):
        df.to_excel(dst, sheet_name=sheet, index=False)
    else:
        df.to_csv(dst, index=False)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Apply the trading universe filter to a signals frame.")
    p.add_argument("--in", dest="src", required=True, help="Source xlsx (with --sheet) or csv.")
    p.add_argument("--out", dest="dst", required=True, help="Destination xlsx or csv.")
    p.add_argument("--sheet", default="Apply", help="Sheet name when reading/writing xlsx (default: Apply).")
    p.add_argument("--universe", default=None, help=f"Override {ENV_VAR}. One of: {VALID_UNIVERSES}.")
    args = p.parse_args(argv)

    src = args.src
    if not Path(src).exists():
        print(f"[trading_universe] source not found: {src}", file=sys.stderr)
        return 2

    df = _read(src, args.sheet)
    filtered = filter_apply_df(df, universe=args.universe)
    _write(filtered, args.dst, args.sheet)

    active = get_active_universe(args.universe)
    n_demoted = int((filtered.get("universe_filter_reason", pd.Series([], dtype=str)) != "").sum())
    print(f"[trading_universe] universe={active} demoted_to_hold={n_demoted} rows={len(filtered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
