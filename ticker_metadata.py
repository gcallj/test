"""
Ticker metadata: asset class, sector, and segment mapping for the Swedish
universe (Nasdaq Stockholm Large Cap snapshot in data/universe_se.json).
Single source of truth for segmentation across the pipeline.

Segments are the Nasdaq sector names normalised as keys, e.g. "Basic
Materials" -> "basic_materials".
"""

from market_se import is_equity, load_universe, _sector_key

SECTOR_MAP = {
    row["yf_symbol"]: _sector_key(row.get("sector") or "")
    for row in load_universe()["tickers"]
}


def get_ticker_meta(ticker: str) -> dict:
    """Return asset_class, sector, and segment_key for a given ticker."""
    t = ticker.strip().upper()

    if not is_equity(t):
        return {"asset_class": "context", "sector": "context", "segment_key": "context"}

    sector = SECTOR_MAP.get(t, "other_equity")
    return {"asset_class": "equity", "sector": sector, "segment_key": sector}


def get_segment_for_tickers(tickers):
    """Return dict mapping ticker -> segment_key."""
    return {t: get_ticker_meta(t)["segment_key"] for t in tickers}
