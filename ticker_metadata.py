"""
Ticker metadata: asset class, sector, and segment mapping for B3 universe.
Single source of truth for segmentation across the pipeline.
"""

SECTOR_MAP = {
    # ── Commodity Largecap ──
    "VALE3.SA": "commodity_largecap",
    "PETR4.SA": "commodity_largecap",
    "CSNA3.SA": "commodity_largecap",
    "GGBR4.SA": "commodity_largecap",
    "GOAU4.SA": "commodity_largecap",
    "USIM5.SA": "commodity_largecap",
    "BRKM5.SA": "commodity_largecap",
    "SUZB3.SA": "commodity_largecap",
    "KLBN11.SA": "commodity_largecap",
    "SLCE3.SA": "commodity_largecap",
    "SMTO3.SA": "commodity_largecap",
    "UNIP6.SA": "commodity_largecap",
    "PRIO3.SA": "commodity_largecap",
    # ── Financials ──
    "ITUB4.SA": "financials",
    "BBDC4.SA": "financials",
    "BBAS3.SA": "financials",
    "SANB11.SA": "financials",
    "ITSA4.SA": "financials",
    "BPAC11.SA": "financials",
    "BBSE3.SA": "financials",
    "BRSR6.SA": "financials",
    "ABCB4.SA": "financials",
    # ── Utilities / Infra ──
    "CMIG4.SA": "utilities_infra",
    "CPFE3.SA": "utilities_infra",
    "EGIE3.SA": "utilities_infra",
    "EQTL3.SA": "utilities_infra",
    "ENGI11.SA": "utilities_infra",
    "TAEE11.SA": "utilities_infra",
    "SBSP3.SA": "utilities_infra",
    "NEOE3.SA": "utilities_infra",
    "ISAE4.SA": "utilities_infra",
    "CSMG3.SA": "utilities_infra",
    "ALUP11.SA": "utilities_infra",
    "TIMS3.SA": "utilities_infra",
    "VIVT3.SA": "utilities_infra",
    # ── Domestic Cyclicals ──
    "LREN3.SA": "domestic_cyclicals",
    "MGLU3.SA": "domestic_cyclicals",
    "CYRE3.SA": "domestic_cyclicals",
    "CSAN3.SA": "domestic_cyclicals",
    "RENT3.SA": "domestic_cyclicals",
    "MULT3.SA": "domestic_cyclicals",
    "JHSF3.SA": "domestic_cyclicals",
    "DIRR3.SA": "domestic_cyclicals",
    "CURY3.SA": "domestic_cyclicals",
    "LAVV3.SA": "domestic_cyclicals",
    "VULC3.SA": "domestic_cyclicals",
    "ECOR3.SA": "domestic_cyclicals",
    "UGPA3.SA": "domestic_cyclicals",
    "RAIL3.SA": "domestic_cyclicals",
    "WIZC3.SA": "domestic_cyclicals",
    # ── Defensives / Health / Consumer Staples ──
    "RADL3.SA": "defensives_health",
    "FLRY3.SA": "defensives_health",
    "ABEV3.SA": "defensives_health",
    "MDIA3.SA": "defensives_health",
    "ODPV3.SA": "defensives_health",
    "LEVE3.SA": "defensives_health",
    "GRND3.SA": "defensives_health",
    "PSSA3.SA": "defensives_health",
    # ── Industrials / Exporters ──
    "WEGE3.SA": "industrials_exporters",
    "POMO4.SA": "industrials_exporters",
    "TGMA3.SA": "industrials_exporters",
    "RANI3.SA": "industrials_exporters",
    # ── Agro ──
    "AGRO3.SA": "agro",
    "BEEF3.SA": "agro",
    "MBRF3.SA": "agro",
    # ── Tech / Growth ──
    "CXSE3.SA": "tech_growth",
    "SYNE3.SA": "tech_growth",
    "AXIA3.SA": "tech_growth",
    "AXIA6.SA": "tech_growth",
    "ASAI3.SA": "tech_growth",
}

# FII segment (all 11-suffix .SA not in SECTOR_MAP)
_FII_KNOWN = {
    "HGLG11.SA", "HGBS11.SA", "HGRE11.SA", "MXRF11.SA", "KFOF11.SA",
    "BRCR11.SA", "BTLG11.SA", "CPTS11.SA", "RBRP11.SA", "RCRB11.SA",
    "XPML11.SA",
}


def get_ticker_meta(ticker: str) -> dict:
    """Return asset_class, sector, and segment_key for a given ticker."""
    t = ticker.strip().upper()

    # Crypto
    if t.endswith("-USD"):
        return {"asset_class": "crypto", "sector": "crypto", "segment_key": "crypto"}

    # Known FIIs or 11-suffix .SA not in equity map
    if t in _FII_KNOWN or (t.endswith("11.SA") and t not in SECTOR_MAP):
        return {"asset_class": "fii", "sector": "fii", "segment_key": "fii"}

    # Equity with known sector
    sector = SECTOR_MAP.get(t, "other_equity")
    return {"asset_class": "equity", "sector": sector, "segment_key": sector}


def get_segment_for_tickers(tickers):
    """Return dict mapping ticker -> segment_key."""
    return {t: get_ticker_meta(t)["segment_key"] for t in tickers}
