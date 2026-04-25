"""
Ticker metadata: asset class, sector, and segment mapping for OMX Stockholm universe.
Single source of truth for segmentation across the pipeline.

Branch experimental: substitui mapeamento BR (.SA) por SE (.ST). Os 8 segmentos
sao preservados (commodity_largecap, financials, utilities_infra,
domestic_cyclicals, defensives_health, industrials_exporters, agro, tech_growth)
para nao quebrar consumidores em ga_run.py / stock_final_output.py.
"""

SECTOR_MAP = {
    # Commodity Largecap (mining / paper / steel / oil)
    "BOL.ST": "commodity_largecap",
    "SSAB-A.ST": "commodity_largecap",
    "SSAB-B.ST": "commodity_largecap",
    "SCA-B.ST": "commodity_largecap",
    "STORA-A.ST": "commodity_largecap",
    "HOLM-B.ST": "commodity_largecap",
    "BILL.ST": "commodity_largecap",
    "LUNDIN.ST": "commodity_largecap",
    "SBB-B.ST": "commodity_largecap",
    "SAGA-B.ST": "commodity_largecap",

    # Financials
    "SEB-A.ST": "financials",
    "SHB-A.ST": "financials",
    "SWED-A.ST": "financials",
    "NDA-SE.ST": "financials",
    "INVE-B.ST": "financials",
    "KINV-B.ST": "financials",
    "EQT.ST": "financials",

    # Utilities / Telecom / Infra
    "TELIA.ST": "utilities_infra",
    "TEL2-B.ST": "utilities_infra",

    # Domestic Cyclicals (consumer / construction / retail)
    "HM-B.ST": "domestic_cyclicals",
    "ELUX-B.ST": "domestic_cyclicals",
    "SKA-B.ST": "domestic_cyclicals",
    "BHG.ST": "domestic_cyclicals",
    "MTRS.ST": "domestic_cyclicals",
    "EMBRAC-B.ST": "domestic_cyclicals",

    # Defensives / Health / Consumer Staples
    "AZN.ST": "defensives_health",
    "ESSITY-B.ST": "defensives_health",
    "GETI-B.ST": "defensives_health",
    "ICA.ST": "defensives_health",

    # Industrials / Exporters
    "ABB.ST": "industrials_exporters",
    "ALFA.ST": "industrials_exporters",
    "ASSA-B.ST": "industrials_exporters",
    "ATCO-A.ST": "industrials_exporters",
    "ATCO-B.ST": "industrials_exporters",
    "NIBE-B.ST": "industrials_exporters",
    "SAND.ST": "industrials_exporters",
    "SKF-B.ST": "industrials_exporters",
    "VOLV-A.ST": "industrials_exporters",
    "VOLV-B.ST": "industrials_exporters",
    "TREL-B.ST": "industrials_exporters",
    "BEIJ-B.ST": "industrials_exporters",
    "LIFCO-B.ST": "industrials_exporters",
    "EPI-A.ST": "industrials_exporters",

    # Tech / Growth
    "ERIC-B.ST": "tech_growth",
    "EVO.ST": "tech_growth",
    "HEXA-B.ST": "tech_growth",
    "SINCH.ST": "tech_growth",
    "TIETO.ST": "tech_growth",
}

# Sweden has no FII-equivalent listed instruments; keep set empty to disable
# the legacy FII-by-suffix heuristic (was triggered by 11.SA suffix in BR).
_FII_KNOWN = set()


def get_ticker_meta(ticker: str) -> dict:
    """Return asset_class, sector, and segment_key for a given ticker."""
    t = ticker.strip().upper()

    # Crypto
    if t.endswith("-USD"):
        return {"asset_class": "crypto", "sector": "crypto", "segment_key": "crypto"}

    # Equity with known sector
    sector = SECTOR_MAP.get(t, "other_equity")
    return {"asset_class": "equity", "sector": sector, "segment_key": sector}


def get_segment_for_tickers(tickers):
    """Return dict mapping ticker -> segment_key."""
    return {t: get_ticker_meta(t)["segment_key"] for t in tickers}
