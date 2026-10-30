"""
Mercado sueco (Nasdaq Stockholm): fonte unica das definicoes de mercado.

Tudo o que muda de um mercado para outro mora aqui: universo, sufixo dos
tickers, indice de referencia, custos, imposto e regra de execucao. O motor
(`ga_run.py`), o ETL (`stock_etl.py`) e os replays leem destas constantes.

Decisoes registradas em `docs/sweden/plano.md` e `docs/adr/`.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent
UNIVERSE_PATH = REPO_ROOT / "data" / "universe_se.json"

MARKET_NAME = "Nasdaq Stockholm"
CURRENCY = "SEK"
TICKER_SUFFIX = ".ST"

# Indice amplo de referencia para features de regime (^OMXS30 esta indisponivel
# no Yahoo desde abr/2026). E um indice de PRECO: nao reinveste dividendos.
INDEX_SYMBOL = "^OMXSPI"
INDEX_FALLBACKS = ("EWD", "^OMX")

# Index benchmark do portfolio replay: OMX Stockholm Benchmark GROSS index,
# que reinveste dividendos — o que um fundo de indice na ISK de fato rende.
# Comparar com o indice de preco daria ~3 pp/ano de vantagem falsa.
BENCHMARK_SYMBOL = "^OMXSBGI"

# --------------------------------------------------------------------------
# Custos (ISK, courtage Mini) — fracoes do valor da posicao
# --------------------------------------------------------------------------
# Brokerage Mini (Avanza/Nordnet): 0,25% por ordem, minimo 1 SEK. Cenario
# otimista (Avanza Start / campanha Nordnet) via SE_BROKERAGE_PCT=0.
BROKERAGE_PCT_PER_ORDER = float(os.getenv("SE_BROKERAGE_PCT", "0.0025"))
SLIPPAGE_PCT_PER_SIDE = float(os.getenv("SE_SLIPPAGE_PCT", "0.0010"))
BUY_SIDE_COST_PCT = BROKERAGE_PCT_PER_ORDER + SLIPPAGE_PCT_PER_SIDE
SELL_SIDE_COST_PCT = BROKERAGE_PCT_PER_ORDER + SLIPPAGE_PCT_PER_SIDE
ROUND_TRIP_COST_PCT = BUY_SIDE_COST_PCT + SELL_SIDE_COST_PCT

# ISK: tributacao anual sobre o saldo, nunca por operacao. A taxa anual incide
# igualmente sobre o index benchmark e se cancela na comparacao.
TAX_PER_TRADE_PCT = 0.0

# --------------------------------------------------------------------------
# Regra de execucao (ADR 0001)
# --------------------------------------------------------------------------
# Ordem limitada na entrada, valida por N pregoes apos o sinal; cancelada se
# o fechamento cruzar o stop antes de executar. Expirada = missed entry.
ORDER_VALIDITY_DAYS = int(os.getenv("SE_ORDER_VALIDITY_DAYS", "3"))

# --------------------------------------------------------------------------
# Carteira
# --------------------------------------------------------------------------
CAPITAL_SEK = 30_000.0
POSITION_SIZE_SEK = 5_000.0
MAX_POSITIONS = 6

# Piso de liquidez usado na montagem do universo (ADR 0003).
MIN_MEDIAN_TURNOVER_SEK = 10_000_000.0

# --------------------------------------------------------------------------
# Contexto macro baixado pelo ETL (features causais de regime)
# --------------------------------------------------------------------------
CONTEXT_INDICES: List[str] = [
    INDEX_SYMBOL, *INDEX_FALLBACKS,
    "^STOXX", "^STOXX50E", "^FTSE", "^GDAXI", "^FCHI",
    "^GSPC", "^DJI", "^IXIC", "^NDX", "^RUT",
    "^VIX", "^VVIX",
    "^IRX", "^FVX", "^TNX", "^TYX",
    "^N225", "^HSI",
]
CONTEXT_FX_COMMODITIES: List[str] = [
    "DX-Y.NYB",
    "SEK=X", "EURSEK=X", "EURUSD=X", "GBPUSD=X", "JPY=X", "CHF=X",
    "GC=F", "SI=F", "HG=F", "CL=F", "BZ=F", "NG=F",
    "UUP", "FXE", "FXY", "GLD", "USO", "COPX",
]


@lru_cache(maxsize=1)
def load_universe() -> Dict:
    """Snapshot congelado do universo (data/universe_se.json)."""
    with open(UNIVERSE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def universe_tickers() -> List[str]:
    """Simbolos Yahoo do universo, na ordem do snapshot."""
    return [row["yf_symbol"] for row in load_universe()["tickers"]]


def sector_of(ticker: str) -> str:
    """Setor Nasdaq do ticker, normalizado como chave (ex.: 'industrials')."""
    for row in load_universe()["tickers"]:
        if row["yf_symbol"] == ticker:
            return _sector_key(row.get("sector") or "")
    return "other_equity"


def _sector_key(sector: str) -> str:
    key = "".join(ch if ch.isalnum() else "_" for ch in sector.strip().lower())
    key = "_".join(part for part in key.split("_") if part)
    return key or "other_equity"


def is_equity(ticker: str) -> bool:
    t = str(ticker).upper().strip()
    return t.endswith(TICKER_SUFFIX) and not t.startswith("^") and "=" not in t
