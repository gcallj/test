#!/usr/bin/env python3
"""
Precos diarios do universo sueco e do benchmark, ajustados por dividendos.

    python prices_se.py download     # baixa do Yahoo para output/prices_se.parquet (~2 min)
    python prices_se.py info         # cobertura por ticker

A estrategia de rotacao (rotation_se.py) le daqui. Dividendos entram via
`Adj Close`: numa ISK eles sao creditados em caixa, o que equivale a
reinvestir no preco ajustado.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import market_se  # noqa: E402

CACHE = REPO_ROOT / "output" / "prices_se.parquet"
START = "2005-01-01"
FIELDS = ("Open", "Close", "Adj Close", "Volume")


def download(start: str = START) -> pd.DataFrame:
    import yfinance as yf

    symbols = market_se.universe_tickers() + [market_se.BENCHMARK_SYMBOL, market_se.INDEX_SYMBOL]
    frames = []
    for i in range(0, len(symbols), 40):
        chunk = symbols[i:i + 40]
        d = yf.download(chunk, start=start, interval="1d", auto_adjust=False, progress=False,
                        threads=True, group_by="column")
        frames.append(d[list(FIELDS)])
        time.sleep(2)
    raw = pd.concat(frames, axis=1)
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw = raw.sort_index()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(CACHE)
    return raw


def _clean(px: pd.DataFrame) -> pd.DataFrame:
    """Remove ticks absurdos do Yahoo (preco 5x acima/abaixo da mediana de 21 dias)."""
    med = px.rolling(21, min_periods=5, center=True).median()
    bad = (px > 5 * med) | (px < med / 5)
    return px.mask(bad)


def load() -> Dict[str, pd.DataFrame]:
    """Retorna DataFrames data x ticker: open, close, adj, adj_open, volume, turnover."""
    if not CACHE.exists():
        raise SystemExit(f"{CACHE} nao existe. Rode: python prices_se.py download")
    raw = pd.read_parquet(CACHE)
    close = _clean(raw["Close"])
    adj = _clean(raw["Adj Close"])
    opn = _clean(raw["Open"]).where(lambda x: x > 0)
    factor = (adj / close).ffill()
    out = {
        "close": close,
        "adj": adj,
        "open": opn,
        "adj_open": opn * factor,
        "volume": raw["Volume"],
        "turnover": close * raw["Volume"],
    }
    # Dias sem nenhum preco de acao (feriados que o Yahoo deixou) saem.
    eq = [t for t in close.columns if market_se.is_equity(t)]
    keep = close[eq].notna().any(axis=1)
    return {k: v.loc[keep] for k, v in out.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["download", "info"])
    args = ap.parse_args(argv)
    if args.cmd == "download":
        raw = download()
        print(f"[prices] {raw.shape[0]} pregoes, {raw['Close'].shape[1]} simbolos -> {CACHE}")
        return 0
    p = load()
    first = p["adj"].apply(lambda s: s.first_valid_index())
    print(f"{p['adj'].shape[1]} simbolos, {p['adj'].index.min().date()} .. {p['adj'].index.max().date()}")
    print(first.sort_values().tail(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
