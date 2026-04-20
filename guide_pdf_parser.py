#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic Brazilian stock-guide PDF parser.

Returns a DataFrame keyed by normalized ticker with analyst recommendation,
target price, upside, and any provider-specific scores (quality / momentum /
recommendation strength). Values absent from the PDF are returned as NaN.
"""
from __future__ import annotations

import os
import re
import warnings
from datetime import date, datetime
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import pdfplumber
except ImportError as _e:
    raise ImportError(
        "pdfplumber is required for guide_pdf_parser; add it to requirements.txt"
    ) from _e


SCHEMA_COLUMNS = [
    "ticker",
    "guide_rec",
    "guide_target",
    "guide_upside_pct",
    "guide_price_ref",
    "guide_qual_score",
    "guide_momentum_score",
    "guide_rec_score",
    "guide_report_date",
    "guide_pe",
    "guide_dy",
    "guide_pb",
    "guide_fair",
    "guide_date",
    "guide_source",
]

_TICKER_RE = re.compile(r"\b([A-Z]{4}(?:11|3|4|5|6))\b")
_MONEY_RE = re.compile(r"R\$\s*([0-9][0-9\.\,]*)")
_PCT_RE = re.compile(r"(-?\d+[\.,]?\d*)\s*%")
_DATE_RE = re.compile(r"(\d{2}/\d{2}/\d{4})")
_SCORE_RE = re.compile(r"^-?\d+[\.,]?\d*$")

_REC_MAP = [
    ("buy",  ["comprar", "compra", "buy", "outperform", "forte compra", "atrativo"]),
    ("sell", ["vender", "venda", "sell", "underperform", "reduzir", "evitar"]),
    ("hold", ["manter", "neutra", "neutro", "hold", "market perform", "segurar"]),
]


def _normalize_ticker(raw: str) -> str | None:
    """Strip garbage and append .SA. Returns None if no valid ticker found."""
    if not raw:
        return None
    up = str(raw).upper()
    m = _TICKER_RE.search(up)
    if not m:
        return None
    t = m.group(1)
    # Filter false positives like MA200 or WS200 from technical-analysis text.
    if t.startswith(("MA", "SMA", "EMA", "WMA", "ATR", "RSI", "MACD")):
        return None
    return f"{t}.SA"


def _parse_money(s: str) -> float:
    if s is None:
        return np.nan
    m = _MONEY_RE.search(str(s))
    if not m:
        # Allow bare numerics like "15,33"
        if re.match(r"^\s*-?\d", str(s)):
            return _parse_num(s)
        return np.nan
    return _parse_num(m.group(1))


def _parse_pct(s: str) -> float:
    if s is None:
        return np.nan
    m = _PCT_RE.search(str(s))
    if not m:
        return np.nan
    return _parse_num(m.group(1))


def _parse_num(s: str) -> float:
    if s is None:
        return np.nan
    t = str(s).strip().replace("R$", "").replace("%", "").strip()
    # pt-BR: "1.234,56" -> "1234.56"
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".")
    elif "," in t:
        t = t.replace(",", ".")
    try:
        return float(t)
    except (ValueError, TypeError):
        return np.nan


def _parse_date(s: str) -> date | None:
    if not s:
        return None
    m = _DATE_RE.search(str(s))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%d/%m/%Y").date()
    except ValueError:
        return None


def _classify_rec(text: str) -> str | None:
    if not text:
        return None
    low = str(text).lower().strip()
    # Exact-word priority (VENDA/COMPRA/NEUTRA are single-cell tokens in tables).
    for label, keywords in _REC_MAP:
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", low):
                return label
    return None


def _row_has_ticker(row: Iterable) -> tuple[str | None, int]:
    """Return (normalized_ticker, column_index) for first ticker-shaped cell."""
    for i, cell in enumerate(row or []):
        t = _normalize_ticker(cell or "")
        if t:
            return t, i
    return None, -1


def _parse_table_row(row: list, source_name: str) -> dict | None:
    """Extract one ticker record from a table row. Returns None if no ticker."""
    if not row:
        return None
    ticker, t_idx = _row_has_ticker(row)
    if not ticker:
        return None

    # Recommendation: scan all cells for COMPRA/VENDA/NEUTRA keywords.
    rec = None
    rec_cell_idx = -1
    for i, cell in enumerate(row):
        r = _classify_rec(cell or "")
        if r is not None:
            rec = r
            rec_cell_idx = i
            break

    # Target price: first money/numeric cell AFTER the ticker that's not also the current price.
    # Blue3 layout puts [ticker, preço_atual, preço_alvo, upside%, data, qual, mom, rec_score, rec_label]
    money_vals: list[tuple[int, float]] = []
    pct_val = np.nan
    date_val = None
    scores: list[tuple[int, float]] = []
    for i, cell in enumerate(row):
        if i <= t_idx:
            continue
        cell_s = str(cell or "")
        if _MONEY_RE.search(cell_s):
            money_vals.append((i, _parse_money(cell_s)))
        elif _PCT_RE.search(cell_s):
            if np.isnan(pct_val):
                pct_val = _parse_pct(cell_s)
        elif _DATE_RE.search(cell_s):
            if date_val is None:
                date_val = _parse_date(cell_s)
        elif _SCORE_RE.match(cell_s.strip()):
            v = _parse_num(cell_s)
            if not np.isnan(v) and 0.0 <= v <= 10.0:
                scores.append((i, v))

    price_ref = money_vals[0][1] if len(money_vals) >= 1 else np.nan
    target = money_vals[1][1] if len(money_vals) >= 2 else np.nan
    # If only one money cell and we have an upside pct, fall back to that.
    if np.isnan(target) and not np.isnan(price_ref) and not np.isnan(pct_val):
        target = price_ref * (1.0 + pct_val / 100.0)

    qual = mom = rec_sc = np.nan
    if len(scores) >= 3:
        qual = scores[0][1]
        mom = scores[1][1]
        rec_sc = scores[2][1]
    elif len(scores) == 2:
        qual, mom = scores[0][1], scores[1][1]
    elif len(scores) == 1:
        qual = scores[0][1]

    return {
        "ticker": ticker,
        "guide_rec": rec,
        "guide_target": target,
        "guide_upside_pct": pct_val,
        "guide_price_ref": price_ref,
        "guide_qual_score": qual,
        "guide_momentum_score": mom,
        "guide_rec_score": rec_sc,
        "guide_report_date": date_val,
        "guide_pe": np.nan,
        "guide_dy": np.nan,
        "guide_pb": np.nan,
        "guide_fair": target,
        "guide_source": source_name,
    }


def _parse_text_fallback(text: str, source_name: str) -> list[dict]:
    """Line-by-line text fallback for PDFs where extract_tables returns nothing."""
    records: list[dict] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        ticker = _normalize_ticker(line)
        if not ticker:
            continue
        rec = _classify_rec(line)
        moneys = [_parse_num(m.group(1)) for m in _MONEY_RE.finditer(line)]
        pct = _parse_pct(line)
        dt = _parse_date(line)
        price_ref = moneys[0] if len(moneys) >= 1 else np.nan
        target = moneys[1] if len(moneys) >= 2 else np.nan
        if np.isnan(target) and not np.isnan(price_ref) and not np.isnan(pct):
            target = price_ref * (1.0 + pct / 100.0)
        records.append({
            "ticker": ticker,
            "guide_rec": rec,
            "guide_target": target,
            "guide_upside_pct": pct,
            "guide_price_ref": price_ref,
            "guide_qual_score": np.nan,
            "guide_momentum_score": np.nan,
            "guide_rec_score": np.nan,
            "guide_report_date": dt,
            "guide_pe": np.nan,
            "guide_dy": np.nan,
            "guide_pb": np.nan,
            "guide_fair": target,
            "guide_source": source_name,
        })
    return records


def _report_date_from_filename(path: str) -> date | None:
    """Best-effort extract date from filename stem like 'Stock.Guide.18.abr.pdf'."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    m = _DATE_RE.search(stem)
    if m:
        return _parse_date(m.group(1))
    months = {
        "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
        "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
    }
    m2 = re.search(r"(\d{1,2})[._-]?(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)", stem)
    if m2:
        d, mo = int(m2.group(1)), months[m2.group(2)]
        y = datetime.now().year
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    return None


def parse_guide(pdf_path: str) -> pd.DataFrame:
    """Parse a Brazilian stock-guide PDF. Returns a DataFrame with SCHEMA_COLUMNS."""
    if not os.path.exists(pdf_path):
        warnings.warn(f"[guide_parser] file not found: {pdf_path}")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    source_name = os.path.splitext(os.path.basename(pdf_path))[0]
    report_date = _report_date_from_filename(pdf_path)
    records: list[dict] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    tables = page.extract_tables() or []
                except Exception as _e:
                    tables = []
                got_from_tables = False
                for tbl in tables:
                    for row in tbl or []:
                        rec = _parse_table_row(row, source_name)
                        if rec is not None:
                            records.append(rec)
                            got_from_tables = True
                if not got_from_tables:
                    try:
                        txt = page.extract_text() or ""
                    except Exception:
                        txt = ""
                    records.extend(_parse_text_fallback(txt, source_name))
    except Exception as e:
        warnings.warn(f"[guide_parser] failed to read {pdf_path}: {e}")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    if not records:
        warnings.warn(f"[guide_parser] no ticker records extracted from {pdf_path}")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    df = pd.DataFrame(records)
    df["guide_date"] = report_date

    # Dedup: when a ticker appears multiple times, keep the row with most non-null fields.
    df["_nn"] = df.notna().sum(axis=1)
    df = df.sort_values("_nn", ascending=False).drop_duplicates("ticker", keep="first")
    df = df.drop(columns=["_nn"]).reset_index(drop=True)

    for c in SCHEMA_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[SCHEMA_COLUMNS]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "guides/Stock.Guide.18.abr.pdf"
    out = parse_guide(path)
    print(f"rows: {len(out)} | tickers: {out['ticker'].nunique()}")
    print(f"recs: {out['guide_rec'].value_counts(dropna=False).to_dict()}")
    print(out.head(20).to_string())
