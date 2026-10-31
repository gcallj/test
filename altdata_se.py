#!/usr/bin/env python3
"""
Dados alternativos do mercado sueco (estrategia v4, docs/adr/0008).

Tres fontes publicas, cada registro com o instante em que ficou publico:

- Finansinspektionen, registro de posicoes vendidas (blankningsregistret):
  posicoes liquidas >= 0,5% do capital, por detentor, desde 2012-11 (regulamento
  europeu de vendas a descoberto). A posicao da data T e publicada em T+1; o
  sinal so vale a partir da abertura de T+2.
- Finansinspektionen, registro de insiders (insynshandel, MAR): toda transacao
  de pessoa em posicao de lideranca, desde 2016-07, com hora de publicacao. O
  sinal vale a partir da abertura do primeiro pregao depois do dia da publicacao.
- Nasdaq Nordic, comunicados das empresas (titulo, categoria, hora), desde 2005.
  Datas de balancos e texto para analise de sentimento. O sinal vale a partir
  da abertura do primeiro pregao depois do dia do comunicado.

Os arquivos brutos ficam em output/altdata/raw (cache); as tabelas limpas em
output/altdata/*.parquet. Nada aqui le precos.

    python altdata_se.py shorts        # registro de posicoes vendidas
    python altdata_se.py insiders      # registro de insiders (baixa em janelas)
    python altdata_se.py news          # comunicados da Nasdaq Nordic por empresa
    python altdata_se.py coverage      # cobertura por fonte e por ano
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import market_se  # noqa: E402

OUT = REPO_ROOT / "output" / "altdata"
RAW = OUT / "raw"
SHORTS_PATH = OUT / "fi_shorts.parquet"
INSIDERS_PATH = OUT / "fi_insiders.parquet"
NEWS_PATH = OUT / "news.parquet"
NEWS_MAP_PATH = REPO_ROOT / "data" / "news_companies_se.json"

FI_SHORT_URL = "https://www.fi.se/BlankningsRegister/{name}"
FI_INSIDER_URL = ("https://marknadssok.fi.se/Publiceringsklient/sv-SE/Search/Search?SearchFunctionType=Insyn"
                  "&Utgivare=&PersonILedandeStallningNamn=&Transaktionsdatum.From=&Transaktionsdatum.To="
                  "&Publiceringsdatum.From={start}&Publiceringsdatum.To={end}&button=export")
FI_EXPORT_CAP = 1000          # a exportacao do FI devolve no maximo 1000 linhas por consulta
INSIDER_START = "2016-07-01"  # inicio do MAR; antes disso a exportacao vem vazia
NEWS_API = "https://api.news.eu.nasdaq.com/news/query.action"
NEWS_PAGE = 200               # a API devolve no maximo 200 itens por pagina
USER_AGENT = "Mozilla/5.0 (research; trading-signals SE)"


# --------------------------------------------------------------------------
# HTTP com repeticao
# --------------------------------------------------------------------------
def _get(url: str, params: Optional[Dict] = None, timeout: int = 120, tries: int = 5) -> bytes:
    import requests
    err = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                return r.content
            err = RuntimeError(f"HTTP {r.status_code} {url}")
        except Exception as e:  # rede instavel: tenta de novo
            err = e
        time.sleep(2 ** k)
    raise RuntimeError(f"falha ao baixar {url}: {err}")


# --------------------------------------------------------------------------
# Leitor minimo de planilhas ODS (evita a dependencia odfpy)
# --------------------------------------------------------------------------
_T = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
_O = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
_TXT = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"


def read_ods(data: bytes) -> List[List[Optional[str]]]:
    """Linhas da primeira aba de um arquivo .ods, como texto (valor bruto da celula)."""
    from xml.etree import ElementTree as ET
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = ET.fromstring(z.read("content.xml"))
    table = next(root.iter(_T + "table"))
    rows: List[List[Optional[str]]] = []
    for tr in table.iter(_T + "table-row"):
        row: List[Optional[str]] = []
        for tc in tr:
            if tc.tag not in (_T + "table-cell", _T + "covered-table-cell"):
                continue
            rep = min(int(tc.get(_T + "number-columns-repeated", "1")), 64)
            v = tc.get(_O + "value") or tc.get(_O + "date-value")
            if v is None:
                txt = "\n".join("".join(p.itertext()) for p in tc.findall(_TXT + "p"))
                v = txt if txt else None
            row.extend([v] * rep)
        while row and row[-1] is None:
            row.pop()
        if row:
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Nomes de emissores -> tickers do universo
# --------------------------------------------------------------------------
_LEGAL = r"\b(ab|publ|aktiebolag|aktiebolaget|asa|oyj|plc|ltd|limited|inc|corp|corporation|sa|nv|se|the|group|holding|holdings)\b"


def norm_name(s: str) -> str:
    """Nome de empresa normalizado: sem acentos, forma juridica, pontuacao ou espacos."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[().,'\-/]", " ", s)
    s = re.sub(_LEGAL, " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def universe() -> List[Dict]:
    return market_se.load_universe()["tickers"]


class IssuerMap:
    """ISIN, LEI e nome do emissor -> ticker do universo (uma classe de acao por emissor).

    O ISIN atual do snapshot e a chave exata. Para linhas com outro ISIN (antes de
    um desdobramento, outra classe de acao) exige-se LEI *e* nome do emissor do
    proprio ticker: o registro do FI tem linhas com o LEI de outra empresa
    digitado por engano, entao nenhum dos dois sozinho basta. Os nomes aceitos
    sao os que aparecem com frequencia nas linhas do ISIN atual.
    """

    MIN_NAME_SHARE = 0.02
    MIN_NAME_ROWS = 3

    def __init__(self, insiders: Optional[pd.DataFrame] = None, extra: Optional[Dict[str, str]] = None):
        u = universe()
        self.by_isin = {t["isin"]: t["yf_symbol"] for t in u}
        self.lei_of: Dict[str, str] = {}
        names: Dict[str, set] = {t["yf_symbol"]: {norm_name(re.sub(r"\s+(A|B|C|D|R|SDB|ser\.?\s*\w)$", "", t["name"]))}
                                 for t in u}
        if insiders is not None and len(insiders):
            core = insiders[insiders["isin"].isin(list(self.by_isin))]
            for isin, g in core.groupby("isin"):
                tk = self.by_isin[isin]
                leis = g["lei"][g["lei"].astype(str).str.len() > 0]
                if len(leis):
                    self.lei_of[tk] = leis.mode().iat[0]
                cnt = g["issuer"].map(norm_name).value_counts()
                keep = cnt[(cnt >= self.MIN_NAME_ROWS) & (cnt >= self.MIN_NAME_SHARE * len(g))].index
                names[tk] |= set(keep)
        for k, v in (extra or {}).items():
            names.setdefault(v, set()).add(norm_name(k))
        owners: Dict[str, set] = {}
        for tk, ns in names.items():
            for n in ns:
                if n:
                    owners.setdefault(n, set()).add(tk)
        # nome que aponta para dois tickers e ambiguo: descartado
        self.by_name = {n: next(iter(ts)) for n, ts in owners.items() if len(ts) == 1}
        self.names = names

    def consistent(self, tk: str, name: Optional[str]) -> bool:
        """O nome do emissor e uma grafia do proprio ticker (igual, ou mesmo prefixo de 4 letras)."""
        n = norm_name(name) if name else ""
        for m in self.names.get(tk, ()):
            if n == m or (len(n) >= 4 and len(m) >= 4 and (n.startswith(m[:4]) or m.startswith(n[:4]))):
                return True
        return False

    def ticker(self, isin: Optional[str] = None, lei: Optional[str] = None, name: Optional[str] = None) -> Optional[str]:
        if isin and isin in self.by_isin:
            # ISIN certo, mas o emissor (ou o LEI, no registro de insiders) precisa concordar:
            # evita linhas de insiders de outra empresa e ISINs digitados errado.
            tk = self.by_isin[isin]
            if (lei and self.lei_of.get(tk) == lei) or self.consistent(tk, name):
                return tk
            return None
        tk = self.by_name.get(norm_name(name)) if name else None
        if tk is None:
            return None
        if lei is not None:  # registro de insiders: o LEI precisa confirmar o nome
            return tk if lei and self.lei_of.get(tk) == lei else None
        return tk


# --------------------------------------------------------------------------
# Calendario: a partir de quando um fato publico pode ser negociado
# --------------------------------------------------------------------------
def next_session_after(days: pd.DatetimeIndex, stamps: pd.Series, lag_sessions: int = 1) -> pd.Series:
    """Primeiro pregao depois do dia de `stamps`, mais `lag_sessions - 1` pregoes.

    `lag_sessions=1`: publicado no dia D (a qualquer hora) -> abertura do primeiro
    pregao com data > D. A ordem e enviada depois do fechamento, como no resto do
    sistema, entao nunca se negocia no proprio dia da publicacao.
    """
    d = pd.to_datetime(stamps).dt.normalize()
    pos = days.searchsorted(d.values, side="right") + (lag_sessions - 1)
    out = pd.Series(pd.NaT, index=stamps.index, dtype="datetime64[ns]")
    ok = pos < len(days)
    out[ok] = days[pos[ok]]
    return out


# --------------------------------------------------------------------------
# Registro de posicoes vendidas (blankningsregistret)
# --------------------------------------------------------------------------
def _parse_short_file(data: bytes) -> pd.DataFrame:
    rows = read_ods(data)
    head = next(i for i, r in enumerate(rows) if r and str(r[0]).startswith("Innehavare"))
    body = [r + [None] * (6 - len(r)) for r in rows[head + 1:]]
    df = pd.DataFrame([r[:6] for r in body], columns=["holder", "issuer", "isin", "pct", "position_date", "comment"])
    df["issuer"] = df["issuer"].astype(str).str.replace("\xa0", " ").str.strip()
    df["holder"] = df["holder"].astype(str).str.replace("\xa0", " ").str.strip()
    raw = df["pct"].astype(str).str.replace(",", ".").str.strip()
    df["below_threshold"] = raw.str.startswith("<")
    df["pct"] = pd.to_numeric(raw.where(~df["below_threshold"], "0"), errors="coerce")
    df["position_date"] = pd.to_datetime(df["position_date"].astype(str).str[:10], errors="coerce")
    return df.dropna(subset=["position_date", "pct"])


def download_shorts(refresh: bool = True) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    parts = []
    for name in ("GetHistFile", "GetAktuellFile"):
        path = RAW / f"fi_{name}.ods"
        if refresh or not path.exists():
            path.write_bytes(_get(FI_SHORT_URL.format(name=name)))
        parts.append(_parse_short_file(path.read_bytes()))
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["holder", "isin", "position_date", "pct"])
    return df.sort_values(["position_date", "issuer", "holder"]).reset_index(drop=True)


def map_shorts(df: pd.DataFrame, imap: IssuerMap) -> pd.DataFrame:
    df = df.copy()
    df["ticker"] = [imap.ticker(isin=i, name=n) for i, n in zip(df["isin"], df["issuer"])]
    # Um emissor com nome reconhecido em alguma linha herda o ticker nas demais grafias.
    name_tk = df.dropna(subset=["ticker"]).assign(k=lambda x: x["issuer"].map(norm_name)).groupby("k")["ticker"].agg(
        lambda s: s.mode().iat[0])
    miss = df["ticker"].isna()
    df.loc[miss, "ticker"] = df.loc[miss, "issuer"].map(norm_name).map(name_tk)
    return df


def short_interest_panel(df: pd.DataFrame, days: pd.DatetimeIndex, tickers: List[str]) -> pd.DataFrame:
    """Soma das posicoes vendidas divulgadas (% do capital) por ticker, conhecida na abertura de cada pregao.

    Cada detentor mantem a ultima posicao informada ate a proxima; "<0,5" zera a
    posicao. A posicao da data T entra na abertura de T+2 (publicacao em T+1).
    """
    d = df.dropna(subset=["ticker"]).copy()
    d["avail"] = next_session_after(days, d["position_date"], lag_sessions=2)
    d = d.dropna(subset=["avail"]).sort_values(["avail", "position_date"])
    out = pd.DataFrame(0.0, index=days, columns=tickers)
    for tk, g in d.groupby("ticker"):
        if tk not in out.columns:
            continue
        # ultima posicao de cada detentor em cada data de disponibilidade
        last = g.groupby(["avail", "holder"])["pct"].last().unstack("holder")
        last = last.reindex(days).ffill().fillna(0.0)
        out[tk] = last.sum(axis=1)
    return out


# --------------------------------------------------------------------------
# Registro de insiders (insynshandel)
# --------------------------------------------------------------------------
INSIDER_COLS = {
    "Publiceringsdatum": "published", "Emittent": "issuer", "LEI-kod": "lei", "Anmälningsskyldig": "notifier",
    "Person i ledande ställning": "person", "Befattning": "role", "Närstående": "related",
    "Korrigering": "correction", "Beskrivning av korrigering": "correction_text",
    "Är förstagångsrapportering": "first_report", "Är kopplad till aktieprogram": "share_program",
    "Karaktär": "nature", "Instrumenttyp": "instrument_type", "Instrumentnamn": "instrument",
    "ISIN": "isin", "Transaktionsdatum": "trade_date", "Volym": "volume", "Volymsenhet": "volume_unit",
    "Pris": "price", "Valuta": "currency", "Handelsplats": "venue", "Status": "status",
}


def _decode_fi_csv(data: bytes) -> pd.DataFrame:
    for enc in ("utf-16", "utf-16-le", "utf-8-sig", "cp1252"):
        try:
            txt = data.decode(enc)
        except UnicodeDecodeError:
            continue
        if txt.lstrip("﻿").startswith("Publiceringsdatum"):
            return pd.read_csv(io.StringIO(txt.lstrip("﻿")), sep=";", dtype=str, keep_default_na=False)
    raise ValueError("exportacao do FI em formato desconhecido")


def _insider_chunk(start: pd.Timestamp, end: pd.Timestamp, cache: Path) -> List[pd.DataFrame]:
    """Baixa [start, end]; se bater no limite de 1000 linhas, divide a janela ao meio."""
    path = cache / f"insyn_{start.date()}_{end.date()}.csv"
    if path.exists():
        data = path.read_bytes()
    else:
        data = _get(FI_INSIDER_URL.format(start=start.date(), end=end.date()))
        time.sleep(0.3)
    df = _decode_fi_csv(data)
    if len(df) >= FI_EXPORT_CAP:
        if start == end:
            raise RuntimeError(f"mais de {FI_EXPORT_CAP} publicacoes em {start.date()}: exportacao truncada")
        mid = start + (end - start) // 2
        mid = mid.normalize()
        if path.exists():
            path.unlink()
        return _insider_chunk(start, mid, cache) + _insider_chunk(mid + pd.Timedelta(days=1), end, cache)
    if not path.exists() and end < pd.Timestamp.today().normalize():
        path.write_bytes(data)  # janela fechada: guarda no cache
    return [df]


def download_insiders(start: str = INSIDER_START, end: Optional[str] = None) -> pd.DataFrame:
    cache = RAW / "fi_insyn"
    cache.mkdir(parents=True, exist_ok=True)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    parts: List[pd.DataFrame] = []
    for m in pd.date_range(pd.Timestamp(start), end_ts, freq="MS"):
        m_end = min(m + pd.offsets.MonthEnd(0), end_ts)
        parts += _insider_chunk(m, m_end, cache)
    df = pd.concat(parts, ignore_index=True).rename(columns=INSIDER_COLS)
    df = df[[c for c in INSIDER_COLS.values() if c in df.columns]]
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"].str[:10], errors="coerce")
    for c in ("volume", "price"):
        df[c] = pd.to_numeric(df[c].str.replace("\xa0", "").str.replace(" ", "").str.replace(",", "."), errors="coerce")
    for c in ("related", "correction", "first_report", "share_program"):
        df[c] = df[c].str.strip().str.lower().eq("ja")
    df = df.drop_duplicates()
    return df.sort_values("published").reset_index(drop=True)


def map_insiders(df: pd.DataFrame, imap: IssuerMap) -> pd.DataFrame:
    df = df.copy()
    df["ticker"] = [imap.ticker(isin=i, lei=l, name=n) for i, l, n in zip(df["isin"], df["lei"], df["issuer"])]
    return df


# --------------------------------------------------------------------------
# Comunicados da Nasdaq Nordic
# --------------------------------------------------------------------------
_NEWS_BASE = {"type": "json", "showAttachments": "false", "showCnsSpecific": "true", "showCompany": "true",
              "countResults": "true", "globalGroup": "exchangeNotice", "globalName": "NordicAllMarkets",
              "displayLanguage": "en", "timeZone": "CET", "dateMask": "yyyy-MM-dd HH:mm:ss", "dir": "DESC"}


def _news_query(**kw) -> Dict:
    params = dict(_NEWS_BASE, **{k: str(v) for k, v in kw.items()})
    return json.loads(_get(NEWS_API, params=params, timeout=90))


def discover_news_companies(imap: IssuerMap) -> Dict[str, str]:
    """Nome da empresa na API de comunicados -> ticker, por busca textual e nome normalizado."""
    found: Dict[str, str] = {}
    for t in universe():
        base = re.sub(r"\s+(A|B|C|D|R|SDB)$", "", t["name"]).strip()
        hits = _news_query(freeText=base, limit=100, start=0)["results"]["item"]
        cands = {h["company"] for h in hits if "Stockholm" in str(h.get("market", "")) or "Sweden" in str(h.get("market", ""))}
        cands |= {h["company"] for h in hits}
        target = norm_name(base)
        best = [c for c in cands if norm_name(c) == target or imap.ticker(name=c) == t["yf_symbol"]]
        if not best:
            best = [c for c in cands if target and (norm_name(c).startswith(target) or target.startswith(norm_name(c)))]
        for c in best:
            found[c] = t["yf_symbol"]
        time.sleep(0.2)
    return found


def download_news(companies: Dict[str, str]) -> pd.DataFrame:
    cache = RAW / "news"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, tk in sorted(companies.items()):
        path = cache / f"{re.sub(r'[^A-Za-z0-9]+', '_', name)}.json"
        if path.exists():
            items = json.loads(path.read_text())
        else:
            items, start = [], 0
            while True:
                res = _news_query(company=name, limit=NEWS_PAGE, start=start)["results"]["item"]
                items += res
                if len(res) < NEWS_PAGE:
                    break
                start += NEWS_PAGE
                time.sleep(0.2)
            path.write_text(json.dumps(items))
        for it in items:
            rows.append({"ticker": tk, "company": name, "id": it.get("disclosureId"),
                         "released": it.get("releaseTime"), "category": it.get("cnsCategory"),
                         "language": it.get("language"), "headline": it.get("headline"),
                         "url": it.get("messageUrl")})
    df = pd.DataFrame(rows)
    df["released"] = pd.to_datetime(df["released"], errors="coerce")
    return df.drop_duplicates("id").sort_values("released").reset_index(drop=True)


# Categorias de comunicado (nomenclatura da Nasdaq desde 2016) usadas pela v4.
# Balancos: trimestrais, semestral e o de fim de ano (bokslutskommuniké). O relatorio anual sai
# meses depois do balanco de fim de ano e nao traz numero novo: fica de fora.
# "Quarterly report" e o nome usado ate 2016; "Inside information" existe desde 2017.
EARNINGS_CATEGORIES = ("Interim report (Q1 and Q3)", "Half Year financial report", "Financial Statement Release",
                       "Quarterly report")
INSIDE_CATEGORY = "Inside information"
TONE_START = "2015-07-01"   # um ano antes do periodo de teste, para o percentil movel ter historia
BODIES_DIR = RAW / "bodies"
TONE_PATH = OUT / "news_tone.parquet"
FINBERT = "ProsusAI/finbert"
FINBERT_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"   # versao usada na avaliacao da v4
LEAD_CHARS = 2000       # trecho inicial do texto analisado (resumo, citacao do CEO, numeros principais)
MAX_SENTENCES = 12


def _clean_body(html: str, headline: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    txt = " ".join(soup.get_text(" ").split())
    # remove o cabecalho da pagina (botoes de compartilhar, data, categoria) ate o titulo repetido
    k = txt.find(headline, len(headline)) if headline else -1
    if k > 0:
        txt = txt[k + len(headline):]
    m = re.search(r"Published:\s*\S+\s+\S+\s+\S+", txt)
    if m and m.start() < 400:
        txt = txt[m.end():]
    return txt.strip()


def fetch_bodies(news: pd.DataFrame, workers: int = 10) -> Dict[int, str]:
    """Texto dos comunicados (cache em disco por id). So comunicados em ingles."""
    from concurrent.futures import ThreadPoolExecutor
    BODIES_DIR.mkdir(parents=True, exist_ok=True)

    def one(row) -> Tuple[int, str]:
        path = BODIES_DIR / f"{int(row.id)}.txt"
        if path.exists():
            return int(row.id), path.read_text(encoding="utf-8")
        try:
            html = _get(row.url, timeout=60, tries=4).decode("utf-8", "ignore")
            txt = _clean_body(html, str(row.headline))
        except Exception:
            return int(row.id), ""
        path.write_text(txt, encoding="utf-8")
        return int(row.id), txt

    rows = list(news.itertuples())
    out: Dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (k, v) in enumerate(ex.map(one, rows)):
            out[k] = v
            if (i + 1) % 500 == 0:
                print(f"  textos: {i + 1}/{len(rows)}", flush=True)
    return out


def split_sentences(text: str) -> List[str]:
    lead = text[:LEAD_CHARS]
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“])", lead)
    return [p.strip() for p in parts if len(p.strip()) >= 25][:MAX_SENTENCES]


def tone_documents(news: pd.DataFrame, bodies: Dict[int, str], batch_size: int = 32) -> pd.DataFrame:
    """Tom FinBERT de cada comunicado: media de P(positivo) - P(negativo) sobre o titulo e as frases iniciais."""
    from transformers import pipeline
    clf = pipeline("text-classification", model=FINBERT, revision=FINBERT_REVISION, top_k=None, truncation=True,
                   max_length=256)
    texts, owner = [], []
    for row in news.itertuples():
        sents = [str(row.headline)] + split_sentences(bodies.get(int(row.id), ""))
        texts += sents
        owner += [int(row.id)] * len(sents)
    scores = clf(texts, batch_size=batch_size)
    s = pd.DataFrame({"id": owner, "pos": [next(d["score"] for d in o if d["label"] == "positive") for o in scores],
                      "neg": [next(d["score"] for d in o if d["label"] == "negative") for o in scores]})
    s["net"] = s["pos"] - s["neg"]
    agg = s.groupby("id").agg(tone=("net", "mean"), n_sent=("net", "size"))
    out = news.set_index("id").join(agg, how="inner").reset_index()
    out["body_chars"] = out["id"].map(lambda i: len(bodies.get(int(i), "")))
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _load_insiders_raw() -> pd.DataFrame:
    if INSIDERS_PATH.exists():
        return pd.read_parquet(INSIDERS_PATH)
    return pd.DataFrame(columns=["isin", "lei", "issuer"])


def coverage() -> Dict:
    out: Dict = {}
    n_univ = len(universe())
    if SHORTS_PATH.exists():
        s = pd.read_parquet(SHORTS_PATH)
        m = s.dropna(subset=["ticker"])
        out["shorts"] = {"rows": len(s), "rows_universe": len(m), "tickers": int(m["ticker"].nunique()),
                         "of": n_univ, "first": str(s["position_date"].min().date()),
                         "last": str(s["position_date"].max().date()),
                         "by_year": m.groupby(m["position_date"].dt.year).size().to_dict()}
    if INSIDERS_PATH.exists():
        i = pd.read_parquet(INSIDERS_PATH)
        m = i.dropna(subset=["ticker"])
        out["insiders"] = {"rows": len(i), "rows_universe": len(m), "tickers": int(m["ticker"].nunique()),
                           "of": n_univ, "first": str(i["published"].min()), "last": str(i["published"].max()),
                           "nature": m["nature"].value_counts().head(12).to_dict(),
                           "by_year": m.groupby(m["published"].dt.year).size().to_dict()}
    if NEWS_PATH.exists():
        n = pd.read_parquet(NEWS_PATH)
        out["news"] = {"rows": len(n), "tickers": int(n["ticker"].nunique()), "of": n_univ,
                       "first": str(n["released"].min()), "last": str(n["released"].max()),
                       "language": n["language"].value_counts().to_dict(),
                       "category": n["category"].value_counts().head(15).to_dict(),
                       "by_year": n.groupby(n["released"].dt.year).size().to_dict()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["shorts", "insiders", "news", "tone", "coverage"])
    ap.add_argument("--end", default=None, help="ultimo dia de publicacao (padrao: hoje)")
    ap.add_argument("--rediscover", action="store_true", help="refaz o mapa empresa->ticker dos comunicados")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if a.cmd == "insiders":
        df = download_insiders(end=a.end)
        df = map_insiders(df, IssuerMap(df))
        df.to_parquet(INSIDERS_PATH, index=False)
        print(f"insiders: {len(df)} linhas, {df['ticker'].notna().sum()} no universo -> {INSIDERS_PATH}")
    elif a.cmd == "shorts":
        df = map_shorts(download_shorts(), IssuerMap(_load_insiders_raw()))
        df.to_parquet(SHORTS_PATH, index=False)
        print(f"posicoes vendidas: {len(df)} linhas, {df['ticker'].notna().sum()} no universo -> {SHORTS_PATH}")
    elif a.cmd == "news":
        imap = IssuerMap(_load_insiders_raw())
        if a.rediscover or not NEWS_MAP_PATH.exists():
            comp = discover_news_companies(imap)
            NEWS_MAP_PATH.write_text(json.dumps(dict(sorted(comp.items())), indent=1, ensure_ascii=False))
        comp = json.loads(NEWS_MAP_PATH.read_text())
        df = download_news(comp)
        df.to_parquet(NEWS_PATH, index=False)
        print(f"comunicados: {len(df)} de {df['ticker'].nunique()} tickers -> {NEWS_PATH}")
    elif a.cmd == "tone":
        news = pd.read_parquet(NEWS_PATH)
        sel = news[(news["language"] == "en") & (news["released"] >= pd.Timestamp(TONE_START))
                   & news["category"].isin(EARNINGS_CATEGORIES + (INSIDE_CATEGORY,))]
        print(f"tom: {len(sel)} comunicados em ingles (balancos + informacao privilegiada)", flush=True)
        bodies = fetch_bodies(sel)
        tone = tone_documents(sel, bodies)
        tone.to_parquet(TONE_PATH, index=False)
        print(f"tom: {len(tone)} documentos -> {TONE_PATH}")
    else:
        print(json.dumps(coverage(), indent=1, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
