#!/usr/bin/env python3
"""
Revisoes de analistas e dicas da imprensa sueca (estrategia v5, docs/adr/0009).

Fonte: Börsvärlden (borsvarlden.com), que publica cada mudanca de
recomendacao e de preco-alvo das corretoras ("SEB hojer riktkursen for Volvo
till 300 kronor (280) och upprepar kop"). O sitemap publico lista todos os
artigos; os de analistas sao escolhidos pelo endereco, baixados com cache e
lidos com a hora de publicacao (datePublished). O sinal so vale a partir da
abertura do primeiro pregao depois do dia da publicacao, como na v4.

    python analyst_se.py fetch      # sitemaps + artigos de analistas (cache em output/analyst/raw)
    python analyst_se.py parse      # acoes por artigo -> output/analyst/actions.parquet
    python analyst_se.py coverage   # contagens por ano, tipo e ticker
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import altdata_se as alt  # noqa: E402

OUT = REPO_ROOT / "output" / "analyst"
RAW = OUT / "raw"
ARTICLES_PATH = OUT / "articles.parquet"
ACTIONS_PATH = OUT / "actions.parquet"
SITEMAP_INDEX = "https://borsvarlden.com/sitemap.xml"
WORKERS = 2

# Enderecos de artigos de analistas e dicas (escolhidos pelo texto do endereco).
CANDIDATE = re.compile(
    r"riktkurs|motiverat-varde|koprad|saljrad|kop-rad|salj-rad|uppgradera|nedgradera|inleder-bevakning|ny-bevakning|"
    r"initierar-bevakning|(hojer|sanker|hojs|sanks|lyfter|sankte|hojde)-.*-till-(kop|salj|behall|oka|minska|neutral|"
    r"overvikt|undervikt|outperform|underperform|marknadsvikt|jamvikt)|rekar|aktierad|kopstampel|saljstampel|"
    r"kopsignal-fran|analytiker", re.I)


# --------------------------------------------------------------------------
# Coleta
# --------------------------------------------------------------------------
def sitemap_urls() -> pd.DataFrame:
    import requests
    idx = requests.get(SITEMAP_INDEX, timeout=60, headers={"User-Agent": alt.USER_AGENT}).text
    maps = re.findall(r"<loc>([^<]+)</loc>", idx)
    rows = []
    for m in maps:
        t = alt._get(m, timeout=120).decode("utf-8", "ignore")
        rows += re.findall(r"<loc>([^<]+)</loc>\s*<lastmod>([^<]+)</lastmod>", t)
        time.sleep(0.3)
    return pd.DataFrame(rows, columns=["url", "lastmod"]).drop_duplicates("url")


EXCLUDE_SLUG = re.compile(r"motiverat-varde|affarsvarlden|borsveckan|aktiespararna|analytikerkonsensus|dagens-industri|"
                          r"stockpicker|placera")


def _slug(x: str) -> str:
    import unicodedata
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower().replace("&", "")
    return re.sub(r"[^a-z0-9]+", "-", x).strip("-")


def largecap_candidates(sm: pd.DataFrame) -> pd.DataFrame:
    """Artigos de corretoras que citam no endereco uma empresa do universo (pesquisa encomendada e
    dicas de revista ficam de fora: nao entram em nenhuma hipotese). O site limita a taxa de acesso
    (HTTP 429), por isso a lista e restrita ao que o teste usa."""
    import analyst_parse
    names = {_slug(n) for n, _, _ in analyst_parse.build_aliases()} | {"hm", "h-m", "saab", "volvo-cars"}
    names = {n for n in names if len(n) >= 2}
    rx = re.compile(r"(?:^|-)(?:" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")(?:-|$)")
    slugs = sm["url"].str.split("/artiklar/").str[-1]
    ok = sm["url"].str.contains("/artiklar/") & slugs.map(lambda s: bool(CANDIDATE.search(s)))
    ok &= slugs.map(lambda s: bool(rx.search(s)) and not EXCLUDE_SLUG.search(s))
    return sm[ok]


def _key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def _extract(url: str, html: str) -> Dict:
    from bs4 import BeautifulSoup
    s = BeautifulSoup(html, "lxml")
    published, headline = None, None
    for sc in s.find_all("script", type="application/ld+json"):
        try:
            d = json.loads(sc.string or "")
        except Exception:
            continue
        for x in (d if isinstance(d, list) else [d]):
            if isinstance(x, dict) and x.get("datePublished"):
                published, headline = x.get("datePublished"), x.get("headline")
                break
        if published:
            break
    lead = s.find("div", class_="lead-wrapper")
    body = s.find("div", class_="bv-content-left")
    return {"url": url, "published": published, "headline": headline or (s.title.string if s.title else ""),
            "lead": " ".join(lead.get_text(" ").split()) if lead else "",
            "body": " ".join(body.get_text(" ").split())[:4000] if body else ""}


def fetch_articles(urls: List[str]) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)

    def one(url: str) -> Optional[Dict]:
        path = RAW / f"{_key(url)}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        try:
            html = alt._get(url, timeout=60, tries=4).decode("utf-8", "ignore")
        except Exception:
            return None
        rec = _extract(url, html)
        path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        time.sleep(0.25)
        return rec

    out = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for k, rec in enumerate(ex.map(one, urls)):
            if rec:
                out.append(rec)
            if (k + 1) % 1000 == 0:
                print(f"  artigos: {k + 1}/{len(urls)}", flush=True)
    df = pd.DataFrame(out)
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True).dt.tz_convert("Europe/Stockholm").dt.tz_localize(None)
    return df


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "parse", "coverage"])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "fetch":
        smp = OUT / "sitemap.parquet"
        sm = pd.read_parquet(smp) if smp.exists() else sitemap_urls()
        sm.to_parquet(smp, index=False)
        cand = largecap_candidates(sm)
        print(f"sitemap: {len(sm)} enderecos, {len(cand)} candidatos", flush=True)
        df = fetch_articles(cand["url"].tolist())
        df.to_parquet(ARTICLES_PATH, index=False)
        print(f"artigos: {len(df)} -> {ARTICLES_PATH}")
    elif a.cmd == "parse":
        import analyst_parse
        acts = analyst_parse.actions(pd.read_parquet(ARTICLES_PATH))
        acts.to_parquet(ACTIONS_PATH, index=False)
        print(f"acoes: {len(acts)} -> {ACTIONS_PATH}")
    else:
        acts = pd.read_parquet(ACTIONS_PATH)
        print(acts.groupby([acts["published"].dt.year, "kind"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
