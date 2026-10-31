"""
Leitura dos artigos de analistas do Börsvärlden (estrategia v5).

Cada artigo vira, por empresa do universo citada, contagens de:
- alta/baixa de recomendacao ("hojer Volvo till kop", "sanker till behall"),
- preco-alvo novo vs. antigo ("hojer riktkursen till 300 kronor (280)" ou "fran 280"),
- inicio de cobertura ("inleder bevakning ... med kop"),
e a origem: corretora, pesquisa encomendada ("motiverat varde", paga pela
empresa) ou dica de revista (Affarsvarlden, Borsveckan, Aktiespararna...).

O texto e dividido em oracoes; cada acao e atribuida a empresa citada na
propria oracao ou, se nenhuma, a ultima citada antes (o titulo abre o artigo).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

import altdata_se as alt

# --------------------------------------------------------------------------
# Corretoras, revistas e verbos
# --------------------------------------------------------------------------
BROKERS = [
    "DNB Carnegie", "Carnegie", "SEB", "Handelsbanken", "SHB", "Swedbank", "Nordea", "Danske Bank", "Danske",
    "ABG Sundal Collier", "ABG", "Pareto", "Kepler Cheuvreux", "Kepler", "Jefferies", "Morgan Stanley",
    "Goldman Sachs", "JP Morgan", "JPMorgan", "J.P. Morgan", "Citi", "Citigroup", "UBS", "Barclays", "Bernstein",
    "Deutsche Bank", "HSBC", "BNP Paribas Exane", "BNP Paribas", "Exane", "Berenberg", "Redeye", "Erik Penser",
    "Penser", "DNB Markets", "DNB", "Nordnet", "Bank of America", "BofA", "RBC", "Credit Suisse", "Société Générale",
    "Societe Generale", "SocGen", "Oddo", "Stifel", "Bryan Garnier", "Kempen", "ING", "Inderes", "Arctic",
    "SB1 Markets", "SB1", "Mangold", "Remium", "Evli", "Nykredit", "Sparebank 1", "Clarksons", "Fearnley",
    "Mizuhu", "Mizuho", "Wells Fargo", "Bernstein Research", "Investec", "Liberum", "Panmure", "Peel Hunt",
    "Jyske", "Svenska Handelsbanken", "Skandinaviska Enskilda Banken", "Pareto Securities", "Kepler Cheuvreux",
]
MEDIA = ["Affärsvärlden", "Börsveckan", "Aktiespararna", "Dagens industri", "Dagens Industri", "Privata Affärer",
         "Placera", "Stockpicker", "Analysguiden", "Veckans Affärer", "Realtid"]
POS = r"köp|köpa|öka|övervikt|outperform|buy|add|accumulate|ackumulera|köprek\w*"
NEU = r"neutral|behåll|marknadsvikt|jämvikt|hold|equal[- ]weight|market perform|sector perform|reducera"
NEG = r"sälj|sälja|minska|undervikt|underperform|sell|reduce|säljrek\w*"
RATING = rf"(?:{POS}|{NEU}|{NEG})"
NUM = r"\d[\d\s]*(?:[.,]\d+)?"
CUR = r"(?:(?:svenska |danska |norska )?kronor|kr|sek|euro|eur|dollar|usd|pence|pund|gbp|chf|franc|nok|dkk)"

RX_TARGET = [
    re.compile(rf"till\s+(?P<new>{NUM})\s*{CUR}?\s*\(\s*(?P<old>{NUM})\s*\)", re.I),
    re.compile(rf"till\s+(?P<new>{NUM})\s*{CUR}?\s+från\s+(?P<old>{NUM})", re.I),
    re.compile(rf"från\s+(?P<old>{NUM})\s*{CUR}?\s+till\s+(?P<new>{NUM})", re.I),
]
RX_UP = re.compile(rf"\b(höjer|uppgraderar|lyfter|höjs|uppgraderas|höjde|uppgraderade|höjt|uppgraderat)\b[^.;:]{{0,70}}?\btill\s+(?:en\s+)?({RATING})\b", re.I)
RX_DOWN = re.compile(rf"\b(sänker|nedgraderar|sänks|nedgraderas|sänkte|nedgraderade|sänkt|nedgraderat)\b[^.;:]{{0,70}}?\btill\s+(?:en\s+)?({RATING})\b", re.I)
RX_INIT = re.compile(rf"\b(inleder|initierar|startar|påbörjar|tar upp)\b[^.;:]{{0,40}}?\bbevakning\b[^.;:]{{0,80}}?\b({RATING})\b", re.I)
RX_T_UP_VERB = re.compile(r"\b(höjer|höjt|höjd|höjda|höjs|lyfter)\s+(?:sin\s+|sina\s+)?riktkurs", re.I)
RX_T_DOWN_VERB = re.compile(r"\b(sänker|sänkt|sänkta|sänks|kapar|skär ned)\s+(?:sin\s+|sina\s+)?riktkurs", re.I)
RX_COMMISSIONED = re.compile(r"motiverat\s+värde|motiverade\s+värde|uppdragsanalys|betald analys", re.I)
RX_MEDIA_BUY = re.compile(r"köpråd|köpläge|ger köp|köpvärd|köpstämpel|köpsignal", re.I)
RX_MEDIA_SELL = re.compile(r"säljråd|ger sälj|säljstämpel|slopar köpråd|slopar köprådet|avråder", re.I)
VERB_AFTER_BROKER = re.compile(r"^\s*(?:Markets\s+|Securities\s+|Research\s+)?(höjer|sänker|inleder|initierar|upprepar|ger|ser|lyfter|"
                               r"uppgraderar|nedgraderar|justerar|tror|spår|räknar|landar|slopar|behåller|har|bedömer|"
                               r"startar|påbörjar|tar|sätter)\b", re.I)


def _num(s: str) -> float:
    return float(re.sub(r"\s", "", s).replace(",", "."))


def _rating_sign(r: str) -> int:
    r = r.lower()
    if re.fullmatch(POS, r):
        return 1
    if re.fullmatch(NEG, r):
        return -1
    return 0


# --------------------------------------------------------------------------
# Empresas do universo
# --------------------------------------------------------------------------
MANUAL_ALIASES = {
    "H&M": "HM-B.ST", "Hennes & Mauritz": "HM-B.ST", "Sobi": "SOBI.ST", "Swedish Orphan Biovitrum": "SOBI.ST",
    "Volvo Cars": "VOLCAR-B.ST", "Volvo Car": "VOLCAR-B.ST", "Volvokoncernen": "VOLV-B.ST", "AB Volvo": "VOLV-B.ST",
    "Handelsbanken": "SHB-A.ST", "Svenska Handelsbanken": "SHB-A.ST", "SHB": "SHB-A.ST",
    "Skandinaviska Enskilda Banken": "SEB-A.ST", "SEB": "SEB-A.ST", "Balder": "BALD-B.ST",
    "Industrivärden": "INDU-C.ST", "Lundbergs": "LUND-B.ST", "Lundbergföretagen": "LUND-B.ST",
    "Latour": "LATO-B.ST", "Investor": "INVE-B.ST", "Kinnevik": "KINV-B.ST", "Ericsson": "ERIC-B.ST",
    "Electrolux Professional": "EPRO-B.ST", "Electrolux": "ELUX-B.ST", "Nordnet": "SAVE.ST", "Avanza": "AZA.ST",
    "Autoliv": "ALIV-SDB.ST", "Atlas Copco": "ATCO-A.ST", "Epiroc": "EPI-A.ST", "Essity": "ESSITY-B.ST",
    "SCA": "SCA-B.ST", "Svenska Cellulosa": "SCA-B.ST", "Stora Enso": "STE-R.ST", "Nordea": "NDA-SE.ST",
    "AstraZeneca": "AZN.ST", "Astra Zeneca": "AZN.ST", "Telia": "TELIA.ST", "Tele2": "TEL2-B.ST", "Sinch": "SINCH.ST",
    "Embracer": "EMBRAC-B.ST", "Traton": "8TRA.ST", "Lundin Mining": "LUMI.ST", "Lundin Gold": "LUG.ST",
    "International Petroleum": "IPCO.ST", "IPC": "IPCO.ST", "Assa Abloy": "ASSA-B.ST", "Clas Ohlson": "CLAS-B.ST",
    "Beijer Ref": "BEIJ-B.ST", "Corem": "CORE-B.ST", "Sagax": "SAGA-B.ST", "Wihlborgs": "WIHL.ST",
    "Hufvudstaden": "HUFV-A.ST", "Bonesupport": "BONEX.ST", "Scandic": "SHOT.ST", "Thule": "THULE.ST",
    "Vitec": "VIT-B.ST", "Storskogen": "STOR-B.ST", "Paradox": "PDX.ST", "Truecaller": "TRUE-B.ST",
    "Munters": "MTRS.ST", "Lindab": "LIAB.ST", "Alimak": "ALIG.ST", "Asker": "ASKER.ST", "Asmodee": "ASMDEE-B.ST",
    "Addnode": "ANOD-B.ST", "AddLife": "ALIF-B.ST", "Addtech": "ADDT-B.ST", "Lagercrantz": "LAGR-B.ST",
    "Bravida": "BRAV.ST", "Dometic": "DOM.ST", "Hemnet": "HEM.ST", "Medicover": "MCOV-B.ST", "Bure": "BURE.ST",
    "Gränges": "GRNG.ST", "Hacksaw": "HACK.ST", "Yubico": "YUBICO.ST", "Röko": "ROKO-B.ST", "Pandox": "PNDX-B.ST",
    "Octave": "OCTV-SDB.ST", "Public Property Invest": "PUBLI.ST", "NOBA": "NOBA.ST", "Nobia": None,
    "Saab": "SAAB-B.ST", "Sectra": "SECT-B.ST", "Hexpol": "HPOL-B.ST", "Nibe": "NIBE-B.ST", "Peab": "PEAB-B.ST",
    "Invisio": "IVSO.ST", "Sweco": "SWEC-B.ST", "Elekta": "EKTA-B.ST", "Getinge": "GETI-B.ST", "Hexagon": "HEXA-B.ST",
}
# Bancos e corretoras que tambem estao no universo: so contam como empresa analisada quando vem
# depois de uma preposicao ou verbo de acao ("hojer riktkursen for SEB", "Pareto sanker Swedbank").
BANK_ALIASES = {"SEB", "SHB", "Handelsbanken", "Svenska Handelsbanken", "Skandinaviska Enskilda Banken", "Swedbank",
                "Nordea", "Avanza", "Nordnet", "Carnegie"}
SUBJECT_AFTER = re.compile(r"^\s+(riktkurs\w*|sin|sina|sitt|rekommendation\w*|motiverat|prognos\w*|också|även|dessutom|"
                           r"upprepar|bedömer|skriver|ser|räknar|tror|har|anser)\b", re.I)
OBJECT_BEFORE = re.compile(r"(för|på|av|om|i|hos|höjer|sänker|uppgraderar|nedgraderar|lyfter|bevakning av)\s+$", re.I)
CASE_SENSITIVE = {"JM", "AQ", "IPC", "SCA", "SEB", "SHB", "SKF", "ABB", "EQT", "NCC", "HMS", "AAK", "EVO"}


def build_aliases() -> List[Tuple[str, str, re.Pattern]]:
    al: Dict[str, Optional[str]] = {}
    for t in alt.universe():
        base = re.sub(r"\s+(A|B|C|D|R|SDB)$", "", t["name"]).strip()
        al.setdefault(base, t["yf_symbol"])
    al.update(MANUAL_ALIASES)
    out = []
    for name, tk in al.items():
        if not tk or len(name) < 2:
            continue
        flags = 0 if name in CASE_SENSITIVE else re.I
        rx = re.compile(rf"(?<![\wÅÄÖåäö-]){re.escape(name)}(?![\wÅÄÖåäö])", flags)
        out.append((name, tk, rx))
    return sorted(out, key=lambda x: -len(x[0]))   # nome mais longo primeiro (Volvo Cars antes de Volvo)


BROKER_RX = [(b, re.compile(rf"(?<![\w-]){re.escape(b)}(?![\w])")) for b in sorted(BROKERS, key=len, reverse=True)]
MEDIA_RX = re.compile("|".join(re.escape(m) for m in MEDIA))


def companies_in(text: str, aliases) -> List[Tuple[int, str]]:
    """(posicao, ticker) das empresas citadas, sem as que aparecem no papel de corretora (antes do verbo)."""
    taken: List[Tuple[int, int]] = []
    found: List[Tuple[int, str]] = []
    for name, tk, rx in aliases:
        for m in rx.finditer(text):
            if any(a <= m.start() < b for a, b in taken):
                continue
            if VERB_AFTER_BROKER.match(text[m.end():]) and any(name.lower() == b.lower() for b in BROKERS):
                taken.append((m.start(), m.end()))
                continue
            if name in BANK_ALIASES and (not OBJECT_BEFORE.search(text[max(0, m.start() - 40):m.start()])
                                         or SUBJECT_AFTER.match(text[m.end():])):
                taken.append((m.start(), m.end()))
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), tk))
    return sorted(found)


def broker_in(text: str) -> str:
    best = None
    for b, rx in BROKER_RX:
        m = rx.search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), b)
    if best is None:
        return "?"
    b = best[1]
    return {"SHB": "Handelsbanken", "Svenska Handelsbanken": "Handelsbanken", "Skandinaviska Enskilda Banken": "SEB",
            "Carnegie": "DNB Carnegie", "DNB": "DNB Carnegie", "DNB Markets": "DNB Carnegie", "JPMorgan": "JP Morgan",
            "J.P. Morgan": "JP Morgan", "Citigroup": "Citi", "BNP Paribas": "BNP Paribas Exane", "Exane": "BNP Paribas Exane",
            "Kepler": "Kepler Cheuvreux", "Penser": "Erik Penser", "SB1": "SB1 Markets", "Danske": "Danske Bank",
            "ABG": "ABG Sundal Collier", "Pareto Securities": "Pareto"}.get(b, b)


def _clean_body(body: str) -> str:
    for stop in ("Finwire / Börsvärlden", "RELATERAT FRÅN", "Publicerad "):
        k = body.find(stop)
        if k > 0:
            body = body[:k]
    return body


def parse_article(rec: Dict, aliases) -> List[Dict]:
    import html
    headline = html.unescape(str(rec.get("headline") or ""))
    lead = html.unescape(str(rec.get("lead") or ""))
    text = headline + ". " + lead + ". " + _clean_body(html.unescape(str(rec.get("body") or "")))
    source = "commissioned" if RX_COMMISSIONED.search(headline + " " + lead) else \
        ("media" if MEDIA_RX.search(headline) else "broker")
    clauses = re.split(r"(?<=[.;!?])\s+|,\s+|\s+–\s+|\s+-\s+|\s+och\s+(?=(?:höjer|sänker|upprepar|uppgraderar|nedgraderar|"
                       r"inleder|lyfter)\b)", text)
    current: Optional[str] = None
    acts: Dict[str, Dict] = {}
    for cl in clauses:
        cos = companies_in(cl, ALIASES if aliases is None else aliases)
        ticks = {tk for _, tk in cos}
        if len(ticks) == 1:
            current = next(iter(ticks))
        elif len(ticks) > 1:
            current = None
        if current is None:
            continue
        a = acts.setdefault(current, {"up": set(), "down": set(), "t_up": set(), "t_down": set(), "init_pos": set(),
                                      "init_neg": set(), "media_buy": 0, "media_sell": 0})
        br = broker_in(cl)
        for m in RX_UP.finditer(cl):
            a["up"].add(br)
        for m in RX_DOWN.finditer(cl):
            a["down"].add(br)
        for m in RX_INIT.finditer(cl):
            s = _rating_sign(m.group(2))
            if s > 0:
                a["init_pos"].add(br)
            elif s < 0:
                a["init_neg"].add(br)
        found_num = False
        if re.search(r"riktkurs|motiverat|värde|kronor|till\s+\d", cl, re.I):
            for rx in RX_TARGET:
                for m in rx.finditer(cl):
                    try:
                        new, old = _num(m.group("new")), _num(m.group("old"))
                    except ValueError:
                        continue
                    found_num = True
                    if new > old:
                        a["t_up"].add(br)
                    elif new < old:
                        a["t_down"].add(br)
        if not found_num:  # "hojer riktkursen" sem numeros: o verbo da a direcao
            if RX_T_UP_VERB.search(cl) and not RX_T_DOWN_VERB.search(cl):
                a["t_up"].add(br)
            elif RX_T_DOWN_VERB.search(cl) and not RX_T_UP_VERB.search(cl):
                a["t_down"].add(br)
        if source == "media":
            a["media_buy"] |= int(bool(RX_MEDIA_BUY.search(cl)))
            a["media_sell"] |= int(bool(RX_MEDIA_SELL.search(cl)))
    out = []
    for tk, a in acts.items():
        # titulo-resumo ("Hojda riktkurser for X") repete o que o texto atribui a corretoras nomeadas
        for k in ("up", "down", "t_up", "t_down", "init_pos", "init_neg"):
            if len(a[k] - {"?"}) > 0:
                a[k].discard("?")
        row = {"url": rec["url"], "published": rec["published"], "headline": headline, "ticker": tk, "source": source,
               "n_up": len(a["up"]), "n_down": len(a["down"]), "n_t_up": len(a["t_up"]), "n_t_down": len(a["t_down"]),
               "n_init_pos": len(a["init_pos"]), "n_init_neg": len(a["init_neg"]),
               "media_buy": a["media_buy"], "media_sell": a["media_sell"]}
        if any(row[k] for k in ("n_up", "n_down", "n_t_up", "n_t_down", "n_init_pos", "n_init_neg", "media_buy", "media_sell")):
            out.append(row)
    return out


ALIASES = None


def actions(articles: pd.DataFrame) -> pd.DataFrame:
    global ALIASES
    ALIASES = build_aliases()
    rows = []
    for rec in articles.to_dict("records"):
        rows += parse_article(rec, ALIASES)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["kind"] = df.apply(_kind, axis=1)
    return df.sort_values("published").reset_index(drop=True)


def _kind(r) -> str:
    if r["source"] == "commissioned":
        return "commissioned"
    if r["source"] == "media":
        return "media_buy" if r["media_buy"] and not r["media_sell"] else ("media_sell" if r["media_sell"] else "media_other")
    up = r["n_up"] + r["n_init_pos"]
    down = r["n_down"] + r["n_init_neg"]
    if up > down:
        return "upgrade"
    if down > up:
        return "downgrade"
    if r["n_t_up"] > r["n_t_down"]:
        return "target_up"
    if r["n_t_down"] > r["n_t_up"]:
        return "target_down"
    return "mixed"
