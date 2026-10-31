#!/usr/bin/env python3
"""
Estrategia v4 do mercado sueco: eventos de informacao publica (docs/adr/0008).

Quatro hipoteses, cada uma tirada da literatura e congelada em
`data/strategy_se_v4.json` antes de qualquer dado alternativo ser ligado a
retornos:

- H1 compras de insiders (registro de insiders do FI): compras em bolsa de
  pessoas em posicao de lideranca antecedem alta (Lakonishok & Lee 2001;
  Cohen, Malloy & Pomorski 2012).
- H2 posicoes vendidas altas (registro de posicoes vendidas do FI): as acoes
  mais vendidas a descoberto ficam para tras (Desai et al. 2002; Boehmer, Jones &
  Zhang 2008; Jones, Reed & Waller 2016 para as divulgacoes europeias).
- H3 tom dos comunicados (Nasdaq Nordic + FinBERT): balancos e comunicados de
  informacao privilegiada de tom positivo antecedem alta (Tetlock, Saar-Tsechansky
  & Macskassy 2008; Heston & Sinha 2017).
- H4 deriva pos-balanco: a reacao de 3 dias ao balanco (EAR) continua nas semanas
  seguintes (Bernard & Thomas 1989; Brandt et al. 2008).

Cada evento e negociado na abertura do primeiro pregao em que a informacao ja
era publica (altdata_se.next_session_after) e medido contra acoes do mesmo
quintil de retorno de 63 dias (controle de reversao, momentum e do vies de
sobrevivencia do universo atual). Os testes e o criterio 1 estao no ADR 0008.

    python events_se.py dry-run     # contagens de eventos, sem nenhum retorno
    python events_se.py evaluate    # avaliacao unica (escreve reports/se_v4)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import altdata_se as alt  # noqa: E402
import market_se  # noqa: E402

SPEC_PATH = REPO_ROOT / "data" / "strategy_se_v4.json"
REPORT_DIR = REPO_ROOT / "reports" / "se_v4"
FAMILIES = ("H1", "H2", "H3", "H4")
LONG_FAMILIES = ("H1", "H3", "H4")


def spec() -> Dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Mercado: precos, elegibilidade e quintis de controle
# --------------------------------------------------------------------------
@dataclass
class Mkt:
    days: pd.DatetimeIndex
    tickers: List[str]
    adj: pd.DataFrame            # fechamento ajustado por dividendos
    opx: pd.DataFrame            # abertura ajustada (fechamento quando falta)
    eligible: pd.DataFrame       # liquidez e historico, conhecido no fechamento do dia
    quint: pd.DataFrame          # quintil (0..4) do retorno de 63 dias no fechamento do dia
    ew_ret: pd.Series            # retorno diario do universo elegivel, peso igual
    bench: pd.Series             # ^OMXSBGI
    isin: Dict[str, str] = field(default_factory=dict)

    def pos(self, d) -> int:
        return int(self.days.get_loc(pd.Timestamp(d)))


def market_from_prices(px: Dict[str, pd.DataFrame]) -> Mkt:
    import selective_se as v3
    d = v3.build(px)
    adj = d.adj
    opx = d.adj_open.where(d.adj_open > 0).reindex_like(adj).fillna(adj)
    rank = d.stock["r63"] + 0.5                                  # percentil em (0, 1]
    quint = np.floor(rank.clip(upper=0.999999) * 5).where(d.eligible)
    isin = {t["yf_symbol"]: t["isin"] for t in alt.universe()}
    bench = d.bench if d.bench is not None else pd.Series(np.nan, index=adj.index)
    return Mkt(d.days, list(d.tickers), adj, opx, d.eligible.fillna(False), quint, d.mkt_ret, bench, isin)


def load_market() -> Mkt:
    import prices_se
    return market_from_prices(prices_se.load())


# --------------------------------------------------------------------------
# Eventos (so informacao publica ate o fechamento anterior a entrada)
# --------------------------------------------------------------------------
EVENT_COLS = ["family", "ticker", "entry", "magnitude", "info"]


def _cooldown(ev: pd.DataFrame, days: pd.DatetimeIndex, sessions: int) -> pd.DataFrame:
    """Um evento por ticker a cada `sessions` pregoes: o primeiro vale, os seguintes sao repeticao."""
    if ev.empty:
        return ev
    ev = ev.sort_values(["ticker", "entry"]).copy()
    pos = days.get_indexer(pd.DatetimeIndex(ev["entry"]))
    keep = np.zeros(len(ev), dtype=bool)
    last: Dict[str, int] = {}
    for k, (t, p) in enumerate(zip(ev["ticker"].values, pos)):
        if t not in last or p - last[t] >= sessions:
            keep[k] = True
            last[t] = p
    return ev[keep].sort_values(["entry", "ticker"]).reset_index(drop=True)


def insider_events(ins: pd.DataFrame, mkt: Mkt, s: Dict) -> pd.DataFrame:
    """H1: compras em bolsa de pessoas em posicao de lideranca, somadas por ticker e dia de publicacao."""
    c = s["signals"]["H1"]
    x = ins.dropna(subset=["ticker"]).copy()
    x = x[x["ticker"].isin(mkt.tickers)]
    # O FI so passou a preencher o tipo de instrumento em 2018: tipo vazio vale como acao
    # quando o ISIN e de acao (aparece como "Aktie" em outra linha ou e o ISIN do universo).
    itype = x["instrument_type"].fillna("").str.strip()
    share_isins = set(ins.loc[ins["instrument_type"].isin(c["instrument_type"]), "isin"]) | set(mkt.isin.values())
    type_ok = itype.isin(c["instrument_type"])
    if c.get("empty_instrument_type_as_share", False):
        type_ok |= (itype == "") & x["isin"].isin(share_isins)
    ok = (x["nature"].isin(c["nature"]) & type_ok
          & (x["status"].isin(c["status"])) & (x["volume_unit"].isin(c["volume_unit"]))
          & (x["currency"].isin(c["currency"])) & ~x["venue"].fillna("").str.strip().str.lower().isin(
              [v.lower() for v in c["exclude_venue"]]))
    if c.get("exclude_share_program", True):
        ok &= ~x["share_program"].fillna(False).astype(bool)
    x = x[ok].copy()
    x["value_sek"] = x["volume"] * x["price"]
    x["entry"] = alt.next_session_after(mkt.days, x["published"], 1)
    x = x.dropna(subset=["entry", "value_sek"])
    g = x.groupby(["ticker", "entry"]).agg(value=("value_sek", "sum"), n=("person", "nunique"),
                                           published=("published", "max")).reset_index()
    g = g[g["value"] >= c["min_value_sek"]]
    # a ultima publicacao do grupo precisa ser anterior a entrada (causalidade)
    g = g[g["published"].dt.normalize() < g["entry"]]
    ev = pd.DataFrame({"family": "H1", "ticker": g["ticker"], "entry": g["entry"], "magnitude": g["value"],
                       "info": g["n"].astype(str) + " pessoa(s)"})
    return _cooldown(ev, mkt.days, c["cooldown_sessions"])


def short_groups(panel: pd.DataFrame, mkt: Mkt, s: Dict) -> pd.DataFrame:
    """H2: a cada semana, as acoes cobertas pelo FI no topo da posicao vendida divulgada.

    `panel` e a posicao conhecida na abertura de cada pregao (altdata_se.short_interest_panel);
    a decisao do fechamento de sexta usa a linha da sexta (conhecida na abertura da sexta),
    e a entrada e na abertura do pregao seguinte.
    """
    import selective_se as v3
    c = s["signals"]["H2"]
    covered = [t for t in mkt.tickers if str(mkt.isin.get(t, "")).startswith(tuple(c["covered_isin_prefix"]))]
    weeks = v3.weekly_dates(mkt.days, s["period"][0], s["period"][1])
    rows = []
    for d in weeks:
        p = mkt.pos(d)
        if p + 1 >= len(mkt.days):
            break
        el = mkt.eligible.loc[d, covered]
        si = panel.loc[d, covered][el.values]
        if si.empty:
            continue
        cut = si.quantile(1 - c["top_fraction"])
        top = si[(si >= cut) & (si >= c["min_pct"])]
        for t, v in top.items():
            rows.append({"family": "H2", "ticker": t, "entry": mkt.days[p + 1], "magnitude": float(v),
                         "info": f"{v:.2f}%"})
    return pd.DataFrame(rows, columns=EVENT_COLS)


def reaction_session(days: pd.DatetimeIndex, released: pd.Series, close_time: str) -> pd.Series:
    """Pregao cujo fechamento ja reflete o comunicado: o proprio dia se saiu antes do fechamento."""
    ts = pd.to_datetime(released)
    hh, mm = (int(v) for v in close_time.split(":"))
    before_close = (ts.dt.hour * 60 + ts.dt.minute) < hh * 60 + mm
    day = ts.dt.normalize()
    i_same = days.searchsorted(day.values, side="left")
    same_is_session = (i_same < len(days)) & (days[np.minimum(i_same, len(days) - 1)] == day.values)
    i_next = days.searchsorted(day.values, side="right")
    idx = np.where(same_is_session & before_close.values, i_same, i_next)
    out = pd.Series(pd.NaT, index=released.index, dtype="datetime64[ns]")
    ok = idx < len(days)
    out[ok] = days[idx[ok]]
    return out


def _trailing_tercile(values: pd.Series, dates: pd.Series, window_days: int, min_obs: int) -> pd.Series:
    """Percentil de cada valor contra os valores dos `window_days` dias anteriores (estritamente antes)."""
    order = np.argsort(dates.values, kind="stable")
    v = values.values[order]
    t = pd.DatetimeIndex(dates.values[order])
    out = np.full(len(v), np.nan)
    lo = 0
    for k in range(len(v)):
        while t[lo] < t[k] - pd.Timedelta(days=window_days):
            lo += 1
        hist = v[lo:k][t[lo:k] < t[k]]
        if len(hist) >= min_obs:
            out[k] = (hist < v[k]).mean()
    res = np.empty(len(v))
    res[order] = out
    return pd.Series(res, index=values.index)


def earnings_events(news: pd.DataFrame, mkt: Mkt, s: Dict) -> pd.DataFrame:
    """H4: reacao de 3 dias ao balanco (fechamento r-2 ao fechamento r+1) contra o universo; entra em r+2."""
    c = s["signals"]["H4"]
    x = news[news["category"].isin(c["categories"]) & news["ticker"].isin(mkt.tickers)].copy()
    x = x.dropna(subset=["released"])
    x["r"] = reaction_session(mkt.days, x["released"], c["close_time"])
    x = x.dropna(subset=["r"]).sort_values("released").drop_duplicates(["ticker", "r"])
    p = mkt.days.get_indexer(pd.DatetimeIndex(x["r"]))
    x = x[(p >= 2) & (p + 2 < len(mkt.days))].copy()
    p = mkt.days.get_indexer(pd.DatetimeIndex(x["r"]))
    adj = mkt.adj
    ew_level = (1 + mkt.ew_ret.fillna(0)).cumprod().values
    col = {t: i for i, t in enumerate(adj.columns)}
    A = adj.values
    ear = []
    for t, i in zip(x["ticker"].values, p):
        j = col[t]
        a0, a1 = A[i - 2, j], A[i + 1, j]
        ear.append(a1 / a0 - 1 - (ew_level[i + 1] / ew_level[i - 2] - 1) if np.isfinite(a0) and np.isfinite(a1) else np.nan)
    x["ear"] = ear
    x["entry"] = mkt.days[p + 2]
    x = x.dropna(subset=["ear"])
    x = _cooldown(x.assign(family="H4", magnitude=x["ear"], info=x["category"]), mkt.days, c["cooldown_sessions"])
    x["pct"] = _trailing_tercile(x["ear"], x["entry"], c["tercile_window_days"], c["min_history_events"])
    x = x[x["pct"] >= 1 - c["top_fraction"]]
    return x[EVENT_COLS].reset_index(drop=True)


def tone_events(tone: pd.DataFrame, mkt: Mkt, s: Dict) -> pd.DataFrame:
    """H3: comunicados de tom no terco superior do ano anterior; entra no pregao seguinte ao dia do comunicado."""
    c = s["signals"]["H3"]
    x = tone[tone["category"].isin(c["categories"]) & tone["ticker"].isin(mkt.tickers)].copy()
    x = x.dropna(subset=["tone", "released"])
    x["entry"] = alt.next_session_after(mkt.days, x["released"], 1)
    x = x.dropna(subset=["entry"])
    g = x.groupby(["ticker", "entry"]).agg(tone=("tone", "mean"), released=("released", "max"),
                                           cats=("category", lambda v: "; ".join(sorted(set(v))))).reset_index()
    g = _cooldown(g.assign(family="H3", magnitude=g["tone"], info=g["cats"]), mkt.days, c["cooldown_sessions"])
    g["pct"] = _trailing_tercile(g["tone"], g["entry"], c["tercile_window_days"], c["min_history_events"])
    g = g[g["pct"] >= 1 - c["top_fraction"]]
    return g[EVENT_COLS].reset_index(drop=True)


# --------------------------------------------------------------------------
# Retorno anormal contra o quintil de 63 dias
# --------------------------------------------------------------------------
def abnormal_returns(mkt: Mkt, ev: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Abertura da entrada -> abertura de entrada+H; controle = mesmas datas, mesmo quintil na vespera."""
    if ev.empty:
        return ev.assign(ret=[], ctrl=[], ar=[], ar_ew=[])
    O = mkt.opx.values
    n = len(mkt.days)
    fwd = np.full_like(O, np.nan)
    fwd[:n - horizon] = O[horizon:] / O[:n - horizon] - 1.0
    Q = mkt.quint.values
    col = {t: i for i, t in enumerate(mkt.tickers)}
    pos = mkt.days.get_indexer(pd.DatetimeIndex(ev["entry"]))
    out = ev.copy()
    rets, ctrls, ews = [], [], []
    for t, p in zip(ev["ticker"].values, pos):
        j = col[t]
        if p < 1 or p + horizon >= n or not np.isfinite(fwd[p, j]):
            rets.append(np.nan); ctrls.append(np.nan); ews.append(np.nan)
            continue
        q = Q[p - 1, j]
        f = fwd[p]
        ok_all = np.isfinite(f) & np.isfinite(Q[p - 1])
        ok_all[j] = False
        ews.append(float(np.mean(f[ok_all])) if ok_all.any() else np.nan)
        if not np.isfinite(q):
            rets.append(fwd[p, j]); ctrls.append(np.nan)
            continue
        same = ok_all & (Q[p - 1] == q)
        rets.append(fwd[p, j])
        ctrls.append(float(np.mean(f[same])) if same.sum() >= 3 else np.nan)
    out["ret"] = rets
    out["ctrl"] = ctrls
    out["ar"] = out["ret"] - out["ctrl"]
    out["ar_ew"] = out["ret"] - np.asarray(ews)
    return out


def weekly_mean(ar: pd.DataFrame, col: str = "ar") -> pd.Series:
    x = ar.dropna(subset=[col])
    if x.empty:
        return pd.Series(dtype=float)
    iso = pd.DatetimeIndex(x["entry"]).isocalendar()
    key = pd.to_datetime(iso.year.astype(str) + "-" + iso.week.astype(str) + "-5", format="%G-%V-%u")
    return x.groupby(key.values)[col].mean().sort_index()


def holm(pvals: Dict[str, float], alpha: float) -> Dict[str, bool]:
    items = sorted(((k, p if np.isfinite(p) else 1.0) for k, p in pvals.items()), key=lambda kv: kv[1])
    m = len(items)
    out = {k: False for k in pvals}
    for i, (k, p) in enumerate(items):
        if np.isfinite(p) and p < alpha / (m - i):
            out[k] = True
        else:
            break
    return out


def test_family(ar: pd.DataFrame, horizon: int, direction: str, s: Dict) -> Dict:
    import selective_se as v3
    sign = 1.0 if direction == "+" else -1.0
    w = weekly_mean(ar)
    lags = max(1, horizon // 5)
    block = float(max(2, horizon // 5))
    x = ar.dropna(subset=["ar"])
    out = {"events": int(len(ar)), "events_measured": int(len(x)), "weeks": int(len(w)),
           "mean_ar": float(x["ar"].mean()) if len(x) else float("nan"),
           "median_ar": float(x["ar"].median()) if len(x) else float("nan"),
           "hit_rate": float((sign * x["ar"] > 0).mean()) if len(x) else float("nan"),
           "mean_ret": float(x["ret"].mean()) if len(x) else float("nan"),
           "mean_ar_vs_universe": float(x["ar_ew"].mean()) if len(x) else float("nan"),
           "weekly_mean": float(w.mean()) if len(w) else float("nan")}
    if len(w) >= 20:
        out["t_nw"] = float(v3.newey_west_t(sign * w, lags))
        out["p_bootstrap"] = float(v3.stationary_bootstrap_p(sign * w, mean_block=block,
                                                             reps=s["tests"]["bootstrap_reps"]))
    else:
        out["t_nw"], out["p_bootstrap"] = float("nan"), float("nan")
    by_year = x.groupby(pd.DatetimeIndex(x["entry"]).year)["ar"].agg(["mean", "size"]) if len(x) else pd.DataFrame()
    out["by_year"] = {int(k): {"mean_ar": float(r["mean"]), "n": int(r["size"])} for k, r in by_year.iterrows()}
    return out


# --------------------------------------------------------------------------
# Carteira: 6 vagas de evento, o resto no fundo de indice
# --------------------------------------------------------------------------
@dataclass
class Result4:
    equity: pd.Series
    trades: pd.DataFrame
    invested: pd.Series
    fund_weight: pd.Series
    slots_used: pd.Series
    skipped: int = 0
    log: List[Dict] = field(default_factory=list)

    @property
    def taken(self) -> pd.DataFrame:
        return self.trades


def simulate(mkt: Mkt, events: pd.DataFrame, horizons: Dict[str, int], start, end, k: int = market_se.MAX_POSITIONS,
             priority: Tuple[str, ...] = ("H1", "H4", "H3"), blocked: Optional[pd.DataFrame] = None,
             capital: float = market_se.CAPITAL_SEK, brokerage: float = market_se.BROKERAGE_PCT_PER_ORDER,
             slippage: float = market_se.SLIPPAGE_PCT_PER_SIDE,
             block_windows: Optional[pd.DataFrame] = None) -> Result4:
    """Conta unica. Um evento com entrada em D compra na abertura de D (decidido no fechamento de D-1)
    e vende na abertura de D+H. Vagas livres ficam no fundo de indice (sem courtage, cota do dia).

    `blocked`: grupos semanais (H2 da v4). `block_windows`: janelas por ticker (colunas ticker,
    start, end) em que o nome nao e comprado (rebaixamentos da v5)."""
    windows: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    if block_windows is not None and not block_windows.empty:
        for r in block_windows.itertuples():
            windows.setdefault(r.ticker, []).append((pd.Timestamp(r.start), pd.Timestamp(r.end)))
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = mkt.days[(mkt.days >= start) & (mkt.days <= end)]
    rank = {f: i for i, f in enumerate(priority)}
    ev = events[events["family"].isin(priority)].copy()
    ev["prio"] = ev["family"].map(rank)
    ev = ev.sort_values(["entry", "prio", "magnitude"], ascending=[True, True, False])
    by_day = {d: g for d, g in ev.groupby("entry")}
    if blocked is not None and not blocked.empty:
        # bloqueio semanal: vale do pregao de entrada ate a proxima decisao semanal
        bl = blocked.groupby("entry")["ticker"].apply(set).sort_index()
    else:
        bl = pd.Series(dtype=object)
    pos_of = {d: i for i, d in enumerate(mkt.days)}

    cash, fund_units = float(capital), 0.0
    pos: Dict[str, Dict] = {}
    trades, log = [], []
    equity, invw, fundw, slots = {}, {}, {}, {}
    skipped = 0
    last_eq = float(capital)

    def nav(d) -> float:
        v = float(mkt.bench.get(d, np.nan))
        return v if np.isfinite(v) else np.nan

    def open_px(t, d) -> float:
        v = mkt.opx.at[d, t]
        return float(v) if np.isfinite(v) and v > 0 else np.nan

    for d in days:
        i = pos_of[d]
        nv = nav(d)
        # 1) saidas na abertura
        for t in [t for t, b in pos.items() if i >= b["exit_i"]]:
            p = open_px(t, d)
            if not np.isfinite(p):
                continue  # sem negocio: tenta no pregao seguinte
            b = pos.pop(t)
            proceeds = b["n"] * p * (1 - slippage) * (1 - brokerage)
            cash += proceeds
            trades.append({"ticker": t, "familia": b["family"], "data_compra": b["date"], "preco_compra": b["px"],
                           "data_venda": str(d.date()), "preco_venda": p, "resultado_liq": proceeds / b["paid"] - 1.0,
                           "status": "fechada", "pregoes": i - b["i"]})
        # 2) entradas na abertura
        cur_block = set()
        if len(bl):
            prev = bl.index[bl.index <= d]
            if len(prev):
                cur_block = bl.loc[prev[-1]]
        if d in by_day:
            slot = last_eq / k
            for r in by_day[d].itertuples():
                if r.ticker in pos:
                    continue
                if len(pos) >= k:
                    skipped += 1
                    continue
                if r.ticker in cur_block:
                    log.append({"date": str(d.date()), "ticker": r.ticker, "family": r.family, "why": "bloqueado H2"})
                    continue
                if any(a <= d <= b for a, b in windows.get(r.ticker, ())):
                    log.append({"date": str(d.date()), "ticker": r.ticker, "family": r.family, "why": "bloqueado"})
                    continue
                p = open_px(r.ticker, d)
                if not np.isfinite(p):
                    continue
                need = slot - cash
                if need > 0 and fund_units > 0 and np.isfinite(nv):
                    sell_u = min(fund_units, need / nv)
                    fund_units -= sell_u
                    cash += sell_u * nv
                amount = min(slot, cash)
                if amount < 0.5 * slot:
                    continue
                pos[r.ticker] = {"n": amount * (1 - brokerage) / (p * (1 + slippage)), "paid": amount,
                                 "date": str(d.date()), "px": p, "i": i, "exit_i": i + horizons[r.family],
                                 "family": r.family, "last": p}
                cash -= amount
        # 3) caixa parado vai para o fundo de indice (cota do dia)
        if cash > 1e-9 and np.isfinite(nv):
            fund_units += cash / nv
            cash = 0.0
        for t, b in pos.items():
            v = mkt.adj.at[d, t]
            if np.isfinite(v):
                b["last"] = float(v)
        stock = sum(b["n"] * b["last"] for b in pos.values())
        fund = fund_units * nv if np.isfinite(nv) else 0.0
        eq = cash + stock + fund
        equity[d], invw[d], fundw[d], slots[d] = eq, stock / eq, fund / eq, len(pos)
        last_eq = eq

    for t, b in pos.items():
        val = b["n"] * b["last"] * (1 - slippage) * (1 - brokerage)
        trades.append({"ticker": t, "familia": b["family"], "data_compra": b["date"], "preco_compra": b["px"],
                       "data_venda": "", "preco_venda": b["last"], "resultado_liq": val / b["paid"] - 1.0,
                       "status": "aberta", "pregoes": pos_of[days[-1]] - b["i"]})
    cols = ["ticker", "familia", "data_compra", "preco_compra", "data_venda", "preco_venda", "resultado_liq",
            "status", "pregoes"]
    return Result4(pd.Series(equity).sort_index(), pd.DataFrame(trades, columns=cols), pd.Series(invw).sort_index(),
                   pd.Series(fundw).sort_index(), pd.Series(slots).sort_index(), skipped, log)


# --------------------------------------------------------------------------
# Dados alternativos -> eventos
# --------------------------------------------------------------------------
def load_altdata() -> Dict[str, pd.DataFrame]:
    out = {}
    for name, path in (("insiders", alt.INSIDERS_PATH), ("shorts", alt.SHORTS_PATH), ("news", alt.NEWS_PATH),
                       ("tone", alt.TONE_PATH)):
        if not path.exists():
            raise SystemExit(f"falta {path}: rode `python altdata_se.py {name if name != 'tone' else 'tone'}`")
        out[name] = pd.read_parquet(path)
    return out


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_events(mkt: Mkt, data: Dict[str, pd.DataFrame], s: Dict) -> Dict[str, pd.DataFrame]:
    lo, hi = pd.Timestamp(s["period"][0]), pd.Timestamp(s["period"][1])
    panel = alt.short_interest_panel(data["shorts"], mkt.days, mkt.tickers)
    ev = {"H1": insider_events(data["insiders"], mkt, s), "H2": short_groups(panel, mkt, s),
          "H3": tone_events(data["tone"], mkt, s), "H4": earnings_events(data["news"], mkt, s)}
    return {f: e[(e["entry"] >= lo) & (e["entry"] <= hi)].reset_index(drop=True) for f, e in ev.items()}


def dry_run(mkt: Mkt, data: Dict[str, pd.DataFrame], s: Dict) -> Dict:
    """Contagens de eventos por familia e ano. Nao le nenhum retorno posterior a entrada."""
    ev = build_events(mkt, data, s)
    out = {}
    for f, e in ev.items():
        yr = pd.DatetimeIndex(e["entry"]).year
        out[f] = {"events": int(len(e)), "tickers": int(e["ticker"].nunique()) if len(e) else 0,
                  "by_year": e.groupby(yr).size().to_dict() if len(e) else {}}
    return out


# --------------------------------------------------------------------------
# Avaliacao unica
# --------------------------------------------------------------------------
def evaluate(mkt: Mkt, data: Dict[str, pd.DataFrame], s: Dict, out_dir: Path = REPORT_DIR) -> Dict:
    import portfolio_se as ps
    import rotation_se as rot
    import selective_se as v3
    import walkforward as wf

    ev = build_events(mkt, data, s)
    H = {f: s["signals"][f]["horizon"] for f in FAMILIES}
    res_tests, ars = {}, {}
    for f in FAMILIES:
        ars[f] = abnormal_returns(mkt, ev[f], H[f])
        res_tests[f] = test_family(ars[f], H[f], s["signals"][f]["direction"], s)
    alpha = s["tests"]["alpha"]
    holm_ok = holm({f: res_tests[f]["p_bootstrap"] for f in FAMILIES}, alpha)
    min_ar = s["tests"]["min_mean_ar_long"]
    for f in FAMILIES:
        r = res_tests[f]
        sign = 1.0 if s["signals"][f]["direction"] == "+" else -1.0
        r["pass_holm"] = bool(holm_ok[f])
        r["pass_t"] = bool(np.isfinite(r["t_nw"]) and r["t_nw"] >= s["tests"]["nw_t_min"])
        r["pass_size"] = bool(f not in LONG_FAMILIES or (np.isfinite(r["mean_ar"]) and sign * r["mean_ar"] >= min_ar))
        r["pass"] = r["pass_holm"] and r["pass_t"] and r["pass_size"]

    used = [f for f in s["strategy"]["priority"] if res_tests[f]["pass"]]
    use_filter = res_tests["H2"]["pass"]
    events_all = pd.concat([ev[f] for f in LONG_FAMILIES], ignore_index=True)
    folds = wf.folds()
    c_start, c_end = folds[0].trade_start, folds[-1].trade_end

    def run(fams, filt, start, end):
        return simulate(mkt, events_all, H, start, end, k=s["strategy"]["k"], priority=tuple(fams) or ("none",),
                        blocked=ev["H2"] if filt else None)

    out = {"spec_version": s["version"], "period": s["period"], "tests": res_tests, "signals_used": used,
           "short_filter": use_filter}
    rc = run(used, use_filter, c_start, c_end)
    rc.trades = rot.assign_folds(rc.trades)
    v = ps.verdict(rc, mkt.bench)
    out["criterion1"] = {k2: val for k2, val in v.items() if k2 != "bench_equity"}
    out["pass"] = bool(len(used) > 0 and v["pass"])
    full = run(used, use_filter, s["period"][0], s["period"][1])
    out["full_period"] = rot.metrics(full.equity, mkt.bench)
    out["full_period"]["trades"] = int(len(full.trades))
    x = v3.excess_series(full.equity, mkt.bench)
    out["full_period"]["excess_p_bootstrap"] = float(v3.stationary_bootstrap_p(x, reps=s["tests"]["bootstrap_reps"])) if len(x) > 50 else float("nan")
    # Diagnostico anunciado antes da rodada (nao decide nada): todas as familias, com filtro.
    # A rodada de 2026-10-31 usou a ordem H1>H3>H4 aqui, diferente da prioridade da
    # especificacao (H1>H4>H3); corrigido depois da rodada, ver reports/se_v4/verdict.md.
    diag = run(s["strategy"]["priority"], True, c_start, c_end)
    diag.trades = rot.assign_folds(diag.trades)
    vd = ps.verdict(diag, mkt.bench)
    out["diag_all_signals_criterion1"] = {k2: vd[k2] for k2 in ("total_return", "bench_return", "mdd", "bench_mdd",
                                                                 "positive_folds", "trades", "pass")}
    out["data_vintage"] = {p.name: file_sha(p) for p in (alt.INSIDERS_PATH, alt.SHORTS_PATH, alt.NEWS_PATH,
                                                         alt.TONE_PATH, REPO_ROOT / "output" / "prices_se.parquet")}
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in FAMILIES:
        ars[f].to_csv(out_dir / f"events_{f}.csv", index=False)
    pd.DataFrame({"sistema": rc.equity, "indice": v["bench_equity"]}).to_csv(out_dir / "equity.csv")
    rc.trades.to_csv(out_dir / "trades.csv", index=False)
    full.trades.to_csv(out_dir / "trades_full_period.csv", index=False)
    (out_dir / "verdict.json").write_text(json.dumps(out, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
    (out_dir / "verdict.md").write_text(verdict_markdown(out, s), encoding="utf-8")
    _chart(full, mkt.bench, out_dir / "equity.png", used)
    return out


def verdict_markdown(o: Dict, s: Dict) -> str:
    def p(x):
        return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:+.2f}%"

    def f3(x):
        return "-" if x is None or not np.isfinite(x) else f"{x:.3f}"

    def ok(b):
        return "PASSA" if b else "NAO PASSA"

    names = {"H1": "compras de insiders (60 pregoes)", "H2": "posicao vendida alta (20 pregoes, previsto negativo)",
             "H3": "tom positivo dos comunicados (20 pregoes)", "H4": "reacao positiva ao balanco (60 pregoes)"}
    L = ["# Avaliacao da estrategia v4 — eventos de informacao publica", "",
         f"Especificacao congelada em `data/strategy_se_v4.json` antes de ligar os dados alternativos a retornos "
         f"(ADR 0008). Periodo dos testes: {o['period'][0]} a {o['period'][1]}.", "",
         "## Testes por hipotese", "",
         "Retorno anormal = retorno da acao (abertura a abertura) menos o das acoes do mesmo quintil de retorno de "
         "63 dias, nas mesmas datas. Serie semanal da media dos eventos; t de Newey-West e p de bootstrap "
         "estacionario no sentido previsto; Holm sobre as quatro hipoteses.", "",
         "| Hipotese | Eventos | Retorno anormal medio | Acerto | t NW | p bootstrap | Holm | Tamanho | |",
         "|---|---:|---:|---:|---:|---:|---|---|---|"]
    for f in FAMILIES:
        r = o["tests"][f]
        L.append(f"| {f} {names[f]} | {r['events_measured']} | {p(r['mean_ar'])} | {p(r['hit_rate'])} | "
                 f"{r['t_nw']:.2f} | {f3(r['p_bootstrap'])} | {ok(r['pass_holm'])} | {ok(r['pass_size'])} | "
                 f"**{ok(r['pass'])}** |")
    c1 = o["criterion1"]
    L += ["", f"Sinais usados na carteira: {', '.join(o['signals_used']) or 'nenhum'}; filtro H2: "
          f"{'sim' if o['short_filter'] else 'nao'}.", "",
          "## Criterio 1 (2023-10-31 a 2026-10-30)", "",
          f"- Retorno {p(c1['total_return'])} vs indice {p(c1['bench_return'])} -> {ok(c1['crit_beats_index'])}",
          f"- Folds com expectativa positiva: {c1['positive_folds']}/6 -> {ok(c1['crit_folds'])}",
          f"- Max drawdown {p(c1['mdd'])} vs indice {p(c1['bench_mdd'])} -> {ok(c1['crit_mdd'])}",
          f"- Trades: {c1['trades']}", "",
          f"**{'APROVADA para paper trading' if o['pass'] else 'REPROVADA'}**", "",
          "## Periodo inteiro", ""]
    fp = o["full_period"]
    L += [f"- CAGR {p(fp['cagr'])} vs indice {p(fp['bench_cagr'])}; max drawdown {p(fp['mdd'])} vs {p(fp['bench_mdd'])}; "
          f"{fp['trades']} trades; excesso diario p bootstrap {f3(fp['excess_p_bootstrap'])}", "",
          "## Diagnostico anunciado (nao decide)", ""]
    d = o["diag_all_signals_criterion1"]
    L += [f"- Todas as familias longas com filtro H2, 2023-2026: retorno {p(d['total_return'])} vs {p(d['bench_return'])}, "
          f"MDD {p(d['mdd'])} vs {p(d['bench_mdd'])}, folds {d['positive_folds']}/6, {d['trades']} trades", "",
          "## Por ano (retorno anormal medio, eventos)", "", "| Ano | " + " | ".join(FAMILIES) + " |",
          "|---|" + "---:|" * len(FAMILIES)]
    years = sorted({y for f in FAMILIES for y in o["tests"][f]["by_year"]})
    for y in years:
        cells = []
        for f in FAMILIES:
            r = o["tests"][f]["by_year"].get(y) or o["tests"][f]["by_year"].get(str(y))
            cells.append(f"{p(r['mean_ar'])} ({r['n']})" if r else "-")
        L.append(f"| {y} | " + " | ".join(cells) + " |")
    L += ["", "Versao dos dados (sha256): " + ", ".join(f"`{k}` {v}" for k, v in o["data_vintage"].items()), ""]
    return "\n".join(L)


def _chart(res: Result4, bench: pd.Series, path: Path, used: List[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eq = res.equity
    b = bench.reindex(eq.index).ffill()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(eq.index, eq.values, label=f"Estrategia v4 ({', '.join(used) or 'so fundo'})", color="#1565c0")
    ax.plot(b.index, eq.iloc[0] * b / b.iloc[0], label=f"Indice {market_se.BENCHMARK_SYMBOL} (com dividendos)", color="#9e9e9e")
    ax.set_ylabel("SEK")
    ax.set_title("Estrategia v4 — eventos de informacao publica vs index benchmark")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["dry-run", "evaluate"])
    a = ap.parse_args(argv)
    s = spec()
    mkt = load_market()
    data = load_altdata()
    if a.cmd == "dry-run":
        print(json.dumps(dry_run(mkt, data, s), indent=1, default=str))
    else:
        o = evaluate(mkt, data, s)
        print((REPORT_DIR / "verdict.md").read_text(encoding="utf-8"))
        return 0 if o["pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
