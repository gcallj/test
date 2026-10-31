#!/usr/bin/env python3
"""
Estrategia v6 do mercado sueco: swing trade derivado da v1, com otimizador
robusto e entradas mais relevantes (docs/adr/0010).

O que vem da v1 (ADR 0001): ordem limite abaixo do fechamento valida 3 pregoes,
stop e alvo em ATR, time stop, parametros escolhidos por otimizacao e
walk-forward de 6 folds com 4 anos de treino antes de cada um.

O que muda, contra as causas da falha da v1 (docs/sweden/diagnostico.md):
- carteira unica com 6 vagas e o resto no fundo de indice (a v1 avaliava cada
  acao sozinha e ficava metade do tempo em caixa);
- score = soma ponderada de 12 entradas ranqueadas *entre* as acoes a cada dia:
  fatores documentados (momentum 12-1, reversao de 1 semana e 1 mes, maxima de
  52 semanas, volatilidade, tendencia de 200 dias) e dados que os precos nao
  contem (compras de insiders, posicao vendida divulgada e sua variacao,
  revisoes de analistas, tom do ultimo balanco, reacao ao ultimo balanco);
- prazo minimo de 10 pregoes antes da saida por score (a v1 girava em 4 dias);
- 18 genes em vez de 38, otimizados por CMA-ES com fitness da carteira:
  excesso anual sobre o indice com penalidade de drawdown, media menos desvio
  entre os 4 anos do treino; no fim, entre os 10 melhores, vence o mais estavel
  sob perturbacao dos genes (otimo "plano").

    python swing_se.py download      # maximas/minimas (output/prices_se_hl.parquet)
    python swing_se.py smoke         # tempo de uma avaliacao e contagem de trades (sem retornos)
    python swing_se.py evaluate      # walk-forward unico (reports/se_v6)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import market_se  # noqa: E402

HL_CACHE = REPO_ROOT / "output" / "prices_se_hl.parquet"
SPEC_PATH = REPO_ROOT / "data" / "strategy_se_v6.json"
REPORT_DIR = REPO_ROOT / "reports" / "se_v6"
ATR_WINDOW = 14
ORDER_VALIDITY = market_se.ORDER_VALIDITY_DAYS          # 3 pregoes, como na v1

PRICE_FEATURES = ["mom12_1", "resid5", "resid21", "hi52", "vol63", "gap200"]
ALT_FEATURES = ["insider60", "short_si", "short_chg20", "analyst_rating20", "analyst_target20", "tone60", "ear60"]
FEATURES = PRICE_FEATURES + ALT_FEATURES

TRADE_GENES = [  # nome, minimo, maximo, inteiro
    ("entry_q", 0.80, 0.99, False),       # entra no topo do score (percentil entre as acoes)
    ("exit_q", 0.20, 0.70, False),        # sai (depois do prazo minimo) se o score cair abaixo
    ("discount_atr", 0.0, 1.0, False),    # limite = fechamento - desconto x ATR
    ("stop_atr", 1.5, 5.0, False),        # stop = limite - m x ATR
    ("reward_risk", 1.5, 6.0, False),     # alvo = limite + RR x (limite - stop)
    ("time_stop", 10, 60, True),          # sai na abertura depois de N pregoes
]
WEIGHT_RANGE = (-1.0, 1.0)


def gene_space() -> List[Tuple[str, float, float, bool]]:
    return [(f"w_{f}", WEIGHT_RANGE[0], WEIGHT_RANGE[1], False) for f in FEATURES] + TRADE_GENES


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
def download_hl(start: str = "2005-01-01") -> pd.DataFrame:
    import yfinance as yf
    symbols = market_se.universe_tickers()
    frames = []
    for i in range(0, len(symbols), 40):
        d = yf.download(symbols[i:i + 40], start=start, interval="1d", auto_adjust=False, progress=False,
                        threads=True, group_by="column")
        frames.append(d[["High", "Low", "Close"]])
        time.sleep(2)
    raw = pd.concat(frames, axis=1)
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw = raw.sort_index()
    HL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(HL_CACHE)
    return raw


@dataclass
class Panel:
    days: pd.DatetimeIndex
    tickers: List[str]
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    ATR: np.ndarray
    NAV: np.ndarray            # ^OMXSBGI (fundo de indice)
    elig: np.ndarray
    X: np.ndarray              # F x T x N, entradas ranqueadas em [-0.5, 0.5] (0 = neutro/ausente)
    names: List[str] = field(default_factory=lambda: list(FEATURES))


def _xrank(df: pd.DataFrame, elig: pd.DataFrame) -> np.ndarray:
    r = df.where(elig).rank(axis=1, pct=True) - 0.5
    return r.fillna(0.0).to_numpy(dtype=np.float64)


def _clean(px: pd.DataFrame) -> pd.DataFrame:
    med = px.rolling(21, min_periods=5, center=True).median()
    return px.mask((px > 5 * med) | (px < med / 5))


def alt_features(days: pd.DatetimeIndex, tickers: List[str], adj: pd.DataFrame, ew_ret: pd.Series) -> Dict[str, pd.DataFrame]:
    """Entradas de dados alternativos conhecidas no fechamento de cada dia (para a ordem do dia seguinte).

    Um fato publicado no dia D entra a partir do fechamento de D (a ordem sai para D+1, como na v4).
    Posicao vendida: usa o painel da v4 (posicao de T a partir de T+2).
    """
    import altdata_se as alt
    import analyst_se
    import events_se as v4
    out: Dict[str, pd.DataFrame] = {}
    zeros = pd.DataFrame(0.0, index=days, columns=tickers)

    def by_day(df: pd.DataFrame, when: str, value: str) -> pd.DataFrame:
        d = df.dropna(subset=["ticker", when]).copy()
        d = d[d["ticker"].isin(tickers)]
        pos = days.searchsorted(pd.to_datetime(d[when]).dt.normalize().values, side="left")
        ok = pos < len(days)
        d = d[ok]
        d["day"] = days[pos[ok]]
        return d.pivot_table(index="day", columns="ticker", values=value, aggfunc="sum").reindex(index=days, columns=tickers).fillna(0.0)

    # compras de insiders (mesmas regras da H1 da v4), soma em SEK dos ultimos 60 pregoes
    ins = pd.read_parquet(alt.INSIDERS_PATH)
    s4 = json.loads((REPO_ROOT / "data" / "strategy_se_v4.json").read_text(encoding="utf-8"))["signals"]["H1"]
    x = ins.dropna(subset=["ticker"]).copy()
    itype = x["instrument_type"].fillna("").str.strip()
    share_isins = set(ins.loc[ins["instrument_type"] == "Aktie", "isin"])
    ok = (x["nature"].isin(s4["nature"]) & (itype.isin(s4["instrument_type"]) | ((itype == "") & x["isin"].isin(list(share_isins))))
          & x["status"].isin(s4["status"]) & x["volume_unit"].isin(s4["volume_unit"]) & x["currency"].isin(s4["currency"])
          & ~x["venue"].fillna("").str.strip().str.lower().isin([v.lower() for v in s4["exclude_venue"]])
          & ~x["share_program"].fillna(False).astype(bool))
    x = x[ok].assign(value=lambda d: d["volume"] * d["price"])
    daily = by_day(x, "published", "value")
    out["insider60"] = np.log1p(daily.rolling(60, min_periods=1).sum())

    # posicao vendida divulgada e variacao de 20 pregoes
    panel = alt.short_interest_panel(pd.read_parquet(alt.SHORTS_PATH), days, tickers)
    out["short_si"] = panel
    out["short_chg20"] = panel - panel.shift(20)

    # revisoes de analistas (corretoras) nos ultimos 20 pregoes
    acts = pd.read_parquet(analyst_se.ACTIONS_PATH)
    a = acts[acts["source"] == "broker"].copy()
    a["rating_net"] = a["n_up"] + a["n_init_pos"] - a["n_down"] - a["n_init_neg"]
    a["target_net"] = a["n_t_up"] - a["n_t_down"]
    out["analyst_rating20"] = by_day(a, "published", "rating_net").rolling(20, min_periods=1).sum()
    out["analyst_target20"] = by_day(a, "published", "target_net").rolling(20, min_periods=1).sum()

    # tom do ultimo balanco/informacao privilegiada (ate 60 pregoes); reacao ao ultimo balanco (EAR, ate 60)
    tone = pd.read_parquet(alt.TONE_PATH)
    t = by_day(tone.assign(v=tone["tone"]), "released", "v")
    cnt = by_day(tone.assign(v=1.0), "released", "v")
    last_tone = (t / cnt.replace(0, np.nan)).ffill(limit=60)
    out["tone60"] = last_tone
    news = pd.read_parquet(alt.NEWS_PATH)
    s4e = json.loads((REPO_ROOT / "data" / "strategy_se_v4.json").read_text(encoding="utf-8"))["signals"]["H4"]
    e = news[news["category"].isin(s4e["categories"]) & news["ticker"].isin(tickers)].dropna(subset=["released"]).copy()
    e["r"] = v4.reaction_session(days, e["released"], s4e["close_time"])
    e = e.dropna(subset=["r"]).drop_duplicates(["ticker", "r"])
    p = days.get_indexer(pd.DatetimeIndex(e["r"]))
    e = e[(p >= 2) & (p + 1 < len(days))]
    p = days.get_indexer(pd.DatetimeIndex(e["r"]))
    lvl = (1 + ew_ret.reindex(days).fillna(0)).cumprod().values
    A = adj.reindex(columns=tickers).to_numpy()
    col = {tk: i for i, tk in enumerate(tickers)}
    ear = [A[i + 1, col[tk]] / A[i - 2, col[tk]] - 1 - (lvl[i + 1] / lvl[i - 2] - 1) for tk, i in zip(e["ticker"], p)]
    e = e.assign(ear=ear, known=days[p + 1])                        # conhecida no fechamento de r+1
    ear_daily = by_day(e.dropna(subset=["ear"]), "known", "ear")
    has = by_day(e.dropna(subset=["ear"]).assign(one=1.0), "known", "one")
    out["ear60"] = (ear_daily / has.replace(0, np.nan)).ffill(limit=60)
    return out


def build_panel(px: Dict[str, pd.DataFrame], hl: pd.DataFrame, with_alt: bool = True) -> Panel:
    import selective_se as v3
    d = v3.build(px)
    days, tickers = d.days, list(d.tickers)
    close = px["close"].reindex(index=days, columns=tickers)
    factor = (d.adj / close).ffill()
    H = _clean(hl["High"]).reindex(index=days, columns=tickers) * factor
    L = _clean(hl["Low"]).reindex(index=days, columns=tickers) * factor
    C = d.adj
    O = d.adj_open.where(d.adj_open > 0)
    H = H.where(H >= np.fmax(O, C)).fillna(np.fmax(O, C))      # coerencia: maxima >= abertura e fechamento
    L = L.where(L <= np.fmin(O, C)).fillna(np.fmin(O, C))
    tr = pd.concat({"a": H - L, "b": (H - C.shift(1)).abs(), "c": (L - C.shift(1)).abs()}).groupby(level=1).max()
    tr = tr.reindex(index=days, columns=tickers)
    atr = tr.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean()
    elig = d.eligible.reindex(index=days, columns=tickers).fillna(False)
    feats = [_xrank(d.stock[f] + 0.5, elig) if f in d.stock else None for f in PRICE_FEATURES]
    if with_alt:
        alt = alt_features(days, tickers, d.adj, d.mkt_ret)
        feats += [_xrank(alt[f], elig) for f in ALT_FEATURES]
    else:
        feats += [np.zeros((len(days), len(tickers))) for _ in ALT_FEATURES]
    X = np.stack(feats)
    nav = d.bench.reindex(days).ffill().to_numpy(dtype=np.float64)
    return Panel(days, tickers, O.to_numpy(np.float64), H.to_numpy(np.float64), L.to_numpy(np.float64),
                 C.to_numpy(np.float64), atr.to_numpy(np.float64), nav, elig.to_numpy(bool), X)


def load_panel(with_alt: bool = True) -> Panel:
    import prices_se
    if not HL_CACHE.exists():
        raise SystemExit(f"{HL_CACHE} nao existe: rode `python swing_se.py download`")
    return build_panel(prices_se.load(), pd.read_parquet(HL_CACHE), with_alt=with_alt)


# --------------------------------------------------------------------------
# Motor (numba): regra de execucao da v1 numa conta com 6 vagas e fundo de indice
# --------------------------------------------------------------------------
@njit(cache=True)
def _simulate(O, H, L, C, ATR, NAV, pct, elig, entry_q, exit_q, disc, stop_m, rr, tstop,
              min_hold, k, capital, brokerage, slippage, validity, i0, i1, max_trades):
    T, N = C.shape
    held = np.zeros(N, np.bool_)
    shares = np.zeros(N)
    paid = np.zeros(N)
    epx = np.zeros(N)
    stp = np.zeros(N)
    tgt = np.zeros(N)
    ei = np.zeros(N, np.int64)
    tst = np.zeros(N, np.int64)
    last = np.zeros(N)
    pend = np.zeros(N, np.bool_)
    plim = np.zeros(N)
    pstp = np.zeros(N)
    ptgt = np.zeros(N)
    pexp = np.zeros(N, np.int64)
    pamt = np.zeros(N)
    ptst = np.zeros(N, np.int64)
    cash = capital
    units = 0.0
    eq = np.full(T, np.nan)
    inv = np.zeros(T)
    slots = np.zeros(T, np.int64)
    trades = np.zeros((max_trades, 7))     # ticker, entrada, saida, preco entrada, preco saida, resultado, motivo
    nt = 0
    last_eq = capital
    for i in range(i0, i1 + 1):
        nav = NAV[i]
        # 1) saidas: time stop e score (decididos no fechamento anterior) na abertura; stop e alvo no pregao
        for j in range(N):
            if not held[j]:
                continue
            o = O[i, j]
            if not np.isfinite(o):
                continue
            bars = i - ei[j]
            px = np.nan
            why = 0
            if bars >= tst[j]:
                px = o
                why = 1
            elif bars >= min_hold and pct[i - 1, j] < exit_q[i - 1]:
                px = o
                why = 2
            elif o <= stp[j]:
                px = o
                why = 3
            elif o >= tgt[j]:
                px = o
                why = 4
            elif L[i, j] <= stp[j]:
                px = stp[j]
                why = 3
            elif H[i, j] >= tgt[j]:
                px = tgt[j]
                why = 4
            if np.isfinite(px):
                proceeds = shares[j] * px * (1.0 - slippage) * (1.0 - brokerage)
                cash += proceeds
                if nt < max_trades:
                    trades[nt, 0] = j
                    trades[nt, 1] = ei[j]
                    trades[nt, 2] = i
                    trades[nt, 3] = epx[j]
                    trades[nt, 4] = px
                    trades[nt, 5] = proceeds / paid[j] - 1.0
                    trades[nt, 6] = why
                    nt += 1
                held[j] = False
        # 2) ordens limite pendentes
        for j in range(N):
            if not pend[j]:
                continue
            if i > pexp[j]:
                pend[j] = False
                continue
            o = O[i, j]
            if not np.isfinite(o):
                continue
            fill = np.nan
            if o <= plim[j]:
                fill = o
            elif L[i, j] <= plim[j]:
                fill = plim[j]
            if not np.isfinite(fill):
                continue
            pend[j] = False
            need = pamt[j] - cash
            if need > 0 and units > 0 and np.isfinite(nav):
                u = min(units, need / nav)
                units -= u
                cash += u * nav
            amt = min(pamt[j], cash)
            if amt < 0.5 * pamt[j]:
                continue
            shares[j] = amt * (1.0 - brokerage) / (fill * (1.0 + slippage))
            paid[j] = amt
            epx[j] = fill
            stp[j] = pstp[j]
            tgt[j] = ptgt[j]
            ei[j] = i
            tst[j] = ptst[j]
            last[j] = fill
            held[j] = True
            cash -= amt
            if L[i, j] <= stp[j]:   # stop no proprio dia da compra (conservador)
                px = stp[j] if fill > stp[j] else fill
                proceeds = shares[j] * px * (1.0 - slippage) * (1.0 - brokerage)
                cash += proceeds
                if nt < max_trades:
                    trades[nt, 0] = j
                    trades[nt, 1] = i
                    trades[nt, 2] = i
                    trades[nt, 3] = fill
                    trades[nt, 4] = px
                    trades[nt, 5] = proceeds / amt - 1.0
                    trades[nt, 6] = 3
                    nt += 1
                held[j] = False
        # 3) fechamento: caixa parado vai para o fundo; marcacao
        if cash > 1e-9 and np.isfinite(nav):
            units += cash / nav
            cash = 0.0
        stock = 0.0
        nh = 0
        for j in range(N):
            if held[j]:
                c = C[i, j]
                if np.isfinite(c):
                    last[j] = c
                stock += shares[j] * last[j]
                nh += 1
        fund = units * nav if np.isfinite(nav) else 0.0
        e = cash + stock + fund
        eq[i] = e
        inv[i] = stock / e if e > 0 else 0.0
        slots[i] = nh
        last_eq = e
        # 4) novas ordens para o proximo pregao
        if i >= i1:
            continue
        npend = 0
        for j in range(N):
            if pend[j]:
                npend += 1
        free = k - nh - npend
        if free <= 0:
            continue
        row = pct[i]
        order = np.argsort(-np.where(np.isfinite(row), row, -1.0))
        for jj in range(N):
            if free <= 0:
                break
            j = order[jj]
            s = row[j]
            if not np.isfinite(s) or s < entry_q[i]:
                break
            if held[j] or pend[j] or not elig[i, j]:
                continue
            a = ATR[i, j]
            c = C[i, j]
            if not (np.isfinite(a) and np.isfinite(c)) or a <= 0:
                continue
            lim = c - disc[i] * a
            st = lim - stop_m[i] * a
            if st <= 0 or lim <= 0:
                continue
            pend[j] = True
            plim[j] = lim
            pstp[j] = st
            ptgt[j] = lim + rr[i] * (lim - st)
            pexp[j] = i + validity
            pamt[j] = last_eq / k
            ptst[j] = tstop[i]
            free -= 1
    return eq, inv, slots, trades[:nt]


def score_pct(panel: Panel, w: np.ndarray, lo: int = 0, hi: Optional[int] = None) -> np.ndarray:
    """Percentil (0..1) do score entre as acoes elegiveis em cada dia (linhas lo:hi); NaN fora do universo elegivel."""
    hi = len(panel.days) if hi is None else hi
    s = np.tensordot(w, panel.X[:, lo:hi], axes=(0, 0))
    s = np.where(panel.elig[lo:hi], s, np.nan)
    key = np.where(np.isfinite(s), s, np.inf)
    order = np.argsort(key, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(s.shape[0])[:, None]
    ranks[rows, order] = np.arange(s.shape[1])[None, :]
    cnt = np.isfinite(s).sum(axis=1, keepdims=True)
    pct = (ranks + 1.0) / np.maximum(cnt, 1)
    return np.where(np.isfinite(s), pct, np.nan)


@dataclass
class Genome:
    w: np.ndarray
    entry_q: float
    exit_q: float
    discount_atr: float
    stop_atr: float
    reward_risk: float
    time_stop: int

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "Genome":
        return Genome(np.array([d[f"w_{f}"] for f in FEATURES]), d["entry_q"], d["exit_q"], d["discount_atr"],
                      d["stop_atr"], d["reward_risk"], int(round(d["time_stop"])))

    def to_dict(self) -> Dict[str, float]:
        out = {f"w_{f}": float(v) for f, v in zip(FEATURES, self.w)}
        out.update({"entry_q": self.entry_q, "exit_q": self.exit_q, "discount_atr": self.discount_atr,
                    "stop_atr": self.stop_atr, "reward_risk": self.reward_risk, "time_stop": int(self.time_stop)})
        return out


@dataclass
class Run:
    equity: pd.Series
    invested: pd.Series
    slots_used: pd.Series
    trades: pd.DataFrame
    skipped: int = 0

    @property
    def taken(self) -> pd.DataFrame:
        return self.trades


def run(panel: Panel, genomes: List[Tuple[pd.Timestamp, Genome]], start, end, min_hold: int = 10,
        k: int = market_se.MAX_POSITIONS, capital: float = market_se.CAPITAL_SEK,
        brokerage: float = market_se.BROKERAGE_PCT_PER_ORDER, slippage: float = market_se.SLIPPAGE_PCT_PER_SIDE) -> Run:
    """Simula de `start` a `end`; `genomes` = [(a partir de, genoma)], o genoma vale ate o proximo."""
    days = panel.days
    i0 = int(days.searchsorted(pd.Timestamp(start)))
    i1 = int(days.searchsorted(pd.Timestamp(end), side="right")) - 1
    T = len(days)
    pct = np.full((T, len(panel.tickers)), np.nan)
    i0 = max(i0, 1)
    arr = {n: np.zeros(T) for n in ("entry_q", "exit_q", "disc", "stop", "rr")}
    tst = np.zeros(T, np.int64)
    bounds = [int(days.searchsorted(pd.Timestamp(a))) for a, _ in genomes] + [T]
    for (a, g), lo, hi in zip(genomes, bounds[:-1], bounds[1:]):
        lo2 = max(lo - 1, 0)    # a saida por score do 1o dia usa o percentil da vespera
        hi2 = min(hi, i1 + 1)
        if hi2 > lo2:
            pct[lo2:hi2] = score_pct(panel, g.w, lo2, hi2)
        arr["entry_q"][lo2:hi], arr["exit_q"][lo2:hi], arr["disc"][lo2:hi] = g.entry_q, g.exit_q, g.discount_atr
        arr["stop"][lo2:hi], arr["rr"][lo2:hi], tst[lo2:hi] = g.stop_atr, g.reward_risk, g.time_stop
    eq, inv, slots, tr = _simulate(panel.O, panel.H, panel.L, panel.C, panel.ATR, panel.NAV, pct, panel.elig,
                                   arr["entry_q"], arr["exit_q"], arr["disc"], arr["stop"], arr["rr"], tst,
                                   min_hold, k, float(capital), brokerage, slippage, ORDER_VALIDITY, i0, i1, 20000)
    idx = days[i0:i1 + 1]
    t = pd.DataFrame(tr, columns=["j", "ei", "xi", "preco_compra", "preco_venda", "resultado_liq", "motivo"])
    if len(t):
        t["ticker"] = [panel.tickers[int(j)] for j in t["j"]]
        t["data_compra"] = [str(days[int(i)].date()) for i in t["ei"]]
        t["data_venda"] = [str(days[int(i)].date()) for i in t["xi"]]
        t["pregoes"] = (t["xi"] - t["ei"]).astype(int)
        t["status"] = "fechada"
        t["motivo"] = t["motivo"].map({1: "time stop", 2: "score", 3: "stop", 4: "alvo"})
    return Run(pd.Series(eq[i0:i1 + 1], index=idx), pd.Series(inv[i0:i1 + 1], index=idx),
               pd.Series(slots[i0:i1 + 1], index=idx), t)


# --------------------------------------------------------------------------
# Fitness robusta e otimizacao
# --------------------------------------------------------------------------
def block_scores(eq: pd.Series, nav: pd.Series, blocks: int, dd_weight: float) -> np.ndarray:
    eq, nav = eq.dropna(), nav.reindex(eq.dropna().index).ffill()
    out = []
    for part in np.array_split(np.arange(len(eq)), blocks):
        if len(part) < 20:
            continue
        e, b = eq.iloc[part], nav.iloc[part]
        x = e.pct_change().fillna(0) - b.pct_change().fillna(0)
        mdd_e = float((e / e.cummax() - 1).min())
        mdd_b = float((b / b.cummax() - 1).min())
        out.append(252 * float(x.mean()) - dd_weight * max(0.0, abs(mdd_e) - abs(mdd_b)))
    return np.array(out)


def fitness(panel: Panel, g: Genome, start, end, s: Dict) -> float:
    r = run(panel, [(pd.Timestamp(start), g)], start, end, min_hold=s["engine"]["min_hold"])
    nav = pd.Series(panel.NAV, index=panel.days)
    sc = block_scores(r.equity, nav, s["optimizer"]["blocks"], s["optimizer"]["dd_weight"])
    if len(sc) == 0:
        return -1.0
    return float(sc.mean() - s["optimizer"]["std_weight"] * sc.std())


def optimize(panel: Panel, start, end, s: Dict, seed: int) -> Tuple[Genome, Dict]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = gene_space()
    o = s["optimizer"]

    def objective(trial):
        d = {}
        for name, lo, hi, is_int in space:
            d[name] = trial.suggest_int(name, int(lo), int(hi)) if is_int else trial.suggest_float(name, lo, hi)
        return fitness(panel, Genome.from_dict(d), start, end, s)

    sampler = optuna.samplers.CmaEsSampler(seed=seed, n_startup_trials=o["startup_trials"], restart_strategy=None)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=o["trials"], show_progress_bar=False)
    done = [t for t in study.trials if t.value is not None]
    top = sorted(done, key=lambda t: t.value, reverse=True)[:o["top_for_robustness"]]
    rng = np.random.default_rng(seed)
    best, best_rob, rob_table = None, -np.inf, []
    for t in top:
        vals = []
        for _ in range(o["perturbations"]):
            d = dict(t.params)
            for name, lo, hi, is_int in space:
                v = d[name] + rng.normal(0, o["perturbation_sd"] * (hi - lo))
                v = float(np.clip(v, lo, hi))
                d[name] = int(round(v)) if is_int else v
            vals.append(fitness(panel, Genome.from_dict(d), start, end, s))
        rob = float(np.mean(vals))
        rob_table.append({"trial": t.number, "fitness": float(t.value), "robust_fitness": rob})
        if rob > best_rob:
            best, best_rob = t, rob
    info = {"best_trial": best.number, "fitness": float(best.value), "robust_fitness": best_rob,
            "trials": len(done), "top": rob_table}
    return Genome.from_dict(best.params), info


# --------------------------------------------------------------------------
# Walk-forward unico e veredito
# --------------------------------------------------------------------------
def spec() -> Dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def evaluate(panel: Panel, s: Dict, out_dir: Path = REPORT_DIR) -> Dict:
    import portfolio_se as ps
    import rotation_se as rot
    import selective_se as v3
    import walkforward as wf

    folds = wf.folds()
    genomes, info = [], []
    for f in folds:
        train_end = pd.Timestamp(f.train_end)
        train_start = train_end - pd.DateOffset(years=s["optimizer"]["train_years"])
        t0 = time.time()
        g, inf = optimize(panel, train_start, train_end, s, seed=s["optimizer"]["seed"] + f.k)
        inf.update({"fold": f.k, "train": [str(train_start.date()), str(train_end.date())],
                    "trade": [f.trade_start, f.trade_end], "genome": g.to_dict(), "seconds": round(time.time() - t0, 1)})
        print(f"fold {f.k}: treino {inf['train']} fitness {inf['fitness']:+.3f} robusta {inf['robust_fitness']:+.3f} "
              f"({inf['seconds']}s)", flush=True)
        genomes.append((pd.Timestamp(f.trade_start), g))
        info.append(inf)
    res = run(panel, genomes, folds[0].trade_start, folds[-1].trade_end, min_hold=s["engine"]["min_hold"])
    res.trades = rot.assign_folds(res.trades)
    nav = pd.Series(panel.NAV, index=panel.days)
    v = ps.verdict(res, nav)
    x = v3.excess_series(res.equity, nav)
    p_boot = float(v3.stationary_bootstrap_p(x, reps=s["tests"]["bootstrap_reps"]))
    t = res.trades
    out = {"spec_version": s["version"], "folds": info,
           "criterion1": {k: val for k, val in v.items() if k != "bench_equity"},
           "excess_daily_mean": float(x.mean()), "excess_p_bootstrap": p_boot,
           "trades": int(len(t)), "trades_per_year": float(len(t) / 3.0),
           "mean_hold": float(t["pregoes"].mean()) if len(t) else float("nan"),
           "gross_per_trade": float((t["preco_venda"] / t["preco_compra"] - 1).mean()) if len(t) else float("nan"),
           "net_per_trade": float(t["resultado_liq"].mean()) if len(t) else float("nan"),
           "invested_mean": float(res.invested.mean()),
           "exit_reasons": t["motivo"].value_counts().to_dict() if len(t) else {}}
    out["tests"] = {"C1_criterion1": bool(v["pass"]), "S_excess_bootstrap": bool(p_boot < s["tests"]["alpha"])}
    out["pass"] = all(out["tests"].values())
    w = pd.DataFrame([i["genome"] for i in info], index=[f"fold {i['fold']}" for i in info])
    out_dir.mkdir(parents=True, exist_ok=True)
    w.to_csv(out_dir / "genomes.csv")
    pd.DataFrame({"sistema": res.equity, "indice": v["bench_equity"]}).to_csv(out_dir / "equity.csv")
    t.to_csv(out_dir / "trades.csv", index=False)
    (out_dir / "verdict.json").write_text(json.dumps(out, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
    (out_dir / "verdict.md").write_text(verdict_markdown(out, w), encoding="utf-8")
    _chart(res, v["bench_equity"], out_dir / "equity.png")
    return out


def verdict_markdown(o: Dict, w: pd.DataFrame) -> str:
    def p(x):
        return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:+.2f}%"

    def ok(b):
        return "PASSA" if b else "NAO PASSA"

    c1 = o["criterion1"]
    L = ["# Avaliacao da estrategia v6 — swing derivado da v1 com CMA-ES e entradas novas", "",
         "Especificacao congelada em `data/strategy_se_v6.json` antes da rodada (ADR 0010). Walk-forward de 6 folds "
         "(2023-10-31 a 2026-10-30), cada genoma treinado so nos 4 anos anteriores ao fold.", "",
         "| Teste | Resultado | Exigido | |", "|---|---:|---:|---|",
         f"| C1 criterio 1 | retorno {p(c1['total_return'])} vs {p(c1['bench_return'])}; folds {c1['positive_folds']}/6; "
         f"MDD {p(c1['mdd'])} vs {p(c1['bench_mdd'])} | os 3 | {ok(o['tests']['C1_criterion1'])} |",
         f"| S excesso diario sobre o indice (bootstrap) | p = {o['excess_p_bootstrap']:.3f} | p < 0,05 | "
         f"{ok(o['tests']['S_excess_bootstrap'])} |", "",
         f"**{'APROVADA para paper trading' if o['pass'] else 'REPROVADA'}**", "",
         "## Comportamento", "",
         f"- Trades: {o['trades']} ({o['trades_per_year']:.0f}/ano); prazo medio {o['mean_hold']:.1f} pregoes; "
         f"exposicao media em acoes {100 * o['invested_mean']:.0f}%",
         f"- Resultado medio por trade: bruto {p(o['gross_per_trade'])}, liquido {p(o['net_per_trade'])}",
         f"- Saidas: {', '.join(f'{k} {v}' for k, v in o['exit_reasons'].items())}", "",
         "## Treino por fold", "", "| Fold | Treino | Fitness | Robusta | Tempo |", "|---|---|---:|---:|---:|"]
    for f in o["folds"]:
        L.append(f"| {f['fold']} | {f['train'][0]} a {f['train'][1]} | {f['fitness']:+.3f} | {f['robust_fitness']:+.3f} | {f['seconds']:.0f}s |")
    L += ["", "## Genomas (pesos das entradas e genes de operacao)", "", w.round(3).T.to_markdown(), ""]
    return "\n".join(L)


def _chart(res: Run, bench_eq: pd.Series, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(res.equity.index, res.equity.values, label="Estrategia v6", color="#1565c0")
    ax.plot(bench_eq.index, bench_eq.values, label=f"Indice {market_se.BENCHMARK_SYMBOL}", color="#9e9e9e")
    ax.set_ylabel("SEK")
    ax.set_title("Estrategia v6 — walk-forward 2023-2026 vs index benchmark")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["download", "smoke", "evaluate"])
    a = ap.parse_args(argv)
    if a.cmd == "download":
        raw = download_hl()
        print(f"maximas/minimas: {raw.shape} -> {HL_CACHE}")
        return 0
    s = spec()
    panel = load_panel()
    if a.cmd == "smoke":
        # Tempo e contagem de trades de genomas aleatorios num periodo antigo; nenhum retorno e impresso.
        rng = np.random.default_rng(0)
        for _ in range(3):
            d = {n: (int(rng.integers(lo, hi + 1)) if it else float(rng.uniform(lo, hi))) for n, lo, hi, it in gene_space()}
            t0 = time.time()
            r = run(panel, [(pd.Timestamp("2015-01-01"), Genome.from_dict(d))], "2015-01-01", "2018-12-31")
            print(f"avaliacao: {time.time() - t0:.3f}s, {len(r.trades)} trades, vagas medias {r.slots_used.mean():.1f}")
        return 0
    o = evaluate(panel, s)
    print((REPORT_DIR / "verdict.md").read_text(encoding="utf-8"))
    return 0 if o["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
