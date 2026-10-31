#!/usr/bin/env python3
"""
Estrategia v7 do mercado sueco: calendario, com pouquissimos trades (docs/adr/0011).

Duas hipoteses da literatura, sem nenhum parametro a ajustar, congeladas em
`data/strategy_se_v7.json` antes de qualquer retorno ser calculado:

- T1 "Halloween" (Bouman & Jacobsen 2002; Jacobsen & Zhang 2012): o indice
  rende mais de novembro a abril do que de maio a outubro. Estrategia: fundo
  de indice de novembro a abril, caixa a juro do Riksbank de maio a outubro.
  Duas trocas de fundo por ano, sem courtage.
- T2 mes do dividendo (Hartzmark & Solomon 2013): acoes rendem mais no mes em
  que se espera o pagamento do dividendo. Estrategia: no ultimo pregao de cada
  mes, as 6 acoes elegiveis que tiveram ex-dividendo no mesmo mes do ano
  anterior, pelo maior dividend yield de 12 meses; compra na abertura do
  primeiro pregao do mes, venda na abertura do primeiro pregao do mes seguinte.
  O resto do capital fica no fundo de indice.

    python seasonal_se.py dry-run     # contagens (meses, pagadores previstos, trocas), sem retornos
    python seasonal_se.py evaluate    # avaliacao unica (reports/se_v7)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import market_se  # noqa: E402

SPEC_PATH = REPO_ROOT / "data" / "strategy_se_v7.json"
REPORT_DIR = REPO_ROOT / "reports" / "se_v7"
RATE_PATH = REPO_ROOT / "output" / "riksbank_policy_rate.parquet"
RATE_URL = "https://api.riksbank.se/swea/v1/Observations/SECBREPOEFF/{start}/{end}"
WINTER = (11, 12, 1, 2, 3, 4)
DIV_MIN_YIELD = 0.002          # salto do fator de ajuste abaixo disso e ruido, nao dividendo


def spec() -> Dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
@dataclass
class Data:
    days: pd.DatetimeIndex
    tickers: List[str]
    adj: pd.DataFrame
    opx: pd.DataFrame            # abertura ajustada (fechamento quando falta)
    eligible: pd.DataFrame
    quint: pd.DataFrame          # quintil do retorno de 63 dias (controle, como na v4)
    dy: pd.DataFrame             # dividend yield no dia ex (0 nos outros dias)
    trailing_yield: pd.DataFrame
    nav: pd.Series               # ^OMXSBGI
    price_index: pd.Series       # ^OMXSPI
    ew_ret: pd.Series
    rate: pd.Series              # juro do Riksbank (% a.a.), por pregao


def policy_rate(days: pd.DatetimeIndex) -> pd.Series:
    if not RATE_PATH.exists():
        import requests
        r = requests.get(RATE_URL.format(start="2005-01-01", end=str(days[-1].date())), timeout=60).json()
        df = pd.DataFrame(r)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date")["value"].astype(float).to_frame("policy_rate_pct").to_parquet(RATE_PATH)
    s = pd.read_parquet(RATE_PATH)["policy_rate_pct"]
    return s.reindex(s.index.union(days)).ffill().reindex(days).fillna(0.0)


def ex_dividend_yield(close: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """Yield pago no dia ex, pelo salto do fator Adj Close / Close do Yahoo (so dividendos >= 0,2%)."""
    f = (adj / close).where(close > 0)
    dy = (1.0 - f.shift(1) / f).where(f.notna() & f.shift(1).notna())
    return dy.where(dy >= DIV_MIN_YIELD, 0.0).fillna(0.0)


def build(px: Dict[str, pd.DataFrame]) -> Data:
    import events_se as v4
    import selective_se as v3
    d = v3.build(px)
    tickers = list(d.tickers)
    close = px["close"].reindex(index=d.days, columns=tickers)
    adj = px["adj"].reindex(index=d.days, columns=tickers)
    dy = ex_dividend_yield(close, adj)
    trailing = dy.rolling(252, min_periods=200).sum()
    m = v4.market_from_prices(px)
    nav = px["adj"][market_se.BENCHMARK_SYMBOL].reindex(d.days).ffill()
    pidx = px["adj"][market_se.INDEX_SYMBOL].reindex(d.days).ffill()
    return Data(d.days, tickers, d.adj, m.opx, d.eligible.fillna(False), m.quint, dy, trailing, nav, pidx,
                d.mkt_ret, policy_rate(d.days))


def load() -> Data:
    import prices_se
    return build(prices_se.load())


# --------------------------------------------------------------------------
# Calendario
# --------------------------------------------------------------------------
def month_starts(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Primeiro pregao de cada mes."""
    s = pd.Series(days, index=days)
    return pd.DatetimeIndex(s.groupby([days.year, days.month]).min().values)


def is_winter(d: pd.Timestamp) -> bool:
    return d.month in WINTER


def predicted_payers(data: Data, decision_day: pd.Timestamp, month_start: pd.Timestamp, k: int) -> List[str]:
    """No fechamento de `decision_day`: elegiveis com ex-dividendo no mesmo mes do ano anterior,
    pelo maior yield de 12 meses (desempate alfabetico), ate k."""
    y, m = month_start.year - 1, month_start.month
    win = data.dy[(data.dy.index.year == y) & (data.dy.index.month == m)]
    if win.empty:
        return []
    paid = win.gt(0).any(axis=0)
    el = data.eligible.loc[decision_day]
    ty = data.trailing_yield.loc[decision_day]
    cand = [t for t in data.tickers if paid.get(t, False) and bool(el.get(t, False)) and np.isfinite(ty.get(t, np.nan))]
    cand.sort(key=lambda t: (-float(ty[t]), t))
    return cand[:k]


# --------------------------------------------------------------------------
# Simulacao: T1 (fundo/caixa por estacao) e T2 (mes do dividendo), isoladas ou juntas
# --------------------------------------------------------------------------
@dataclass
class Result7:
    equity: pd.Series
    trades: pd.DataFrame
    invested: pd.Series
    fund_weight: pd.Series
    cash_weight: pd.Series
    slots_used: pd.Series
    switches: List[Dict] = field(default_factory=list)
    skipped: int = 0

    @property
    def taken(self) -> pd.DataFrame:
        return self.trades


def simulate(data: Data, start, end, t1: bool, t2: bool, k: int = market_se.MAX_POSITIONS,
             capital: float = market_se.CAPITAL_SEK, brokerage: float = market_se.BROKERAGE_PCT_PER_ORDER,
             slippage: float = market_se.SLIPPAGE_PCT_PER_SIDE, cash_floor: float = 0.0) -> Result7:
    """Capital ocioso: fundo de indice (cota do dia, sem courtage) ou, com T1 ligada, caixa a juro do
    Riksbank de maio a outubro. Trocas de fundo na cota do primeiro pregao da estacao. T2: compras na
    abertura do primeiro pregao do mes (decididas no fechamento do ultimo pregao do mes anterior) e
    vendas na abertura do primeiro pregao do mes seguinte."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = data.days[(data.days >= start) & (data.days <= end)]
    pos_of = {d: i for i, d in enumerate(data.days)}
    starts = set(month_starts(data.days))
    cash, units = float(capital), 0.0
    pos: Dict[str, Dict] = {}
    trades, switches = [], []
    equity, invw, fundw, cashw, slots = {}, {}, {}, {}, {}
    pending: List[str] = []
    in_fund: Optional[bool] = None
    last_eq = float(capital)

    def opx(t, d) -> float:
        v = data.opx.at[d, t]
        return float(v) if np.isfinite(v) and v > 0 else np.nan

    for d in days:
        i = pos_of[d]
        nav = float(data.nav.get(d, np.nan))
        # juro do caixa (pregao anterior ate hoje)
        if cash > 0:
            r = max(float(data.rate.get(d, 0.0)), cash_floor) / 100.0
            cash *= 1.0 + r / 252.0
        # T1: no primeiro pregao de novembro/maio, troca fundo <-> caixa na cota do dia
        want_fund = (not t1) or is_winter(d)
        if in_fund is None or (d in starts and want_fund != in_fund):
            if want_fund and cash > 0 and np.isfinite(nav):
                units += cash / nav
                cash = 0.0
                switches.append({"date": str(d.date()), "to": "fundo"})
            elif not want_fund and units > 0 and np.isfinite(nav):
                cash += units * nav
                units = 0.0
                switches.append({"date": str(d.date()), "to": "caixa"})
            in_fund = want_fund
        # T2: no primeiro pregao do mes, vende as posicoes do mes anterior e compra as novas
        if t2 and d in starts:
            for t in list(pos):
                p = opx(t, d)
                if not np.isfinite(p):
                    continue
                b = pos.pop(t)
                proceeds = b["n"] * p * (1 - slippage) * (1 - brokerage)
                cash += proceeds
                trades.append({"ticker": t, "familia": "T2", "data_compra": b["date"], "preco_compra": b["px"],
                               "data_venda": str(d.date()), "preco_venda": p, "resultado_liq": proceeds / b["paid"] - 1.0,
                               "status": "fechada", "pregoes": i - b["i"]})
            slot = last_eq / k
            for t in pending:
                if t in pos or len(pos) >= k:
                    continue
                p = opx(t, d)
                if not np.isfinite(p):
                    continue
                need = slot - cash
                if need > 0 and units > 0 and np.isfinite(nav):
                    u = min(units, need / nav)
                    units -= u
                    cash += u * nav
                amount = min(slot, cash)
                if amount < 0.5 * slot:
                    continue
                pos[t] = {"n": amount * (1 - brokerage) / (p * (1 + slippage)), "paid": amount, "date": str(d.date()),
                          "px": p, "i": i, "last": p}
                cash -= amount
            pending = []
        # caixa parado volta para onde o capital ocioso deve ficar
        if in_fund and cash > 1e-9 and np.isfinite(nav):
            units += cash / nav
            cash = 0.0
        for t, b in pos.items():
            v = data.adj.at[d, t]
            if np.isfinite(v):
                b["last"] = float(v)
        stock = sum(b["n"] * b["last"] for b in pos.values())
        fund = units * nav if np.isfinite(nav) else 0.0
        eq = cash + stock + fund
        equity[d], invw[d], fundw[d], cashw[d], slots[d] = eq, stock / eq, fund / eq, cash / eq, len(pos)
        last_eq = eq
        # decisao de T2 para o mes seguinte, no ultimo pregao do mes
        if t2 and i + 1 < len(data.days) and data.days[i + 1] in starts and data.days[i + 1] <= end:
            pending = predicted_payers(data, d, data.days[i + 1], k)
    for t, b in pos.items():
        val = b["n"] * b["last"] * (1 - slippage) * (1 - brokerage)
        trades.append({"ticker": t, "familia": "T2", "data_compra": b["date"], "preco_compra": b["px"], "data_venda": "",
                       "preco_venda": b["last"], "resultado_liq": val / b["paid"] - 1.0, "status": "aberta",
                       "pregoes": pos_of[days[-1]] - b["i"]})
    cols = ["ticker", "familia", "data_compra", "preco_compra", "data_venda", "preco_venda", "resultado_liq", "status", "pregoes"]
    return Result7(pd.Series(equity).sort_index(), pd.DataFrame(trades, columns=cols), pd.Series(invw).sort_index(),
                   pd.Series(fundw).sort_index(), pd.Series(cashw).sort_index(), pd.Series(slots).sort_index(), switches)


# --------------------------------------------------------------------------
# Testes
# --------------------------------------------------------------------------
def monthly_returns(level: pd.Series, start, end) -> pd.Series:
    s = level.dropna()
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    m = s.groupby([s.index.year, s.index.month]).last()
    m.index = pd.to_datetime([f"{y}-{mo:02d}-01" for y, mo in m.index])
    return np.log(m).diff().dropna()


def halloween_test(r: pd.Series, s: Dict) -> Dict:
    """z = +r nos meses de inverno, -r no verao; H0: media(z) <= 0 (inverno nao rende mais que verao)."""
    import selective_se as v3
    sign = pd.Series([1.0 if m in WINTER else -1.0 for m in r.index.month], index=r.index)
    z = sign * r
    w, su = r[sign > 0], r[sign < 0]
    return {"months": int(len(r)), "winter_mean": float(w.mean()), "summer_mean": float(su.mean()),
            "winter_minus_summer": float(w.mean() - su.mean()),
            "t_nw": float(v3.newey_west_t(z, s["tests"]["T1"]["nw_lags"])),
            "p_bootstrap": float(v3.stationary_bootstrap_p(z, mean_block=s["tests"]["T1"]["bootstrap_block"],
                                                           reps=s["tests"]["bootstrap_reps"])),
            "by_year": {int(y): {"winter": float(w[w.index.year == y].sum()) if (w.index.year == y).any() else None,
                                 "summer": float(su[su.index.year == y].sum()) if (su.index.year == y).any() else None}
                        for y in sorted(set(r.index.year))}}


def dividend_month_events(data: Data, start, end, k: int) -> pd.DataFrame:
    """Um evento por acao escolhida e mes: retorno abertura(1o pregao do mes) -> abertura(1o do mes seguinte),
    menos o das acoes do mesmo quintil de retorno de 63 dias no fechamento da decisao."""
    starts = month_starts(data.days)
    starts = starts[(starts >= pd.Timestamp(start)) & (starts <= pd.Timestamp(end))]
    rows = []
    for a, b in zip(starts[:-1], starts[1:]):
        ia = data.days.get_loc(a)
        dec = data.days[ia - 1]
        chosen = predicted_payers(data, dec, a, k)
        if not chosen:
            continue
        fwd = data.opx.loc[b] / data.opx.loc[a] - 1.0
        q = data.quint.loc[dec]
        el = data.eligible.loc[dec]
        for t in chosen:
            if not np.isfinite(fwd.get(t, np.nan)):
                continue
            same = (q == q[t]) & el & fwd.notna()
            same[t] = False
            ctrl = float(fwd[same].mean()) if same.sum() >= 3 else np.nan
            rows.append({"family": "T2", "ticker": t, "entry": a, "exit": b, "ret": float(fwd[t]), "ctrl": ctrl,
                         "ar": float(fwd[t]) - ctrl, "ar_ew": float(fwd[t] - fwd[el & fwd.notna()].drop(t).mean())})
    return pd.DataFrame(rows)


def dividend_test(ev: pd.DataFrame, s: Dict) -> Dict:
    import selective_se as v3
    x = ev.dropna(subset=["ar"])
    m = x.groupby(pd.DatetimeIndex(x["entry"]).to_period("M"))["ar"].mean()
    out = {"events": int(len(ev)), "events_measured": int(len(x)), "months": int(len(m)),
           "mean_ar": float(x["ar"].mean()) if len(x) else float("nan"),
           "hit_rate": float((x["ar"] > 0).mean()) if len(x) else float("nan"),
           "mean_ret": float(x["ret"].mean()) if len(x) else float("nan"),
           "mean_ar_vs_universe": float(x["ar_ew"].mean()) if len(x) else float("nan")}
    if len(m) >= 12:
        out["t_nw"] = float(v3.newey_west_t(m, s["tests"]["T2"]["nw_lags"]))
        out["p_bootstrap"] = float(v3.stationary_bootstrap_p(m, mean_block=s["tests"]["T2"]["bootstrap_block"],
                                                             reps=s["tests"]["bootstrap_reps"]))
    else:
        out["t_nw"], out["p_bootstrap"] = float("nan"), float("nan")
    by = x.groupby(pd.DatetimeIndex(x["entry"]).year)["ar"].agg(["mean", "size"]) if len(x) else pd.DataFrame()
    out["by_year"] = {int(y): {"mean_ar": float(r["mean"]), "n": int(r["size"])} for y, r in by.iterrows()}
    out["by_month"] = {int(mo): int(n) for mo, n in x.groupby(pd.DatetimeIndex(x["entry"]).month).size().items()} if len(x) else {}
    return out


def fold_excess(eq: pd.Series, bench: pd.Series, only_cash: Optional[pd.Series] = None) -> List[Dict]:
    """Excesso sobre o indice por fold; com `only_cash`, so nos pregoes em caixa (T1)."""
    import walkforward as wf
    b = bench.reindex(eq.index).ffill()
    xs = eq.pct_change() - b.pct_change()
    out = []
    for f in wf.folds():
        m = (xs.index >= pd.Timestamp(f.trade_start)) & (xs.index <= pd.Timestamp(f.trade_end))
        x = xs[m]
        if only_cash is not None:
            x = x[only_cash.reindex(x.index).fillna(0) > 0.5]
        out.append({"fold": f.k, "sessions": int(len(x)), "excess": float(x.sum()) if len(x) else None})
    return out


# --------------------------------------------------------------------------
# Avaliacao unica
# --------------------------------------------------------------------------
def dry_run(data: Data, s: Dict) -> Dict:
    k = s["strategy"]["k"]
    lo, hi = s["period"]
    starts = month_starts(data.days)
    starts = starts[(starts >= pd.Timestamp(lo)) & (starts <= pd.Timestamp(hi))]
    payers = {}
    for a in starts:
        dec = data.days[data.days.get_loc(a) - 1]
        y, m = a.year - 1, a.month
        win = data.dy[(data.dy.index.year == y) & (data.dy.index.month == m)]
        n_all = int(win.gt(0).any(axis=0).sum()) if not win.empty else 0
        payers[str(a.date())] = {"predicted": n_all, "chosen": len(predicted_payers(data, dec, a, k))}
    per_month = pd.DataFrame(payers).T
    ex_by_year = data.dy.gt(0).sum(axis=1).groupby(data.dy.index.year).sum()
    r = monthly_returns(data.price_index, *s["period"])
    return {"ex_dividend_events_by_year": {int(y): int(n) for y, n in ex_by_year.items() if y >= 2006},
            "chosen_by_calendar_month": per_month.groupby(pd.DatetimeIndex(per_month.index).month)["chosen"].sum().to_dict(),
            "months_with_positions": int((per_month["chosen"] > 0).sum()), "months": int(len(per_month)),
            "trades_per_year_T2": float(per_month["chosen"].sum() / max(1, len(per_month) / 12)),
            "index_months_for_T1": int(len(r)), "first_month": str(r.index[0].date()), "last_month": str(r.index[-1].date()),
            "fund_switches_per_year_T1": 2}


def evaluate(data: Data, s: Dict, out_dir: Path = REPORT_DIR) -> Dict:
    import portfolio_se as ps
    import rotation_se as rot
    import selective_se as v3
    import walkforward as wf
    import events_se as v4

    k = s["strategy"]["k"]
    lo, hi = s["period"]
    tests: Dict[str, Dict] = {}
    tests["T1"] = halloween_test(monthly_returns(data.price_index, lo, hi), s)
    ew_level = (1 + data.ew_ret.fillna(0)).cumprod()
    tests["T1_diag_ew_universe"] = halloween_test(monthly_returns(ew_level, s["tests"]["T1"]["diag_start"], hi), s)
    ev = dividend_month_events(data, lo, hi, k)
    tests["T2"] = dividend_test(ev, s)
    holm = v4.holm({f: tests[f]["p_bootstrap"] for f in ("T1", "T2")}, s["tests"]["alpha"])
    for f in ("T1", "T2"):
        r = tests[f]
        r["pass_holm"] = bool(holm[f])
        r["pass_t"] = bool(np.isfinite(r["t_nw"]) and r["t_nw"] >= s["tests"]["nw_t_min"])
        r["pass_size"] = bool(f == "T1" or (np.isfinite(r["mean_ar"]) and r["mean_ar"] >= s["tests"]["min_mean_ar_long"]))
        r["pass"] = r["pass_holm"] and r["pass_t"] and r["pass_size"]
    use_t1, use_t2 = tests["T1"]["pass"], tests["T2"]["pass"]
    folds = wf.folds()
    c_start, c_end = folds[0].trade_start, folds[-1].trade_end

    def crit(res: Result7, t1: bool, t2: bool) -> Dict:
        res.trades = rot.assign_folds(res.trades)
        v = ps.verdict(res, data.nav)
        out = {kk: val for kk, val in v.items() if kk != "bench_equity"}
        if t1 and not t2:
            # T1 nao tem trades: fold positivo = excesso sobre o indice > 0 nos pregoes em caixa (folds de verao)
            fx = fold_excess(res.equity, data.nav, res.cash_weight)
            summer = [f for f in fx if f["sessions"] > 0]
            out["per_fold_excess_in_cash"] = fx
            out["positive_folds"] = sum(1 for f in summer if f["excess"] > 0)
            out["crit_folds"] = bool(out["positive_folds"] >= s["criterion1_T1"]["min_positive_summer_folds"])
            out["pass"] = bool(out["crit_beats_index"] and out["crit_folds"] and out["crit_mdd"])
        return out

    res = simulate(data, c_start, c_end, use_t1, use_t2, k=k)
    c1 = crit(res, use_t1, use_t2)
    out = {"spec_version": s["version"], "period": s["period"], "tests": tests,
           "signals_used": [f for f, u in (("T1", use_t1), ("T2", use_t2)) if u], "criterion1": c1,
           "pass": bool((use_t1 or use_t2) and c1["pass"])}
    full = simulate(data, lo, hi, use_t1, use_t2, k=k)
    out["full_period"] = rot.metrics(full.equity, data.nav)
    out["full_period"]["trades"] = int(len(full.trades))
    out["full_period"]["fund_switches"] = int(len(full.switches))
    # Diagnosticos anunciados: cada hipotese sozinha e as duas juntas, no criterio 1 e no periodo inteiro
    diag = {}
    for name, a, b in (("T1", True, False), ("T2", False, True), ("T1+T2", True, True)):
        r6 = simulate(data, c_start, c_end, a, b, k=k)
        c = crit(r6, a, b)
        rf = simulate(data, lo, hi, a, b, k=k)
        mf = rot.metrics(rf.equity, data.nav)
        diag[name] = {"criterion1": {kk: c[kk] for kk in ("total_return", "bench_return", "mdd", "bench_mdd", "positive_folds", "trades", "pass")},
                      "full_cagr": mf["cagr"], "full_bench_cagr": mf["bench_cagr"], "full_mdd": mf["mdd"], "full_bench_mdd": mf["bench_mdd"],
                      "trades": int(len(rf.trades)), "switches": int(len(rf.switches))}
        if name == "T1":
            pd.DataFrame({"sistema": rf.equity, "indice": data.nav.reindex(rf.equity.index).ffill()}).to_csv(out_dir / "equity_T1_full.csv") if out_dir.exists() else None
    out["diagnostics"] = diag
    out["data_vintage"] = {"prices_se.parquet": v4.file_sha(REPO_ROOT / "output" / "prices_se.parquet"),
                           "riksbank_policy_rate.parquet": v4.file_sha(RATE_PATH)}
    out_dir.mkdir(parents=True, exist_ok=True)
    ev.to_csv(out_dir / "events_T2.csv", index=False)
    pd.DataFrame({"sistema": res.equity, "indice": data.nav.reindex(res.equity.index).ffill() * market_se.CAPITAL_SEK /
                  data.nav.reindex(res.equity.index).ffill().iloc[0]}).to_csv(out_dir / "equity.csv")
    res.trades.to_csv(out_dir / "trades.csv", index=False)
    pd.DataFrame(full.switches).to_csv(out_dir / "switches_full_period.csv", index=False)
    (out_dir / "verdict.json").write_text(json.dumps(out, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
    (out_dir / "verdict.md").write_text(verdict_markdown(out, s), encoding="utf-8")
    _chart(full, data.nav, out_dir / "equity.png", out["signals_used"])
    return out


def verdict_markdown(o: Dict, s: Dict) -> str:
    def p(x):
        return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:+.2f}%"

    def f3(x):
        return "-" if x is None or not np.isfinite(x) else f"{x:.3f}"

    def ok(b):
        return "PASSA" if b else "NAO PASSA"

    t1, t1d, t2 = o["tests"]["T1"], o["tests"]["T1_diag_ew_universe"], o["tests"]["T2"]
    c1 = o["criterion1"]
    L = ["# Avaliacao da estrategia v7 — calendario (Halloween e mes do dividendo)", "",
         f"Especificacao congelada em `data/strategy_se_v7.json` antes de qualquer retorno (ADR 0011). "
         f"Periodo: {o['period'][0]} a {o['period'][1]}. Nenhum parametro foi ajustado.", "",
         "## Testes", "",
         "| Hipotese | Amostra | Efeito | t NW | p bootstrap | Holm | |", "|---|---:|---:|---:|---:|---|---|",
         f"| T1 Halloween: inverno (nov-abr) - verao (mai-out), ^OMXSPI mensal | {t1['months']} meses | "
         f"{p(t1['winter_minus_summer'])}/mes (inverno {p(t1['winter_mean'])}, verao {p(t1['summer_mean'])}) | "
         f"{t1['t_nw']:.2f} | {f3(t1['p_bootstrap'])} | {ok(t1['pass_holm'])} | **{ok(t1['pass'])}** |",
         f"| T2 mes do dividendo: retorno anormal vs mesmo quintil | {t2['events_measured']} eventos em {t2['months']} meses | "
         f"{p(t2['mean_ar'])}/evento (acerto {p(t2['hit_rate'])}) | {t2['t_nw']:.2f} | {f3(t2['p_bootstrap'])} | "
         f"{ok(t2['pass_holm'])} | **{ok(t2['pass'])}** (tamanho {ok(t2['pass_size'])}) |",
         "", f"Diagnostico T1 no universo de peso igual desde {s['tests']['T1']['diag_start']}: "
         f"{p(t1d['winter_minus_summer'])}/mes, t {t1d['t_nw']:.2f}, p {f3(t1d['p_bootstrap'])} ({t1d['months']} meses).", "",
         f"Sinais usados na carteira: {', '.join(o['signals_used']) or 'nenhum'}.", "",
         "## Criterio 1 (2023-10-31 a 2026-10-30)", "",
         f"- Retorno {p(c1['total_return'])} vs indice {p(c1['bench_return'])} -> {ok(c1['crit_beats_index'])}",
         f"- Folds positivos: {c1['positive_folds']} -> {ok(c1['crit_folds'])}",
         f"- Max drawdown {p(c1['mdd'])} vs indice {p(c1['bench_mdd'])} -> {ok(c1['crit_mdd'])}",
         f"- Trades: {c1['trades']}", "", f"**{'APROVADA para paper trading' if o['pass'] else 'REPROVADA'}**", "",
         "## Periodo inteiro", ""]
    fp = o["full_period"]
    L += [f"- CAGR {p(fp['cagr'])} vs indice {p(fp['bench_cagr'])}; max drawdown {p(fp['mdd'])} vs {p(fp['bench_mdd'])}; "
          f"{fp['trades']} trades; {fp['fund_switches']} trocas de fundo", "",
          "## Diagnosticos anunciados (nao decidem)", "",
          "| Carteira | 2023-2026 | Folds | MDD | CAGR periodo inteiro | Trades (periodo) |", "|---|---:|---:|---:|---:|---:|"]
    for name, d in o["diagnostics"].items():
        c = d["criterion1"]
        L.append(f"| {name} | {p(c['total_return'])} vs {p(c['bench_return'])} | {c['positive_folds']} | "
                 f"{p(c['mdd'])} vs {p(c['bench_mdd'])} | {p(d['full_cagr'])} vs {p(d['full_bench_cagr'])} | {d['trades']} |")
    L += ["", "## T1 por ano (soma dos retornos mensais do ^OMXSPI)", "", "| Ano | Inverno | Verao |", "|---|---:|---:|"]
    for y, r in t1["by_year"].items():
        L.append(f"| {y} | {p(r['winter'])} | {p(r['summer'])} |")
    L += ["", "## T2 por ano (retorno anormal medio, eventos)", "", "| Ano | T2 |", "|---|---:|"]
    for y, r in t2["by_year"].items():
        L.append(f"| {y} | {p(r['mean_ar'])} ({r['n']}) |")
    L += ["", f"T2, eventos por mes do calendario: {t2['by_month']}", "",
          "Versao dos dados (sha256): " + ", ".join(f"`{k}` {v}" for k, v in o["data_vintage"].items()), ""]
    return "\n".join(L)


def _chart(res: Result7, nav: pd.Series, path: Path, used: List[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eq = res.equity
    b = nav.reindex(eq.index).ffill()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(eq.index, eq.values, label=f"Estrategia v7 ({', '.join(used) or 'so fundo'})", color="#1565c0")
    ax.plot(b.index, eq.iloc[0] * b / b.iloc[0], label=f"Indice {market_se.BENCHMARK_SYMBOL} (com dividendos)", color="#9e9e9e")
    ax.set_ylabel("SEK")
    ax.set_title("Estrategia v7 — calendario vs index benchmark")
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
    data = load()
    if a.cmd == "dry-run":
        print(json.dumps(dry_run(data, s), indent=1, default=str))
        return 0
    o = evaluate(data, s)
    print((REPORT_DIR / "verdict.md").read_text(encoding="utf-8"))
    return 0 if o["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
