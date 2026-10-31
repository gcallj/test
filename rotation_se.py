#!/usr/bin/env python3
"""
Estrategia v2 do mercado sueco: rotacao mensal por momentum relativo
(docs/adr/0005, docs/sweden/diagnostico.md).

Regra, avaliada no ultimo pregao de cada mes (sinal no fechamento, execucao na
abertura do pregao seguinte):

1. Elegivel: historico suficiente e giro mediano de 60 dias >= 10 MSEK
   (liquidez medida na data, nao no snapshot).
2. Score: retorno de `lookback` pregoes ate `skip` pregoes atras (momentum
   12-1 por padrao), opcionalmente dividido pela volatilidade diaria.
3. Filtro de tendencia da acao (opcional): preco acima da media de 200 dias e
   momentum positivo.
4. Filtro de mercado (opcional): se o regime e desfavoravel a carteira vai
   inteira para caixa. `regime_rule`: "sma200" (benchmark acima da media de 200
   dias) ou "abs12" (retorno de 12 meses do benchmark positivo).
5. Carteira: ate `k` acoes, peso igual. Uma acao ja na carteira fica enquanto
   estiver entre as `buffer` melhores (giro menor); vagas novas vao para as
   melhores fora da carteira.
6. Volatility targeting (opcional): exposicao em acoes = min(1, alvo / vol
   realizada de 63 dias da cesta escolhida); o resto fica em caixa. Posicoes
   mantidas so sao redimensionadas quando se afastam mais que `resize_band`.
7. Vagas sem acao: fundo de indice (`fill="index"`, sem courtage, cota do dia
   da execucao) ou caixa.

Custos: courtage + slippage por ordem de acao (market_se.py). Precos
ajustados por dividendos (prices_se.py).

    python rotation_se.py run --start 2014-01-01 --end 2023-10-30 [--spec '{...}']
    python rotation_se.py evaluate      # avaliacao unica da especificacao congelada
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import market_se  # noqa: E402

MIN_TURNOVER_SEK = market_se.MIN_MEDIAN_TURNOVER_SEK
FFILL_LIMIT = 5      # dias sem negocio tolerados ao marcar a mercado
VOL_WINDOW = 63      # pregoes para a volatilidade realizada da cesta


@dataclass(frozen=True)
class Spec:
    lookback: int = 252
    skip: int = 21
    risk_adjust: bool = False
    k: int = market_se.MAX_POSITIONS
    buffer: int = 12
    stock_trend: bool = True
    market_regime: bool = True
    regime_rule: str = "sma200"     # "sma200" | "abs12"
    vol_target: float = 0.0         # 0 = desligado; ex.: 0.20 = 20% a.a.
    resize_band: float = 0.25
    fill: str = "index"             # "index" | "cash"

    def label(self) -> str:
        reg = f"-{self.regime_rule}" if self.market_regime else ""
        vt = f"-vt{int(round(100 * self.vol_target))}" if self.vol_target > 0 else ""
        return (f"L{self.lookback}-S{self.skip}{'-ra' if self.risk_adjust else ''}-K{self.k}-B{self.buffer}"
                f"{'-tr' if self.stock_trend else ''}{reg}{vt}-{self.fill}")


@dataclass
class Costs:
    brokerage: float = market_se.BROKERAGE_PCT_PER_ORDER
    slippage: float = market_se.SLIPPAGE_PCT_PER_SIDE


@dataclass
class SimResult:
    equity: pd.Series
    trades: pd.DataFrame          # one row per stock position or partial sale
    rebalances: List[Dict]        # what was decided at each signal date
    fund_weight: pd.Series        # share of equity in the index fund, per day
    invested: pd.Series           # share of equity in stocks, per day
    skipped: int = 0
    slots_used: pd.Series = field(default_factory=pd.Series)
    holdings: Dict[str, Dict] = field(default_factory=dict)  # open at the end

    @property
    def taken(self) -> pd.DataFrame:  # compatibility with portfolio_se.verdict
        return self.trades


# --------------------------------------------------------------------------
# Signals (everything at date d uses data up to d)
# --------------------------------------------------------------------------
class Market:
    """Precos e indicadores pre-calculados, compartilhados entre especificacoes."""

    def __init__(self, px: Dict[str, pd.DataFrame]):
        self.px = px
        eq = [t for t in px["adj"].columns if market_se.is_equity(t)]
        self.tickers = eq
        self.days = px["adj"].index
        self.adj = px["adj"][eq].ffill(limit=FFILL_LIMIT)
        self.raw_adj = px["adj"][eq]
        self.adj_open = px["adj_open"][eq]
        self.ret = self.adj.pct_change(fill_method=None)
        self.sma200 = self.adj.rolling(200, min_periods=150).mean()
        self.liquid = px["turnover"][eq].rolling(60, min_periods=40).median() >= MIN_TURNOVER_SEK
        b = px["adj"][market_se.BENCHMARK_SYMBOL].ffill(limit=FFILL_LIMIT)
        self.bench = b
        self.regime = {
            "sma200": b > b.rolling(200, min_periods=150).mean(),
            "abs12": (b / b.shift(252) - 1.0) > 0,
        }
        self._cache: Dict[Tuple, pd.DataFrame] = {}

    def raw_momentum(self, lookback: int, skip: int) -> pd.DataFrame:
        key = ("raw", lookback, skip)
        if key not in self._cache:
            enough = self.raw_adj.notna().rolling(lookback + 1, min_periods=1).sum() >= 0.9 * (lookback + 1)
            self._cache[key] = (self.adj.shift(skip) / self.adj.shift(lookback) - 1.0).where(enough)
        return self._cache[key]

    def score(self, lookback: int, skip: int, risk_adjust: bool) -> pd.DataFrame:
        key = ("score", lookback, skip, risk_adjust)
        if key not in self._cache:
            mom = self.raw_momentum(lookback, skip)
            if risk_adjust:
                vol = self.ret.rolling(lookback, min_periods=int(0.8 * lookback)).std()
                mom = mom / vol.replace(0.0, np.nan)
            self._cache[key] = mom
        return self._cache[key]

    def basket_vol(self, d: pd.Timestamp, names: List[str]) -> float:
        """Vol anualizada de 63 dias de uma cesta de peso igual, ate d."""
        if not names:
            return np.nan
        i = self.days.get_loc(d)
        r = self.ret.iloc[max(0, i - VOL_WINDOW + 1): i + 1][names].mean(axis=1)
        return float(r.std() * np.sqrt(252)) if r.notna().sum() >= 20 else np.nan

    def month_ends(self, start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
        d = self.days[(self.days >= start) & (self.days <= end)]
        s = pd.Series(d, index=d)
        return list(s.groupby([d.year, d.month]).max())


def target_portfolio(mkt: Market, spec: Spec, d: pd.Timestamp, held: List[str]) -> Tuple[List[str], float, Dict]:
    """Acoes que a carteira deve ter apos o sinal de `d`, a exposicao em acoes e o porque."""
    info: Dict = {"date": str(d.date()), "spec": spec.label()}
    if spec.market_regime and not bool(mkt.regime[spec.regime_rule].get(d, False)):
        info["regime"] = f"risk-off ({spec.regime_rule})"
        return [], 0.0, info
    info["regime"] = "risk-on" if spec.market_regime else "sem filtro"
    sc = mkt.score(spec.lookback, spec.skip, spec.risk_adjust).loc[d]
    ok = sc.notna() & mkt.liquid.loc[d].fillna(False)
    if spec.stock_trend:
        raw = mkt.raw_momentum(spec.lookback, spec.skip).loc[d]
        ok &= (mkt.adj.loc[d] > mkt.sma200.loc[d]) & (raw > 0)
    ranked = sc[ok].sort_values(ascending=False)
    rank = {t: i + 1 for i, t in enumerate(ranked.index)}
    keep = [t for t in held if rank.get(t, 10 ** 9) <= spec.buffer]
    new = [t for t in ranked.index if t not in keep][: max(0, spec.k - len(keep))]
    target = keep + new
    exposure = 1.0
    if spec.vol_target > 0 and target:
        v = mkt.basket_vol(d, target)
        if np.isfinite(v) and v > 0:
            exposure = float(min(1.0, spec.vol_target / v))
        info["basket_vol"] = round(v, 4) if np.isfinite(v) else None
    info.update({"eligible": int(ok.sum()), "keep": keep, "buy": new, "exposure": round(exposure, 4),
                 "sell": [t for t in held if t not in target],
                 "top": [(t, round(float(ranked[t]), 4)) for t in ranked.index[: spec.buffer]]})
    return target, exposure, info


# --------------------------------------------------------------------------
# Portfolio simulation
# --------------------------------------------------------------------------
def simulate(mkt: Market, spec, start, end, capital: float = market_se.CAPITAL_SEK,
             costs: Costs = Costs(), initial: Optional[Dict] = None) -> SimResult:
    """Conta unica, pregao a pregao. `spec` pode ser uma funcao data -> Spec (walk-forward)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    spec_at: Callable[[pd.Timestamp], Spec] = spec if callable(spec) else (lambda _d, s=spec: s)
    days = mkt.days[(mkt.days >= start) & (mkt.days <= end)]
    if len(days) < 2:
        raise ValueError("periodo curto demais")
    signals = set([days[0]] + [d for d in mkt.month_ends(start, end) if d < days[-1]])

    cash = float(capital)
    shares: Dict[str, float] = {}
    basis: Dict[str, Dict] = {}
    fund_units = 0.0
    trades: List[Dict] = []
    rebalances: List[Dict] = []
    equity, fundw, invw, slots = {}, {}, {}, {}
    pending = None  # (target, exposure, equity at signal, spec, risk_off) decided at a close

    def nav(d) -> float:
        v = float(mkt.bench.get(d, np.nan))
        return v if np.isfinite(v) else np.nan

    def close_px(t, d) -> float:
        v = mkt.adj.at[d, t]
        return float(v) if np.isfinite(v) else np.nan

    def exec_px(t, d) -> float:
        p = mkt.adj_open.at[d, t]
        return float(p) if np.isfinite(p) and p > 0 else close_px(t, d)

    def sell(t, d, n_sell, status) -> None:
        nonlocal cash
        p = exec_px(t, d)
        if not np.isfinite(p) or n_sell <= 0:
            return
        frac = min(1.0, n_sell / shares[t])
        proceeds = n_sell * p * (1 - costs.slippage) * (1 - costs.brokerage)
        cash += proceeds
        b = basis[t]
        paid_part = b["paid"] * frac
        trades.append({"ticker": t, "data_compra": b["date"], "preco_compra": b["px"], "data_venda": str(d.date()),
                       "preco_venda": p, "resultado_liq": proceeds / paid_part - 1.0, "status": status,
                       "peso": paid_part})
        b["paid"] -= paid_part
        shares[t] -= n_sell
        if frac >= 1.0 - 1e-12:
            del shares[t]
            del basis[t]

    def buy(t, d, amount) -> None:
        nonlocal cash
        p = exec_px(t, d)
        amount = min(amount, cash)
        if not np.isfinite(p) or amount <= 0:
            return
        n = amount * (1 - costs.brokerage) / (p * (1 + costs.slippage))
        if t in shares:
            shares[t] += n
            basis[t]["paid"] += amount
        else:
            shares[t] = n
            basis[t] = {"date": str(d.date()), "px": p, "paid": amount, "last": p}
        cash -= amount

    for d in days:
        if pending is not None:
            target, exposure, eq_sig, sp, risk_off = pending
            pending = None
            slot = eq_sig * exposure / sp.k
            for t in [t for t in list(shares) if t not in target]:
                sell(t, d, shares[t], "fechada")
            for t in [t for t in target if t in shares]:  # trim oversized keepers
                p = exec_px(t, d)
                if np.isfinite(p) and shares[t] * p > slot * (1 + sp.resize_band):
                    sell(t, d, shares[t] - slot / p, "parcial")
            nv = nav(d)
            want_fund = sp.fill == "index" and np.isfinite(nv) and not risk_off
            fund_target = ((sp.k - len(target)) / sp.k) * eq_sig * exposure if want_fund else 0.0
            if np.isfinite(nv) and fund_units * nv > fund_target:
                cut = fund_units - fund_target / nv
                cash += cut * nv
                fund_units -= cut
            for t in target:
                p = exec_px(t, d)
                if not np.isfinite(p):
                    continue
                have = shares.get(t, 0.0) * p
                if t not in shares:
                    if min(slot, cash) >= 0.2 * slot:
                        buy(t, d, slot)
                elif have < slot * (1 - sp.resize_band):
                    buy(t, d, slot - have)
            if np.isfinite(nv) and fund_target > fund_units * nv:
                add = min(fund_target - fund_units * nv, cash)
                fund_units += add / nv
                cash -= add

        for t in shares:
            p = close_px(t, d)
            if np.isfinite(p):
                basis[t]["last"] = p
        nv = nav(d)
        stock_val = sum(n * basis[t]["last"] for t, n in shares.items())
        fund_val = fund_units * nv if np.isfinite(nv) else 0.0
        eq_now = cash + stock_val + fund_val
        equity[d] = eq_now
        fundw[d] = fund_val / eq_now if eq_now > 0 else 0.0
        invw[d] = stock_val / eq_now if eq_now > 0 else 0.0
        slots[d] = len(shares)

        if d in signals and d != days[-1]:
            sp = spec_at(d)
            target, exposure, info = target_portfolio(mkt, sp, d, list(shares))
            info["equity"] = round(float(eq_now), 2)
            rebalances.append(info)
            pending = (target, exposure, eq_now, sp, info["regime"].startswith("risk-off"))

    holdings = {}
    for t, n in shares.items():
        b = basis[t]
        val = n * b["last"] * (1 - costs.slippage) * (1 - costs.brokerage)
        trades.append({"ticker": t, "data_compra": b["date"], "preco_compra": b["px"], "data_venda": "",
                       "preco_venda": b["last"], "resultado_liq": val / b["paid"] - 1.0, "status": "aberta",
                       "peso": b["paid"]})
        holdings[t] = {"shares": n, "since": b["date"], "entry_px": b["px"], "last": b["last"]}
    return SimResult(pd.Series(equity).sort_index(), pd.DataFrame(trades), rebalances,
                     pd.Series(fundw).sort_index(), pd.Series(invw).sort_index(), 0,
                     pd.Series(slots).sort_index(), holdings)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metrics(eq: pd.Series, bench: pd.Series) -> Dict[str, float]:
    b = bench.reindex(eq.index).ffill().dropna()
    e = eq.reindex(b.index)
    yrs = (e.index[-1] - e.index[0]).days / 365.25

    def cagr(s):
        return float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) if yrs > 0 else 0.0

    def mdd(s):
        return float((s / s.cummax() - 1).min())

    out = {"cagr": cagr(e), "bench_cagr": cagr(b), "mdd": mdd(e), "bench_mdd": mdd(b),
           "total": float(e.iloc[-1] / e.iloc[0] - 1), "bench_total": float(b.iloc[-1] / b.iloc[0] - 1)}
    out["excess_cagr"] = out["cagr"] - out["bench_cagr"]
    out["score"] = out["excess_cagr"] - max(0.0, abs(out["mdd"]) - abs(out["bench_mdd"]))
    return out


SPEC_PATH = REPO_ROOT / "data" / "strategy_se_v2.json"
REPORT_DIR = REPO_ROOT / "reports" / "se_v2"


def frozen_spec() -> Spec:
    """A especificacao congelada (docs/adr/0006)."""
    return Spec(**json.loads(SPEC_PATH.read_text(encoding="utf-8"))["spec"])


def assign_folds(trades: pd.DataFrame) -> pd.DataFrame:
    import walkforward as wf
    out = trades.copy()
    if out.empty:
        out["fold"] = []
        return out
    entry = pd.to_datetime(out["data_compra"])
    out["fold"] = 0
    for f in wf.folds():
        m = (entry >= pd.Timestamp(f.trade_start)) & (entry <= pd.Timestamp(f.trade_end))
        out.loc[m, "fold"] = f.k
    return out


def evaluate(mkt: Market, spec: Spec, out_dir: Path = REPORT_DIR) -> Dict:
    """Roda o periodo de avaliacao (6 folds) e escreve o veredito do criterio 1."""
    import portfolio_se as ps
    import walkforward as wf

    folds = wf.folds()
    start, end = folds[0].trade_start, folds[-1].trade_end
    out = {}
    for name, costs in (("base", Costs()), ("courtage_zero", Costs(brokerage=0.0))):
        res = simulate(mkt, spec, start, end, costs=costs)
        res.trades = assign_folds(res.trades)
        out[name] = (res, ps.verdict(res, mkt.bench))
    res, v = out["base"]
    _, v0 = out["courtage_zero"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sistema": res.equity, "indice": v["bench_equity"]}).to_csv(out_dir / "equity.csv")
    res.trades.to_csv(out_dir / "trades.csv", index=False)
    pd.DataFrame([{**r, "keep": ", ".join(r.get("keep", [])), "buy": ", ".join(r.get("buy", [])),
                   "sell": ", ".join(r.get("sell", [])),
                   "top": ", ".join(f"{t} ({sc:.2f})" for t, sc in r.get("top", []))}
                  for r in res.rebalances]).to_csv(out_dir / "rebalances.csv", index=False)
    (out_dir / "verdict.md").write_text(verdict_markdown(spec, res, v, v0), encoding="utf-8")
    summary = {k: x for k, x in v.items() if k != "bench_equity"}
    summary["optimistic_pass"] = v0["pass"]
    summary["optimistic_total_return"] = v0["total_return"]
    (out_dir / "verdict.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    _chart(res, v, out_dir / "equity.png")
    return v


def verdict_markdown(spec: Spec, res: SimResult, v: Dict, v0: Dict) -> str:
    def p(x):
        return "-" if x is None or not np.isfinite(x) else f"{100 * x:+.1f}%"
    def ok(b):
        return "PASSA" if b else "NAO PASSA"
    t = res.trades
    yrs = (res.equity.index[-1] - res.equity.index[0]).days / 365.25
    full = t[t["status"] != "parcial"] if not t.empty else t
    L = ["# Avaliacao da estrategia v2 — rotacao mensal por momentum", "",
         f"Especificacao congelada em `data/strategy_se_v2.json` (`{spec.label()}`), escolhida so com dados ate "
         f"2023-10-30 e avaliada aqui uma unica vez (ADR 0006).", "",
         f"Periodo: {v['start']} a {v['end']} · conta ISK de {market_se.CAPITAL_SEK:,.0f} SEK · "
         f"index benchmark `{market_se.BENCHMARK_SYMBOL}` (com dividendos)", "",
         "## Veredito (criterio 1 do plano)", "",
         "| Criterio | Sistema | Indice | Resultado |", "|---|---:|---:|---|",
         f"| Retorno total depois de custos | {p(v['total_return'])} | {p(v['bench_return'])} | {ok(v['crit_beats_index'])} |",
         f"| Folds com expectancy positiva | {v['positive_folds']} de 6 | precisa >= 4 | {ok(v['crit_folds'])} |",
         f"| Max drawdown da conta | {p(v['mdd'])} | {p(v['bench_mdd'])} | {ok(v['crit_mdd'])} |", "",
         f"**{'APROVADO para paper trading' if v['pass'] else 'REPROVADO'}**", "",
         "## Conta", "",
         f"- Patrimonio final: {v['final_equity']:,.0f} SEK (indice: {v['bench_final']:,.0f} SEK)",
         f"- CAGR: {p(v['cagr'])} (indice: {p(v['bench_cagr'])})",
         f"- Posicoes abertas por ano: {len(full) / yrs:.1f} · exposicao media em acoes: {100 * res.invested.mean():.0f}%",
         "", "## Por fold (trades pela data de compra; posicoes abertas no fim marcadas a mercado)", "",
         "| Fold | Trades | Win rate | Expectancy | Max drawdown da conta |", "|---:|---:|---:|---:|---:|"]
    for r in v["per_fold"]:
        L.append(f"| {r['fold']} | {r['trades']} | {p(r['win_rate'])} | {p(r['expectancy'])} | {p(r['mdd'])} |")
    L += ["", "## Cenario com courtage zero", "",
          f"Retorno {p(v0['total_return'])} · folds positivos {v0['positive_folds']} de 6 · MDD {p(v0['mdd'])} · "
          f"{'passaria' if v0['pass'] else 'nao passaria'} no criterio 1.", "",
          "Arquivos: `equity.csv`, `equity.png`, `trades.csv`, `rebalances.csv` (o que foi decidido em cada "
          "fim de mes e por que).", ""]
    return "\n".join(L)


def _chart(res: SimResult, v: Dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import walkforward as wf
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(res.equity.index, res.equity.values, label="Estrategia v2 (conta ISK)", color="#1565c0")
    be = v["bench_equity"]
    ax.plot(be.index, be.values, label=f"Indice {market_se.BENCHMARK_SYMBOL} (com dividendos)", color="#9e9e9e")
    for f in wf.folds():
        ax.axvline(pd.Timestamp(f.trade_start), color="#bdbdbd", linestyle="--", linewidth=0.8)
    ax.set_ylabel("SEK")
    ax.set_title("Estrategia v2 — avaliacao fora da amostra vs index benchmark")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def load_market() -> Market:
    import prices_se
    return Market(prices_se.load())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--start", required=True)
    r.add_argument("--end", required=True)
    r.add_argument("--spec", help="JSON com campos de Spec")
    sub.add_parser("evaluate", help="avaliacao unica da especificacao congelada (ADR 0006)")
    args = ap.parse_args(argv)
    mkt = load_market()
    if args.cmd == "evaluate":
        v = evaluate(mkt, frozen_spec())
        print((REPORT_DIR / "verdict.md").read_text(encoding="utf-8"))
        return 0
    spec = Spec(**json.loads(args.spec)) if args.spec else Spec()
    res = simulate(mkt, spec, args.start, args.end)
    m = metrics(res.equity, mkt.bench)
    print(spec.label(), {k: round(v, 4) for k, v in m.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
