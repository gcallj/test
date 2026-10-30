#!/usr/bin/env python3
"""
Portfolio replay: todas as ordens do replay walk-forward numa unica conta ISK,
comparada ao index benchmark (fundo do indice amplo sueco na mesma ISK).

    python portfolio_se.py                    # cenario base (courtage Mini 0,25%)
    python portfolio_se.py --brokerage 0      # cenario otimista (courtage zero)

Le reports/se/orders.csv (gerado por `python replay_se.py all`) e escreve
reports/se/portfolio*.{md,csv,png}.

Regras da conta (docs/sweden/plano.md):
- capital inicial 30.000 SEK, posicoes de 5.000 SEK, no maximo 6 ao mesmo tempo;
- uma ordem aberta reserva a vaga e o saldo do dia seguinte ao sinal ate
  executar ou expirar (na ISK a corretora bloqueia o saldo de ordens abertas);
- quando ha mais ordens que vagas, entram as de maior forca de sinal;
- sem imposto por trade (ISK); a taxa anual da ISK incide igual no benchmark.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import market_se  # noqa: E402
import walkforward as wf  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "se"
BENCHMARK_CACHE = REPO_ROOT / "output" / "benchmark_close.csv"


@dataclass
class Account:
    capital: float = market_se.CAPITAL_SEK
    size: float = market_se.POSITION_SIZE_SEK
    max_positions: int = market_se.MAX_POSITIONS


@dataclass
class Result:
    equity: pd.Series
    taken: pd.DataFrame
    skipped: int
    slots_used: pd.Series
    notes: List[str] = field(default_factory=list)


def adjust_for_brokerage(orders: pd.DataFrame, brokerage: float) -> pd.DataFrame:
    """Recalcula o resultado liquido para outra courtage.

    Com custos percentuais, o resultado ja paga a courtage base nas duas pontas
    (as saidas parciais somam 100% da posicao), entao a troca e exata.
    """
    out = orders.copy()
    delta = 2.0 * (market_se.BROKERAGE_PCT_PER_ORDER - brokerage)
    out["resultado_liq"] = out["resultado_liq"].astype(float) + np.where(out["resultado_liq"].notna(), delta, 0.0)
    return out


def simulate(orders: pd.DataFrame, closes: pd.DataFrame, days: pd.DatetimeIndex,
             account: Account = Account()) -> Result:
    """Roda a conta pregao a pregao.

    `orders` usa as colunas de reports/se/orders.csv; `closes` e um DataFrame
    data x ticker com fechamentos para marcar as posicoes a mercado.
    """
    o = orders.copy()
    for col in ("data_sinal", "data_compra", "data_venda", "data_fim_ordem"):
        o[col] = pd.to_datetime(o[col].replace("", np.nan), errors="coerce")
    o = o.sort_values(["data_sinal", "forca"], ascending=[True, False]).reset_index(drop=True)
    by_signal: Dict[pd.Timestamp, pd.DataFrame] = {d: g for d, g in o.groupby("data_sinal")}

    cash = account.capital
    live: List[Dict] = []        # accepted orders not yet resolved
    taken_rows: List[Dict] = []
    skipped = 0
    equity = {}
    slots = {}

    for t in days:
        # 1) resolve what ends today
        still = []
        for a in live:
            if a["state"] == "position" and a["exit"] == t:
                cash += account.size * (1.0 + a["ret"])
                taken_rows.append(a["row"])
            elif a["state"] == "order" and a["fill"] == t:
                a["state"] = "position"
                still.append(a)
            elif a["state"] == "order" and a["end"] == t:
                cash += account.size  # reservation released
            else:
                still.append(a)
        live = still
        # A position filled today can also exit today (stop on the fill bar).
        still = []
        for a in live:
            if a["state"] == "position" and a["exit"] == t and a["fill"] == t:
                cash += account.size * (1.0 + a["ret"])
                taken_rows.append(a["row"])
            else:
                still.append(a)
        live = still

        # 2) new orders decided at today's close
        todays = by_signal.get(t)
        if todays is not None:
            for _, r in todays.iterrows():
                if len(live) >= account.max_positions or cash < account.size - 1e-9:
                    skipped += 1
                    continue
                status = r["status"]
                if status == "aberta":
                    # still open at the end of the data: reserve until the end
                    end = days[-1] + pd.Timedelta(days=1)
                    fill = r["data_compra"] if pd.notna(r["data_compra"]) else None
                    ret = 0.0
                elif status == "executada":
                    fill, end, ret = r["data_compra"], None, float(r["resultado_liq"])
                else:
                    fill, end, ret = None, r["data_fim_ordem"], 0.0
                cash -= account.size
                live.append({"state": "order", "fill": fill, "end": end, "exit": r["data_venda"],
                             "ret": ret, "ticker": r["ticker"], "entry": r["preco_compra"], "row": r})
                if fill is not None and status == "aberta":
                    live[-1]["state"] = "position" if fill <= t else "order"

        # 3) mark to market at the close
        mtm = cash
        for a in live:
            if a["state"] == "position" and pd.notna(a["entry"]) and a["ticker"] in closes.columns:
                px = closes.at[t, a["ticker"]] if t in closes.index else np.nan
                mtm += account.size * (px / a["entry"]) if np.isfinite(px) else account.size
            else:
                mtm += account.size
        equity[t] = mtm
        slots[t] = len(live)

    # positions still open at the end: count them at their last mark
    taken = pd.DataFrame([a for a in taken_rows])
    return Result(pd.Series(equity).sort_index(), taken, skipped, pd.Series(slots).sort_index())


def max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    return float((series / peak - 1.0).min())


def cagr(series: pd.Series) -> float:
    years = (series.index[-1] - series.index[0]).days / 365.25
    return float((series.iloc[-1] / series.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0


def load_benchmark(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if BENCHMARK_CACHE.exists():
        s = pd.read_csv(BENCHMARK_CACHE, index_col=0, parse_dates=True).iloc[:, 0]
        if s.index.min() <= start and s.index.max() >= end - pd.Timedelta(days=5):
            return s
    import yfinance as yf
    d = yf.download(market_se.BENCHMARK_SYMBOL, start=str((start - pd.Timedelta(days=10)).date()),
                    end=str((end + pd.Timedelta(days=3)).date()), auto_adjust=False, progress=False)
    s = d["Close"].iloc[:, 0] if isinstance(d["Close"], pd.DataFrame) else d["Close"]
    s.index = pd.to_datetime(s.index).normalize()
    BENCHMARK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    s.rename(market_se.BENCHMARK_SYMBOL).to_csv(BENCHMARK_CACHE)
    return s


def load_closes(tickers: List[str]) -> pd.DataFrame:
    import replay_se
    cache = replay_se.load_cache()
    cols = {}
    for tk in tickers:
        if tk in cache:
            p = cache[tk]
            cols[tk] = pd.Series(np.asarray(p["close"], dtype=float),
                                 index=pd.to_datetime(pd.Series(p["dates"])).dt.normalize().to_numpy())
    return pd.DataFrame(cols).sort_index()


def verdict(res: Result, bench: pd.Series) -> Dict:
    eq = res.equity
    b = bench.reindex(eq.index).ffill().dropna()
    eq = eq.reindex(b.index)
    bench_eq = market_se.CAPITAL_SEK * b / b.iloc[0]
    folds = wf.folds()
    per_fold = []
    t = res.taken.copy()
    if not t.empty:
        t["resultado_liq"] = t["resultado_liq"].astype(float)
    for f in folds:
        g = t[t["fold"] == f.k] if not t.empty else t
        per_fold.append({
            "fold": f.k, "trades": int(len(g)),
            "expectancy": float(g["resultado_liq"].mean()) if len(g) else np.nan,
            "win_rate": float((g["resultado_liq"] > 0).mean()) if len(g) else np.nan,
        })
    pos_folds = sum(1 for r in per_fold if np.isfinite(r["expectancy"]) and r["expectancy"] > 0)
    out = {
        "start": eq.index[0].date().isoformat(), "end": eq.index[-1].date().isoformat(),
        "final_equity": float(eq.iloc[-1]), "total_return": float(eq.iloc[-1] / market_se.CAPITAL_SEK - 1.0),
        "cagr": cagr(eq), "mdd": max_drawdown(eq),
        "bench_final": float(bench_eq.iloc[-1]), "bench_return": float(bench_eq.iloc[-1] / market_se.CAPITAL_SEK - 1.0),
        "bench_cagr": cagr(bench_eq), "bench_mdd": max_drawdown(bench_eq),
        "trades": int(len(t)), "skipped": res.skipped,
        "avg_slots": float(res.slots_used.mean()),
        "per_fold": per_fold, "positive_folds": pos_folds,
        "bench_equity": bench_eq,
    }
    out["crit_beats_index"] = out["total_return"] > out["bench_return"]
    out["crit_folds"] = pos_folds >= 4
    out["crit_mdd"] = abs(out["mdd"]) < abs(out["bench_mdd"])
    out["pass"] = bool(out["crit_beats_index"] and out["crit_folds"] and out["crit_mdd"])
    return out


def _p(x: float) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:+.1f}%"


def markdown(base: Dict, optimistic: Optional[Dict]) -> str:
    def mark(ok: bool) -> str:
        return "PASSA" if ok else "NAO PASSA"
    L = ["# Portfolio replay — mercado sueco (walk-forward)", "",
         f"Periodo: {base['start']} a {base['end']} · conta ISK de 30.000 SEK · "
         f"ate {market_se.MAX_POSITIONS} posicoes de {market_se.POSITION_SIZE_SEK:,.0f} SEK · "
         f"index benchmark `{market_se.BENCHMARK_SYMBOL}` (com dividendos)", "",
         "## Veredito (criterio 1 do plano)", "",
         "| Criterio | Sistema | Indice | Resultado |", "|---|---:|---:|---|",
         f"| Retorno total depois de custos | {_p(base['total_return'])} | {_p(base['bench_return'])} | {mark(base['crit_beats_index'])} |",
         f"| Folds com expectancy positiva | {base['positive_folds']} de 6 | precisa >= 4 | {mark(base['crit_folds'])} |",
         f"| Max drawdown da conta | {_p(base['mdd'])} | {_p(base['bench_mdd'])} | {mark(base['crit_mdd'])} |",
         "", f"**{'APROVADO para paper trading' if base['pass'] else 'REPROVADO: a resposta honesta e o fundo de indice'}**", "",
         "## Conta", "",
         f"- Patrimonio final: {base['final_equity']:,.0f} SEK (indice: {base['bench_final']:,.0f} SEK)",
         f"- CAGR: {_p(base['cagr'])} (indice: {_p(base['bench_cagr'])})",
         f"- Trades executados: {base['trades']} · ordens sem vaga: {base['skipped']} · "
         f"vagas ocupadas em media: {base['avg_slots']:.1f} de {market_se.MAX_POSITIONS}", "",
         "## Por fold", "", "| Fold | Trades | Win rate | Expectancy |", "|---:|---:|---:|---:|"]
    for r in base["per_fold"]:
        L.append(f"| {r['fold']} | {r['trades']} | {_p(r['win_rate'])} | {_p(r['expectancy'])} |")
    if optimistic is not None:
        L += ["", "## Cenario otimista (courtage zero)", "",
              f"Retorno {_p(optimistic['total_return'])} vs indice {_p(optimistic['bench_return'])} · "
              f"folds positivos {optimistic['positive_folds']} de 6 · MDD {_p(optimistic['mdd'])} · "
              f"{'passaria' if optimistic['pass'] else 'tambem nao passaria'} no criterio 1.",
              "", "Se o sistema so passa com courtage zero, isso e sinal de alerta (plano, Q19)."]
    L += ["", "Gerado por `python portfolio_se.py`. Curva da conta: `portfolio_equity.csv`, `portfolio.png`.", ""]
    return "\n".join(L)


def chart(base: Dict, res: Result, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(res.equity.index, res.equity.values, label="Sistema (conta ISK)", color="#1565c0")
    be = base["bench_equity"]
    ax.plot(be.index, be.values, label=f"Indice {market_se.BENCHMARK_SYMBOL} (com dividendos)", color="#9e9e9e")
    for f in wf.folds():
        ax.axvline(pd.Timestamp(f.trade_start), color="#bdbdbd", linestyle="--", linewidth=0.8)
    ax.set_ylabel("SEK")
    ax.set_title("Portfolio replay walk-forward vs index benchmark")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", default=str(REPORT_DIR / "orders.csv"))
    args = ap.parse_args(argv)

    orders = pd.read_csv(args.orders, keep_default_na=True)
    for col in ("data_compra", "data_venda", "data_fim_ordem"):
        orders[col] = orders[col].fillna("")
    folds = wf.folds()
    start, end = pd.Timestamp(folds[0].trade_start), pd.Timestamp(folds[-1].trade_end)
    closes = load_closes(sorted(orders["ticker"].unique()))
    days = closes.index[(closes.index >= start) & (closes.index <= end)]
    bench = load_benchmark(start, end)

    res = simulate(orders, closes, days)
    base = verdict(res, bench)
    res0 = simulate(adjust_for_brokerage(orders, 0.0), closes, days)
    opt = verdict(res0, bench)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sistema": res.equity, "indice": base["bench_equity"]}).to_csv(REPORT_DIR / "portfolio_equity.csv")
    if not res.taken.empty:
        res.taken.to_csv(REPORT_DIR / "portfolio_trades.csv", index=False)
    (REPORT_DIR / "portfolio.md").write_text(markdown(base, opt), encoding="utf-8")
    chart(base, res, REPORT_DIR / "portfolio.png")
    print(markdown(base, opt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
