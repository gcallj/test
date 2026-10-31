#!/usr/bin/env python3
"""
Estrategia v5 do mercado sueco: revisoes de analistas (docs/adr/0009).

Fonte: artigos do Börsvärlden lidos por `analyst_se.py` / `analyst_parse.py`.
Tres hipoteses da literatura, congeladas em `data/strategy_se_v5.json` antes
de ligar as revisoes a retornos:

- R1 elevacao de recomendacao (ou inicio de cobertura com compra) antecede
  alta (Womack 1996; Jegadeesh & Kim 2006; Irvine 2003 para inicios);
- R2 rebaixamento antecede queda por meses (Womack 1996); numa ISK serve como
  filtro: o nome rebaixado nao e comprado;
- R3 mais precos-alvo elevados do que cortados no dia antecede alta
  (Brav & Lehavy 2003; Asquith, Mikhail & Au 2005).

So artigos de corretoras: pesquisa encomendada ("motiverat varde", paga pela
empresa) e dicas de revista ficam de fora. Entrada na abertura do primeiro
pregao depois do dia da publicacao; retorno anormal contra o mesmo quintil de
retorno de 63 dias (events_se.abnormal_returns); testes e criterio 1 como na v4.

    python revisions_se.py dry-run    # contagens, sem nenhum retorno
    python revisions_se.py evaluate   # avaliacao unica (reports/se_v5)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import altdata_se as alt  # noqa: E402
import analyst_se  # noqa: E402
import events_se as v4  # noqa: E402
import market_se  # noqa: E402

SPEC_PATH = REPO_ROOT / "data" / "strategy_se_v5.json"
REPORT_DIR = REPO_ROOT / "reports" / "se_v5"
FAMILIES = ("R1", "R2", "R3")
LONG_FAMILIES = ("R1", "R3")


def spec() -> Dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def build_events(mkt: v4.Mkt, acts: pd.DataFrame, s: Dict) -> Dict[str, pd.DataFrame]:
    """Eventos por ticker e pregao de entrada (somando os artigos de corretoras daquele dia)."""
    x = acts[(acts["source"] == "broker") & acts["ticker"].isin(mkt.tickers)].dropna(subset=["published"]).copy()
    x["entry"] = alt.next_session_after(mkt.days, x["published"], 1)
    x = x.dropna(subset=["entry"])
    g = x.groupby(["ticker", "entry"]).agg(up=("n_up", "sum"), down=("n_down", "sum"), ip=("n_init_pos", "sum"),
                                           ineg=("n_init_neg", "sum"), tup=("n_t_up", "sum"), tdown=("n_t_down", "sum"),
                                           n=("url", "nunique")).reset_index()
    g["rating_net"] = g["up"] + g["ip"] - g["down"] - g["ineg"]
    g["target_net"] = g["tup"] - g["tdown"]
    lo, hi = pd.Timestamp(s["period"][0]), pd.Timestamp(s["period"][1])
    c = s["signals"]
    sel = {
        "R1": g[g["rating_net"] >= c["R1"]["min_rating_net"]],
        "R2": g[g["rating_net"] <= -c["R2"]["min_rating_net"]],
        "R3": g[(g["target_net"] >= c["R3"]["min_target_net"]) & (g["rating_net"] >= 0)],
    }
    out = {}
    for f, e in sel.items():
        ev = pd.DataFrame({"family": f, "ticker": e["ticker"], "entry": e["entry"],
                           "magnitude": (e["rating_net"] if f != "R3" else e["target_net"]).abs().astype(float),
                           "info": e["n"].astype(str) + " artigo(s)"})
        ev = v4._cooldown(ev, mkt.days, c[f]["cooldown_sessions"])
        out[f] = ev[(ev["entry"] >= lo) & (ev["entry"] <= hi)].reset_index(drop=True)
    return out


def block_windows(mkt: v4.Mkt, ev: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Rebaixamento em D bloqueia a compra do nome de D ate D+H pregoes."""
    if ev.empty:
        return pd.DataFrame(columns=["ticker", "start", "end"])
    p = mkt.days.get_indexer(pd.DatetimeIndex(ev["entry"]))
    end = mkt.days[np.minimum(p + horizon, len(mkt.days) - 1)]
    return pd.DataFrame({"ticker": ev["ticker"].values, "start": ev["entry"].values, "end": end})


def dry_run(mkt: v4.Mkt, acts: pd.DataFrame, s: Dict) -> Dict:
    ev = build_events(mkt, acts, s)
    return {f: {"events": int(len(e)), "tickers": int(e["ticker"].nunique()) if len(e) else 0,
                "by_year": e.groupby(pd.DatetimeIndex(e["entry"]).year).size().to_dict() if len(e) else {}}
            for f, e in ev.items()}


def evaluate(mkt: v4.Mkt, acts: pd.DataFrame, s: Dict, out_dir: Path = REPORT_DIR) -> Dict:
    import portfolio_se as ps
    import rotation_se as rot
    import selective_se as v3
    import walkforward as wf

    ev = build_events(mkt, acts, s)
    H = {f: s["signals"][f]["horizon"] for f in FAMILIES}
    tests, ars = {}, {}
    for f in FAMILIES:
        ars[f] = v4.abnormal_returns(mkt, ev[f], H[f])
        tests[f] = v4.test_family(ars[f], H[f], s["signals"][f]["direction"], s)
    holm_ok = v4.holm({f: tests[f]["p_bootstrap"] for f in FAMILIES}, s["tests"]["alpha"])
    for f in FAMILIES:
        r = tests[f]
        sign = 1.0 if s["signals"][f]["direction"] == "+" else -1.0
        r["pass_holm"] = bool(holm_ok[f])
        r["pass_t"] = bool(np.isfinite(r["t_nw"]) and r["t_nw"] >= s["tests"]["nw_t_min"])
        r["pass_size"] = bool(f not in LONG_FAMILIES or (np.isfinite(r["mean_ar"]) and sign * r["mean_ar"] >= s["tests"]["min_mean_ar_long"]))
        r["pass"] = r["pass_holm"] and r["pass_t"] and r["pass_size"]

    used = [f for f in s["strategy"]["priority"] if tests[f]["pass"]]
    use_filter = tests["R2"]["pass"]
    events_all = pd.concat([ev[f] for f in LONG_FAMILIES], ignore_index=True)
    blocks = block_windows(mkt, ev["R2"], H["R2"])
    folds = wf.folds()
    c_start, c_end = folds[0].trade_start, folds[-1].trade_end

    def run(fams, filt, start, end):
        return v4.simulate(mkt, events_all, H, start, end, k=s["strategy"]["k"], priority=tuple(fams) or ("none",),
                           block_windows=blocks if filt else None)

    out = {"spec_version": s["version"], "period": s["period"], "tests": tests, "signals_used": used,
           "downgrade_filter": use_filter}
    rc = run(used, use_filter, c_start, c_end)
    rc.trades = rot.assign_folds(rc.trades)
    v = ps.verdict(rc, mkt.bench)
    out["criterion1"] = {k: val for k, val in v.items() if k != "bench_equity"}
    out["pass"] = bool(len(used) > 0 and v["pass"])
    full = run(used, use_filter, s["period"][0], s["period"][1])
    out["full_period"] = rot.metrics(full.equity, mkt.bench)
    out["full_period"]["trades"] = int(len(full.trades))
    x = v3.excess_series(full.equity, mkt.bench)
    out["full_period"]["excess_p_bootstrap"] = float(v3.stationary_bootstrap_p(x, reps=s["tests"]["bootstrap_reps"])) if len(x) > 50 else float("nan")
    # Diagnostico anunciado (nao decide): as duas familias longas com o filtro, na prioridade da especificacao.
    diag = run(s["strategy"]["priority"], True, c_start, c_end)
    diag.trades = rot.assign_folds(diag.trades)
    vd = ps.verdict(diag, mkt.bench)
    out["diag_all_signals_criterion1"] = {k: vd[k] for k in ("total_return", "bench_return", "mdd", "bench_mdd",
                                                             "positive_folds", "trades", "pass")}
    out["data_vintage"] = {p.name: v4.file_sha(p) for p in (analyst_se.ACTIONS_PATH, analyst_se.ARTICLES_PATH,
                                                           REPO_ROOT / "output" / "prices_se.parquet")}
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in FAMILIES:
        ars[f].to_csv(out_dir / f"events_{f}.csv", index=False)
    pd.DataFrame({"sistema": rc.equity, "indice": v["bench_equity"]}).to_csv(out_dir / "equity.csv")
    rc.trades.to_csv(out_dir / "trades.csv", index=False)
    full.trades.to_csv(out_dir / "trades_full_period.csv", index=False)
    (out_dir / "verdict.json").write_text(json.dumps(out, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
    (out_dir / "verdict.md").write_text(verdict_markdown(out), encoding="utf-8")
    v4._chart(full, mkt.bench, out_dir / "equity.png", used)
    return out


def verdict_markdown(o: Dict) -> str:
    def p(x):
        return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:+.2f}%"

    def f3(x):
        return "-" if x is None or not np.isfinite(x) else f"{x:.3f}"

    def ok(b):
        return "PASSA" if b else "NAO PASSA"

    names = {"R1": "elevacao de recomendacao (20 pregoes)", "R2": "rebaixamento (60 pregoes, previsto negativo)",
             "R3": "mais precos-alvo elevados que cortados (20 pregoes)"}
    L = ["# Avaliacao da estrategia v5 — revisoes de analistas", "",
         f"Especificacao congelada em `data/strategy_se_v5.json` antes de ligar as revisoes a retornos (ADR 0009). "
         f"Periodo dos testes: {o['period'][0]} a {o['period'][1]}.", "",
         "Retorno anormal = abertura a abertura menos o das acoes do mesmo quintil de retorno de 63 dias. Serie semanal; "
         "t de Newey-West e p de bootstrap no sentido previsto; Holm sobre as tres hipoteses.", "",
         "| Hipotese | Eventos | Retorno anormal medio | Acerto | t NW | p bootstrap | Holm | Tamanho | |",
         "|---|---:|---:|---:|---:|---:|---|---|---|"]
    for f in FAMILIES:
        r = o["tests"][f]
        L.append(f"| {f} {names[f]} | {r['events_measured']} | {p(r['mean_ar'])} | {p(r['hit_rate'])} | "
                 f"{r['t_nw']:.2f} | {f3(r['p_bootstrap'])} | {ok(r['pass_holm'])} | {ok(r['pass_size'])} | **{ok(r['pass'])}** |")
    c1 = o["criterion1"]
    L += ["", f"Sinais usados: {', '.join(o['signals_used']) or 'nenhum'}; filtro de rebaixamento: "
          f"{'sim' if o['downgrade_filter'] else 'nao'}.", "",
          "## Criterio 1 (2023-10-31 a 2026-10-30)", "",
          f"- Retorno {p(c1['total_return'])} vs indice {p(c1['bench_return'])} -> {ok(c1['crit_beats_index'])}",
          f"- Folds com expectativa positiva: {c1['positive_folds']}/6 -> {ok(c1['crit_folds'])}",
          f"- Max drawdown {p(c1['mdd'])} vs indice {p(c1['bench_mdd'])} -> {ok(c1['crit_mdd'])}",
          f"- Trades: {c1['trades']}", "", f"**{'APROVADA para paper trading' if o['pass'] else 'REPROVADA'}**", "",
          "## Periodo inteiro", ""]
    fp = o["full_period"]
    L += [f"- CAGR {p(fp['cagr'])} vs indice {p(fp['bench_cagr'])}; max drawdown {p(fp['mdd'])} vs {p(fp['bench_mdd'])}; "
          f"{fp['trades']} trades", "", "## Diagnostico anunciado (nao decide)", ""]
    d = o["diag_all_signals_criterion1"]
    L += [f"- R1+R3 com filtro de rebaixamento, 2023-2026: retorno {p(d['total_return'])} vs {p(d['bench_return'])}, "
          f"MDD {p(d['mdd'])} vs {p(d['bench_mdd'])}, folds {d['positive_folds']}/6, {d['trades']} trades", "",
          "## Por ano (retorno anormal medio, eventos)", "", "| Ano | " + " | ".join(FAMILIES) + " |",
          "|---|" + "---:|" * len(FAMILIES)]
    years = sorted({int(y) for f in FAMILIES for y in o["tests"][f]["by_year"]})
    for y in years:
        cells = []
        for f in FAMILIES:
            r = o["tests"][f]["by_year"].get(y) or o["tests"][f]["by_year"].get(str(y))
            cells.append(f"{p(r['mean_ar'])} ({r['n']})" if r else "-")
        L.append(f"| {y} | " + " | ".join(cells) + " |")
    L += ["", "Versao dos dados (sha256): " + ", ".join(f"`{k}` {v}" for k, v in o["data_vintage"].items()), ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["dry-run", "evaluate"])
    a = ap.parse_args(argv)
    s = spec()
    mkt = v4.load_market()
    acts = pd.read_parquet(analyst_se.ACTIONS_PATH)
    if a.cmd == "dry-run":
        print(json.dumps(dry_run(mkt, acts, s), indent=1, default=str))
        return 0
    o = evaluate(mkt, acts, s)
    print((REPORT_DIR / "verdict.md").read_text(encoding="utf-8"))
    return 0 if o["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
