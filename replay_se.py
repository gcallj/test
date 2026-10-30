#!/usr/bin/env python3
"""
Replay walk-forward do mercado sueco: o que o sistema teria feito, pregao a
pregao, usando em cada fold so o genoma treinado antes dele.

    python replay_se.py store                  # monta os payloads com o historico completo
    python replay_se.py ticker VOLV-B.ST       # tabela + grafico + resumo de um ticker
    python replay_se.py all                    # todos os tickers -> reports/se/orders.csv
    python replay_se.py all --charts           # idem, com um grafico por ticker

    # Ilustrativo (in-sample): um unico genoma sobre os mesmos 3 anos.
    python replay_se.py ticker VOLV-B.ST --genome global_ga_checkpoint.json

Saidas em reports/se/ (ou reports/se_insample/ com --genome). O portfolio
replay (portfolio_se.py) le reports/se/orders.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import walkforward as wf  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "se"
INSAMPLE_DIR = REPO_ROOT / "reports" / "se_insample"

REASONS_PT = {
    "in_position": "ja comprado",
    "order_live": "ordem anterior ainda ativa",
    "cooldown": "pausa apos perdas seguidas ou drawdown",
    "warmup": "historico insuficiente para os indicadores",
    "volatility_too_low": "volatilidade baixa demais",
    "volatility_too_high": "volatilidade alta demais",
    "illiquid": "ATR minimo nao atingido",
    "signal_too_weak": "sinal fraco (forca abaixo do minimo)",
    "below_entry_threshold": "forca do sinal abaixo do limiar de entrada",
    "volume_below_average": "volume abaixo da media",
    "negative_momentum": "momento negativo nos ultimos dias",
    "unfavourable_regime": "regime de mercado desfavoravel",
    "no_buy_vote": "indicadores nao votaram compra ou score abaixo do gatilho",
    "awaiting_confirmation": "aguardando confirmacao (dias seguidos de sinal)",
    "below_long_average": "preco abaixo da media longa",
    "below_pivot": "fechou abaixo do pivo",
    "overextended_above_resistance": "esticado acima da resistencia R1",
    "support_broken": "suporte S1 rompido",
}
EXIT_PT = {"take": "alvo", "stop": "stop", "trailing": "stop movel", "time": "tempo", "hard_loss": "perda maxima"}
STATUS_PT = {"filled": "executada", "missed": "perdida", "cancelled": "cancelada", "pending": "aberta"}


# --------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------
def build_store() -> int:
    """Monta output/ga_memmap a partir do ETL, sem rodar o GA."""
    env = dict(os.environ)
    env.update({"GA_BUILD_STORE_ONLY": "1", "GA_KEEP_STORE": "1", "GA_RUN_MODE": "load",
                "GA_DISABLE_TELEGRAM_SEND": "1"})
    env.pop("GA_TRAIN_END", None)
    env.pop("GA_MEMMAP_DIR", None)
    return subprocess.run([sys.executable, "-u", "ga_run.py"], cwd=REPO_ROOT, env=env).returncode


def load_cache() -> Dict[str, Dict]:
    import ga_run as gr
    from payload_store import PayloadStore
    import run_local_ga_staged as staged

    store_dir = os.path.abspath(gr.GA_MEMMAP_DIR)
    if not os.path.exists(os.path.join(store_dir, "meta.json")):
        raise SystemExit(f"Store nao encontrado em {store_dir}. Rode: python replay_se.py store")
    store = PayloadStore(store_dir, mode="r")
    return staged.build_ticker_arrays_cache(store)


# --------------------------------------------------------------------------
# Genomes
# --------------------------------------------------------------------------
def walkforward_plan() -> List[Tuple[wf.Fold, dict]]:
    plan = []
    missing = []
    for f in wf.folds():
        g = wf.load_fold_genome(f.k)
        if g is None:
            missing.append(f.k)
        else:
            plan.append((f, g))
    if missing:
        raise SystemExit(f"Faltam genomas dos folds {missing}. Rode: python walkforward.py train --all "
                         f"(ou o workflow SE Walk-forward).")
    return plan


def insample_plan(genome_path: str) -> List[Tuple[wf.Fold, dict]]:
    g = json.loads(Path(genome_path).read_text(encoding="utf-8"))
    return [(f, g) for f in wf.folds()]


def _params(genome: List[float]):
    import ga_run as gr
    return gr.decode_global_params(gr.sanitize_global_genome(list(genome)))


# --------------------------------------------------------------------------
# Replay of one ticker
# --------------------------------------------------------------------------
def _fmt_pct(x: Optional[float]) -> str:
    return "" if x is None or not np.isfinite(x) else f"{100 * x:+.2f}%"


def replay_ticker(tk: str, p: Dict, plan: List[Tuple[wf.Fold, dict]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (tabela diaria, ordens) do ticker ao longo dos folds."""
    import ga_run as gr

    dates = pd.to_datetime(pd.Series(p["dates"])).dt.normalize().to_numpy()
    n = len(dates)
    daily_rows: List[Dict] = []
    order_rows: List[Dict] = []

    for f, g in plan:
        lo_d, hi_d = np.datetime64(pd.Timestamp(f.trade_start)), np.datetime64(pd.Timestamp(f.trade_end))
        idx = np.where((dates >= lo_d) & (dates <= hi_d))[0]
        if len(idx) == 0:
            continue
        lo, hi = int(idx[0]), int(idx[-1])
        trace: List[Dict] = []
        orders: List[Dict] = []
        gr.backtest_stats_global_intraday(
            p["open"], p["high"], p["low"], p["close"], p["score_matrix"], p["atr"],
            _params(g["genome"]), precomputed=p, trace=trace, trades_out=orders,
        )
        by_i = {r["i"]: r for r in trace}

        for j in range(lo, hi + 1):
            today = by_i.get(j, {})
            after = by_i.get(j + 1)  # decision taken at the close of j
            ev = today.get("event", "")
            happened = []
            if "fill" in ev:
                happened.append(f"comprou a {today['fill_px']:.2f}")
            if "exit" in ev:
                happened.append(f"vendeu por {EXIT_PT.get(today.get('exit_type'), today.get('exit_type'))} "
                                f"a {today['exit_px']:.2f} ({_fmt_pct(today.get('ret'))} liquido)")
            if "missed" in ev:
                happened.append("ordem expirou sem executar (missed entry)")
            if "cancelled" in ev:
                happened.append("ordem cancelada: fechou abaixo do stop")

            holding_at_close = today.get("state") == "long" and "exit" not in ev
            order_live_at_close = today.get("state") == "pending"
            if holding_at_close:
                if after is not None and "stop_px" in after and "order" not in after.get("event", ""):
                    decision = f"manter posicao (stop {after['stop_px']:.2f}, alvo {after['target_px']:.2f})"
                else:
                    decision = "manter posicao"
            elif order_live_at_close:
                decision = "ordem de compra continua ativa"
            elif after is None:
                decision = "(fim dos dados)"
            elif "order" in after.get("event", ""):
                decision = (f"ORDEM: comprar limite {after['limit']:.2f}, stop {after['stop_px']:.2f}, "
                            f"alvo {after['target_px']:.2f}, valida {wf_validity()} pregoes")
            else:
                decision = "nao compra: " + REASONS_PT.get(after.get("reason", ""), after.get("reason", "") or "-")

            src = after or {}
            daily_rows.append({
                "data": pd.Timestamp(dates[j]).date().isoformat(),
                "fold": f.k,
                "fechamento": round(float(p["close"][j]), 4),
                "situacao": {"long": "comprado", "pending": "ordem ativa", "flat": "sem posicao"}.get(today.get("state", "flat"), today.get("state", "")),
                "aconteceu_no_dia": "; ".join(happened),
                "decisao_no_fechamento": decision,
                "votos": src.get("votes"), "votos_min": src.get("votes_needed"),
                "score": src.get("score"), "gatilho": src.get("score_trigger"),
                "forca": src.get("strength"),
                "regime": src.get("regime"), "regime_min": src.get("regime_needed"),
                "vol_rank": src.get("vol_rank"),
                "motivo": src.get("reason", ""),
            })

        for o in orders:
            s = o["signal_idx"]
            if not (lo <= s <= hi):
                continue
            def _d(k):
                return pd.Timestamp(dates[k]).date().isoformat() if k is not None and 0 <= k < n else ""
            status = o.get("status")
            if status == "filled" and "exit_idx" not in o:
                status = "pending"  # still open at the end of the data
            order_rows.append({
                "ticker": tk, "fold": f.k,
                "data_sinal": _d(s),
                "status": STATUS_PT.get(status, status),
                "limite": round(o["limit"], 4), "stop": round(o["stop_px"], 4), "alvo": round(o["target_px"], 4),
                "forca": round(float(o.get("strength", 0.0)), 4),
                "data_compra": _d(o.get("fill_idx")),
                "preco_compra": round(o["entry_px"], 4) if "entry_px" in o else None,
                "data_venda": _d(o.get("exit_idx")) if o.get("status") == "filled" else "",
                # Ultimo dia em que a ordem nao executada reservou saldo.
                "data_fim_ordem": _d(o.get("exit_idx")) if o.get("status") in ("missed", "cancelled") else "",
                "preco_venda": round(o["exit_px"], 4) if "exit_px" in o else None,
                "saida": EXIT_PT.get(o.get("exit_type"), ""),
                "resultado_liq": o.get("ret") if o.get("status") == "filled" and "exit_px" in o else None,
                "pregoes": o.get("hold_bars"),
            })
    return pd.DataFrame(daily_rows), pd.DataFrame(order_rows)


def wf_validity() -> int:
    import market_se
    return market_se.ORDER_VALIDITY_DAYS


# --------------------------------------------------------------------------
# Summaries and charts
# --------------------------------------------------------------------------
def fold_summary(orders: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in orders.groupby("fold"):
        closed = g[g["resultado_liq"].notna()]
        n_orders = int(len(g[g["status"] != "aberta"]))
        filled = int((g["status"] == "executada").sum())
        rets = closed["resultado_liq"].astype(float)
        rows.append({
            "fold": int(k), "ordens": n_orders,
            "executadas": filled, "perdidas": int((g["status"] == "perdida").sum()),
            "canceladas": int((g["status"] == "cancelada").sum()),
            "fill_rate": filled / n_orders if n_orders else np.nan,
            "trades_fechados": int(len(closed)),
            "win_rate": float((rets > 0).mean()) if len(rets) else np.nan,
            "expectancy": float(rets.mean()) if len(rets) else np.nan,
            "soma_resultados": float(rets.sum()) if len(rets) else 0.0,
        })
    return pd.DataFrame(rows)


def ticker_markdown(tk: str, daily: pd.DataFrame, orders: pd.DataFrame, mode: str) -> str:
    lines = [f"# Replay {mode}: {tk}", ""]
    if orders.empty:
        lines += ["Nenhuma ordem no periodo.", ""]
    else:
        fs = fold_summary(orders)
        lines += ["| Fold | Ordens | Executadas | Perdidas | Canceladas | Fill rate | Trades | Win rate | Expectancy |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in fs.iterrows():
            lines.append(f"| {r.fold} | {r.ordens} | {r.executadas} | {r.perdidas} | {r.canceladas} | "
                         f"{_fmt_pct(r.fill_rate) if np.isfinite(r.fill_rate) else '-'} | {r.trades_fechados} | "
                         f"{_fmt_pct(r.win_rate) if np.isfinite(r.win_rate) else '-'} | "
                         f"{_fmt_pct(r.expectancy) if np.isfinite(r.expectancy) else '-'} |")
        lines.append("")
    flat = daily[daily["decisao_no_fechamento"].str.startswith("nao compra")]
    if len(flat):
        lines += ["## Por que nao comprou (dias sem posicao)", ""]
        for reason, cnt in Counter(flat["motivo"]).most_common(8):
            lines.append(f"- {REASONS_PT.get(reason, reason)}: {cnt} pregoes ({cnt / len(flat):.0%})")
        lines.append("")
    lines += ["Tabela pregao a pregao: `" + f"{_safe(tk)}_diario.csv" + "`. Ordens: `" + f"{_safe(tk)}_ordens.csv" + "`.", ""]
    return "\n".join(lines)


def _safe(tk: str) -> str:
    return tk.replace("^", "").replace(".", "_").replace("-", "_")


def render_chart(tk: str, p: Dict, orders: pd.DataFrame, plan, path: Path, title_suffix: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = pd.to_datetime(pd.Series(p["dates"])).dt.normalize()
    start = pd.Timestamp(plan[0][0].trade_start)
    end = pd.Timestamp(plan[-1][0].trade_end)
    m = (dates >= start) & (dates <= end)
    if m.sum() < 5:
        return
    d = dates[m].to_numpy()
    o, h, l, c = (np.asarray(p[k])[m.to_numpy()] for k in ("open", "high", "low", "close"))
    x = np.arange(len(d))
    pos_of = {pd.Timestamp(v).date().isoformat(): i for i, v in enumerate(d)}

    fig, ax = plt.subplots(figsize=(18, 7))
    up = c >= o
    ax.vlines(x, l, h, color="#888888", linewidth=0.5)
    ax.vlines(x[up], o[up], c[up], color="#2e7d32", linewidth=2.2)
    ax.vlines(x[~up], c[~up], o[~up], color="#c62828", linewidth=2.2)

    colors = {"alvo": "#1565c0", "stop": "#c62828", "stop movel": "#ef6c00", "tempo": "#6d6d6d", "perda maxima": "#000000"}
    for _, r in orders.iterrows():
        if r["status"] == "executada" and r["data_compra"] in pos_of:
            ax.scatter(pos_of[r["data_compra"]], r["preco_compra"], marker="^", s=70, color="#2e7d32", zorder=5)
            if r["data_venda"] in pos_of and r["preco_venda"] is not None and np.isfinite(r["preco_venda"]):
                ax.scatter(pos_of[r["data_venda"]], r["preco_venda"], marker="v", s=70,
                           color=colors.get(r["saida"], "#6d6d6d"), zorder=5)
        elif r["status"] in ("perdida", "cancelada") and r["data_sinal"] in pos_of:
            ax.scatter(pos_of[r["data_sinal"]] + 1, r["limite"], marker="x", s=40, color="#9e9e9e", zorder=4)

    for f, _ in plan:
        k = np.searchsorted(d, np.datetime64(pd.Timestamp(f.trade_start)))
        ax.axvline(k, color="#9e9e9e", linestyle="--", linewidth=0.8)
        ax.text(k + 2, np.nanmax(h), f"fold {f.k}", fontsize=8, color="#616161", va="top")

    ticks = np.linspace(0, len(d) - 1, 12).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([pd.Timestamp(d[t]).strftime("%Y-%m") for t in ticks], fontsize=8)
    ax.set_title(f"{tk} — replay{title_suffix}   ▲ compra   ▼ venda (azul alvo, vermelho stop, laranja stop movel, "
                 f"cinza tempo)   × ordem nao executada", fontsize=10)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def run_ticker(tk: str, cache: Dict, plan, out_dir: Path, mode: str, chart: bool = True) -> pd.DataFrame:
    if tk not in cache:
        raise SystemExit(f"{tk} nao esta no store (tickers disponiveis: {len(cache)}).")
    daily, orders = replay_ticker(tk, cache[tk], plan)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir / f"{_safe(tk)}_diario.csv", index=False)
    orders.to_csv(out_dir / f"{_safe(tk)}_ordens.csv", index=False)
    (out_dir / f"{_safe(tk)}.md").write_text(ticker_markdown(tk, daily, orders, mode), encoding="utf-8")
    if chart:
        render_chart(tk, cache[tk], orders, plan, out_dir / f"{_safe(tk)}.png",
                     "" if mode == "walk-forward" else " (in-sample, ilustrativo)")
    return orders


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("store")
    t = sub.add_parser("ticker")
    t.add_argument("ticker")
    t.add_argument("--genome", help="genoma unico (modo in-sample, ilustrativo)")
    a = sub.add_parser("all")
    a.add_argument("--genome", help="genoma unico (modo in-sample, ilustrativo)")
    a.add_argument("--charts", action="store_true", help="gera um grafico por ticker")
    args = ap.parse_args(argv)

    if args.cmd == "store":
        return build_store()

    plan = insample_plan(args.genome) if args.genome else walkforward_plan()
    mode = "in-sample (ilustrativo)" if args.genome else "walk-forward"
    out_dir = INSAMPLE_DIR if args.genome else REPORT_DIR
    cache = load_cache()

    if args.cmd == "ticker":
        tk = args.ticker.upper()
        if not tk.endswith(".ST") and not tk.startswith("^"):
            tk += ".ST"
        run_ticker(tk, cache, plan, out_dir / "tickers", mode)
        print(f"[replay] {tk}: {out_dir / 'tickers'}")
        return 0

    all_orders = []
    for i, tk in enumerate(sorted(cache), 1):
        all_orders.append(run_ticker(tk, cache, plan, out_dir / "tickers", mode, chart=args.charts))
        if i % 20 == 0:
            print(f"[replay] {i}/{len(cache)} tickers", flush=True)
    orders = pd.concat([o for o in all_orders if not o.empty], ignore_index=True)
    orders.to_csv(out_dir / "orders.csv", index=False)
    summ = fold_summary(orders)
    summ.to_csv(out_dir / "folds.csv", index=False)
    print(summ.to_string(index=False))
    print(f"[replay] {len(orders)} ordens -> {out_dir / 'orders.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
