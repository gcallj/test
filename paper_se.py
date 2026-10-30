#!/usr/bin/env python3
"""
Paper trading e ramp-up do mercado sueco (docs/sweden/plano.md).

    python paper_se.py start                       # etapa 1: paper, conta de 30.000 SEK
    python paper_se.py start --stage real1         # etapa 2: dinheiro real, 10.000 SEK, 2 posicoes
    python paper_se.py start --stage real2         # etapa 3: dinheiro real, 30.000 SEK, 6 posicoes

    python paper_se.py fill VOLV-B.ST 254.30       # a ordem executou a este preco
    python paper_se.py missed VOLV-B.ST            # a ordem expirou sem executar
    python paper_se.py cancelled VOLV-B.ST         # cancelada (fechou abaixo do stop)
    python paper_se.py exit VOLV-B.ST 261.10 --reason alvo   # saiu (alvo|stop|stop_movel|tempo|perda_maxima)

    python paper_se.py check                       # regras de parada + etapa do ramp-up

Cada registro guarda o preco planejado (limite, stop ou alvo) tirado do ultimo
relatorio diario (reports/se/daily/<data>_book.csv), para medir slippage.
O paper trading e um mechanics check: confere a operacao, nao prova edge.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import market_se  # noqa: E402

PAPER_DIR = REPO_ROOT / "data" / "paper"
ACCOUNT_PATH = PAPER_DIR / "se_account.json"
LOG_PATH = PAPER_DIR / "se_executions.jsonl"
DAILY_DIR = REPO_ROOT / "reports" / "se" / "daily"
PORTFOLIO_SUMMARY = REPO_ROOT / "reports" / "se" / "portfolio.json"

STAGES = {
    "paper": {"capital": 30_000.0, "max_positions": 6, "real": False, "next": "real1"},
    "real1": {"capital": 10_000.0, "max_positions": 2, "real": True, "next": "real2"},
    "real2": {"capital": 30_000.0, "max_positions": 6, "real": True, "next": None},
}
PAPER_MIN_SESSIONS = 10       # etapa 1: 10 pregoes de mechanics check
REAL1_MIN_TRADES = 20         # etapa 2: 20 trades reais sem stop rule
FILL_RATE_MAX_GAP = 0.15      # stop rule: fill rate 15 pp abaixo do replay
SLIPPAGE_MAX_PER_SIDE = 0.002  # stop rule: slippage medio > 0,20% por lado
MIN_SAMPLE = 5                # amostra minima para avaliar fill rate e slippage


# --------------------------------------------------------------------------
# Account and log
# --------------------------------------------------------------------------
def load_account() -> dict:
    if not ACCOUNT_PATH.exists():
        raise SystemExit("Conta nao iniciada. Rode: python paper_se.py start")
    return json.loads(ACCOUNT_PATH.read_text(encoding="utf-8"))


def start(stage: str, when: Optional[str]) -> dict:
    cfg = STAGES[stage]
    acct = {"stage": stage, "start": when or date.today().isoformat(), "capital": cfg["capital"],
            "size": market_se.POSITION_SIZE_SEK, "max_positions": cfg["max_positions"], "real_money": cfg["real"]}
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    ACCOUNT_PATH.write_text(json.dumps(acct, indent=2), encoding="utf-8")
    return acct


def read_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=["kind", "ticker", "date", "price", "planned", "reason", "stage"])
    rows = [json.loads(l) for l in LOG_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows)


def _latest_book() -> pd.DataFrame:
    files = sorted(DAILY_DIR.glob("*_book.csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])


def _planned(ticker: str, kind: str, reason: Optional[str]) -> Optional[float]:
    book = _latest_book()
    if book.empty:
        return None
    rows = book[book["ticker"] == ticker]
    if kind == "fill":
        rows = rows[rows["tipo"].isin(["ordem_nova", "ordem_ativa"])]
        return float(rows["limite"].iloc[0]) if len(rows) else None
    if kind == "exit":
        rows = rows[rows["tipo"] == "posicao"]
        if not len(rows):
            return None
        col = "alvo_amanha" if reason == "alvo" else "stop_amanha"
        v = rows[col].iloc[0]
        return float(v) if pd.notna(v) else None
    return None


def record(kind: str, ticker: str, price: Optional[float], when: Optional[str], reason: Optional[str]) -> dict:
    acct = load_account()
    tk = ticker.upper() if ticker.upper().endswith(".ST") else ticker.upper() + ".ST"
    entry = {"kind": kind, "ticker": tk, "date": when or date.today().isoformat(), "price": price,
             "planned": _planned(tk, kind, reason), "reason": reason, "stage": acct["stage"],
             "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


# --------------------------------------------------------------------------
# Stop rules and ramp-up
# --------------------------------------------------------------------------
def slippage(row: pd.Series) -> Optional[float]:
    """Custo extra vs o preco planejado, como fracao (positivo = pior que o planejado)."""
    if row.get("planned") is None or pd.isna(row.get("planned")) or row.get("price") is None:
        return None
    planned, price = float(row["planned"]), float(row["price"])
    if planned <= 0:
        return None
    if row["kind"] == "fill":
        return (price - planned) / planned          # pagou mais que o limite
    return (planned - price) / planned              # vendeu abaixo do nivel planejado


def realized_trades(log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    open_fill: Dict[str, pd.Series] = {}
    for _, r in log.sort_values("recorded_at").iterrows():
        if r["kind"] == "fill":
            open_fill[r["ticker"]] = r
        elif r["kind"] == "exit" and r["ticker"] in open_fill:
            f = open_fill.pop(r["ticker"])
            gross = float(r["price"]) / float(f["price"]) - 1.0
            rows.append({"ticker": r["ticker"], "entrada": f["date"], "saida": r["date"],
                         "resultado_liq": gross - market_se.ROUND_TRIP_COST_PCT, "stage": r["stage"]})
    return pd.DataFrame(rows)


def check(log: pd.DataFrame, acct: dict, replay: dict, sessions: int) -> dict:
    stage_log = log[log["stage"] == acct["stage"]] if not log.empty else log
    orders = stage_log[stage_log["kind"].isin(["fill", "missed", "cancelled"])] if not stage_log.empty else stage_log
    n_orders = len(orders)
    fill_rate = float((orders["kind"] == "fill").mean()) if n_orders else None
    replay_fill = replay.get("replay_fill_rate")
    rules = []

    if n_orders >= MIN_SAMPLE and replay_fill is not None:
        ok = fill_rate >= replay_fill - FILL_RATE_MAX_GAP
        rules.append(("fill rate", ok, f"{fill_rate:.0%} real vs {replay_fill:.0%} no replay (tolerancia 15 pp)"))
    else:
        rules.append(("fill rate", True, f"amostra insuficiente ({n_orders} ordens; minimo {MIN_SAMPLE})"))

    slips = [s for s in (slippage(r) for _, r in stage_log.iterrows()) if s is not None] if not stage_log.empty else []
    if len(slips) >= MIN_SAMPLE:
        mean_slip = float(np.mean(slips))
        ok = mean_slip <= SLIPPAGE_MAX_PER_SIDE
        rules.append(("slippage", ok, f"media {mean_slip:+.2%} por lado (limite +0,20%)"))
    else:
        rules.append(("slippage", True, f"amostra insuficiente ({len(slips)} execucoes; minimo {MIN_SAMPLE})"))

    trades = realized_trades(stage_log) if not stage_log.empty else pd.DataFrame()
    worst = replay.get("worst_fold_mdd")
    if len(trades):
        curve = acct["capital"] + (trades["resultado_liq"] * acct["size"]).cumsum()
        curve = pd.concat([pd.Series([acct["capital"]]), curve], ignore_index=True)
        dd = float((curve / curve.cummax() - 1.0).min())
        ok = worst is None or dd >= worst
        rules.append(("drawdown", ok, f"{dd:.1%} realizado vs pior fold do replay {worst:.1%}" if worst is not None
                      else f"{dd:.1%} realizado (sem referencia do replay)"))
    else:
        rules.append(("drawdown", True, "nenhum trade fechado ainda"))

    breached = [r for r in rules if not r[1]]
    nxt = STAGES[acct["stage"]]["next"]
    if breached:
        advice = "PARAR: voltar ao paper trading e investigar (" + ", ".join(r[0] for r in breached) + ")"
    elif acct["stage"] == "paper":
        advice = (f"pode iniciar a etapa {nxt} (`python paper_se.py start --stage {nxt}`)"
                  if sessions >= PAPER_MIN_SESSIONS else f"continuar o paper trading ({sessions}/{PAPER_MIN_SESSIONS} pregoes)")
    elif acct["stage"] == "real1":
        advice = (f"pode iniciar a etapa {nxt} (`python paper_se.py start --stage {nxt}`)"
                  if len(trades) >= REAL1_MIN_TRADES else f"continuar com 10.000 SEK ({len(trades)}/{REAL1_MIN_TRADES} trades)")
    else:
        advice = "capital cheio; seguir monitorando as regras de parada"
    return {"rules": rules, "advice": advice, "fill_rate": fill_rate, "trades": int(len(trades)), "sessions": sessions}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--stage", choices=list(STAGES), default="paper")
    s.add_argument("--date")
    for kind in ("fill", "exit"):
        p = sub.add_parser(kind)
        p.add_argument("ticker")
        p.add_argument("price", type=float)
        p.add_argument("--date")
        if kind == "exit":
            p.add_argument("--reason", required=True,
                           choices=["alvo", "stop", "stop_movel", "tempo", "perda_maxima", "manual"])
    for kind in ("missed", "cancelled"):
        p = sub.add_parser(kind)
        p.add_argument("ticker")
        p.add_argument("--date")
    sub.add_parser("check")
    args = ap.parse_args(argv)

    if args.cmd == "start":
        print(json.dumps(start(args.stage, args.date), indent=2))
        return 0
    if args.cmd in ("fill", "exit"):
        print(json.dumps(record(args.cmd, args.ticker, args.price, args.date, getattr(args, "reason", None)), indent=2))
        return 0
    if args.cmd in ("missed", "cancelled"):
        print(json.dumps(record(args.cmd, args.ticker, None, args.date, None), indent=2))
        return 0

    acct = load_account()
    replay = json.loads(PORTFOLIO_SUMMARY.read_text(encoding="utf-8")) if PORTFOLIO_SUMMARY.exists() else {}
    sessions = len(sorted({p.name[:10] for p in DAILY_DIR.glob("*_book.csv") if p.name[:10] >= acct["start"]}))
    out = check(read_log(), acct, replay, sessions)
    print(f"Etapa: {acct['stage']} desde {acct['start']} · {out['sessions']} pregoes com relatorio · "
          f"{out['trades']} trades fechados")
    for name, ok, msg in out["rules"]:
        print(f"  [{'ok' if ok else 'STOP'}] {name}: {msg}")
    print(f"=> {out['advice']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
