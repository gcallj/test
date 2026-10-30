#!/usr/bin/env python3
"""
Operacao diaria do mercado sueco: o que fazer amanha.

    python daily_se.py run       # ETL + payloads + relatorio (+ Telegram se configurado)
    python daily_se.py report    # so o relatorio, com os payloads ja montados

Requisitos:
- um genoma aceito em data/genome_se.json  (python walkforward.py accept ...)
- uma conta iniciada em data/paper/se_account.json  (python paper_se.py start)

O relatorio sai do mesmo motor do backtest e do replay (docs/adr/0001): ao
historico de cada ticker soma-se um pregao vazio (amanha) e o motor decide no
fechamento de hoje, como decidiu em todos os dias do replay. A conta e a
continuacao do portfolio replay a partir da data de inicio (portfolio_se.py).

Telegram (opcional): SE_TELEGRAM_BOT_TOKEN e SE_TELEGRAM_CHAT_ID.
Saidas: reports/se/daily/<data>.md e <data>_book.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import market_se  # noqa: E402
import walkforward as wf  # noqa: E402

GENOME_PATH = wf.GENOME_PATH
DAILY_DIR = REPO_ROOT / "reports" / "se" / "daily"


def load_genome() -> dict:
    if not GENOME_PATH.exists():
        raise SystemExit("Nenhum genoma aceito em data/genome_se.json. Leia o replay e rode "
                         "`python walkforward.py accept data/walkforward/live.json`.")
    return json.loads(GENOME_PATH.read_text(encoding="utf-8"))


def with_tomorrow(p: Dict) -> Dict:
    """Payload com um pregao vazio no fim: o motor decide no fechamento de hoje."""
    out = {}
    n = len(p["close"])
    last = pd.Timestamp(pd.to_datetime(pd.Series(p["dates"])).iloc[-1])
    tomorrow = (last + pd.offsets.BDay(1)).to_datetime64()
    for k, v in p.items():
        if k == "dates":
            out[k] = np.append(np.asarray(v, dtype="datetime64[ns]"), np.datetime64(tomorrow, "ns"))
        elif k == "score_matrix":
            v = np.asarray(v, dtype=np.float64)
            out[k] = np.vstack([v, np.full((1, v.shape[1]), np.nan)])
        elif k == "valid_mask":
            out[k] = np.append(np.asarray(v, dtype=bool), False)
        elif hasattr(v, "__len__") and not isinstance(v, str) and len(v) == n:
            out[k] = np.append(np.asarray(v, dtype=np.float64), np.nan)
        else:
            out[k] = v
    return out


def tomorrow_levels(p_aug: Dict, gp) -> Dict:
    """Stop/alvo validos amanha para uma posicao aberta (ultimo registro do trace)."""
    import ga_run as gr
    trace: List[Dict] = []
    gr.backtest_stats_global_intraday(p_aug["open"], p_aug["high"], p_aug["low"], p_aug["close"],
                                      p_aug["score_matrix"], p_aug["atr"], gp, precomputed=p_aug, trace=trace)
    return trace[-1] if trace else {}


def build_report(cache: Dict, genome: dict, account: dict) -> Tuple[str, pd.DataFrame, str]:
    import replay_se
    import portfolio_se as ps
    import ga_run as gr

    gp = gr.decode_global_params(gr.sanitize_global_genome(genome["genome"]))
    last_day = max(pd.Timestamp(pd.to_datetime(pd.Series(p["dates"])).iloc[-1]) for p in cache.values()).normalize()
    start = min(pd.Timestamp(account["start"]), last_day)
    fold = wf.Fold(k=0, train_end="", trade_start=str(start.date()), trade_end=str(last_day.date()))

    augmented = {tk: with_tomorrow(p) for tk, p in cache.items()}
    orders = []
    closes = {}
    for tk, p in augmented.items():
        _, o = replay_se.replay_ticker(tk, p, [(fold, genome)])
        if not o.empty:
            orders.append(o)
        p0 = cache[tk]
        closes[tk] = pd.Series(np.asarray(p0["close"], dtype=float),
                               index=pd.to_datetime(pd.Series(p0["dates"])).dt.normalize().to_numpy())
    orders = pd.concat(orders, ignore_index=True) if orders else pd.DataFrame(columns=["ticker"])
    closes = pd.DataFrame(closes).sort_index()
    days = closes.index[(closes.index >= start) & (closes.index <= last_day)]

    acct = ps.Account(capital=float(account["capital"]), size=float(account["size"]),
                      max_positions=int(account["max_positions"]))
    for col in ("data_compra", "data_venda", "data_fim_ordem"):
        if col in orders:
            orders[col] = orders[col].fillna("")
    res = ps.simulate(orders, closes, days, acct)
    book = res.open_book.copy()
    today = last_day.date().isoformat()

    rows = []
    for _, r in book.iterrows():
        tk = r["ticker"]
        if r["estado"] == "posicao":
            lv = tomorrow_levels(augmented[tk], gp)
            rows.append({"tipo": "posicao", "ticker": tk, "desde": r["data_compra"], "preco_compra": r["preco_compra"],
                         "fechamento": float(closes[tk].dropna().iloc[-1]),
                         "stop_amanha": lv.get("stop_px"), "alvo_amanha": lv.get("target_px")})
        elif r["data_sinal"] == today:
            rows.append({"tipo": "ordem_nova", "ticker": tk, "limite": r["limite"], "stop": r["stop"],
                         "alvo": r["alvo"], "forca": r["forca"], "validade": market_se.ORDER_VALIDITY_DAYS})
        else:
            sig = pd.Timestamp(r["data_sinal"])
            used = int(((days > sig) & (days <= last_day)).sum())
            rows.append({"tipo": "ordem_ativa", "ticker": tk, "limite": r["limite"], "stop": r["stop"],
                         "alvo": r["alvo"], "forca": r["forca"],
                         "validade": max(0, market_se.ORDER_VALIDITY_DAYS - used), "data_sinal": r["data_sinal"]})
    book_df = pd.DataFrame(rows)

    signals_today = orders[orders["data_sinal"] == today] if not orders.empty else orders
    accepted = set(book_df.loc[book_df.get("tipo", pd.Series(dtype=str)) == "ordem_nova", "ticker"]) if not book_df.empty else set()
    left_out = signals_today[~signals_today["ticker"].isin(accepted)] if len(signals_today) else signals_today

    equity = float(res.equity.iloc[-1]) if len(res.equity) else acct.capital
    mdd = ps.max_drawdown(res.equity) if len(res.equity) > 1 else 0.0
    used = int(len(book_df)) if not book_df.empty else 0

    L = [f"# Ordens para o pregao seguinte a {today}", "",
         f"Etapa: **{account['stage']}** · conta {acct.capital:,.0f} SEK · patrimonio {equity:,.0f} SEK · "
         f"drawdown desde o inicio {100 * mdd:.1f}% · vagas {used}/{acct.max_positions}", ""]
    def table(kind, cols, title):
        sub = book_df[book_df["tipo"] == kind] if not book_df.empty else book_df
        L.append(f"## {title}")
        L.append("")
        if sub is None or sub.empty:
            L.extend(["Nenhuma.", ""])
            return
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "---|" * len(cols))
        for _, r in sub.iterrows():
            L.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
        L.append("")
    table("ordem_nova", ["ticker", "limite", "stop", "alvo", "forca", "validade"],
          "Novas ordens (comprar com ordem limitada, valida pelos pregoes indicados)")
    table("ordem_ativa", ["ticker", "limite", "stop", "alvo", "validade", "data_sinal"],
          "Ordens de dias anteriores que continuam ativas")
    table("posicao", ["ticker", "desde", "preco_compra", "fechamento", "stop_amanha", "alvo_amanha"],
          "Posicoes abertas (atualizar stop e alvo para amanha)")
    if len(left_out):
        L += ["## Sinais que ficaram de fora por falta de vaga", "",
              ", ".join(f"{r.ticker} (forca {r.forca:.2f})" for r in left_out.itertuples()), ""]
    L += ["Regra: cancelar a ordem se o preco fechar abaixo do stop antes de executar; ordem que nao "
          "executa na validade e uma entrada perdida (nao comprar a outro preco).", "",
          "Registrar o que aconteceu: `python paper_se.py fill|missed|exit ...`; conferir: `python paper_se.py check`.", ""]
    return "\n".join(L), book_df, today


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def telegram(text: str) -> None:
    token, chat = os.environ.get("SE_TELEGRAM_BOT_TOKEN"), os.environ.get("SE_TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[daily] Telegram nao configurado (SE_TELEGRAM_BOT_TOKEN/SE_TELEGRAM_CHAT_ID) — pulando envio")
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text[:4000]}).encode()
    urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=30).read()
    print("[daily] Telegram enviado")


def main(argv=None) -> int:
    import paper_se
    import replay_se

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "report"])
    args = ap.parse_args(argv)

    if args.cmd == "run":
        from stock_etl import run_etl
        run_etl()
        if replay_se.build_store() != 0:
            raise SystemExit("falha ao montar os payloads")
    genome = load_genome()
    account = paper_se.load_account()
    text, book, today = build_report(replay_se.load_cache(), genome, account)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    (DAILY_DIR / f"{today}.md").write_text(text, encoding="utf-8")
    book.to_csv(DAILY_DIR / f"{today}_book.csv", index=False)
    print(text)
    telegram(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
