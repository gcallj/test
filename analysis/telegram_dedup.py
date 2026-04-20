#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dedup guard para envios de summary_latest.xlsx ao Telegram.

Problema: multiplos pontos enviavam a mesma planilha (ga_run auto-send,
predict_daily.py manual, send_telegram.py manual, GitHub Actions cron).
Usuario recebia copias identicas.

Solucao: antes de enviar, verifica em `.telegram_last_sent.json` o
estado do ultimo envio. So envia se:
  - a data dos dados (Apply sheet latest Date) for MAIS NOVA que a ultima enviada, OU
  - o genome_sha for DIFERENTE (modelo mudou), OU
  - o fit_delta for > min_improvement_pp (default 0.05pp) — modelo melhorou notavelmente, OU
  - override via env GA_FORCE_TELEGRAM_SEND=1

Escrita do tracker e atomica (tmp file + rename).

API:
  should_send_telegram(xlsx_path, checkpoint_path=None) -> (bool, reason: str)
  record_telegram_sent(xlsx_path, checkpoint_path=None) -> None
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACKER = ROOT / ".telegram_last_sent.json"
DEFAULT_CHECKPOINT = ROOT / "global_ga_checkpoint.json"
DEFAULT_XLSX = ROOT / "summary_latest.xlsx"


def _genome_sha(checkpoint: dict) -> str:
    genome = checkpoint.get("genome") or checkpoint.get("best_genome") or []
    if not genome:
        return "unknown"
    canon = json.dumps(genome, separators=(",", ":"),
                       default=lambda o: round(float(o), 10) if hasattr(o, "__float__") else o)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]


def _checkpoint_fit(checkpoint: dict) -> float:
    snap = checkpoint.get("metrics_snapshot") or {}
    cand = snap.get("candidate") or snap.get("incumbent") or {}
    for key in ("fit", "fitness"):
        if key in cand:
            try:
                return float(cand[key])
            except (TypeError, ValueError):
                pass
    for key in ("fit", "fitness", "best_fitness"):
        if key in checkpoint:
            try:
                return float(checkpoint[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def _latest_xlsx_date(xlsx_path: Path) -> str | None:
    """Le a coluna 'Date' do Apply sheet e retorna a data mais recente."""
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        df = pd.read_excel(xlsx_path, sheet_name="Apply")
    except (FileNotFoundError, ValueError, KeyError):
        return None
    col = "Date" if "Date" in df.columns else (
        "date" if "date" in df.columns else None
    )
    if col is None:
        return None
    try:
        ts = pd.to_datetime(df[col], errors="coerce").dropna().max()
        if ts is None or ts is pd.NaT:
            return None
        return str(ts.date())
    except Exception:
        return None


def _load_checkpoint(checkpoint_path: Path | None) -> dict:
    path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _load_tracker() -> dict:
    if not TRACKER.exists():
        return {}
    try:
        return json.loads(TRACKER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_tracker(data: dict) -> None:
    tmp = TRACKER.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        tmp.replace(TRACKER)
    except OSError:
        # fallback nao-atomico no Windows se replace falhar por lock
        TRACKER.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.unlink(missing_ok=True)


def _business_days_between(d1, d2) -> int:
    """Dias uteis entre d1 (mais antigo) e d2 (mais novo). Aproximado — conta
    dias mas subtrai 2/7 estimativa para weekends. Zero se formatos invalidos."""
    try:
        from datetime import date
        if isinstance(d1, str):
            d1 = date.fromisoformat(d1[:10])
        if isinstance(d2, str):
            d2 = date.fromisoformat(d2[:10])
        days = (d2 - d1).days
        # Subtract weekend days (approx)
        full_weeks = days // 7
        remainder = days % 7
        # remainder could overlap weekend; pessimistic = subtract all possible
        weekend_days = full_weeks * 2 + max(0, min(2, remainder))
        return max(0, days - weekend_days)
    except Exception:
        return 0


def should_send_telegram(xlsx_path: Path | None = None,
                         checkpoint_path: Path | None = None,
                         min_fit_improvement: float = 0.05,
                         max_staleness_bdays: int = 5) -> tuple[bool, str]:
    """
    Decide se o envio e permitido. Retorna (should_send, reason).

    Politica (combinada, AND para BLOCK; OR para ALLOW):

    BLOCK (nunca envia se):
      - xlsx nao existe
      - Data do xlsx e STALE (>= max_staleness_bdays dias uteis atras). Evita
        enviar dados velhos se o cron ETL falhou por muitos dias.
      - Fit do checkpoint REGREDIU >= min_fit_improvement vs ultimo envio.
        Nao spammar usuario com modelo pior.

    ALLOW (envia se qualquer):
      - GA_FORCE_TELEGRAM_SEND=1 no env (override manual)
      - Primeiro envio (tracker vazio)
      - Data do xlsx e MAIS NOVA que ultimo envio E fit nao regrediu
      - genome_sha mudou E fit nao regrediu (modelo updated)
      - fit melhorou >= min_fit_improvement vs ultimo

    Sem nenhum desses gatilhos: DONT SEND (duplicata sem melhoria).
    """
    if os.environ.get("GA_FORCE_TELEGRAM_SEND", "") in ("1", "true", "yes", "on"):
        return True, "GA_FORCE_TELEGRAM_SEND=1"

    xlsx = Path(xlsx_path) if xlsx_path else DEFAULT_XLSX
    if not xlsx.exists():
        return False, f"xlsx nao existe: {xlsx.name}"

    ck = _load_checkpoint(checkpoint_path)
    current = {
        "data_date": _latest_xlsx_date(xlsx),
        "genome_sha": _genome_sha(ck),
        "fit": _checkpoint_fit(ck),
    }

    # GUARD 1: data do xlsx nao pode estar muito antiga (evita envio de xlsx
    # stale se o ETL falhou dias seguidos).
    if current["data_date"]:
        from datetime import date, timedelta
        today = date.today()
        try:
            xlsx_date = date.fromisoformat(current["data_date"][:10])
            bdays_old = _business_days_between(xlsx_date, today)
            if bdays_old > max_staleness_bdays:
                return False, (
                    f"dados stale: xlsx date={current['data_date']} "
                    f"({bdays_old} dias uteis atras, max={max_staleness_bdays}). "
                    f"Rode ETL fresh antes de enviar."
                )
        except (ValueError, TypeError):
            pass  # fall through se nao conseguir parsear

    last = _load_tracker()

    if not last:
        return True, f"primeiro envio (data={current['data_date']}, fit={current['fit']:.3f})"

    # GUARD 2: fit NUNCA pode ter regredido (nao envia downgrade).
    last_fit = float(last.get("fit", 0.0) or 0.0)
    delta = current["fit"] - last_fit
    if delta < -min_fit_improvement:
        return False, (
            f"fit REGREDIU: {last_fit:.3f} -> {current['fit']:.3f} "
            f"(delta {delta:+.3f}). NAO enviar modelo pior que o ultimo."
        )

    # ALLOW 1: data mais nova que ultima enviada (AND fit nao regrediu)
    if current["data_date"] and last.get("data_date"):
        if current["data_date"] > last["data_date"]:
            return True, (
                f"data nova: {last['data_date']} -> {current['data_date']} "
                f"(fit {last_fit:.3f} -> {current['fit']:.3f})"
            )

    # ALLOW 2: genome mudou (modelo atualizado) AND fit nao regrediu
    if current["genome_sha"] != last.get("genome_sha"):
        return True, (
            f"genome mudou: {last.get('genome_sha','?')[:8]} -> "
            f"{current['genome_sha'][:8]} (fit {last_fit:.3f} -> {current['fit']:.3f})"
        )

    # ALLOW 3: fit melhorou significativamente (>= threshold)
    if delta >= min_fit_improvement:
        return True, (
            f"fit melhorou: {last_fit:.3f} -> {current['fit']:.3f} "
            f"(delta {delta:+.3f})"
        )

    # BLOCK default: nada mudou o suficiente
    return False, (
        f"duplicata (ultimo: data={last.get('data_date')}, "
        f"sha={last.get('genome_sha','?')[:8]}, fit={last_fit:.3f} | "
        f"atual: data={current['data_date']}, "
        f"sha={current['genome_sha'][:8]}, fit={current['fit']:.3f}, "
        f"delta={delta:+.4f})"
    )


def record_telegram_sent(xlsx_path: Path | None = None,
                          checkpoint_path: Path | None = None) -> None:
    """Grava estado do envio atual em .telegram_last_sent.json."""
    from datetime import datetime, timezone
    xlsx = Path(xlsx_path) if xlsx_path else DEFAULT_XLSX
    ck = _load_checkpoint(checkpoint_path)
    data = {
        "data_date": _latest_xlsx_date(xlsx),
        "genome_sha": _genome_sha(ck),
        "fit": _checkpoint_fit(ck),
        "xlsx_size": xlsx.stat().st_size if xlsx.exists() else 0,
        "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_tracker(data)


def main():
    """CLI: inspeciona o tracker atual e reporta se enviaria."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--check", action="store_true",
                        help="Apenas reporta se enviaria (nao modifica nada)")
    parser.add_argument("--record", action="store_true",
                        help="Forca registrar estado atual (simula envio)")
    parser.add_argument("--xlsx", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    if args.record:
        record_telegram_sent(args.xlsx, args.checkpoint)
        print(f"[TELEGRAM_DEDUP] tracker updated: {TRACKER}")
        print(TRACKER.read_text(encoding="utf-8"))
        return 0

    should, reason = should_send_telegram(args.xlsx, args.checkpoint)
    print(f"[TELEGRAM_DEDUP] should_send={should}")
    print(f"[TELEGRAM_DEDUP] reason: {reason}")
    if TRACKER.exists():
        print("\nLast-sent tracker:")
        print(TRACKER.read_text(encoding="utf-8"))
    return 0 if should else 1  # exit 1 = skip send


if __name__ == "__main__":
    sys.exit(main())
