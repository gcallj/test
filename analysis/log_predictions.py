#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal logger (Phase C1 do plano de overfitting prevention).

Le summary_latest.xlsx (Apply sheet) + global_ga_checkpoint.json, e anexa
cada sinal em analysis/predictions_log.jsonl. Cada linha JSON contem:

    date, ticker, signal, entrada, stop, alvo, confidence,
    genome_sha (sha1 dos genes do checkpoint no momento da emissao),
    checkpoint_fit, wf_oos_alpha, logged_at

O log e append-only. Sinais do mesmo (date, ticker, genome_sha) sao
idempotentes: nao duplicam se o script rodar duas vezes no mesmo dia.

Chamado por predict_daily.py no fim do pipeline. Pode tambem ser rodado
manualmente:

    python analysis/log_predictions.py            # le summary_latest.xlsx
    python analysis/log_predictions.py --dry-run  # mostra o que seria logado
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_XLSX = ROOT / "summary_latest.xlsx"
CHECKPOINT = ROOT / "global_ga_checkpoint.json"
LOG_PATH = ROOT / "analysis" / "predictions_log.jsonl"


def _genome_sha(checkpoint: dict) -> str:
    """sha1 dos genes ordenados. Permite segmentar realized outcomes por versao."""
    genome = checkpoint.get("genome") or checkpoint.get("best_genome") or []
    if not genome:
        return "unknown"
    # Canonicalize: round floats to 10 decimal places, sort if dict else preserve list order
    canonical = json.dumps(genome, sort_keys=False, separators=(",", ":"),
                           default=lambda o: round(float(o), 10) if hasattr(o, "__float__") else o)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def _checkpoint_fit(checkpoint: dict) -> float:
    """Best-effort extract do fit atual do checkpoint."""
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


def _checkpoint_alpha(checkpoint: dict) -> float:
    """Best-effort extract do WF-OOS alpha_ann atual."""
    snap = checkpoint.get("metrics_snapshot") or {}
    cand = snap.get("candidate") or snap.get("incumbent") or {}
    for key in ("mean_alpha_ann", "alpha_ann_med", "alpha_ann"):
        if key in cand:
            try:
                return float(cand[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def _existing_keys(log_path: Path) -> set[tuple]:
    """Le log existente e retorna set de (date, ticker, genome_sha) ja registrados."""
    if not log_path.exists():
        return set()
    keys: set[tuple] = set()
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                keys.add((rec.get("date"), rec.get("ticker"), rec.get("genome_sha")))
            except json.JSONDecodeError:
                continue
    return keys


def _read_apply_sheet(xlsx_path: Path):
    """Le sheet 'Apply' do summary_latest.xlsx. Retorna DataFrame ou None."""
    try:
        import pandas as pd
    except ImportError:
        print("[LOG_PREDICTIONS] pandas nao disponivel", file=sys.stderr)
        return None
    if not xlsx_path.exists():
        print(f"[LOG_PREDICTIONS] {xlsx_path} nao existe (skip)", file=sys.stderr)
        return None
    try:
        # Apply e o sheet mais comum (gerado em ga_run.py:4603 / 4717)
        df = pd.read_excel(xlsx_path, sheet_name="Apply")
        return df
    except (ValueError, KeyError) as e:
        print(f"[LOG_PREDICTIONS] sheet Apply nao encontrada em {xlsx_path}: {e}",
              file=sys.stderr)
        return None


def _normalize_row(row: dict) -> dict:
    """Mapeia colunas do Apply sheet para schema canonico."""
    def _get(keys, default=None):
        for k in keys:
            if k in row and row[k] is not None:
                return row[k]
        return default

    def _fmt_date(v):
        if v is None:
            return None
        try:
            import pandas as pd
            ts = pd.to_datetime(v)
            return ts.strftime("%Y-%m-%d") if not (hasattr(ts, "isnat") and ts is None) else None
        except Exception:
            return str(v)[:10]

    def _num(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "date": _fmt_date(_get(("date", "Date", "data", "trading_date"))),
        "ticker": _get(("ticker", "Ticker", "symbol", "Symbol")),
        "signal": _get(("signal", "Signal", "recomendacao", "Recomendacao",
                        "action", "Action")),
        "entrada": _num(_get(("entrada", "Entrada", "entry", "price", "close"))),
        "stop": _num(_get(("stop", "Stop", "stop_loss", "sl"))),
        "alvo": _num(_get(("alvo", "Alvo", "target", "tp", "alvo_price"))),
        "confidence": _num(_get(("confidence", "Confidence", "confianca",
                                  "proba", "score"))),
        "win_rate": _num(_get(("win_rate", "WR", "wr", "hist_wr"))),
        "potential_pct": _num(_get(("potencial_pct", "potential_pct", "potential"))),
        "rank": _num(_get(("rank", "Rank"))),
    }


def append_log(dry_run: bool = False) -> int:
    """Reads summary_latest.xlsx, appends new rows to predictions_log.jsonl.

    Returns number of rows appended (0 if file missing or nothing new).
    """
    df = _read_apply_sheet(SUMMARY_XLSX)
    if df is None or df.empty:
        return 0

    # Load checkpoint for genome_sha + fit/alpha
    try:
        ckpt = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"[LOG_PREDICTIONS] {CHECKPOINT} nao existe, usando genome_sha=unknown",
              file=sys.stderr)
        ckpt = {}
    except json.JSONDecodeError as e:
        print(f"[LOG_PREDICTIONS] checkpoint invalido: {e}", file=sys.stderr)
        ckpt = {}

    gsha = _genome_sha(ckpt)
    cfit = _checkpoint_fit(ckpt)
    calpha = _checkpoint_alpha(ckpt)
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    existing = _existing_keys(LOG_PATH)
    to_write: list[str] = []

    for _, row in df.iterrows():
        normalized = _normalize_row(row.to_dict())
        key = (normalized["date"], normalized["ticker"], gsha)
        if key[0] is None or key[1] is None:
            continue
        if key in existing:
            continue
        record = {
            **normalized,
            "genome_sha": gsha,
            "checkpoint_fit": cfit,
            "wf_oos_alpha": calpha,
            "logged_at": now_iso,
        }
        to_write.append(json.dumps(record, ensure_ascii=False))

    if not to_write:
        print(f"[LOG_PREDICTIONS] nothing to log (all {len(df)} rows already logged)")
        return 0

    if dry_run:
        print(f"[LOG_PREDICTIONS] DRY RUN — would append {len(to_write)} rows")
        for line in to_write[:3]:
            print(f"  {line}")
        if len(to_write) > 3:
            print(f"  ... ({len(to_write) - 3} more)")
        return len(to_write)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        for line in to_write:
            f.write(line + "\n")

    print(f"[LOG_PREDICTIONS] appended {len(to_write)} rows to {LOG_PATH.name} "
          f"(genome_sha={gsha}, fit={cfit:.2f}, alpha={calpha:.4f})")
    return len(to_write)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra o que seria logado sem escrever")
    args = parser.parse_args()
    try:
        n = append_log(dry_run=args.dry_run)
    except Exception as e:
        # Nao queremos quebrar o predict_daily.py por falha de logging.
        print(f"[LOG_PREDICTIONS] erro (nao-fatal): {e}", file=sys.stderr)
        return 0
    return 0 if n >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
