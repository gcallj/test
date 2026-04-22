#!/usr/bin/env python3
"""Best-ever tracker + long-term regression guard.

Problema que resolve: o acceptance gate atual usa `baseline_main_ref` como
referencia. Quando main degrada step-by-step (cada promocao aceita uma queda
pequena vs baseline atual), a degradacao se acumula -- "death by a thousand
cuts". Observado: fit 26.46 -> 22.21 em poucas horas mesmo com gates ativos.

Solucao: mantem um `best_ever_tracker.json` com as melhores metricas historicas
observadas. Toda promocao e checada contra ele: se piorar mais que tolerance
vs best_ever, REJEITA, independente de passar no gate vs main.

Uso:
  from analysis.best_ever_guard import (
      load_best_ever, check_regression_vs_best, update_best_ever
  )

  bev = load_best_ever()
  ok, reason = check_regression_vs_best(cand_metrics, bev)
  if not ok:
      raise SystemExit(f"Regression guard: {reason}")
  # ... aplica candidate ...
  update_best_ever(cand_metrics, genome, checkpoint_sha="abc123")
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
TRACKER_PATH = ROOT / "analysis" / "best_ever_tracker.json"

# Tolerancia relativa (em fracao). fit pode cair ate (1-TOL)*best_ever.
# WR/alpha usam tolerancia ABSOLUTA pra nao ser enganoso (WR=70% -> 66% e pior).
FIT_TOL_RELATIVE = 0.04         # fit >= best * 0.96 (aceita ate 4% queda)
WR_TOL_ABSOLUTE = 0.015         # WR >= best_wr - 1.5pp
ALPHA_TOL_ABSOLUTE = 0.010      # alpha_ann >= best_alpha - 1.0pp
N_GE_70_TOL_ABSOLUTE = 5        # n_ge_70 >= best_n_ge_70 - 5

MAX_HISTORY = 50  # mantem ultimas 50 atualizacoes do best_ever


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_best_ever() -> Dict[str, Any]:
    """Retorna dict best_ever; default vazio se nao existe ainda."""
    if not TRACKER_PATH.exists():
        return {}
    try:
        return json.loads(TRACKER_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[best_ever_guard] Warn: falha lendo {TRACKER_PATH.name}: {e}")
        return {}


def _save(data: Dict[str, Any]) -> None:
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(TRACKER_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, TRACKER_PATH)


def check_regression_vs_best(cand_metrics: Dict[str, Any],
                             best: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Retorna (ok, reason). ok=False se candidato regride vs best_ever."""
    if best is None:
        best = load_best_ever()
    if not best or "best_fit" not in best:
        # Primeira vez — sem baseline, aceita qualquer coisa
        return True, "no_best_ever_yet"

    cand_fit = float(cand_metrics.get("fit", 0.0) or 0.0)
    cand_wr = float(cand_metrics.get("wr_med_all", 0.0) or 0.0)
    cand_alpha = float(cand_metrics.get("mean_alpha_ann", 0.0) or 0.0)
    cand_n70 = int(cand_metrics.get("n_ge_70", 0) or 0)

    best_fit = float(best.get("best_fit", 0.0) or 0.0)
    best_wr = float(best.get("best_wr_med_all", 0.0) or 0.0)
    best_alpha = float(best.get("best_alpha_ann", 0.0) or 0.0)
    best_n70 = int(best.get("best_n_ge_70", 0) or 0)

    # Cada criterio e um possivel rejeite. So precisa de 1 pra falhar.
    checks = []
    if cand_fit < best_fit * (1.0 - FIT_TOL_RELATIVE):
        checks.append(
            f"fit {cand_fit:.4f} < {best_fit * (1.0 - FIT_TOL_RELATIVE):.4f} "
            f"(best={best_fit:.4f} * {1-FIT_TOL_RELATIVE:.2f})"
        )
    if cand_wr < best_wr - WR_TOL_ABSOLUTE:
        checks.append(
            f"WR {cand_wr:.4f} < {best_wr - WR_TOL_ABSOLUTE:.4f} "
            f"(best={best_wr:.4f} - {WR_TOL_ABSOLUTE})"
        )
    if cand_alpha < best_alpha - ALPHA_TOL_ABSOLUTE:
        checks.append(
            f"alpha {cand_alpha:+.4f} < {best_alpha - ALPHA_TOL_ABSOLUTE:+.4f} "
            f"(best={best_alpha:+.4f} - {ALPHA_TOL_ABSOLUTE})"
        )
    if cand_n70 < best_n70 - N_GE_70_TOL_ABSOLUTE:
        checks.append(
            f"n_ge_70 {cand_n70} < {best_n70 - N_GE_70_TOL_ABSOLUTE} "
            f"(best={best_n70} - {N_GE_70_TOL_ABSOLUTE})"
        )

    if checks:
        return False, "; ".join(checks)
    return True, "within_tolerance"


def update_best_ever(cand_metrics: Dict[str, Any],
                     genome: List[float],
                     checkpoint_sha: str = "",
                     reason: str = "") -> bool:
    """Atualiza best_ever se candidato superar em >= 1 metrica principal.

    Returns True se o tracker foi atualizado.
    """
    best = load_best_ever()
    cand_fit = float(cand_metrics.get("fit", 0.0) or 0.0)
    cand_wr = float(cand_metrics.get("wr_med_all", 0.0) or 0.0)
    cand_alpha = float(cand_metrics.get("mean_alpha_ann", 0.0) or 0.0)
    cand_n70 = int(cand_metrics.get("n_ge_70", 0) or 0)

    # Atualiza cada campo INDEPENDENTEMENTE — manter o melhor historico por metrica.
    # Isso evita o problema de "melhor fit mas WR pior" ficar preso ao snapshot antigo.
    updated = False
    new_best = dict(best) if best else {}
    for metric_key, cand_val, best_key, better_fn in [
        ("fit", cand_fit, "best_fit", lambda c, b: c > b),
        ("wr_med_all", cand_wr, "best_wr_med_all", lambda c, b: c > b),
        ("alpha_ann", cand_alpha, "best_alpha_ann", lambda c, b: c > b),
        ("n_ge_70", cand_n70, "best_n_ge_70", lambda c, b: c > b),
    ]:
        existing = new_best.get(best_key)
        if existing is None or better_fn(cand_val, existing):
            new_best[best_key] = cand_val
            updated = True

    # Se melhorou, grava o genome tambem (snapshot do "mais recente best").
    # Nota: genome refletira SO o ultimo update, nao necessariamente uma
    # combinacao Pareto-dominante em todas as metricas.
    if updated:
        new_best["last_best_genome"] = list(genome)
        new_best["last_update"] = _now_iso()
        new_best["last_checkpoint_sha"] = checkpoint_sha
        new_best["last_reason"] = reason
        history = list(new_best.get("history", []))
        history.append({
            "when": _now_iso(),
            "fit": cand_fit,
            "wr": cand_wr,
            "alpha": cand_alpha,
            "n_ge_70": cand_n70,
            "sha": checkpoint_sha,
            "reason": reason[:80] if reason else "",
        })
        # Trunca historico pra nao crescer infinitamente
        new_best["history"] = history[-MAX_HISTORY:]
        _save(new_best)
        print(f"[best_ever] UPDATED: "
              f"fit={new_best.get('best_fit',0):.4f} "
              f"WR={new_best.get('best_wr_med_all',0):.4f} "
              f"alpha={new_best.get('best_alpha_ann',0):+.4f} "
              f"n_ge_70={new_best.get('best_n_ge_70',0)}")
    return updated


def format_best_ever_summary(best: Optional[Dict[str, Any]] = None) -> str:
    """Retorna string legivel com estado atual."""
    if best is None:
        best = load_best_ever()
    if not best:
        return "(no best_ever tracker yet)"
    return (
        f"BEST EVER: fit={best.get('best_fit', 0):.4f} "
        f"WR={best.get('best_wr_med_all', 0):.4f} "
        f"alpha={best.get('best_alpha_ann', 0):+.4f} "
        f"n_ge_70={best.get('best_n_ge_70', 0)} "
        f"(last_update={best.get('last_update', '?')})"
    )


if __name__ == "__main__":
    # CLI: python analysis/best_ever_guard.py
    # Mostra status atual.
    print(format_best_ever_summary())
    bev = load_best_ever()
    if bev.get("history"):
        print(f"\nLast {min(5, len(bev['history']))} updates:")
        for e in bev["history"][-5:]:
            print(f"  {e.get('when')}: fit={e.get('fit',0):.4f} "
                  f"WR={e.get('wr',0):.4f} sha={e.get('sha','')[:8]}")
