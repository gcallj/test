#!/usr/bin/env python3
"""
Aggregate todos os sweep results JSON e o log markdown em
analysis/sweep_learnings_summary.md.

Le:
  - local_fullmetric_sweep_result_*.json (todos os sweeps historicos)
  - analysis/optimization_log.md (log da rotina continua)
  - continuous_improvement_state.json (rotacao state)
  - global_ga_checkpoint.json (incumbent atual)

Produz analysis/sweep_learnings_summary.md com:
  - Estatisticas globais (n_attempts, n_promotions, taxa)
  - Top promocoes ja aplicadas (cronologia)
  - "Graveyard": moves que foram tentados N vezes e nunca promoveram
  - "Hot zones": genes onde toda tentativa promoveu (ainda vale tentar)
  - Sugestao concreta para o proximo sweep
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
LOG = ANALYSIS / "optimization_log.md"
SUMMARY = ANALYSIS / "sweep_learnings_summary.md"
STATE = ROOT / "continuous_improvement_state.json"
CHECKPOINT = ROOT / "global_ga_checkpoint.json"


def _load_all_sweeps() -> list[tuple[Path, dict]]:
    """Load all sweep result JSONs (continuous + manual codex sweeps)."""
    out = []
    patterns = [
        "local_fullmetric_sweep_result*.json",
        "local_fullmetric_sweep_continuous_*.json",
    ]
    seen = set()
    for pat in patterns:
        for p in ROOT.glob(pat):
            if p.name in seen:
                continue
            seen.add(p.name)
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                out.append((p, data))
            except Exception as e:
                print(f"[WARN] could not parse {p.name}: {e}")
    out.sort(key=lambda t: t[0].stat().st_mtime)
    return out


def _extract_promotions(sweeps: list[tuple[Path, dict]]) -> list[dict]:
    """Pick sweeps that had a promoted candidate."""
    promos = []
    for p, sweep in sweeps:
        best = sweep.get("best_promoted")
        if not best:
            continue
        promos.append({
            "file": p.name,
            "started_at": sweep.get("started_at", ""),
            "moves": best.get("moves", []),
            "metrics": best.get("metrics", {}),
            "label": best.get("label", ""),
        })
    return promos


def _gene_attempt_counts(sweeps: list[tuple[Path, dict]]) -> dict:
    """Count how many times each gene appeared as a sweep target.

    Returns: {gene: {"attempts": int, "promotions": int}}
    """
    counts = defaultdict(lambda: {"attempts": 0, "promotions": 0})
    for _path, sweep in sweeps:
        focus = sweep.get("focus_genes") or []
        promoted_genes = set()
        if sweep.get("best_promoted"):
            for mv in sweep["best_promoted"].get("moves", []):
                promoted_genes.add(mv["gene"])
        for g in focus:
            counts[g]["attempts"] += 1
            if g in promoted_genes:
                counts[g]["promotions"] += 1
    return dict(counts)


def _graveyard(gene_counts: dict, min_attempts: int = 2) -> list[tuple[str, dict]]:
    """Genes attempted >= min_attempts times with zero promotions."""
    return sorted(
        [(g, c) for g, c in gene_counts.items()
         if c["attempts"] >= min_attempts and c["promotions"] == 0],
        key=lambda t: -t[1]["attempts"],
    )


def _hot_zones(gene_counts: dict) -> list[tuple[str, dict]]:
    """Genes with at least one promotion."""
    return sorted(
        [(g, c) for g, c in gene_counts.items() if c["promotions"] > 0],
        key=lambda t: (-t[1]["promotions"], t[1]["attempts"]),
    )


def _suggest_next(state: dict, gene_counts: dict) -> str:
    """Recommend the next sweep target."""
    cats = state.get("categories", {})
    if not cats:
        return "rotina ainda nao iniciada — rodar `python run_continuous_improvement.py` para criar state inicial."

    # Never-swept categories first
    never = [c for c, info in cats.items() if not info.get("last_swept")]
    if never:
        return f"categoria nao explorada: **{sorted(never)[0]}** (rodar `python run_continuous_improvement.py --category {sorted(never)[0].split('_')[0]}`)"

    # Otherwise oldest category
    oldest = min(cats.items(), key=lambda t: t[1].get("last_swept") or "")
    return (f"categoria mais antiga: **{oldest[0]}** (ultima vez: {oldest[1].get('last_swept')}) "
            f"-> `python run_continuous_improvement.py --category {oldest[0].split('_')[0]}`")


def _format_metrics(m: dict) -> str:
    if not m:
        return "n/a"
    return (f"fit={m.get('fit', 0):.3f} "
            f"WR={(m.get('wr_med_all', 0) or 0)*100:.1f}% "
            f"alpha={(m.get('mean_alpha_ann', 0) or 0)*100:+.2f}% "
            f"MDD={(m.get('mdd_med', 0) or 0)*100:.1f}%")


def main() -> None:
    sweeps = _load_all_sweeps()
    promos = _extract_promotions(sweeps)
    counts = _gene_attempt_counts(sweeps)
    graveyard = _graveyard(counts)
    hot = _hot_zones(counts)

    state = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass

    checkpoint_metrics = {}
    checkpoint_fitness = None
    if CHECKPOINT.exists():
        try:
            ck = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
            checkpoint_fitness = ck.get("fitness")
            checkpoint_metrics = ck.get("metrics_snapshot", {}).get("candidate") or \
                               ck.get("metrics_snapshot", {}).get("incumbent") or {}
        except Exception:
            pass

    md = []
    md.append("# Sweep Learnings Summary\n\n")
    md.append(f"_Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}_\n\n")
    md.append("Aggregates all `local_fullmetric_sweep_result*.json` files plus the\n")
    md.append("continuous-improvement state. Use this file to plan the next cycle.\n\n")

    # Section 1: Snapshot
    md.append("## Current snapshot\n\n")
    md.append(f"- Checkpoint fitness: `{checkpoint_fitness}`\n")
    md.append(f"- Total sweeps seen: **{len(sweeps)}**\n")
    md.append(f"- Total promotions: **{len(promos)}**\n")
    if sweeps:
        promo_rate = len(promos) / max(1, len(sweeps))
        md.append(f"- Promotion rate: **{promo_rate*100:.1f}%**\n")
    md.append(f"- Continuous cycles run: {state.get('total_cycles', 0)}\n")
    md.append(f"- Continuous promotions: {state.get('total_promotions', 0)}\n\n")

    # Section 2: Suggested next action
    md.append("## Next suggested sweep\n\n")
    md.append(f"{_suggest_next(state, counts)}\n\n")

    # Section 3: Promotion timeline
    md.append("## Promotion timeline (chronological)\n\n")
    if not promos:
        md.append("_No promotions recorded yet._\n\n")
    else:
        md.append("| Date | Sweep file | Move | Resulting metrics |\n")
        md.append("|---|---|---|---|\n")
        for p in promos:
            moves_str = "; ".join(f"{m['gene']} {m['from']}->{m['to']}" for m in p["moves"][:3])
            if len(p["moves"]) > 3:
                moves_str += f" (+{len(p['moves']) - 3} more)"
            md.append(f"| {p['started_at'][:10]} | `{p['file']}` | `{moves_str}` | "
                      f"{_format_metrics(p['metrics'])} |\n")
        md.append("\n")

    # Section 4: Graveyard
    md.append("## Graveyard (genes que falharam multiplas vezes)\n\n")
    md.append("Genes tentados >= 2 vezes sem nenhuma promocao. Considere SKIP no proximo ciclo.\n\n")
    if not graveyard:
        md.append("_Vazio (todo gene tentado teve pelo menos 1 promocao OU foi tentado so 1 vez)._\n\n")
    else:
        md.append("| Gene | Attempts | Promotions |\n")
        md.append("|---|---:|---:|\n")
        for g, c in graveyard[:20]:
            md.append(f"| `{g}` | {c['attempts']} | {c['promotions']} |\n")
        md.append("\n")

    # Section 5: Hot zones
    md.append("## Hot zones (genes que ja promoveram)\n\n")
    md.append("Genes onde alguma combinacao funcionou. Vale revisitar com novos vizinhos.\n\n")
    if not hot:
        md.append("_Nenhuma promocao registrada ainda._\n\n")
    else:
        md.append("| Gene | Promotions | Attempts | Hit rate |\n")
        md.append("|---|---:|---:|---:|\n")
        for g, c in hot:
            rate = c['promotions'] / c['attempts'] if c['attempts'] else 0
            md.append(f"| `{g}` | {c['promotions']} | {c['attempts']} | {rate*100:.0f}% |\n")
        md.append("\n")

    # Section 6: Category status
    md.append("## Category rotation status\n\n")
    cats = state.get("categories", {})
    if not cats:
        md.append("_Continuous improvement state nao inicializado._\n\n")
    else:
        md.append("| Category | Last swept | Runs | Promotions |\n")
        md.append("|---|---|---:|---:|\n")
        for cat, info in sorted(cats.items()):
            md.append(f"| {cat} | {info.get('last_swept') or 'never'} | "
                      f"{info.get('n_runs', 0)} | {info.get('n_promotions', 0)} |\n")
        md.append("\n")

    md.append("---\n\n_Run again with: `python analysis/sweep_learnings.py`_\n")

    SUMMARY.write_text("".join(md), encoding="utf-8")
    print(f"[LEARNINGS] Wrote {SUMMARY}")
    print(f"[LEARNINGS] Sweeps: {len(sweeps)}  Promotions: {len(promos)}  "
          f"Graveyard: {len(graveyard)}  Hot zones: {len(hot)}")


if __name__ == "__main__":
    main()
