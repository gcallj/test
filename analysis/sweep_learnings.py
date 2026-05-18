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
CODEX_HISTORY = ANALYSIS / "codex_attempts_history.json"


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


def _load_codex_history() -> dict:
    """Load parsed codex attempts history (written by codex_attempts_sync.py).

    Returns {} if the file is absent (e.g. remote runner with no codex source).
    """
    if not CODEX_HISTORY.exists():
        return {}
    try:
        return json.loads(CODEX_HISTORY.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] could not parse {CODEX_HISTORY.name}: {e}")
        return {}


def _local_sweeps_with_promotion(sweeps: list[tuple[Path, dict]]) -> dict:
    """Index local sweeps by filename, flagging which ones already encode a promotion.

    Returns: {filename: {"has_promoted": bool, "promoted_genes": set[str]}}

    This lets the codex merge detect three cases per attempt:
      (a) local JSON missing entirely         -> codex is the only record
      (b) local JSON present, has_promoted    -> dedup: skip codex counts
      (c) local JSON present, NOT promoted    -> codex adds promotion info
                                                  (the JSON was imported
                                                  without best_promoted set)
    """
    idx: dict = {}
    for path, sweep in sweeps:
        bp = sweep.get("best_promoted") or {}
        idx[path.name] = {
            "has_promoted": bool(bp),
            "promoted_genes": {m["gene"] for m in bp.get("moves", [])},
        }
    return idx


def _extract_promotions(
    sweeps: list[tuple[Path, dict]],
    codex_history: dict | None = None,
) -> list[dict]:
    """Pick sweeps that had a promoted candidate.

    Merges codex promotions:
      - if the local sweep JSON already has best_promoted   -> skip (dedup)
      - if the local JSON is silent or absent                -> add codex claim
    """
    promos: list[dict] = []
    local_idx = _local_sweeps_with_promotion(sweeps)
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
            "from_codex": False,
        })

    if codex_history:
        for att in codex_history.get("attempts", []):
            if not att.get("promoted"):
                continue
            sweep_file = att.get("sweep_file") or ""
            local_info = local_idx.get(sweep_file)
            # dedup only if local JSON already encodes a promotion
            if local_info and local_info["has_promoted"]:
                continue
            promos.append({
                "file": sweep_file or f"codex_attempt_{att['attempt_id']}",
                "started_at": att.get("timestamp", ""),
                "moves": att.get("moves", []),
                "metrics": att.get("metrics", {}).get("candidate", {}),
                "label": att.get("section_title", ""),
                "from_codex": True,
            })
    promos.sort(key=lambda d: d.get("started_at") or "")
    return promos


def _gene_attempt_counts(
    sweeps: list[tuple[Path, dict]],
    codex_history: dict | None = None,
) -> dict:
    """Count how many times each gene appeared as a sweep target.

    Dedup rule:
      - if codex attempt's sweep_file exists locally AND that local JSON
        already has best_promoted, the local count is authoritative — skip
        codex for that attempt.
      - else the codex attempt contributes attempts / promotions (counted
        once in `attempts`/`promotions` AND separately in `codex_*` columns).

    Returns: {gene: {"attempts": int, "promotions": int,
                     "codex_attempts": int, "codex_promotions": int}}
    """
    counts: dict = defaultdict(
        lambda: {"attempts": 0, "promotions": 0,
                 "codex_attempts": 0, "codex_promotions": 0}
    )
    local_idx = _local_sweeps_with_promotion(sweeps)
    for path, sweep in sweeps:
        focus = sweep.get("focus_genes") or []
        promoted_genes = set()
        if sweep.get("best_promoted"):
            for mv in sweep["best_promoted"].get("moves", []):
                promoted_genes.add(mv["gene"])
        for g in focus:
            counts[g]["attempts"] += 1
            if g in promoted_genes:
                counts[g]["promotions"] += 1

    if codex_history:
        for att in codex_history.get("attempts", []):
            sweep_file = att.get("sweep_file") or ""
            local_info = local_idx.get(sweep_file)
            # If local JSON is present and already promoted, full dedup
            if local_info and local_info["has_promoted"]:
                continue
            genes = att.get("genes") or []
            promoted = bool(att.get("promoted"))
            moved_genes = {m["gene"] for m in att.get("moves", []) if m.get("gene")}
            # If local JSON is present (no promotion), it already counted the
            # attempts for its focus_genes via the loop above. Avoid
            # double-counting: add only promotions in that case.
            already_counted_attempts = bool(local_info)
            for g in genes:
                if not already_counted_attempts:
                    counts[g]["attempts"] += 1
                    counts[g]["codex_attempts"] += 1
                else:
                    # Flag it as also seen by codex for visibility, but do
                    # not double-count.
                    counts[g]["codex_attempts"] += 1
                if promoted and g in moved_genes:
                    counts[g]["promotions"] += 1
                    counts[g]["codex_promotions"] += 1
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
    codex_history = _load_codex_history()
    promos = _extract_promotions(sweeps, codex_history)
    counts = _gene_attempt_counts(sweeps, codex_history)
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
    codex_meta = (codex_history or {}).get("codex_sync_metadata", {})
    codex_attempts = codex_meta.get("n_attempts_parsed", 0)
    codex_promoted = codex_meta.get("n_promoted", 0)
    codex_from_codex_only = sum(1 for p in promos if p.get("from_codex"))

    md.append("## Current snapshot\n\n")
    md.append(f"- Checkpoint fitness: `{checkpoint_fitness}`\n")
    md.append(f"- Total sweeps seen: **{len(sweeps)}**\n")
    md.append(f"- Total promotions: **{len(promos)}** "
              f"(including {codex_from_codex_only} from codex log)\n")
    if sweeps:
        promo_rate = len(promos) / max(1, len(sweeps) + codex_attempts)
        md.append(f"- Promotion rate: **{promo_rate*100:.1f}%** "
                  f"(over {len(sweeps) + codex_attempts} total attempts)\n")
    md.append(f"- Continuous cycles run: {state.get('total_cycles', 0)}\n")
    md.append(f"- Continuous promotions: {state.get('total_promotions', 0)}\n")
    if codex_meta:
        md.append(f"- Codex attempts synced: **{codex_attempts}** "
                  f"({codex_promoted} promoted) from "
                  f"`{Path(codex_meta.get('source_file', '')).name or 'codex'}`\n")
    md.append("\n")

    # Section 2: Suggested next action
    md.append("## Next suggested sweep\n\n")
    md.append(f"{_suggest_next(state, counts)}\n\n")

    # Section 3: Promotion timeline
    md.append("## Promotion timeline (chronological)\n\n")
    if not promos:
        md.append("_No promotions recorded yet._\n\n")
    else:
        md.append("| Date | Source | Sweep file | Move | Resulting metrics |\n")
        md.append("|---|---|---|---|---|\n")
        for p in promos:
            moves_str = "; ".join(f"{m['gene']} {m['from']}->{m['to']}" for m in p["moves"][:3])
            if len(p["moves"]) > 3:
                moves_str += f" (+{len(p['moves']) - 3} more)"
            source = "codex" if p.get("from_codex") else "local"
            md.append(f"| {p['started_at'][:10]} | {source} | `{p['file']}` | "
                      f"`{moves_str}` | {_format_metrics(p['metrics'])} |\n")
        md.append("\n")

    # Section 4: Graveyard
    md.append("## Graveyard (genes que falharam multiplas vezes)\n\n")
    md.append("Genes tentados >= 2 vezes sem nenhuma promocao. Considere SKIP no proximo ciclo.\n")
    md.append("(Colunas `codex` contam tentativas importadas do log do GPT codex — dedup por sweep file.)\n\n")
    if not graveyard:
        md.append("_Vazio (todo gene tentado teve pelo menos 1 promocao OU foi tentado so 1 vez)._\n\n")
    else:
        md.append("| Gene | Attempts | Promotions | (codex) Attempts | (codex) Promotions |\n")
        md.append("|---|---:|---:|---:|---:|\n")
        for g, c in graveyard[:25]:
            md.append(
                f"| `{g}` | {c['attempts']} | {c['promotions']} | "
                f"{c.get('codex_attempts', 0)} | {c.get('codex_promotions', 0)} |\n"
            )
        md.append("\n")

    # Section 5: Hot zones
    md.append("## Hot zones (genes que ja promoveram)\n\n")
    md.append("Genes onde alguma combinacao funcionou. Vale revisitar com novos vizinhos.\n\n")
    if not hot:
        md.append("_Nenhuma promocao registrada ainda._\n\n")
    else:
        md.append("| Gene | Promotions | Attempts | Hit rate | (codex) Promotions |\n")
        md.append("|---|---:|---:|---:|---:|\n")
        for g, c in hot:
            rate = c['promotions'] / c['attempts'] if c['attempts'] else 0
            md.append(
                f"| `{g}` | {c['promotions']} | {c['attempts']} | "
                f"{rate*100:.0f}% | {c.get('codex_promotions', 0)} |\n"
            )
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

    # Section 7: Codex sync history (last 5 attempts)
    if codex_history and codex_history.get("attempts"):
        md.append("## Codex sync history (latest attempts)\n\n")
        meta = codex_history["codex_sync_metadata"]
        src_name = Path(meta.get("source_file", "")).name or "(unknown)"
        md.append(f"Parsed `{src_name}` at {meta.get('last_parsed', '?')}. ")
        md.append(f"Total attempts: **{meta.get('n_attempts_parsed', 0)}**, "
                  f"promoted: **{meta.get('n_promoted', 0)}**. ")
        md.append("Updates via `python analysis/codex_attempts_sync.py` at the start "
                  "of every continuous cycle (auto) or on demand.\n\n")
        md.append("| Attempt | Timestamp | Promoted | Move(s) | Learning (snippet) |\n")
        md.append("|---|---|:-:|---|---|\n")
        latest = sorted(
            codex_history["attempts"],
            key=lambda a: (a.get("timestamp") or "", a.get("attempt_id") or ""),
            reverse=True,
        )[:5]
        for a in latest:
            moves_str = "; ".join(f"{m['gene']} {m['from']}->{m['to']}"
                                  for m in (a.get("moves") or [])[:2])
            if not moves_str:
                moves_str = "(no moves)"
            learning = (a.get("learning") or "").replace("\n", " ").replace("|", "/")[:120]
            if learning and len(a.get("learning", "")) > 120:
                learning += "..."
            promoted_mark = "YES" if a.get("promoted") else "no"
            md.append(
                f"| #{a.get('attempt_id', '?')} | "
                f"{(a.get('timestamp') or '').replace('|', '/')} | "
                f"{promoted_mark} | `{moves_str}` | {learning} |\n"
            )
        md.append("\n")

    # D3: per-segment breakdown when summary_latest.xlsx is available.
    summary_xlsx = ROOT / "summary_latest.xlsx"
    if summary_xlsx.exists():
        try:
            from analysis.sector_breakdown import breakdown_by_segment, format_markdown, from_summary_xlsx
            per_ticker = from_summary_xlsx(summary_xlsx)
            seg_df = breakdown_by_segment(per_ticker)
            md.append("\n## Per-segment breakdown (latest summary)\n\n")
            md.append(format_markdown(seg_df, title="Universe x segment"))
            md.append("\n")
        except Exception as e:  # missing openpyxl, malformed sheet, etc.
            print(f"[LEARNINGS] sector breakdown skipped: {e}")

    md.append("---\n\n_Run again with: `python analysis/sweep_learnings.py`_\n")

    SUMMARY.write_text("".join(md), encoding="utf-8")
    print(f"[LEARNINGS] Wrote {SUMMARY}")
    codex_count = codex_meta.get("n_attempts_parsed", 0)
    print(f"[LEARNINGS] Sweeps: {len(sweeps)}  Promotions: {len(promos)}  "
          f"Graveyard: {len(graveyard)}  Hot zones: {len(hot)}  "
          f"Codex attempts: {codex_count}")


if __name__ == "__main__":
    main()
