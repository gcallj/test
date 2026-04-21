#!/usr/bin/env python3
"""Gene effect diagnostic — detecta genes com zero variance em sweeps recentes.

Um gene "morto" e aquele que produz O MESMO fit/WR em multiplos valores testados
no sweep. Isso indica que o gate/logica do gene nunca dispara no backtest —
provavel bug de implementacao (ex: threshold errado, range inadequado).

Uso:
  python analysis/gene_effect_diagnostic.py                    # ultimos 10 sweeps
  python analysis/gene_effect_diagnostic.py --tail 20          # ultimos 20
  python analysis/gene_effect_diagnostic.py --output report.json

Saida:
  - analysis/gene_effect_report.json  — estrutura machine-readable
  - stdout resumo humano + exit code != 0 se houver flags criticos

Hooks automaticos:
  - Chamado de run_continuous_improvement.py apos cada ciclo
  - Chamado do workflow gene_monitor.yml a cada 2h
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
SWEEPS_GLOB = "local_fullmetric_sweep_continuous_*.json"

# Limiar: gene com stddev(fit) < este valor EM TODOS os valores testados e "morto"
ZERO_VARIANCE_EPSILON = 1e-4

# Gene-specific thresholds quando aplicavel (ex: int-only genes tem saltos maiores)
CRITICAL_MIN_VALUES_TESTED = 3  # precisa de >=3 valores distintos pra conclusao


def _iter_recent_sweeps(tail: int = 10) -> List[Path]:
    """Retorna os `tail` sweeps mais recentes ordenados por mtime decrescente."""
    all_sweeps = sorted(ROOT.glob(SWEEPS_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    return all_sweeps[:tail]


def _collect_gene_observations(sweep_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Parse sweep JSON e coleta {gene_name: [{value, fit, wr, ...}]} para single_results.

    Ignora combo_results (multiple genes movendo juntos confunde atribuicao de efeito).
    """
    try:
        data = json.loads(sweep_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] falha ao ler {sweep_path.name}: {e}")
        return {}

    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in data.get("single_results", []):
        moves = r.get("moves", [])
        # Single = exatamente 1 move
        if len(moves) != 1:
            continue
        m = moves[0]
        gene = m.get("gene")
        if not gene:
            continue
        metrics = r.get("metrics", {})
        fit = metrics.get("fit")
        if fit is None:
            continue
        out.setdefault(gene, []).append({
            "sweep": sweep_path.name,
            "value": m.get("to"),
            "fit": float(fit),
            "wr": float(metrics.get("wr_med_all", 0.0) or 0.0),
            "alpha": float(metrics.get("mean_alpha_ann", 0.0) or 0.0),
            "n_ge_70": int(metrics.get("n_ge_70", 0) or 0),
        })
    return out


def _compute_variance(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcula stddev de fit/wr/alpha entre as observacoes deste gene."""
    if not observations:
        return {"n": 0}
    fits = [o["fit"] for o in observations]
    wrs = [o["wr"] for o in observations]
    alphas = [o["alpha"] for o in observations]
    distinct_values = sorted({round(float(o["value"]), 6) for o in observations})

    def _std(xs: List[float]) -> float:
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    return {
        "n": len(observations),
        "distinct_values": distinct_values,
        "n_distinct": len(distinct_values),
        "fit_mean": round(sum(fits) / len(fits), 6),
        "fit_std": round(_std(fits), 6),
        "fit_range": round(max(fits) - min(fits), 6),
        "wr_std": round(_std(wrs), 6),
        "alpha_std": round(_std(alphas), 6),
    }


def diagnose(tail: int = 10, epsilon: float = ZERO_VARIANCE_EPSILON) -> Dict[str, Any]:
    """Agrega observacoes de `tail` sweeps recentes e gera diagnostico.

    Returns dict machine-readable com:
      - critical_flags: genes sem variance em >=3 valores
      - warning_flags: genes com baixa variance
      - healthy: genes com variance aceitavel
    """
    sweeps = _iter_recent_sweeps(tail)
    if not sweeps:
        return {
            "generated_at": _now_iso(),
            "sweeps_analyzed": 0,
            "critical_flags": [],
            "warning_flags": [],
            "healthy": [],
            "note": "no sweeps found",
        }

    # Agrega observacoes por gene ACROSS todos os sweeps recentes
    all_obs: Dict[str, List[Dict[str, Any]]] = {}
    for sp in sweeps:
        per_sweep = _collect_gene_observations(sp)
        for gene, obs_list in per_sweep.items():
            all_obs.setdefault(gene, []).extend(obs_list)

    critical = []
    warning = []
    healthy = []
    for gene, obs in all_obs.items():
        stats = _compute_variance(obs)
        record = {"gene": gene, **stats}
        if stats["n"] == 0 or stats.get("n_distinct", 0) < CRITICAL_MIN_VALUES_TESTED:
            # Poucos valores distintos — nao podemos concluir, classificar como healthy.
            healthy.append(record)
        elif stats["fit_std"] < epsilon and stats["wr_std"] < epsilon and stats["alpha_std"] < epsilon:
            # Zero variance em TODAS as 3 metricas principais = gene morto
            record["verdict"] = "ZERO_VARIANCE_GENE_MAY_HAVE_NO_EFFECT"
            record["recommendation"] = (
                "Verificar implementacao do gate/lógica deste gene. "
                "Threshold pode estar em magnitude errada (ex: percent em vez de ATR-based)."
            )
            critical.append(record)
        elif stats["fit_std"] < epsilon * 10:
            record["verdict"] = "LOW_VARIANCE_MARGINAL_EFFECT"
            warning.append(record)
        else:
            healthy.append(record)

    return {
        "generated_at": _now_iso(),
        "sweeps_analyzed": len(sweeps),
        "sweep_files": [p.name for p in sweeps],
        "epsilon": epsilon,
        "critical_flags": sorted(critical, key=lambda r: r["gene"]),
        "warning_flags": sorted(warning, key=lambda r: r["gene"]),
        "healthy": sorted(healthy, key=lambda r: r["gene"]),
    }


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _print_human_summary(report: Dict[str, Any]) -> None:
    print(f"=== GENE EFFECT DIAGNOSTIC ===")
    print(f"generated_at: {report['generated_at']}")
    print(f"sweeps_analyzed: {report['sweeps_analyzed']}")
    crit = report.get("critical_flags", [])
    warn = report.get("warning_flags", [])
    heal = report.get("healthy", [])
    print(f"critical={len(crit)}  warning={len(warn)}  healthy={len(heal)}")
    print()
    if crit:
        print("CRITICAL — genes com ZERO variance (provavel bug de implementacao):")
        for r in crit:
            print(f"  [{r['gene']}] n={r['n']} distinct={r['n_distinct']}  "
                  f"fit_mean={r['fit_mean']}  fit_std={r['fit_std']}  fit_range={r['fit_range']}")
            print(f"    valores={r['distinct_values']}")
            print(f"    recomendacao: {r['recommendation']}")
        print()
    if warn:
        print("WARNING — genes com baixa variance (efeito marginal):")
        for r in warn:
            print(f"  [{r['gene']}] fit_std={r['fit_std']}  distinct={r['n_distinct']}")
        print()
    if heal:
        print(f"HEALTHY — {len(heal)} genes com variance adequada (top 5 por fit_std):")
        top = sorted(heal, key=lambda r: -r.get("fit_std", 0.0))[:5]
        for r in top:
            if r.get("n_distinct", 0) >= CRITICAL_MIN_VALUES_TESTED:
                print(f"  [{r['gene']}] fit_std={r['fit_std']}  distinct={r['n_distinct']}")


def _append_to_optimization_log(report: Dict[str, Any]) -> None:
    """Append warning block to optimization_log.md se houver flags criticas."""
    crit = report.get("critical_flags", [])
    if not crit:
        return
    log_path = ROOT / "analysis" / "optimization_log.md"
    if not log_path.exists():
        return
    ts = report["generated_at"]
    block = [f"\n## {ts}  GENE_EFFECT_WARNING\n"]
    block.append(f"Detector automatico: {len(crit)} gene(s) mostraram zero variance em "
                 f"{report['sweeps_analyzed']} sweeps recentes.\n\n")
    for r in crit:
        block.append(f"- **{r['gene']}**: testado em {r['n']} observacoes, "
                     f"{r['n_distinct']} valores distintos {r['distinct_values']}, "
                     f"fit_std={r['fit_std']}. {r['recommendation']}\n")
    block.append("\nRecomenda revisar implementacao. "
                 "Diagnostico em `analysis/gene_effect_report.json`.\n\n---\n")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("".join(block))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tail", type=int, default=10, help="N ultimos sweeps a analisar (default 10)")
    ap.add_argument("--epsilon", type=float, default=ZERO_VARIANCE_EPSILON,
                    help=f"Threshold de stddev para flag (default {ZERO_VARIANCE_EPSILON})")
    ap.add_argument("--output", type=str, default="analysis/gene_effect_report.json",
                    help="Caminho do JSON report")
    ap.add_argument("--no-log-append", action="store_true",
                    help="Nao anexar a optimization_log.md mesmo se houver critical flags")
    ap.add_argument("--exit-on-critical", action="store_true",
                    help="Exit code != 0 se houver critical flags")
    args = ap.parse_args()

    report = diagnose(tail=args.tail, epsilon=args.epsilon)

    # JSON report
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {out_path}")
    print()

    _print_human_summary(report)

    if not args.no_log_append:
        _append_to_optimization_log(report)

    if args.exit_on_critical and report.get("critical_flags"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
