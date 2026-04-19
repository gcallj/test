#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostico retrospectivo de overfitting (Fase A do plano).

Analisa dados JA DISPONIVEIS no repo (nao requer re-rodar o GA) para responder:

  1. As promocoes foram estatisticamente significativas?
     -> Wilson score interval em WR dado `trades_med`
  2. Multiple testing: em N attempts, o best-of-N bate baseline por acaso?
     -> Bonferroni-corrected pass rate
  3. Trajetoria de alpha: esta piorando com o tempo?
     -> serie temporal de `mean_alpha_ann` por ciclo
  4. Gene drift: algum gene foi movido >=3 vezes na mesma direcao?
     -> micro-drift detection via optimization_log.md
  5. Qual o gain real por attempt? (se cada attempt adiciona < ruido, overfit.)
     -> delta fit / attempt vs sigma estimado

Fontes de dados lidas:
  - analysis/optimization_log.md (historico de promocoes)
  - local_fullmetric_sweep_*.json (metricas de cada sweep)
  - analysis/codex_attempts_history.json (attempts do codex)
  - global_ga_checkpoint.json (estado atual)

Output: analysis/overfitting_diagnostic_report.md

Esse script NAO substitui o pristine holdout (Fase A1 completa requer rerun do
GA com cutoff temporal). Mas responde quantitativamente se o sistema esta
overfit baseado nos DADOS JA OBSERVADOS, o que ja e 70% da informacao.

Para a Fase A1 completa (rerun com holdout temporal), veja instrucoes no fim
deste arquivo.

Uso:
    python analysis/overfitting_diagnostic.py
    python analysis/overfitting_diagnostic.py --out custom_report.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
DEFAULT_REPORT = ANALYSIS / "overfitting_diagnostic_report.md"
LOG_MD = ANALYSIS / "optimization_log.md"
CODEX_JSON = ANALYSIS / "codex_attempts_history.json"
CHECKPOINT = ROOT / "global_ga_checkpoint.json"


# --------------------------- math helpers ---------------------------

def wilson_score_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% IC para proporcao. Mais correto que normal approx
    quando p esta perto de 0 ou 1, ou n e pequeno."""
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    """Threshold corrigido para multiple testing."""
    if n_tests <= 0:
        return alpha
    return alpha / n_tests


# --------------------------- data loaders ---------------------------

@dataclass
class PromotionEntry:
    timestamp: str
    category: str
    cycle_num: int | None
    promoted: bool
    genes: list[str]
    moves: list[dict[str, Any]]
    source: str  # "claude" or "codex"


def _parse_optimization_log(text: str) -> list[PromotionEntry]:
    """Parseia analysis/optimization_log.md. Cada entry comeca com `## <iso>`."""
    entries: list[PromotionEntry] = []
    blocks = re.split(r"(?m)^## (\d{4}-\d{2}-\d{2}T[\d:+-]+)", text)
    # blocks = [preamble, ts1, body1, ts2, body2, ...]
    for i in range(1, len(blocks) - 1, 2):
        ts = blocks[i].strip()
        body = blocks[i + 1]
        cat_match = re.search(r"^\s*-\s*([\w_]+)\s*$", body, re.MULTILINE)
        # Mais robusto: primeira linha apos o ts costuma ter " - <category>"
        cat_match2 = re.match(r"\s*-\s*([\w_]+)", body)
        category = cat_match2.group(1) if cat_match2 else "unknown"
        cycle_match = re.search(r"\*\*Cycle\*\*:\s*#(\d+)", body)
        cycle_num = int(cycle_match.group(1)) if cycle_match else None
        promoted = bool(re.search(r"\*\*Promoted\*\*:\s*YES", body))
        genes_match = re.search(r"\*\*Genes swept\*\*:\s*`([^`]+)`", body)
        genes = [g.strip() for g in genes_match.group(1).split(",")] if genes_match else []
        moves: list[dict[str, Any]] = []
        for mv_line in re.findall(r"^-\s*`([\w_]+)`:\s*`([^`]+)`\s*->\s*`([^`]+)`", body,
                                   re.MULTILINE):
            moves.append({"gene": mv_line[0], "from": mv_line[1], "to": mv_line[2]})
        entries.append(PromotionEntry(timestamp=ts, category=category, cycle_num=cycle_num,
                                       promoted=promoted, genes=genes, moves=moves,
                                       source="claude"))
    return entries


def _parse_category_line(body: str) -> str:
    m = re.search(r"^\s*(?:##\s*[^\n]*)?\n?\s*([A-F]_[\w_]+|[A-F])", body, re.MULTILINE)
    return m.group(1) if m else "unknown"


def load_claude_history() -> list[PromotionEntry]:
    if not LOG_MD.exists():
        return []
    txt = LOG_MD.read_text(encoding="utf-8")
    # Melhor parser: capturar bloco entre "## <iso> - <cat>" e "---"
    entries: list[PromotionEntry] = []
    pattern = re.compile(
        r"^## (?P<ts>\d{4}-\d{2}-\d{2}T[\d:+-]+)\s*-\s*(?P<cat>[\w_]+)\s*\n(?P<body>.*?)(?=^---$|^## |\Z)",
        re.MULTILINE | re.DOTALL)
    for m in pattern.finditer(txt):
        body = m.group("body")
        promoted = bool(re.search(r"\*\*Promoted\*\*:\s*YES", body))
        cycle_match = re.search(r"\*\*Cycle\*\*:\s*#(\d+)", body)
        genes_match = re.search(r"\*\*Genes swept\*\*:\s*`([^`]+)`", body)
        genes = [g.strip() for g in genes_match.group(1).split(",")] if genes_match else []
        moves = []
        for mv in re.findall(r"^-\s*`([\w_]+)`:\s*`([^`]+)`\s*->\s*`([^`]+)`", body,
                              re.MULTILINE):
            moves.append({"gene": mv[0], "from": mv[1], "to": mv[2]})
        entries.append(PromotionEntry(
            timestamp=m.group("ts"),
            category=m.group("cat"),
            cycle_num=int(cycle_match.group(1)) if cycle_match else None,
            promoted=promoted, genes=genes, moves=moves, source="claude"))
    return entries


def load_codex_history() -> list[PromotionEntry]:
    if not CODEX_JSON.exists():
        return []
    data = json.loads(CODEX_JSON.read_text(encoding="utf-8"))
    attempts = data.get("attempts") or []
    entries: list[PromotionEntry] = []
    for att in attempts:
        moves = att.get("moves") or []
        entries.append(PromotionEntry(
            timestamp=att.get("timestamp") or "",
            category=att.get("section_title", "codex")[:40],
            cycle_num=None,
            promoted=bool(att.get("promoted", False)),
            genes=list(att.get("genes") or []),
            moves=[{"gene": m.get("gene"), "from": m.get("from"), "to": m.get("to")}
                   for m in moves],
            source="codex"))
    return entries


def load_checkpoint_metrics() -> dict[str, float]:
    if not CHECKPOINT.exists():
        return {}
    ck = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    snap = ck.get("metrics_snapshot") or {}
    cand = snap.get("candidate") or snap.get("incumbent") or {}
    out = {}
    for k in ("fit", "wr_med_all", "wr_target_med_all", "mean_alpha_ann",
              "mdd_med", "trades_med"):
        v = cand.get(k)
        if v is not None:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def load_all_sweep_metrics() -> list[dict[str, Any]]:
    """Cada sweep tem base_metrics (main), seed_metrics (incumbent),
    best_promoted (candidato vencedor). Retorna lista cronologica."""
    out = []
    for path in sorted(glob.glob(str(ROOT / "local_fullmetric_sweep_*.json"))):
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        base = d.get("base_metrics") or {}
        seed = d.get("seed_metrics") or {}
        bp = d.get("best_promoted") or {}
        out.append({
            "file": Path(path).name,
            "started_at": d.get("started_at"),
            "base_alpha": base.get("mean_alpha_ann"),
            "seed_alpha": seed.get("mean_alpha_ann"),
            "cand_alpha": (bp.get("metrics") or {}).get("mean_alpha_ann"),
            "base_fit": base.get("fit"),
            "seed_fit": seed.get("fit"),
            "cand_fit": (bp.get("metrics") or {}).get("fit"),
            "base_wr": base.get("wr_med_all"),
            "cand_wr": (bp.get("metrics") or {}).get("wr_med_all"),
            "base_trades": base.get("trades_med"),
            "cand_trades": (bp.get("metrics") or {}).get("trades_med"),
            "n_candidates": d.get("single_count", 0) + d.get("combo_count", 0),
            "best_label": (bp.get("label") if bp else None),
        })
    return out


# --------------------------- analyses ---------------------------

def analyze_wilson_ci(metrics: dict[str, float]) -> str:
    wr = metrics.get("wr_med_all")
    n = metrics.get("trades_med")
    if wr is None or n is None or n <= 0:
        return "- Wilson CI: dados insuficientes (wr/trades missing)\n"
    lo, hi = wilson_score_interval(wr, int(n))
    baseline = 0.50
    significant = lo > baseline
    verdict = ("significativamente melhor que 50%" if significant
               else f"NAO significativamente melhor que 50% (IC cobre baseline)")
    return (
        f"- **WR observado**: {wr:.4f} ({wr*100:.2f}%), com n={int(n)} trades\n"
        f"- **Wilson 95% IC**: [{lo:.4f}, {hi:.4f}] = [{lo*100:.2f}%, {hi*100:.2f}%]\n"
        f"- **Veredito**: {verdict}\n"
        f"- **Interpretacao**: com n={int(n)} trades, o IC tem largura {(hi-lo)*100:.2f}pp — "
        f"{'amostra pequena, alta variancia' if (hi-lo) > 0.15 else 'amostra razoavel'}\n"
    )


def analyze_multiple_testing(promotions: list[PromotionEntry]) -> str:
    n_attempts = len(promotions)
    n_promoted = sum(1 for p in promotions if p.promoted)
    if n_attempts == 0:
        return "- Sem historico de promocoes\n"
    bonf = bonferroni_threshold(n_attempts, 0.05)
    rate = n_promoted / n_attempts
    # Sob H0 (zero skill), cada attempt bate baseline com p~0.5 se o gate fosse so sorte,
    # mas com gate acceptance_vs_main o p real e menor (~0.1-0.2). Usamos 0.10 como null rate.
    null_rate = 0.10
    # binomial test via normal approximation
    mu = n_attempts * null_rate
    sd = math.sqrt(n_attempts * null_rate * (1 - null_rate))
    z_obs = (n_promoted - mu) / sd if sd > 0 else 0.0
    p_obs = 0.5 * (1 - math.erf(abs(z_obs) / math.sqrt(2))) if z_obs > 0 else 1.0
    return (
        f"- **N attempts totais**: {n_attempts}\n"
        f"- **N promovidos**: {n_promoted} (taxa {rate*100:.1f}%)\n"
        f"- **Bonferroni alpha={0.05:.2f}/N**: p < {bonf:.4g} exigido por attempt individual\n"
        f"- **Binomial vs null rate 10%**: z={z_obs:.2f}, one-sided p={p_obs:.4g}\n"
        f"- **Interpretacao**: {'Taxa ALTA — forte evidencia contra null (mas pode ser skill OU overfit a mesma janela)' if p_obs < 0.01 else 'Taxa nao excepcional vs null'}\n"
    )


def analyze_alpha_trajectory(sweeps: list[dict]) -> str:
    rows: list[tuple[str, float, float, float]] = []
    for s in sweeps:
        a = s.get("cand_alpha") or s.get("seed_alpha") or s.get("base_alpha")
        f = s.get("cand_fit") or s.get("seed_fit") or s.get("base_fit")
        w = s.get("base_wr") or 0.0
        if a is not None and f is not None:
            rows.append((s.get("started_at") or s["file"], float(a), float(f), float(w)))
    if not rows:
        return "- Sem trajetoria de sweeps\n"
    first_alpha = rows[0][1]
    last_alpha = rows[-1][1]
    delta_alpha_pp = (last_alpha - first_alpha) * 100
    direction = "piorou" if delta_alpha_pp < -0.01 else ("melhorou" if delta_alpha_pp > 0.01 else "estavel")
    lines = [
        f"- **Alpha inicial** (primeiro sweep): {first_alpha*100:.2f}%\n",
        f"- **Alpha atual** (ultimo sweep): {last_alpha*100:.2f}%\n",
        f"- **Delta total**: {delta_alpha_pp:+.2f}pp ({direction})\n",
    ]
    if last_alpha < 0:
        lines.append(f"- **RED FLAG**: alpha atual ({last_alpha*100:.2f}%) e NEGATIVO — "
                     "sistema subperforma buy-and-hold em termos anualizados\n")
    if delta_alpha_pp < -0.5:
        lines.append(f"- **RED FLAG**: alpha degradou {delta_alpha_pp:.2f}pp ao longo dos sweeps — "
                     "sinal de curve-fitting progressivo\n")
    # Sample milestones
    sample_idx = [0, len(rows)//3, 2*len(rows)//3, len(rows)-1]
    lines.append("\n**Amostragem cronologica**:\n\n")
    lines.append("| idx | timestamp | fit | alpha_ann |\n|---:|---|---:|---:|\n")
    for i in sample_idx:
        if 0 <= i < len(rows):
            ts, a, f, _ = rows[i]
            lines.append(f"| {i} | {ts[:19]} | {f:.2f} | {a*100:.2f}% |\n")
    return "".join(lines)


def analyze_gene_drift(promotions: list[PromotionEntry]) -> str:
    """Detecta genes que foram movidos >=3 vezes na mesma direcao."""
    gene_moves: dict[str, list[tuple[str, float, float]]] = {}  # gene -> [(ts, from, to)]
    for p in promotions:
        if not p.promoted:
            continue
        for mv in p.moves:
            try:
                fv = float(mv["from"])
                tv = float(mv["to"])
                gene_moves.setdefault(mv["gene"], []).append((p.timestamp, fv, tv))
            except (TypeError, ValueError, KeyError):
                continue
    if not gene_moves:
        return "- Sem moves numericos detectados nos logs\n"
    lines = ["\n**Genes movidos multiplas vezes (red flag se >=3 na mesma direcao)**:\n\n"]
    lines.append("| gene | #promos | movimentos | direcao |\n|---|---:|---|---|\n")
    suspect_count = 0
    for gene, moves in sorted(gene_moves.items(), key=lambda x: -len(x[1])):
        if len(moves) < 2:
            continue
        deltas = [(tv - fv) for _, fv, tv in moves]
        all_up = all(d > 0 for d in deltas)
        all_down = all(d < 0 for d in deltas)
        direction = "ALL UP" if all_up else ("ALL DOWN" if all_down else "mixed")
        suspect = len(moves) >= 3 and (all_up or all_down)
        if suspect:
            suspect_count += 1
        flag = " 🚩" if suspect else ""
        moves_str = " → ".join(f"{fv:.3f}→{tv:.3f}" for _, fv, tv in moves[-3:])
        lines.append(f"| `{gene}` | {len(moves)} | {moves_str} | {direction}{flag} |\n")
    if suspect_count > 0:
        lines.append(f"\n**{suspect_count} gene(s) com drift suspeito** — veja Fase B4 do plano.\n")
    return "".join(lines)


def analyze_gain_per_attempt(sweeps: list[dict]) -> str:
    """Calcula ganho medio de fit por sweep bem-sucedido; compara com baseline variance."""
    fits = [s.get("cand_fit") or s.get("seed_fit") for s in sweeps
            if (s.get("cand_fit") or s.get("seed_fit"))]
    if len(fits) < 3:
        return "- Sweeps insuficientes para analise de ganho/attempt\n"
    deltas = [fits[i+1] - fits[i] for i in range(len(fits)-1)]
    mean_delta = sum(deltas) / len(deltas)
    variance = sum((d - mean_delta)**2 for d in deltas) / max(1, len(deltas)-1)
    sd = math.sqrt(variance)
    snr = mean_delta / sd if sd > 0 else 0.0
    return (
        f"- **Ganho medio de fit por sweep**: {mean_delta:+.4f}\n"
        f"- **Desvio padrao dos deltas**: {sd:.4f}\n"
        f"- **Signal-to-noise ratio**: {snr:.2f}\n"
        f"- **Interpretacao**: {'SNR > 1 sugere que ganhos sao maiores que ruido' if snr > 1 else 'SNR < 1 sugere que os ganhos sao indistinguiveis de ruido — overfit provavel'}\n"
    )


# --------------------------- main ---------------------------

def generate_report(out_path: Path) -> None:
    claude_hist = load_claude_history()
    codex_hist = load_codex_history()
    promotions = claude_hist + codex_hist
    sweeps = load_all_sweep_metrics()
    ckpt = load_checkpoint_metrics()

    md: list[str] = []
    md.append("# Overfitting diagnostic report\n\n")
    md.append(f"Generated: {Path(__file__).name} sobre "
              f"{len(promotions)} attempts ({sum(1 for p in promotions if p.promoted)} promoted) + "
              f"{len(sweeps)} sweeps\n\n")

    md.append("## Executive summary\n\n")
    n_prom = sum(1 for p in promotions if p.promoted)
    rate = n_prom / len(promotions) if promotions else 0.0
    alpha_curr = ckpt.get("mean_alpha_ann", 0.0) * 100
    wr_curr = ckpt.get("wr_med_all", 0.0) * 100
    trades_curr = ckpt.get("trades_med", 0.0)
    md.append(f"- Incumbent atual: fit={ckpt.get('fit', 0):.2f}, WR={wr_curr:.1f}%, "
              f"alpha={alpha_curr:.2f}%, trades={int(trades_curr)}\n")
    md.append(f"- Historico: {len(promotions)} attempts, {n_prom} promoted ({rate*100:.1f}%)\n\n")

    md.append("## 1. Wilson 95% CI para WR\n\n")
    md.append(analyze_wilson_ci(ckpt))
    md.append("\n")

    md.append("## 2. Multiple testing correction\n\n")
    md.append(analyze_multiple_testing(promotions))
    md.append("\n")

    md.append("## 3. Trajetoria do alpha ao longo dos sweeps\n\n")
    md.append(analyze_alpha_trajectory(sweeps))
    md.append("\n")

    md.append("## 4. Gene drift detection (retrospective B4 preview)\n\n")
    md.append(analyze_gene_drift(promotions))
    md.append("\n")

    md.append("## 5. Signal-to-noise do ganho por attempt\n\n")
    md.append(analyze_gain_per_attempt(sweeps))
    md.append("\n")

    md.append("## Vereditos e recomendacoes\n\n")
    # Heuristic combinado
    red_flags = 0
    notes = []
    if ckpt.get("mean_alpha_ann", 0) < 0:
        red_flags += 1
        notes.append("- 🔴 Alpha negativo atual (subperforma buy-and-hold)")
    if ckpt.get("trades_med", 0) < 40:
        red_flags += 1
        notes.append("- 🔴 Poucos trades por ticker (< 40, amostra fraca)")
    if rate > 0.20:
        red_flags += 1
        notes.append(f"- 🟡 Taxa de promocao alta ({rate*100:.0f}%) sem correcao multiple testing")
    # drift gene check
    gene_moves: dict[str, list] = {}
    for p in promotions:
        if p.promoted:
            for mv in p.moves:
                gene_moves.setdefault(mv.get("gene", ""), []).append(mv)
    if any(len(v) >= 3 for v in gene_moves.values()):
        red_flags += 1
        notes.append("- 🟡 Genes com drift (>=3 promos consecutivas)")
    md.extend(n + "\n" for n in notes)
    md.append("\n")
    if red_flags >= 3:
        md.append("**VEREDITO**: OVERFIT PROVAVEL. Recomenda-se:\n"
                  "1. Rodar Fase A1 completa (pristine holdout temporal — veja `docs/run_pristine_holdout.md`)\n"
                  "2. Ativar `GA_ALPHA_FLOOR_TOLERANCE_PP=0.0` (ja default apos B2)\n"
                  "3. Considerar reverter incumbent para um commit pre-tuning recente\n"
                  "4. Implementar Fase B3 (bootstrap CI) no apply\n")
    elif red_flags >= 1:
        md.append("**VEREDITO**: OVERFIT MARGINAL. Monitorar com Fase C (signal log + realized).\n")
    else:
        md.append("**VEREDITO**: sem red flags estatisticos obvios. Ainda assim, rodar Fase A1 completa para confirmar.\n")

    md.append("\n---\n\n")
    md.append("## Como rodar Fase A1 completa (pristine holdout temporal)\n\n")
    md.append("Esse script NAO reavalia o GA em uma janela temporal cortada. "
              "Para fazer isso corretamente:\n\n")
    md.append("```bash\n")
    md.append("# 1. Adicione ao ga_run.py: parametro GA_EVAL_CUTOFF_DATE no build_temporal_windows\n")
    md.append("# 2. Rode load mode com cutoff para baseline e incumbent:\n")
    md.append("GA_RUN_MODE=load GA_EVAL_CUTOFF_DATE=2026-01-31 \\\n")
    md.append("  python -c 'from ga_run import run; run()'\n")
    md.append("# 3. Compare fit/WR/alpha/MDD: se incumbent piora mais que baseline no holdout,\n")
    md.append("#    esta overfit.\n")
    md.append("```\n\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(md), encoding="utf-8")
    print(f"[OVERFITTING_DIAGNOSTIC] wrote {out_path}")
    print(f"[OVERFITTING_DIAGNOSTIC] red flags detected: {red_flags}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT,
                        help="output markdown path")
    args = parser.parse_args()
    generate_report(args.out)


if __name__ == "__main__":
    main()
