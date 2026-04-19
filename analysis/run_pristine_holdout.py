#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pristine holdout test (Fase A1 completa do plano de overfitting).

Avalia 2 genomas (baseline vs incumbent) em 3 slices temporais distintos:

  1. full:     dataset inteiro (referencia; deve bater com as metricas reportadas)
  2. pre:      tudo exceto os ultimos N dias (o genome "viu" esses dados)
  3. holdout:  somente os ultimos N dias (o genome NUNCA foi validado aqui)

Principio: se o incumbent se sair MUITO melhor que o baseline em `pre` mas
PIOR (ou marginalmente melhor) que o baseline em `holdout`, ele esta overfit
ao historico visto. Quanto maior o gap `delta_pre - delta_holdout`, mais
overfit.

Pre-requisito: precisa do PayloadStore ja construido (output/ga_memmap/).
Execute `GA_RUN_MODE=load GA_FAST_MODE=true python -u ga_run.py` antes.

Uso:
    python analysis/run_pristine_holdout.py                    # N=90 default
    python analysis/run_pristine_holdout.py --holdout-days 60
    python analysis/run_pristine_holdout.py --baseline-ref HEAD~20  # commit ref
    python analysis/run_pristine_holdout.py --baseline-json path.json

Tempo esperado: ~3-5 min (2 genomas x 3 slices x ~60 tickers).

Output: analysis/pristine_holdout_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
sys.path.insert(0, str(ROOT))


def _load_genome_from_file(path: Path) -> tuple[list, str]:
    """Le genome de um JSON no formato global_ga_checkpoint."""
    data = json.loads(path.read_text(encoding="utf-8"))
    genome = data.get("genome") or data.get("best_genome") or []
    source = f"file:{path.name}"
    return list(genome), source


def _load_genome_from_git(commit_ref: str) -> tuple[list, str]:
    """Busca global_ga_checkpoint.json de um commit especifico."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{commit_ref}:global_ga_checkpoint.json"],
            stderr=subprocess.PIPE,
        )
        data = json.loads(out.decode("utf-8"))
        genome = data.get("genome") or data.get("best_genome") or []
        return list(genome), f"git:{commit_ref}"
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        raise SystemExit(f"falha ao carregar genome de {commit_ref}: {e}")


def _format_delta(a: float, b: float, pct: bool = False, ndigits: int = 4) -> str:
    """Formata delta (b - a) com sinal."""
    d = b - a
    if pct:
        return f"{d*100:+.2f}pp"
    return f"{d:+.{ndigits}f}"


def _run_eval(genome: list, cache: dict, gr, label: str) -> dict:
    """Avalia um genome num cache (slice). Retorna metricas."""
    import run_local_ga_staged as rls
    m = rls.evaluate_genome_full(genome, cache, gr)
    print(f"  [{label}] fit={m['fit']:.3f} WR={m['wr_med_all']*100:.2f}% "
          f"alpha={m['mean_alpha_ann']*100:.2f}% MDD={m['mdd_med']*100:.2f}% "
          f"trades={int(m['trades_med'])} n_tickers={m['n_tickers_total']}",
          flush=True)
    return m


def generate_report(baseline: dict, incumbent: dict, cutoff_date,
                    baseline_source: str, incumbent_source: str,
                    holdout_days: int, out_path: Path) -> int:
    """Gera markdown. Retorna numero de red flags detectados."""
    md: list[str] = []
    md.append("# Pristine holdout report\n\n")
    md.append(f"- **Cutoff date** (final do treino, inicio do holdout): {cutoff_date}\n")
    md.append(f"- **Holdout days**: {holdout_days}\n")
    md.append(f"- **Baseline**: `{baseline_source}`\n")
    md.append(f"- **Incumbent**: `{incumbent_source}`\n\n")

    def _metrics_table(title: str, base: dict, incb: dict):
        md.append(f"### {title}\n\n")
        md.append("| Metric | Baseline | Incumbent | Δ (incb-base) |\n")
        md.append("|---|---:|---:|---:|\n")
        for key, pct, name in (
            ("fit", False, "fit"),
            ("wr_med_all", True, "WR"),
            ("wr_target_med_all", True, "WR_target"),
            ("mean_alpha_ann", True, "alpha_ann"),
            ("mdd_med", True, "MDD"),
            ("trades_med", False, "trades"),
            ("n_tickers_active", False, "n_tickers_active"),
        ):
            bv = base.get(key, 0.0)
            iv = incb.get(key, 0.0)
            if pct:
                md.append(f"| {name} | {bv*100:.2f}% | {iv*100:.2f}% | "
                          f"{_format_delta(bv, iv, pct=True)} |\n")
            else:
                md.append(f"| {name} | {bv:.4f} | {iv:.4f} | "
                          f"{_format_delta(bv, iv, pct=False, ndigits=3)} |\n")
        md.append("\n")

    md.append("## 1. Full dataset (sanity check)\n\n")
    _metrics_table("Full", baseline["full"], incumbent["full"])

    md.append("## 2. Pre-holdout (training range, dataset minus last N days)\n\n")
    _metrics_table("Pre-holdout", baseline["pre"], incumbent["pre"])

    md.append(f"## 3. Holdout only (last {holdout_days} days — UNSEEN)\n\n")
    _metrics_table(f"Holdout ({holdout_days}d)", baseline["holdout"], incumbent["holdout"])

    # ============================================================
    # Veredito
    # ============================================================
    md.append("## Veredito\n\n")
    delta_pre = {
        k: incumbent["pre"].get(k, 0.0) - baseline["pre"].get(k, 0.0)
        for k in ("fit", "wr_med_all", "mean_alpha_ann", "mdd_med")
    }
    delta_holdout = {
        k: incumbent["holdout"].get(k, 0.0) - baseline["holdout"].get(k, 0.0)
        for k in ("fit", "wr_med_all", "mean_alpha_ann", "mdd_med")
    }
    md.append("### Ganho do incumbent vs baseline em cada slice\n\n")
    md.append("| Metric | Δ pre-holdout | Δ holdout | Gap (pre - holdout) |\n")
    md.append("|---|---:|---:|---:|\n")
    for key, pct, name in (
        ("fit", False, "fit"),
        ("wr_med_all", True, "WR"),
        ("mean_alpha_ann", True, "alpha_ann"),
        ("mdd_med", True, "MDD"),
    ):
        dp = delta_pre[key]
        dh = delta_holdout[key]
        gap = dp - dh
        if pct:
            md.append(f"| {name} | {dp*100:+.2f}pp | {dh*100:+.2f}pp | "
                      f"{gap*100:+.2f}pp |\n")
        else:
            md.append(f"| {name} | {dp:+.3f} | {dh:+.3f} | {gap:+.3f} |\n")
    md.append("\n")

    # Red flag logic
    red_flags = 0
    notes = []
    # R1: fit degrada fortemente no holdout
    fit_gap = delta_pre["fit"] - delta_holdout["fit"]
    if fit_gap > 1.0:
        red_flags += 1
        notes.append(f"- 🔴 Fit gap `{fit_gap:.3f}` — incumbent teve vantagem "
                      "grande em pre-holdout que NAO sustenta no holdout")
    # R2: WR degrada no holdout mais que pre
    wr_gap_pp = (delta_pre["wr_med_all"] - delta_holdout["wr_med_all"]) * 100
    if wr_gap_pp > 1.0:
        red_flags += 1
        notes.append(f"- 🔴 WR gap `{wr_gap_pp:+.2f}pp` — WR caiu mais no holdout")
    # R3: alpha_ann degrada no holdout
    alpha_holdout = incumbent["holdout"].get("mean_alpha_ann", 0.0) * 100
    baseline_alpha_holdout = baseline["holdout"].get("mean_alpha_ann", 0.0) * 100
    if alpha_holdout < baseline_alpha_holdout - 0.25:
        red_flags += 1
        notes.append(f"- 🔴 Alpha holdout: incumbent {alpha_holdout:.2f}% < "
                      f"baseline {baseline_alpha_holdout:.2f}% "
                      "(incumbent pior no unseen)")
    # R4: MDD piora no holdout
    mdd_gap_pp = (delta_pre["mdd_med"] - delta_holdout["mdd_med"]) * 100
    if mdd_gap_pp < -1.0:
        # Lembrete: delta_holdout MDD < 0 e BOM (menos risco). Se gap < -1,
        # significa que incumbent foi MUITO melhor em pre e MUITO pior em holdout.
        red_flags += 1
        notes.append(f"- 🔴 MDD gap `{mdd_gap_pp:+.2f}pp` — risco pior no holdout")

    for n in notes:
        md.append(n + "\n")
    md.append("\n")

    # Veredito
    if red_flags == 0:
        md.append("**VEREDITO**: sem evidencia de overfit no holdout. O incumbent "
                  "generaliza bem para os ultimos {} dias nao vistos.\n".format(holdout_days))
    elif red_flags <= 1:
        md.append("**VEREDITO**: OVERFIT MARGINAL. Uma metrica degrada no holdout. "
                  "Monitorar; considerar ativar Fase B1 (holdout no acceptance gate).\n")
    elif red_flags <= 2:
        md.append("**VEREDITO**: OVERFIT DETECTADO. Multiplas metricas degradam no "
                  "holdout. Recomenda-se: ativar Fase B1, considerar reverter para "
                  "um commit pre-tuning.\n")
    else:
        md.append("**VEREDITO**: OVERFIT SEVERO. O incumbent teve ganhos em pre-"
                  "holdout que nao sustenta no unseen data. ACAO: reverter o "
                  "checkpoint para o baseline e reconstruir com Fase B ativa.\n")

    out_path.write_text("".join(md), encoding="utf-8")
    print(f"\n[PRISTINE_HOLDOUT] wrote {out_path}")
    print(f"[PRISTINE_HOLDOUT] red flags: {red_flags}")
    return red_flags


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--holdout-days", type=int, default=90,
                        help="tamanho do holdout em dias (default 90)")
    parser.add_argument("--baseline-ref", type=str, default=None,
                        help="git ref de onde carregar o genome baseline "
                             "(ex: '1bd4217^', 'HEAD~15'). Se omitido, usa "
                             "global_ga_checkpoint_pre_alpha_accept.json.bak se existir.")
    parser.add_argument("--baseline-json", type=Path, default=None,
                        help="path para JSON com genome baseline (alternativa a --baseline-ref)")
    parser.add_argument("--incumbent-json", type=Path,
                        default=ROOT / "global_ga_checkpoint.json",
                        help="path para JSON com incumbent (default = main checkpoint)")
    parser.add_argument("--out", type=Path,
                        default=ANALYSIS / "pristine_holdout_report.md",
                        help="path do relatorio")
    args = parser.parse_args()

    # Load genomes
    print("[PRISTINE_HOLDOUT] Loading genomes...", flush=True)
    inc_genome, inc_source = _load_genome_from_file(args.incumbent_json)
    print(f"  Incumbent: {inc_source} ({len(inc_genome)} genes)", flush=True)

    if args.baseline_json:
        base_genome, base_source = _load_genome_from_file(args.baseline_json)
    elif args.baseline_ref:
        base_genome, base_source = _load_genome_from_git(args.baseline_ref)
    else:
        # Default: procura um .bak antigo
        bak = ROOT / "global_ga_checkpoint_pre_alpha_accept.json.bak"
        if bak.exists():
            base_genome, base_source = _load_genome_from_file(bak)
        else:
            # Fallback: usa git ref = HEAD~30 (30 commits atras)
            base_genome, base_source = _load_genome_from_git("HEAD~30")
    print(f"  Baseline:  {base_source} ({len(base_genome)} genes)", flush=True)

    if base_genome == inc_genome:
        print("[PRISTINE_HOLDOUT] WARNING: baseline == incumbent. "
              "Resultados serao identicos. Especifique --baseline-ref.")

    # Setup PayloadStore (prefer primary ga_memmap; fallback to ga_memmap_staged)
    print("\n[PRISTINE_HOLDOUT] Loading PayloadStore...", flush=True)
    import ga_run as gr
    import run_local_ga_staged as rls
    import pandas as pd
    from payload_store import PayloadStore

    candidates = [
        os.path.abspath(gr.GA_MEMMAP_DIR),
        os.path.abspath("output/ga_memmap_staged"),
    ]
    store_dir = None
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "meta.json")):
            store_dir = cand
            break
    if store_dir is None:
        raise SystemExit(
            f"PayloadStore nao encontrado em {candidates}. Rode antes:\n"
            "  GA_RUN_MODE=load GA_FAST_MODE=true python -u ga_run.py"
        )
    print(f"  Using store: {store_dir}", flush=True)

    store = PayloadStore(store_dir, mode="r")
    print(f"  Store: {len(store.tickers)} tickers, {store.min_date.date()} -> "
          f"{store.max_date.date()}", flush=True)

    # Build cache + slices
    cache_full = rls.build_ticker_arrays_cache(store)
    cutoff_date = store.max_date - pd.Timedelta(days=args.holdout_days)
    cache_pre = rls.slice_ticker_arrays_cache(cache_full, end_date=cutoff_date)
    cache_holdout = rls.slice_ticker_arrays_cache(cache_full, start_date=cutoff_date)
    print(f"  cache_full: {len(cache_full)} tickers", flush=True)
    print(f"  cache_pre (<= {cutoff_date.date()}): {len(cache_pre)} tickers", flush=True)
    print(f"  cache_holdout (>= {cutoff_date.date()}): {len(cache_holdout)} tickers", flush=True)

    # Evaluate 2 genomes x 3 slices = 6 evaluations
    print("\n[PRISTINE_HOLDOUT] Evaluating baseline...", flush=True)
    base_results = {
        "full":    _run_eval(base_genome, cache_full, gr, "base/full"),
        "pre":     _run_eval(base_genome, cache_pre, gr, "base/pre"),
        "holdout": _run_eval(base_genome, cache_holdout, gr, "base/holdout"),
    }
    print("\n[PRISTINE_HOLDOUT] Evaluating incumbent...", flush=True)
    inc_results = {
        "full":    _run_eval(inc_genome, cache_full, gr, "incb/full"),
        "pre":     _run_eval(inc_genome, cache_pre, gr, "incb/pre"),
        "holdout": _run_eval(inc_genome, cache_holdout, gr, "incb/holdout"),
    }

    # Generate report
    print("\n[PRISTINE_HOLDOUT] Generating report...")
    generate_report(base_results, inc_results, cutoff_date.date(),
                    base_source, inc_source, args.holdout_days, args.out)

    store.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
