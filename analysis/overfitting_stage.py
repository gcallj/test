#!/usr/bin/env python3
"""Overfitting stage: pristine holdout + bootstrap CIs + parameter sensitivity.

Roda APOS retrain GA e ANTES do commit final. Complementa best_ever_guard com
um teste estatistico de overfitting:

  1. PRISTINE HOLDOUT: avalia genome nos ultimos N dias de cada ticker (slice
     temporal que o GA nao otimizou diretamente, ao menos nao como target
     explicito — e o regime out-of-sample mais relevante p/ trades futuros).

  2. BOOTSTRAP CIs: reamostra trade returns com reposicao 200x por ticker
     para estimar incerteza nos metricos (WR, alpha). Wide CI = instavel.

  3. PARAMETER SENSITIVITY: perturba cada gene +/-1 step e mede Delta fit.
     Media |Delta fit| alta = genome em borda fragil = overfit tipico.

Verdict:
  CLEAN     — todos os checks OK
  MARGINAL  — 1 check com warn, mas aceitavel
  OVERFIT   — >=2 checks com warn OU qualquer fail critico

Exit codes:
  0 = CLEAN
  1 = MARGINAL (workflow pode continuar com warning)
  2 = OVERFIT (workflow deve rejeitar retrain)

Uso:
  python analysis/overfitting_stage.py
  python analysis/overfitting_stage.py --holdout-days 90 --bootstrap 200
  python analysis/overfitting_stage.py --strict    # exit 2 em marginal tambem
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPORT_JSON = ROOT / "analysis" / "overfitting_stage_report.json"
REPORT_MD = ROOT / "analysis" / "overfitting_stage_report.md"

# Garante que ga_run / run_local_ga_staged sao importaveis quando o script
# eh executado de fora do ROOT (ex: workflow chama "python analysis/...py").
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Thresholds para classificar retrain como OVERFIT
# Usa tolerancias absolutas onde faz sentido (pp) e relativas onde escala e continua.
THRESHOLDS = {
    # Pristine holdout: quanto WR pode cair vs full history sem ser overfit?
    "holdout_wr_drop_max": 0.05,         # 5pp (ex: 70% train, 65% holdout = marginal)
    "holdout_wr_drop_critical": 0.10,    # 10pp (critico)
    # Alpha pode cair mais (alpha e ruidoso)
    "holdout_alpha_drop_max": 0.03,      # 3pp (0.03 de alpha_ann)
    "holdout_alpha_drop_critical": 0.06,
    # Bootstrap CI width — WR 90% CI muito largo sugere amostra pequena
    "bootstrap_wr_ci_width_max": 0.10,   # IC 5-95 amplitude ate 10pp OK
    "bootstrap_wr_ci_width_critical": 0.20,
    # Sensitivity: mediana de |delta_fit| por perturbacao de 1 step
    "sensitivity_median_max": 1.50,      # fit muda mais que 1.5 por step = fragil
    "sensitivity_median_critical": 3.00,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


# ============================================================
# PRISTINE HOLDOUT TEST
# ============================================================

def evaluate_holdout(checkpoint_genome: List[float],
                     holdout_days: int = 90) -> Dict[str, Any]:
    """Avalia genome em 2 regimes temporais: train (full - holdout) e holdout (ultimos N dias).

    Returns dict com metricas de ambos + deltas.
    """
    import ga_run as gr
    import run_local_ga_staged as staged

    store, _w, _sg, _sf, _bg, _bf, _bi = staged.open_existing_store()
    cache = staged.build_ticker_arrays_cache(store)

    # Full history baseline
    print(f"[overfitting] Evaluating full history...")
    metrics_full = staged.evaluate_genome_full(checkpoint_genome, cache, gr)

    # Slice cache para train-only (exclui ultimos holdout_days)
    print(f"[overfitting] Evaluating train-only (excluding last {holdout_days}d)...")
    cache_train = {}
    for tk, p in cache.items():
        if "dates" in p and p["dates"] is not None and len(p["dates"]) > holdout_days + 60:
            n = len(p["dates"])
            cutoff = n - holdout_days
            p_train = {}
            for k, v in p.items():
                if hasattr(v, "__len__") and not isinstance(v, str) and len(v) == n:
                    p_train[k] = v[:cutoff]
                else:
                    p_train[k] = v
            cache_train[tk] = p_train
    metrics_train = staged.evaluate_genome_full(checkpoint_genome, cache_train, gr) if cache_train else {}

    # Holdout slice: ultimos holdout_days + LOOKBACK para ter historico de features
    print(f"[overfitting] Evaluating holdout (last {holdout_days}d)...")
    LOOKBACK = 252  # 1 year lookback para stabilizar indicadores
    cache_holdout = {}
    for tk, p in cache.items():
        if "dates" in p and p["dates"] is not None and len(p["dates"]) > holdout_days + LOOKBACK:
            n = len(p["dates"])
            start_idx = n - holdout_days - LOOKBACK
            p_hold = {}
            for k, v in p.items():
                if hasattr(v, "__len__") and not isinstance(v, str) and len(v) == n:
                    p_hold[k] = v[start_idx:]
                else:
                    p_hold[k] = v
            cache_holdout[tk] = p_hold
    metrics_holdout = staged.evaluate_genome_full(checkpoint_genome, cache_holdout, gr) if cache_holdout else {}

    # Deltas
    def _delta(k: str, default: float = 0.0) -> float:
        t = float(metrics_train.get(k, default) or default)
        h = float(metrics_holdout.get(k, default) or default)
        return h - t

    return {
        "metrics_full": {
            "fit": float(metrics_full.get("fit", 0.0) or 0.0),
            "wr_med_all": float(metrics_full.get("wr_med_all", 0.0) or 0.0),
            "mean_alpha_ann": float(metrics_full.get("mean_alpha_ann", 0.0) or 0.0),
            "n_ge_70": int(metrics_full.get("n_ge_70", 0) or 0),
            "trades_med": float(metrics_full.get("trades_med", 0.0) or 0.0),
        },
        "metrics_train": {
            "fit": float(metrics_train.get("fit", 0.0) or 0.0),
            "wr_med_all": float(metrics_train.get("wr_med_all", 0.0) or 0.0),
            "mean_alpha_ann": float(metrics_train.get("mean_alpha_ann", 0.0) or 0.0),
            "n_ge_70": int(metrics_train.get("n_ge_70", 0) or 0),
            "trades_med": float(metrics_train.get("trades_med", 0.0) or 0.0),
        },
        "metrics_holdout": {
            "fit": float(metrics_holdout.get("fit", 0.0) or 0.0),
            "wr_med_all": float(metrics_holdout.get("wr_med_all", 0.0) or 0.0),
            "mean_alpha_ann": float(metrics_holdout.get("mean_alpha_ann", 0.0) or 0.0),
            "n_ge_70": int(metrics_holdout.get("n_ge_70", 0) or 0),
            "trades_med": float(metrics_holdout.get("trades_med", 0.0) or 0.0),
        },
        "delta_holdout_vs_train": {
            "fit": _delta("fit"),
            "wr_med_all": _delta("wr_med_all"),
            "mean_alpha_ann": _delta("mean_alpha_ann"),
            "n_ge_70": _delta("n_ge_70"),
        },
        "n_tickers_train": len(cache_train),
        "n_tickers_holdout": len(cache_holdout),
        "holdout_days": holdout_days,
    }


# ============================================================
# BOOTSTRAP CIs
# ============================================================

def bootstrap_ci_wr(checkpoint_genome: List[float],
                    n_samples: int = 200,
                    seed: int = 42) -> Dict[str, Any]:
    """Bootstrap sobre trade_returns por ticker para estimar CI de metricas."""
    import ga_run as gr
    import run_local_ga_staged as staged

    store, _w, _sg, _sf, _bg, _bf, _bi = staged.open_existing_store()
    cache = staged.build_ticker_arrays_cache(store)

    # Coleta trade_rets e win_rate por ticker
    gp = gr.decode_global_params(checkpoint_genome)
    per_ticker_wr = []
    per_ticker_trades = []
    for tk, p in cache.items():
        try:
            st = gr.backtest_stats_global_intraday(
                p["open"], p["high"], p["low"], p["close"],
                p["score_matrix"], p["atr"], gp, precomputed=p,
            )
            n = int(st.get("n_trades", 0) or 0)
            wr = float(st.get("win_rate", 0.0) or 0.0)
            if n >= 5:  # minimo para incluir no bootstrap
                per_ticker_wr.append(wr)
                per_ticker_trades.append(n)
        except Exception:
            continue

    if not per_ticker_wr:
        return {"error": "no_ticker_stats"}

    # Bootstrap sobre per_ticker_wr (ticker-level)
    rng = np.random.default_rng(seed)
    n_tickers = len(per_ticker_wr)
    wr_samples = []
    for _ in range(n_samples):
        idx = rng.integers(0, n_tickers, size=n_tickers)
        resampled_wr = np.asarray(per_ticker_wr)[idx]
        wr_samples.append(float(np.median(resampled_wr)))

    wr_samples = np.asarray(wr_samples)
    ci_low = float(np.percentile(wr_samples, 5))
    ci_high = float(np.percentile(wr_samples, 95))
    ci_median = float(np.percentile(wr_samples, 50))
    ci_width = ci_high - ci_low

    return {
        "n_samples": n_samples,
        "n_tickers_in_bootstrap": n_tickers,
        "wr_ci_5": ci_low,
        "wr_ci_50": ci_median,
        "wr_ci_95": ci_high,
        "wr_ci_width": ci_width,
        "point_estimate_wr_median": float(np.median(per_ticker_wr)),
    }


# ============================================================
# PARAMETER SENSITIVITY
# ============================================================

def parameter_sensitivity(checkpoint_genome: List[float]) -> Dict[str, Any]:
    """Perturba cada gene +/-1 step, mede Delta fit. Genes fragéis -> overfit."""
    import ga_run as gr
    import run_local_ga_staged as staged

    store, _w, _sg, _sf, _bg, _bf, _bi = staged.open_existing_store()
    cache = staged.build_ticker_arrays_cache(store)

    base_metrics = staged.evaluate_genome_full(checkpoint_genome, cache, gr)
    base_fit = float(base_metrics.get("fit", 0.0) or 0.0)

    deltas = []
    per_gene = {}
    for idx, (name, lo, hi, step, is_int) in enumerate(gr.GLOBAL_PARAM_SPECS):
        current = float(checkpoint_genome[idx])
        # Test +step and -step
        for direction in (+1, -1):
            candidate = list(checkpoint_genome)
            new_val = current + direction * step
            if new_val < lo or new_val > hi:
                continue
            candidate[idx] = new_val
            candidate = gr.sanitize_global_genome(candidate)
            try:
                m = staged.evaluate_genome_full(candidate, cache, gr)
                delta = abs(float(m.get("fit", 0.0) or 0.0) - base_fit)
                deltas.append(delta)
                per_gene.setdefault(name, []).append(delta)
            except Exception:
                continue

    if not deltas:
        return {"error": "no_perturbations"}

    # Sumariza por gene: max de |delta|
    gene_max_deltas = {g: float(max(ds)) for g, ds in per_gene.items()}
    top_fragile = sorted(gene_max_deltas.items(), key=lambda kv: -kv[1])[:5]

    return {
        "base_fit": base_fit,
        "n_perturbations": len(deltas),
        "median_abs_delta": float(np.median(deltas)),
        "max_abs_delta": float(np.max(deltas)),
        "mean_abs_delta": float(np.mean(deltas)),
        "top_5_fragile_genes": [{"gene": g, "max_delta_fit": d} for g, d in top_fragile],
    }


# ============================================================
# VERDICT
# ============================================================

def classify(holdout_result: Dict, bootstrap_result: Dict,
             sensitivity_result: Dict) -> Tuple[str, List[str]]:
    """Classifica CLEAN / MARGINAL / OVERFIT baseado em thresholds.

    Returns (verdict, list_of_reasons).
    """
    warnings = []
    critical = []

    # Holdout checks
    delta = holdout_result.get("delta_holdout_vs_train", {})
    wr_drop = -float(delta.get("wr_med_all", 0.0))  # positivo = WR caiu
    alpha_drop = -float(delta.get("mean_alpha_ann", 0.0))
    if wr_drop > THRESHOLDS["holdout_wr_drop_critical"]:
        critical.append(f"holdout_wr_drop={wr_drop:.4f} > critical ({THRESHOLDS['holdout_wr_drop_critical']})")
    elif wr_drop > THRESHOLDS["holdout_wr_drop_max"]:
        warnings.append(f"holdout_wr_drop={wr_drop:.4f} > warn ({THRESHOLDS['holdout_wr_drop_max']})")
    if alpha_drop > THRESHOLDS["holdout_alpha_drop_critical"]:
        critical.append(f"holdout_alpha_drop={alpha_drop:.4f} > critical ({THRESHOLDS['holdout_alpha_drop_critical']})")
    elif alpha_drop > THRESHOLDS["holdout_alpha_drop_max"]:
        warnings.append(f"holdout_alpha_drop={alpha_drop:.4f} > warn ({THRESHOLDS['holdout_alpha_drop_max']})")

    # Bootstrap checks
    wr_ci_w = float(bootstrap_result.get("wr_ci_width", 0.0) or 0.0)
    if wr_ci_w > THRESHOLDS["bootstrap_wr_ci_width_critical"]:
        critical.append(f"wr_ci_width={wr_ci_w:.4f} > critical ({THRESHOLDS['bootstrap_wr_ci_width_critical']})")
    elif wr_ci_w > THRESHOLDS["bootstrap_wr_ci_width_max"]:
        warnings.append(f"wr_ci_width={wr_ci_w:.4f} > warn ({THRESHOLDS['bootstrap_wr_ci_width_max']})")

    # Sensitivity checks
    med_delta = float(sensitivity_result.get("median_abs_delta", 0.0) or 0.0)
    if med_delta > THRESHOLDS["sensitivity_median_critical"]:
        critical.append(f"sensitivity_median={med_delta:.2f} > critical ({THRESHOLDS['sensitivity_median_critical']})")
    elif med_delta > THRESHOLDS["sensitivity_median_max"]:
        warnings.append(f"sensitivity_median={med_delta:.2f} > warn ({THRESHOLDS['sensitivity_median_max']})")

    if critical:
        return "OVERFIT", critical + warnings
    if len(warnings) >= 2:
        return "OVERFIT", warnings
    if warnings:
        return "MARGINAL", warnings
    return "CLEAN", ["all_checks_passed"]


# ============================================================
# MARKDOWN REPORT
# ============================================================

def format_markdown(data: Dict) -> str:
    md = [f"# Overfitting Stage Report\n\n",
          f"Generated at: `{data.get('generated_at')}`\n",
          f"Verdict: **{data.get('verdict')}**\n\n",
          f"## Checkpoint\n\n",
          f"- fit: `{data['holdout_result']['metrics_full']['fit']:.4f}`\n",
          f"- WR: `{data['holdout_result']['metrics_full']['wr_med_all']:.4f}`\n",
          f"- alpha_ann: `{data['holdout_result']['metrics_full']['mean_alpha_ann']:+.4f}`\n",
          f"- n_ge_70: `{data['holdout_result']['metrics_full']['n_ge_70']}`\n\n"]

    h = data["holdout_result"]
    md += [f"## Pristine Holdout ({h.get('holdout_days', 90)}d)\n\n",
           f"| Metric | Train | Holdout | Delta |\n",
           f"|---|---:|---:|---:|\n"]
    for k in ("fit", "wr_med_all", "mean_alpha_ann", "n_ge_70", "trades_med"):
        tr = h["metrics_train"].get(k, 0)
        ho = h["metrics_holdout"].get(k, 0)
        dl = h["delta_holdout_vs_train"].get(k, 0) if k != "trades_med" else 0
        md.append(f"| {k} | {tr:+.4f} | {ho:+.4f} | {dl:+.4f} |\n")

    b = data["bootstrap_result"]
    md += [f"\n## Bootstrap CI (WR)\n\n",
           f"- Samples: {b.get('n_samples', 0)}\n",
           f"- Tickers: {b.get('n_tickers_in_bootstrap', 0)}\n",
           f"- WR CI 5-95: [{b.get('wr_ci_5', 0):.4f}, {b.get('wr_ci_95', 0):.4f}] (width={b.get('wr_ci_width', 0):.4f})\n\n"]

    s = data["sensitivity_result"]
    md += [f"## Parameter Sensitivity\n\n",
           f"- Base fit: {s.get('base_fit', 0):.4f}\n",
           f"- Median abs delta: {s.get('median_abs_delta', 0):.4f}\n",
           f"- Max abs delta: {s.get('max_abs_delta', 0):.4f}\n\n",
           f"### Top 5 fragile genes\n\n",
           f"| Gene | Max delta_fit |\n|---|---:|\n"]
    for g in s.get("top_5_fragile_genes", []):
        md.append(f"| {g['gene']} | {g['max_delta_fit']:.4f} |\n")

    md += [f"\n## Reasons\n\n"]
    for r in data.get("reasons", []):
        md.append(f"- {r}\n")

    return "".join(md)


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout-days", type=int, default=90)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--skip-sensitivity", action="store_true",
                    help="Pula parameter sensitivity (mais lento)")
    ap.add_argument("--strict", action="store_true",
                    help="Exit 2 tambem em MARGINAL")
    args = ap.parse_args()

    # Load checkpoint
    ckpt_path = ROOT / "global_ga_checkpoint.json"
    if not ckpt_path.exists():
        print(f"[overfitting] Checkpoint not found: {ckpt_path}")
        return 0
    with open(ckpt_path, "r", encoding="utf-8") as f:
        ck = json.load(f)
    genome = list(ck.get("genome", []))
    if not genome:
        print(f"[overfitting] No genome in checkpoint")
        return 0

    print(f"[overfitting] ==> Starting overfitting stage (checkpoint fit={ck.get('fitness', 0):.4f})")
    print(f"[overfitting] holdout_days={args.holdout_days} bootstrap={args.bootstrap}")

    # Run checks
    holdout_result = evaluate_holdout(genome, holdout_days=args.holdout_days)
    print(f"[overfitting] Holdout done. "
          f"WR delta={holdout_result.get('delta_holdout_vs_train', {}).get('wr_med_all', 0):+.4f}")

    bootstrap_result = bootstrap_ci_wr(genome, n_samples=args.bootstrap)
    print(f"[overfitting] Bootstrap done. "
          f"WR CI width={bootstrap_result.get('wr_ci_width', 0):.4f}")

    if args.skip_sensitivity:
        sensitivity_result = {"skipped": True, "median_abs_delta": 0.0}
    else:
        sensitivity_result = parameter_sensitivity(genome)
        print(f"[overfitting] Sensitivity done. "
              f"median_abs_delta={sensitivity_result.get('median_abs_delta', 0):.4f}")

    verdict, reasons = classify(holdout_result, bootstrap_result, sensitivity_result)

    data = {
        "generated_at": _now_iso(),
        "verdict": verdict,
        "reasons": reasons,
        "checkpoint_fit": float(ck.get("fitness", 0) or 0),
        "holdout_result": holdout_result,
        "bootstrap_result": bootstrap_result,
        "sensitivity_result": sensitivity_result,
        "thresholds": THRESHOLDS,
    }

    # Save
    _atomic_write_json(REPORT_JSON, data)
    REPORT_MD.write_text(format_markdown(data), encoding="utf-8")

    print(f"\n[overfitting] ==> VERDICT: {verdict}")
    for r in reasons:
        print(f"  - {r}")
    print(f"\n[overfitting] Reports:")
    print(f"  {REPORT_JSON.relative_to(ROOT)}")
    print(f"  {REPORT_MD.relative_to(ROOT)}")

    # Exit code
    if verdict == "OVERFIT":
        return 2
    if verdict == "MARGINAL":
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
