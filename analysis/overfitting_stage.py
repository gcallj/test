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
    # Operational Premium/HQ guardrails for the live swing-trade universe.
    "operable_oos_wr_drop_critical": 0.03,       # 3pp vs incumbent/OOS reference
    "operable_oos_alpha_drop_critical": 0.005,   # 0.5pp alpha_ann
    "operable_wr_wilson_lo_min": 0.55,
    "operable_bootstrap_ci_width_max": 0.15,
    "operable_min_tickers": 20,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _metric_view(metrics: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "fit",
        "wr_med_all",
        "wr_target_med_all",
        "mean_alpha_ann",
        "n_ge_70",
        "trades_med",
        "wr_med_operable",
        "wr_target_med_operable",
        "alpha_ann_mean_operable",
        "expectancy_net_med_operable",
        "mdd_med_operable",
        "hold_med_operable",
        "max_hold_bars_p75_operable",
        "expected_hold_score_med_operable",
        "early_take_dependency_med_operable",
        "swing_survival_3d_med_operable",
        "wr_after_3_bars_med_operable",
        "target_wr_after_3_bars_med_operable",
        "n_buy_operable",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        val = metrics.get(key, 0.0)
        if key in ("n_ge_70", "n_buy_operable"):
            out[key] = int(val or 0)
        else:
            out[key] = float(val or 0.0)
    return out


def _wilson_ci(successes: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = float(successes) / float(n)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))
    return float(max(0.0, center - half)), float(min(1.0, center + half))


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

    delta_keys = set(_metric_view(metrics_train)) | set(_metric_view(metrics_holdout))
    return {
        "metrics_full": _metric_view(metrics_full),
        "metrics_train": _metric_view(metrics_train),
        "metrics_holdout": _metric_view(metrics_holdout),
        "delta_holdout_vs_train": {k: _delta(k) for k in sorted(delta_keys)},
        "n_tickers_train": len(cache_train),
        "n_tickers_holdout": len(cache_holdout),
        "holdout_days": holdout_days,
    }


# ============================================================
# BOOTSTRAP CIs
# ============================================================

def bootstrap_ci_wr(checkpoint_genome: List[float],
                    n_samples: int = 200,
                    seed: int = 42,
                    operable_only: bool = False) -> Dict[str, Any]:
    """Bootstrap sobre trade_returns por ticker para estimar CI de metricas."""
    import ga_run as gr
    import run_local_ga_staged as staged

    store, _w, _sg, _sf, _bg, _bf, _bi = staged.open_existing_store()
    cache = staged.build_ticker_arrays_cache(store)

    # Coleta trade_rets e win_rate por ticker
    gp = gr.decode_global_params(checkpoint_genome)
    per_ticker_wr = []
    per_ticker_trades = []
    per_ticker_wilson_lo = []
    for tk, p in cache.items():
        try:
            st = gr.backtest_stats_global_intraday(
                p["open"], p["high"], p["low"], p["close"],
                p["score_matrix"], p["atr"], gp, precomputed=p,
            )
            gr._attach_benchmark_stats(st, p["close"])
            n = int(st.get("n_trades", 0) or 0)
            wr = float(st.get("win_rate", 0.0) or 0.0)
            if operable_only and not gr.is_operational_stat(st):
                continue
            if n >= 5:  # minimo para incluir no bootstrap
                per_ticker_wr.append(wr)
                per_ticker_trades.append(n)
                lo, _hi = _wilson_ci(wr * n, n)
                per_ticker_wilson_lo.append(lo)
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
    pooled_n = float(np.sum(per_ticker_trades))
    pooled_successes = float(np.sum(np.asarray(per_ticker_wr) * np.asarray(per_ticker_trades)))
    wr_wilson_lo, wr_wilson_hi = _wilson_ci(pooled_successes, pooled_n)

    return {
        "n_samples": n_samples,
        "n_tickers_in_bootstrap": n_tickers,
        "operable_only": bool(operable_only),
        "wr_ci_5": ci_low,
        "wr_ci_50": ci_median,
        "wr_ci_95": ci_high,
        "wr_ci_width": ci_width,
        "wr_wilson_lo": wr_wilson_lo,
        "wr_wilson_hi": wr_wilson_hi,
        "wr_wilson_lo_median_ticker": float(np.median(per_ticker_wilson_lo)) if per_ticker_wilson_lo else 0.0,
        "pooled_trades": int(pooled_n),
        "point_estimate_wr_median": float(np.median(per_ticker_wr)),
    }


# ============================================================
# ROLLING OOS WITH EMBARGO
# ============================================================

def evaluate_rolling_oos(checkpoint_genome: List[float],
                         folds: int = 4,
                         test_days: int = 90,
                         embargo_days: int = 5) -> Dict[str, Any]:
    """Evaluate recent rolling OOS windows with a temporal embargo before each test."""
    import pandas as pd
    import ga_run as gr
    import run_local_ga_staged as staged

    store, _w, _sg, _sf, _bg, _bf, _bi = staged.open_existing_store()
    cache = staged.build_ticker_arrays_cache(store)
    all_dates = []
    for p in cache.values():
        dates = p.get("dates")
        if dates is None:
            continue
        try:
            arr = pd.to_datetime(np.asarray(dates), unit="ns")
        except (TypeError, ValueError):
            try:
                arr = pd.to_datetime(np.asarray(dates))
            except (TypeError, ValueError):
                continue
        if len(arr):
            all_dates.append(arr.max())
    if not all_dates:
        return {"error": "no_dates"}

    max_date = pd.Timestamp(max(all_dates)).normalize()
    fold_results = []
    for fold in range(int(folds)):
        test_end = max_date - pd.Timedelta(days=fold * test_days)
        test_start = test_end - pd.Timedelta(days=test_days - 1)
        train_end = test_start - pd.Timedelta(days=embargo_days)
        cache_train = staged.slice_ticker_arrays_cache(cache, end_date=train_end, min_rows=252)
        cache_test = staged.slice_ticker_arrays_cache(cache, start_date=test_start, end_date=test_end, min_rows=60)
        if not cache_train or not cache_test:
            continue
        train_metrics = staged.evaluate_genome_full(checkpoint_genome, cache_train, gr)
        test_metrics = staged.evaluate_genome_full(checkpoint_genome, cache_test, gr)
        fold_results.append({
            "fold": int(fold),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "n_tickers_train": len(cache_train),
            "n_tickers_test": len(cache_test),
            "train": _metric_view(train_metrics),
            "oos": _metric_view(test_metrics),
            "delta_oos_vs_train": {
                k: float(_metric_view(test_metrics).get(k, 0.0) - _metric_view(train_metrics).get(k, 0.0))
                for k in _metric_view(test_metrics)
            },
        })

    if not fold_results:
        return {"error": "no_valid_folds", "folds_requested": int(folds)}

    def _median_delta(key: str) -> float:
        vals = [f["delta_oos_vs_train"].get(key, 0.0) for f in fold_results]
        return float(np.median(vals)) if vals else 0.0

    return {
        "folds_requested": int(folds),
        "folds_evaluated": len(fold_results),
        "test_days": int(test_days),
        "embargo_days": int(embargo_days),
        "folds": fold_results,
        "median_delta_oos_vs_train": {
            "wr_med_operable": _median_delta("wr_med_operable"),
            "alpha_ann_mean_operable": _median_delta("alpha_ann_mean_operable"),
            "wr_med_all": _median_delta("wr_med_all"),
            "mean_alpha_ann": _median_delta("mean_alpha_ann"),
        },
    }


def evaluate_negative_controls(checkpoint_genome: List[float],
                               max_tickers: int = 40,
                               shift_bars: int = 5,
                               seed: int = 42) -> Dict[str, Any]:
    """Evaluate shifted/permuted signal matrices to catch obvious leakage."""
    import ga_run as gr
    import run_local_ga_staged as staged

    store, _w, _sg, _sf, _bg, _bf, _bi = staged.open_existing_store()
    cache = staged.build_ticker_arrays_cache(store)
    items = list(cache.items())[: max(1, int(max_tickers))]
    if not items:
        return {"error": "no_tickers"}

    rng = np.random.default_rng(seed)

    def _with_control_matrix(kind: str) -> Dict[str, Dict[str, np.ndarray]]:
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for tk, payload in items:
            pp = dict(payload)
            sm = np.asarray(payload.get("score_matrix"), dtype=np.float64)
            if kind == "shifted":
                shifted = np.roll(sm, int(shift_bars), axis=0)
                shifted[: int(shift_bars), :] = 0.0
                pp["score_matrix"] = shifted
            else:
                perm = np.array(sm, copy=True)
                order = rng.permutation(perm.shape[0])
                pp["score_matrix"] = perm[order, :]
            out[tk] = pp
        return out

    shifted_metrics = staged.evaluate_genome_full(
        checkpoint_genome, _with_control_matrix("shifted"), gr
    )
    permuted_metrics = staged.evaluate_genome_full(
        checkpoint_genome, _with_control_matrix("permuted"), gr
    )
    return {
        "max_tickers": int(max_tickers),
        "shift_bars": int(shift_bars),
        "shifted": _metric_view(shifted_metrics),
        "permuted": _metric_view(permuted_metrics),
    }


def multiple_testing_summary() -> Dict[str, Any]:
    """Summarize search breadth as a simple PBO/selection-bias risk proxy."""
    sweep_files = list(ROOT.glob("local_fullmetric_sweep*.json"))
    promoted = 0
    for path in sweep_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("best_promoted"):
                promoted += 1
        except Exception:
            continue

    codex_attempts = 0
    codex_promoted = 0
    attempts_path = ROOT / "analysis" / "codex_attempts_history.json"
    if attempts_path.exists():
        try:
            data = json.loads(attempts_path.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else data.get("attempts", [])
            codex_attempts = len(rows)
            codex_promoted = sum(1 for row in rows if row.get("promoted") or row.get("Promoted"))
        except Exception:
            pass

    total_trials = len(sweep_files) + codex_attempts
    total_promoted = promoted + codex_promoted
    promotion_rate = (total_promoted / total_trials) if total_trials else 0.0
    if total_trials >= 200 or promotion_rate >= 0.30:
        risk = "HIGH"
    elif total_trials >= 50 or promotion_rate >= 0.15:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return {
        "sweep_files": len(sweep_files),
        "sweep_promotions": promoted,
        "codex_attempts": codex_attempts,
        "codex_promotions": codex_promoted,
        "total_trials": total_trials,
        "total_promotions": total_promoted,
        "promotion_rate": promotion_rate,
        "pbo_risk_proxy": risk,
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
             sensitivity_result: Dict, rolling_result: Dict | None = None,
             negative_control_result: Dict | None = None) -> Tuple[str, List[str]]:
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

    operable_delta = holdout_result.get("delta_holdout_vs_train", {})
    op_wr_drop = -float(operable_delta.get("wr_med_operable", 0.0) or 0.0)
    op_alpha_drop = -float(operable_delta.get("alpha_ann_mean_operable", 0.0) or 0.0)
    if op_wr_drop > THRESHOLDS["operable_oos_wr_drop_critical"]:
        critical.append(
            f"operable_holdout_wr_drop={op_wr_drop:.4f} > critical "
            f"({THRESHOLDS['operable_oos_wr_drop_critical']})"
        )
    if op_alpha_drop > THRESHOLDS["operable_oos_alpha_drop_critical"]:
        critical.append(
            f"operable_holdout_alpha_drop={op_alpha_drop:.4f} > critical "
            f"({THRESHOLDS['operable_oos_alpha_drop_critical']})"
        )

    # Bootstrap checks
    wr_ci_w = float(bootstrap_result.get("wr_ci_width", 0.0) or 0.0)
    if wr_ci_w > THRESHOLDS["bootstrap_wr_ci_width_critical"]:
        critical.append(f"wr_ci_width={wr_ci_w:.4f} > critical ({THRESHOLDS['bootstrap_wr_ci_width_critical']})")
    elif wr_ci_w > THRESHOLDS["bootstrap_wr_ci_width_max"]:
        warnings.append(f"wr_ci_width={wr_ci_w:.4f} > warn ({THRESHOLDS['bootstrap_wr_ci_width_max']})")
    if bootstrap_result.get("operable_only"):
        op_n = int(bootstrap_result.get("n_tickers_in_bootstrap", 0) or 0)
        op_wilson = float(bootstrap_result.get("wr_wilson_lo", 0.0) or 0.0)
        if op_n < THRESHOLDS["operable_min_tickers"]:
            critical.append(f"operable_bootstrap_tickers={op_n} < minimum ({THRESHOLDS['operable_min_tickers']})")
        if op_wilson < THRESHOLDS["operable_wr_wilson_lo_min"]:
            critical.append(
                f"operable_wr_wilson_lo={op_wilson:.4f} < minimum "
                f"({THRESHOLDS['operable_wr_wilson_lo_min']})"
            )
        if wr_ci_w > THRESHOLDS["operable_bootstrap_ci_width_max"]:
            critical.append(
                f"operable_wr_ci_width={wr_ci_w:.4f} > maximum "
                f"({THRESHOLDS['operable_bootstrap_ci_width_max']})"
            )

    if rolling_result and not rolling_result.get("error"):
        rd = rolling_result.get("median_delta_oos_vs_train", {})
        rolling_wr_drop = -float(rd.get("wr_med_operable", 0.0) or 0.0)
        rolling_alpha_drop = -float(rd.get("alpha_ann_mean_operable", 0.0) or 0.0)
        if rolling_wr_drop > THRESHOLDS["operable_oos_wr_drop_critical"]:
            critical.append(f"rolling_operable_wr_drop={rolling_wr_drop:.4f} > critical")
        if rolling_alpha_drop > THRESHOLDS["operable_oos_alpha_drop_critical"]:
            critical.append(f"rolling_operable_alpha_drop={rolling_alpha_drop:.4f} > critical")
    elif rolling_result and rolling_result.get("error"):
        critical.append(f"rolling_oos_error={rolling_result.get('error')}")

    if negative_control_result and not negative_control_result.get("error"):
        for kind in ("shifted", "permuted"):
            m = negative_control_result.get(kind, {}) or {}
            neg_wr = float(m.get("wr_med_operable", 0.0) or 0.0)
            neg_n = int(m.get("n_buy_operable", 0) or 0)
            if neg_n >= THRESHOLDS["operable_min_tickers"] and neg_wr >= 0.60:
                critical.append(
                    f"negative_control_{kind}_wr_operable={neg_wr:.4f} "
                    f"with n={neg_n} suggests leakage/selection artifact"
                )
            elif neg_wr >= 0.60 and neg_n > 0:
                warnings.append(
                    f"negative_control_{kind}_wr_operable={neg_wr:.4f} "
                    f"with n={neg_n} below sample minimum "
                    f"({THRESHOLDS['operable_min_tickers']})"
                )
    elif negative_control_result and negative_control_result.get("error"):
        critical.append(f"negative_control_error={negative_control_result.get('error')}")

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
    for k in ("fit", "wr_med_all", "mean_alpha_ann", "wr_med_operable",
              "alpha_ann_mean_operable", "n_buy_operable", "trades_med"):
        tr = h["metrics_train"].get(k, 0)
        ho = h["metrics_holdout"].get(k, 0)
        dl = h["delta_holdout_vs_train"].get(k, 0) if k != "trades_med" else 0
        md.append(f"| {k} | {tr:+.4f} | {ho:+.4f} | {dl:+.4f} |\n")

    r = data.get("rolling_oos_result", {})
    md += [f"\n## Rolling OOS + Embargo\n\n"]
    if r.get("error"):
        md.append(f"- Error: `{r.get('error')}`\n\n")
    else:
        rd = r.get("median_delta_oos_vs_train", {})
        md += [
            f"- Folds evaluated: {r.get('folds_evaluated', 0)}\n",
            f"- Test days: {r.get('test_days', 0)} | Embargo days: {r.get('embargo_days', 0)}\n",
            f"- Median operable WR delta: `{rd.get('wr_med_operable', 0):+.4f}`\n",
            f"- Median operable alpha delta: `{rd.get('alpha_ann_mean_operable', 0):+.4f}`\n\n",
        ]

    b = data["bootstrap_result"]
    md += [f"\n## Bootstrap CI (WR)\n\n",
           f"- Samples: {b.get('n_samples', 0)}\n",
           f"- Tickers: {b.get('n_tickers_in_bootstrap', 0)}\n",
           f"- Operable only: {b.get('operable_only', False)}\n",
           f"- WR CI 5-95: [{b.get('wr_ci_5', 0):.4f}, {b.get('wr_ci_95', 0):.4f}] (width={b.get('wr_ci_width', 0):.4f})\n",
           f"- Pooled Wilson WR: [{b.get('wr_wilson_lo', 0):.4f}, {b.get('wr_wilson_hi', 0):.4f}]\n\n"]

    nc = data.get("negative_control_result", {})
    md += [f"## Negative Controls\n\n"]
    if nc.get("error"):
        md.append(f"- Error: `{nc.get('error')}`\n\n")
    else:
        for kind in ("shifted", "permuted"):
            m = nc.get(kind, {}) or {}
            md.append(
                f"- {kind}: WR_op `{m.get('wr_med_operable', 0):.4f}`, "
                f"alpha_op `{m.get('alpha_ann_mean_operable', 0):+.4f}`, "
                f"n_op `{m.get('n_buy_operable', 0)}`\n"
            )
        md.append("\n")

    s = data["sensitivity_result"]
    md += [f"## Parameter Sensitivity\n\n",
           f"- Base fit: {s.get('base_fit', 0):.4f}\n",
           f"- Median abs delta: {s.get('median_abs_delta', 0):.4f}\n",
           f"- Max abs delta: {s.get('max_abs_delta', 0):.4f}\n\n",
           f"### Top 5 fragile genes\n\n",
           f"| Gene | Max delta_fit |\n|---|---:|\n"]
    for g in s.get("top_5_fragile_genes", []):
        md.append(f"| {g['gene']} | {g['max_delta_fit']:.4f} |\n")

    mt = data.get("multiple_testing", {})
    md += [f"\n## Multiple Testing / PBO Proxy\n\n",
           f"- Total trials: {mt.get('total_trials', 0)}\n",
           f"- Total promotions: {mt.get('total_promotions', 0)}\n",
           f"- Promotion rate: {mt.get('promotion_rate', 0):.1%}\n",
           f"- Risk proxy: **{mt.get('pbo_risk_proxy', 'UNKNOWN')}**\n"]

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
    ap.add_argument("--rolling-folds", type=int, default=4)
    ap.add_argument("--embargo-days", type=int, default=5)
    ap.add_argument("--skip-sensitivity", action="store_true",
                    help="Pula parameter sensitivity (mais lento)")
    ap.add_argument("--negative-control-tickers", type=int, default=40,
                    help="Numero maximo de tickers para controles negativos shifted/permuted")
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

    rolling_result = evaluate_rolling_oos(
        genome,
        folds=args.rolling_folds,
        test_days=args.holdout_days,
        embargo_days=args.embargo_days,
    )
    if rolling_result.get("error"):
        print(f"[overfitting] Rolling OOS error: {rolling_result.get('error')}")
    else:
        rd = rolling_result.get("median_delta_oos_vs_train", {})
        print(f"[overfitting] Rolling OOS done. "
              f"operable WR delta={rd.get('wr_med_operable', 0):+.4f}")

    bootstrap_result = bootstrap_ci_wr(genome, n_samples=args.bootstrap, operable_only=True)
    print(f"[overfitting] Bootstrap done. "
          f"WR CI width={bootstrap_result.get('wr_ci_width', 0):.4f}")

    negative_control_result = evaluate_negative_controls(
        genome, max_tickers=args.negative_control_tickers
    )
    if negative_control_result.get("error"):
        print(f"[overfitting] Negative controls error: {negative_control_result.get('error')}")
    else:
        _shift = negative_control_result.get("shifted", {})
        _perm = negative_control_result.get("permuted", {})
        print(
            f"[overfitting] Negative controls done. "
            f"shifted WR_op={_shift.get('wr_med_operable', 0):.4f}, "
            f"permuted WR_op={_perm.get('wr_med_operable', 0):.4f}"
        )

    if args.skip_sensitivity:
        sensitivity_result = {"skipped": True, "median_abs_delta": 0.0}
    else:
        sensitivity_result = parameter_sensitivity(genome)
        print(f"[overfitting] Sensitivity done. "
              f"median_abs_delta={sensitivity_result.get('median_abs_delta', 0):.4f}")

    mt_summary = multiple_testing_summary()
    verdict, reasons = classify(
        holdout_result,
        bootstrap_result,
        sensitivity_result,
        rolling_result,
        negative_control_result,
    )

    data = {
        "generated_at": _now_iso(),
        "verdict": verdict,
        "reasons": reasons,
        "checkpoint_fit": float(ck.get("fitness", 0) or 0),
        "holdout_result": holdout_result,
        "rolling_oos_result": rolling_result,
        "bootstrap_result": bootstrap_result,
        "negative_control_result": negative_control_result,
        "sensitivity_result": sensitivity_result,
        "multiple_testing": mt_summary,
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
