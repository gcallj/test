#!/usr/bin/env python3
"""
Apply the `best_promoted` candidate from a `run_local_fullmetric_sweep.py` result
to `global_ga_checkpoint.json`, using the shared staged-runner guardrails.

This is intended for small, guardrail-safe promotions discovered via targeted
full-metric sweeps (often 1-2 gene moves), without re-running a full GA stage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import ga_run as gr
import run_local_ga_staged as staged


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _m(metrics: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the best candidate from a fullmetric sweep JSON result.")
    parser.add_argument("--sweep", type=str, required=True, help="Path to local_fullmetric_sweep_result_*.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute decision and print details, but do not write any checkpoints.",
    )
    args = parser.parse_args()

    sweep_path = Path(args.sweep).expanduser().resolve()
    if not sweep_path.exists():
        raise SystemExit(f"Missing sweep file: {sweep_path}")

    sweep = _load_json(sweep_path)
    best = sweep.get("best_promoted") or None
    if not best:
        raise SystemExit("Sweep file has no `best_promoted` candidate.")

    label = str(best.get("label") or "").strip() or "unknown"
    moves = list(best.get("moves") or [])
    if not moves:
        raise SystemExit("Sweep `best_promoted` has no moves to apply.")

    # Load current checkpoint (seed) genome.
    seed_ckpt = json.loads(Path(staged.MAIN_CHECKPOINT).read_text(encoding="utf-8-sig"))
    seed_genome = list(seed_ckpt.get("genome") or [])
    if not seed_genome:
        raise SystemExit("Current global checkpoint has no `genome`.")

    # Apply moves.
    candidate_genome = list(seed_genome)
    for mv in moves:
        idx = int(mv["gene_idx"])
        candidate_genome[idx] = float(mv["to"])
    candidate_genome = gr.sanitize_global_genome(candidate_genome)

    # Evaluate baseline/seed/candidate under the current full-metric formula.
    store, _window_plans, seed_genome_store, _seed_fit, baseline_genome, _baseline_fit, baseline_info = (
        staged.open_existing_store()
    )
    cache = staged.build_ticker_arrays_cache(store)

    base_metrics = sweep.get("base_metrics") or staged.evaluate_genome_full(baseline_genome, cache, gr)
    seed_metrics = sweep.get("seed_metrics") or staged.evaluate_genome_full(seed_genome_store, cache, gr)
    cand_metrics = staged.evaluate_genome_full(candidate_genome, cache, gr)

    # B4 drift guard: passa `moves` para que _candidate_beats_incumbent detecte
    # drift direcional de genes. Ativo apenas se GA_DRIFT_GUARD=1 no env.
    proposed_moves = [
        {"gene": mv.get("gene"), "from": mv.get("from"), "to": mv.get("to")}
        for mv in moves
        if mv.get("gene") is not None and mv.get("from") is not None and mv.get("to") is not None
    ]
    decision = staged._candidate_beats_incumbent(cand_metrics, base_metrics, seed_metrics,
                                                  proposed_moves=proposed_moves)
    promote = bool(decision.get("promote"))

    print(f"[SWEEP] {sweep_path.name}")
    print(f"[BEST] label={label} moves={len(moves)} promote={promote}")
    # Log drift status quando guard ativo
    drift_info = decision.get("parameter_drift") or {}
    if decision.get("drift_guard_active") and drift_info.get("drift_detected"):
        drifting_names = [g.get("gene") for g in drift_info.get("drifting_genes", [])]
        print(f"[DRIFT] detected on: {', '.join(drifting_names)} — promotion blocked")
    print(
        f"[CAND] fit={cand_metrics.get('fit'):.6f} "
        f"WR={cand_metrics.get('wr_med_all'):.6f} "
        f"WR_tgt={cand_metrics.get('wr_target_med_all'):.6f} "
        f"alpha={cand_metrics.get('mean_alpha_ann'):.6f} "
        f"MDD={cand_metrics.get('mdd_med'):.6f}"
    )

    if not promote:
        raise SystemExit("Best sweep candidate failed staged guardrails; refusing to apply.")

    # ================================================================
    # LONG-TERM REGRESSION GUARD — best_ever_tracker.json
    # ================================================================
    # Alem do gate vs main, verifica contra o best_ever historico.
    # Isso impede "death by a thousand cuts": cada promocao passa o gate
    # vs main atual, mas main vai degradando step by step. Guard quebra
    # esse loop checando contra o melhor observado ever.
    try:
        from analysis.best_ever_guard import (
            load_best_ever, check_regression_vs_best, format_best_ever_summary
        )
        best_ever = load_best_ever()
        if best_ever:
            print(f"[GUARD] {format_best_ever_summary(best_ever)}")
            ok, reason = check_regression_vs_best(cand_metrics, best_ever)
            if not ok:
                print(f"[GUARD] REJEITA: regressao vs best_ever -> {reason}")
                raise SystemExit(
                    "Long-term regression guard: candidate piora metricas "
                    "vs best_ever. Use analysis/best_ever_tracker.json para "
                    "resetar manual se necessario."
                )
            print(f"[GUARD] PASSA: {reason}")
        else:
            print("[GUARD] Sem best_ever_tracker ainda — primeira vez")
    except SystemExit:
        raise
    except Exception as _e:
        # Nao-fatal: se guard crashar, mantem comportamento antigo (sem guard)
        print(f"[GUARD] WARN: long-term guard falhou: {_e} — seguindo sem guard")

    # ================================================================
    # FAIL-CLOSED OVERFIT GUARD: holdout + rolling OOS + operable CI.
    # ================================================================
    # Promotion is blocked if any test/report fails or if the candidate
    # regresses against the incumbent in the Premium/HQ operational universe.
    try:
        from analysis.overfitting_stage import (
            THRESHOLDS as _OF_THR,
            bootstrap_ci_wr,
            classify,
            evaluate_holdout,
            evaluate_rolling_oos,
            multiple_testing_summary,
        )

        print("[OVERFIT] Evaluating candidate holdout (90d)...")
        holdout_data = evaluate_holdout(candidate_genome, holdout_days=90)
        print("[OVERFIT] Evaluating incumbent holdout (90d)...")
        incumbent_holdout = evaluate_holdout(seed_genome_store, holdout_days=90)
        print("[OVERFIT] Evaluating rolling OOS (4 folds, 5d embargo)...")
        rolling_data = evaluate_rolling_oos(candidate_genome, folds=4, test_days=90, embargo_days=5)
        print("[OVERFIT] Bootstrapping operable WR CI...")
        bootstrap_data = bootstrap_ci_wr(candidate_genome, n_samples=200, operable_only=True)
        mt_summary = multiple_testing_summary()
        verdict, reasons = classify(
            holdout_data,
            bootstrap_data,
            {"skipped": True, "median_abs_delta": 0.0},
            rolling_data,
        )

        cand_ho = holdout_data.get("metrics_holdout", {})
        inc_ho = incumbent_holdout.get("metrics_holdout", {})
        oos_wr_delta = _m(cand_ho, "wr_med_operable") - _m(inc_ho, "wr_med_operable")
        oos_alpha_delta = _m(cand_ho, "alpha_ann_mean_operable") - _m(inc_ho, "alpha_ann_mean_operable")
        ci_width = _m(bootstrap_data, "wr_ci_width")
        wilson_lo = _m(bootstrap_data, "wr_wilson_lo")
        op_n = int(bootstrap_data.get("n_tickers_in_bootstrap", 0) or 0)

        print(
            f"[OVERFIT] verdict={verdict} "
            f"operable_oos_wr_delta={oos_wr_delta:+.4f} "
            f"operable_oos_alpha_delta={oos_alpha_delta:+.4f} "
            f"wilson_lo={wilson_lo:.4f} ci_width={ci_width:.4f} "
            f"trials={mt_summary.get('total_trials', 0)} "
            f"pbo_risk={mt_summary.get('pbo_risk_proxy', 'UNKNOWN')}"
        )

        critical = []
        if verdict == "OVERFIT":
            critical.extend(reasons)
        if oos_wr_delta < -_OF_THR["operable_oos_wr_drop_critical"]:
            critical.append(f"operable OOS WR delta {oos_wr_delta:+.4f} < -3pp vs incumbent")
        if oos_alpha_delta < -_OF_THR["operable_oos_alpha_drop_critical"]:
            critical.append(f"operable OOS alpha delta {oos_alpha_delta:+.4f} < -0.5pp vs incumbent")
        if op_n < _OF_THR["operable_min_tickers"]:
            critical.append(f"operable bootstrap tickers {op_n} < {_OF_THR['operable_min_tickers']}")
        if wilson_lo < _OF_THR["operable_wr_wilson_lo_min"]:
            critical.append(f"operable Wilson lower {wilson_lo:.4f} < 55%")
        if ci_width > _OF_THR["operable_bootstrap_ci_width_max"]:
            critical.append(f"operable bootstrap WR CI width {ci_width:.4f} > 15pp")

        if critical:
            print(f"[OVERFIT] REJEITA: {'; '.join(critical)}")
            raise SystemExit("Overfitting guard failed; refusing to apply candidate.")
        print("[OVERFIT] PASSA: holdout, rolling OOS and operable CI stable")
    except SystemExit:
        raise
    except Exception as _e:
        raise SystemExit(f"Overfitting guard failed closed: {_e}") from _e

    if args.dry_run:
        print("[DRY] Not writing checkpoint files.")
        return

    best_payload = {
        "fit": float(cand_metrics["fit"]),
        "wr_med_all": float(cand_metrics.get("wr_med_all", 0.0) or 0.0),
        "genome": list(candidate_genome),
        "metrics": dict(cand_metrics),
        "found_at_chunk": -1,
    }
    reason = f"fullmetric_sweep:{label} (from {sweep_path.name})"

    prev_path, new_path = staged._write_checkpoint_with_metadata(
        best_payload,
        base_metrics,
        seed_metrics,
        baseline_info,
        decision.get("acceptance_vs_main", {}),
        decision.get("incumbent_wr_guardrail", {}),
        decision.get("incumbent_safety_guardrail", {}),
        reason,
    )

    staged._atomic_json_dump(
        staged.BEST_SOFAR_FILE,
        {
            "fit": best_payload["fit"],
            "wr_med_all": best_payload["wr_med_all"],
            "genome": best_payload["genome"],
            "metrics": best_payload["metrics"],
            "found_at_chunk": 0,
        },
    )

    print(f"[APPLY] Wrote: {os.path.abspath(staged.MAIN_CHECKPOINT)}")
    print(f"[APPLY] Backup: {prev_path}")
    print(f"[APPLY] Snapshot: {new_path}")

    # Atualiza best_ever_tracker (cada metrica rastreada independentemente).
    try:
        from analysis.best_ever_guard import update_best_ever
        update_best_ever(cand_metrics, candidate_genome,
                         checkpoint_sha="", reason=reason)
    except Exception as _e:
        print(f"[GUARD] WARN: update_best_ever falhou: {_e}")


if __name__ == "__main__":
    main()

