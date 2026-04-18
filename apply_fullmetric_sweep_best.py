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

    decision = staged._candidate_beats_incumbent(cand_metrics, base_metrics, seed_metrics)
    promote = bool(decision.get("promote"))

    print(f"[SWEEP] {sweep_path.name}")
    print(f"[BEST] label={label} moves={len(moves)} promote={promote}")
    print(
        f"[CAND] fit={cand_metrics.get('fit'):.6f} "
        f"WR={cand_metrics.get('wr_med_all'):.6f} "
        f"WR_tgt={cand_metrics.get('wr_target_med_all'):.6f} "
        f"alpha={cand_metrics.get('mean_alpha_ann'):.6f} "
        f"MDD={cand_metrics.get('mdd_med'):.6f}"
    )

    if not promote:
        raise SystemExit("Best sweep candidate failed staged guardrails; refusing to apply.")

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


if __name__ == "__main__":
    main()

