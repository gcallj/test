#!/usr/bin/env python3
"""Local Optuna/TPE refinement around the current checkpoint."""
import argparse
import json
import pickle
import time
from datetime import datetime

import numpy as np
import optuna

import os

os.environ["GA_ALPHA_FOCUS"] = os.environ.get("GA_ALPHA_FOCUS", "on")

import ga_run as gr
import ga_run_modular_final as gm
import run_local_ga_staged as staged_runner


OUTPUT_JSON = "local_optuna_search_result.json"


def _compute_window_indices(n_windows: int, window_count: int, window_offset: int) -> list[int]:
    step = n_windows / max(window_count, 1)
    window_indices = [
        min(max(0, int(round(i * step + window_offset)) % n_windows), n_windows - 1)
        for i in range(window_count)
    ]
    window_indices = sorted(set(window_indices))[:window_count]
    while len(window_indices) < window_count:
        for j in range(n_windows):
            if j not in window_indices:
                window_indices.append(j)
                break
    return list(sorted(window_indices))


def _build_local_bounds(seed_genome, radius_steps):
    bounds = []
    for value, (name, lo, hi, step, is_int) in zip(seed_genome, gr.GLOBAL_PARAM_SPECS):
        local_lo = max(lo, float(value) - float(step) * float(radius_steps))
        local_hi = min(hi, float(value) + float(step) * float(radius_steps))
        if local_hi < local_lo:
            local_lo, local_hi = lo, hi
        if is_int:
            local_lo = int(round(local_lo))
            local_hi = int(round(local_hi))
        else:
            min_idx = int(np.ceil((float(local_lo) - float(lo)) / float(step) - 1e-9))
            max_idx = int(np.floor((float(local_hi) - float(lo)) / float(step) + 1e-9))
            local_lo = float(lo) + float(min_idx) * float(step)
            local_hi = float(lo) + float(max_idx) * float(step)
            local_lo = max(float(lo), local_lo)
            local_hi = min(float(hi), local_hi)
        bounds.append({
            "name": name,
            "low": local_lo,
            "high": local_hi,
            "step": step,
            "is_int": bool(is_int),
        })
    return bounds


def _genome_from_trial(trial, seed_genome, bounds):
    genome = []
    for base_value, bound in zip(seed_genome, bounds):
        if abs(float(bound["high"]) - float(bound["low"])) < 1e-12:
            genome.append(base_value)
            continue
        if bound["is_int"]:
            val = trial.suggest_int(bound["name"], int(round(bound["low"])), int(round(bound["high"])), step=int(round(bound["step"])))
        else:
            val = trial.suggest_float(bound["name"], float(bound["low"]), float(bound["high"]), step=float(bound["step"]))
        genome.append(val)
    return gr.sanitize_global_genome(genome)


def _evaluate_train_objective(genome, store_path, window_plans_bytes, window_indices):
    fit, mean_train, mean_oos = gm._evaluate_batch_sequential(
        [list(genome)],
        window_indices,
        store_path,
        window_plans_bytes,
    )[0]
    return {
        "fit": float(fit),
        "mean_train": float(mean_train),
        "mean_oos": float(mean_oos),
    }


def main():
    parser = argparse.ArgumentParser(description="Run local Optuna/TPE refinement around the current checkpoint.")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--window-offset", type=int, default=23)
    parser.add_argument("--trials", type=int, default=18)
    parser.add_argument("--radius-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk-validate", type=int, default=5)
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()

    store, window_plans, seed_genome, _seed_fit, baseline_genome, _baseline_fit, baseline_info = staged_runner.open_existing_store()
    seed_genome = gr.sanitize_global_genome(seed_genome)
    cache = staged_runner.build_ticker_arrays_cache(store)
    window_indices = _compute_window_indices(len(window_plans), args.windows, args.window_offset)
    window_plans_bytes = pickle.dumps(window_plans)
    bounds = _build_local_bounds(seed_genome, int(args.radius_steps))

    print(f"[SETUP] baseline={baseline_info.get('source')} ref={baseline_info.get('ref')}", flush=True)
    print(f"[SETUP] window_indices={window_indices}", flush=True)

    seed_train_eval = _evaluate_train_objective(seed_genome, store.store_path, window_plans_bytes, window_indices)
    trial_records = []

    def objective(trial):
        genome = _genome_from_trial(trial, seed_genome, bounds)
        train_eval = _evaluate_train_objective(genome, store.store_path, window_plans_bytes, window_indices)
        trial.set_user_attr("genome", list(genome))
        trial.set_user_attr("mean_train", float(train_eval["mean_train"]))
        trial.set_user_attr("mean_oos", float(train_eval["mean_oos"]))
        trial_records.append({
            "trial_number": int(trial.number),
            "genome": list(genome),
            "train_eval": train_eval,
        })
        print(
            f"[TPE] trial={trial.number} fit={train_eval['fit']:.4f} "
            f"train={train_eval['mean_train']:.4f} oos={train_eval['mean_oos']:.4f}",
            flush=True,
        )
        return float(train_eval["fit"])

    sampler = optuna.samplers.TPESampler(
        seed=int(args.seed),
        n_startup_trials=min(6, max(2, int(args.trials) // 3)),
        multivariate=True,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial({
        bound["name"]: int(round(value)) if bound["is_int"] else float(value)
        for value, bound in zip(seed_genome, bounds)
    })
    study.optimize(objective, n_trials=int(args.trials), show_progress_bar=False)

    base_metrics = staged_runner.evaluate_genome_full(baseline_genome, cache, gr)
    seed_metrics = staged_runner.evaluate_genome_full(seed_genome, cache, gr)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: float(t.value), reverse=True)
    top_trials = completed[: max(1, int(args.topk_validate))]

    validated_candidates = []
    seen = set()
    for trial in top_trials:
        genome = gr.sanitize_global_genome(list(trial.user_attrs["genome"]))
        key = tuple(round(float(x), 12) for x in genome)
        if key in seen:
            continue
        seen.add(key)
        metrics = staged_runner.evaluate_genome_full(genome, cache, gr)
        decision = staged_runner._candidate_beats_incumbent(metrics, base_metrics, seed_metrics)
        validated_candidates.append({
            "trial_number": int(trial.number),
            "genome": genome,
            "train_eval": {
                "fit": float(trial.value),
                "mean_train": float(trial.user_attrs.get("mean_train", 0.0) or 0.0),
                "mean_oos": float(trial.user_attrs.get("mean_oos", 0.0) or 0.0),
            },
            "metrics": metrics,
            "decision": decision,
        })

    promoted = [c for c in validated_candidates if c["decision"].get("promote")]
    if promoted:
        best_candidate = max(
            promoted,
            key=lambda item: (
                float(item["metrics"].get("fit", -1e18) or -1e18),
                float(item["metrics"].get("mean_alpha_ann", -1e18) or -1e18),
            ),
        )
    else:
        best_candidate = max(
            validated_candidates,
            key=lambda item: (
                float(item["metrics"].get("fit", -1e18) or -1e18),
                float(item["metrics"].get("mean_alpha_ann", -1e18) or -1e18),
            ),
        )

    result = {
        "timestamp": datetime.now().isoformat(),
        "optimizer": "optuna_tpe",
        "window_indices": list(window_indices),
        "window_offset": int(args.window_offset),
        "window_count": int(args.windows),
        "trials": int(args.trials),
        "radius_steps": int(args.radius_steps),
        "seed": int(args.seed),
        "baseline_info": baseline_info,
        "seed_train_eval": seed_train_eval,
        "seed_metrics": seed_metrics,
        "base_metrics": base_metrics,
        "study_best_value": float(study.best_value),
        "study_best_params": dict(study.best_params),
        "trial_records": trial_records,
        "validated_candidates": validated_candidates,
        "best_candidate": best_candidate,
        "candidate_metrics": best_candidate["metrics"],
        "candidate_vs_seed": best_candidate["decision"],
        "elapsed_s": round(time.time() - t0, 1),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    cand = best_candidate["metrics"]
    print(
        f"[RESULT] full seed fit={seed_metrics['fit']:.4f} candidate={cand['fit']:.4f} "
        f"| WR {seed_metrics['wr_med_all']*100:.1f}% -> {cand['wr_med_all']*100:.1f}% "
        f"| WR_tgt {seed_metrics['wr_target_med_all']*100:.1f}% -> {cand['wr_target_med_all']*100:.1f}% "
        f"| alpha {seed_metrics['mean_alpha_ann']*100:.2f}% -> {cand['mean_alpha_ann']*100:.2f}% "
        f"| MDD {seed_metrics['mdd_med']*100:.2f}% -> {cand['mdd_med']*100:.2f}%",
        flush=True,
    )
    print(f"[RESULT] candidate_beats_seed={best_candidate['decision']['promote']}", flush=True)
    print(f"[RESULT] saved to {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
