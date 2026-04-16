#!/usr/bin/env python3
"""Hybrid search: mini-GA exploration followed by Optuna/TPE local refinement."""
import argparse
import json
import os
import pickle
import random
import shutil
import time
from datetime import datetime

import numpy as np
import optuna

os.environ["GA_ALPHA_FOCUS"] = os.environ.get("GA_ALPHA_FOCUS", "on")

import ga_run as gr
import ga_run_modular_final as gm
import run_local_ga_staged as staged_runner


OUTPUT_JSON = "local_hybrid_search_result.json"
HYBRID_CHECKPOINT_DIR = "output/ga_checkpoints_hybrid_local"


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


def _seed_genomes(best_genome, n_seeds, rng_seed):
    random.seed(int(rng_seed))
    np.random.seed(int(rng_seed))
    seeds = [gr.sanitize_global_genome(list(best_genome))]
    while len(seeds) < max(1, int(n_seeds)):
        varied = []
        for g_val, (_, lo, hi, _step, _is_int) in zip(best_genome, gr.GLOBAL_PARAM_SPECS):
            v = float(np.clip(g_val + random.gauss(0.0, 0.05 * (hi - lo)), lo, hi))
            varied.append(v)
        seeds.append(gr.sanitize_global_genome(varied))
    return seeds


def _dedupe_genomes(genomes):
    out = []
    seen = set()
    for genome in genomes:
        key = tuple(round(float(x), 12) for x in genome)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(genome))
    return out


def _run_mini_ga(seed_genome, store, window_plans, window_offset, ga_pop, ga_ngen, windows, eval_topk, rng_seed):
    if os.path.exists(HYBRID_CHECKPOINT_DIR):
        shutil.rmtree(HYBRID_CHECKPOINT_DIR, ignore_errors=True)
    os.makedirs(HYBRID_CHECKPOINT_DIR, exist_ok=True)

    original_ckpt_dir = staged_runner.CHECKPOINT_DIR
    staged_runner.CHECKPOINT_DIR = HYBRID_CHECKPOINT_DIR
    try:
        best_genome, best_fit_train, hof, elapsed, chunk_window_indices = staged_runner.run_chunk(
            chunk_id=1,
            stage=2,
            ngen=int(ga_ngen),
            pop_size=int(ga_pop),
            window_count=int(windows),
            store_path=store.store_path,
            window_plans=window_plans,
            seed_genomes=_seed_genomes(seed_genome, max(6, int(ga_pop) // 2), rng_seed),
            resume_ckpt=None,
            window_offset=int(window_offset),
        )
    finally:
        staged_runner.CHECKPOINT_DIR = original_ckpt_dir

    hof_list = list(hof) if hof is not None else []
    genomes = []
    for ind in hof_list[: max(1, int(eval_topk))]:
        genomes.append(gr.sanitize_global_genome(list(ind)))
    genomes.insert(0, gr.sanitize_global_genome(list(best_genome)))
    genomes = _dedupe_genomes(genomes)
    return {
        "best_train_fit": float(best_fit_train),
        "elapsed_s": float(elapsed),
        "window_indices": list(chunk_window_indices),
        "genomes": genomes,
    }


def _build_local_bounds(center_genome, radius_steps):
    bounds = []
    for value, (name, lo, hi, step, is_int) in zip(center_genome, gr.GLOBAL_PARAM_SPECS):
        local_lo = max(lo, float(value) - float(step) * float(radius_steps))
        local_hi = min(hi, float(value) + float(step) * float(radius_steps))
        if is_int:
            local_lo = int(round(local_lo))
            local_hi = int(round(local_hi))
        else:
            min_idx = int(np.ceil((float(local_lo) - float(lo)) / float(step) - 1e-9))
            max_idx = int(np.floor((float(local_hi) - float(lo)) / float(step) + 1e-9))
            local_lo = max(float(lo), float(lo) + float(min_idx) * float(step))
            local_hi = min(float(hi), float(lo) + float(max_idx) * float(step))
        bounds.append({
            "name": name,
            "low": local_lo,
            "high": local_hi,
            "step": step,
            "is_int": bool(is_int),
        })
    return bounds


def _trial_genome(trial, center_genome, bounds):
    genome = []
    for center_val, bound in zip(center_genome, bounds):
        if abs(float(bound["high"]) - float(bound["low"])) < 1e-12:
            genome.append(center_val)
            continue
        if bound["is_int"]:
            val = trial.suggest_int(bound["name"], int(bound["low"]), int(bound["high"]), step=max(1, int(round(bound["step"]))))
        else:
            val = trial.suggest_float(bound["name"], float(bound["low"]), float(bound["high"]), step=float(bound["step"]))
        genome.append(val)
    return gr.sanitize_global_genome(genome)


def _params_from_genome(genome, bounds):
    params = {}
    for value, bound in zip(genome, bounds):
        if bound["is_int"]:
            clipped = int(np.clip(int(round(value)), int(bound["low"]), int(bound["high"])))
            params[bound["name"]] = clipped
        else:
            step = float(bound["step"])
            low = float(bound["low"])
            high = float(bound["high"])
            idx = int(round((float(value) - low) / step))
            snapped = low + idx * step
            snapped = min(max(snapped, low), high)
            params[bound["name"]] = float(round(snapped, 10))
    return params


def _validate_candidates(candidate_payloads, base_metrics, seed_metrics, cache):
    validated = []
    seen = set()
    for payload in candidate_payloads:
        genome = gr.sanitize_global_genome(list(payload["genome"]))
        key = tuple(round(float(x), 12) for x in genome)
        if key in seen:
            continue
        seen.add(key)
        metrics = staged_runner.evaluate_genome_full(genome, cache, gr)
        decision = staged_runner._candidate_beats_incumbent(metrics, base_metrics, seed_metrics)
        validated.append({
            "label": payload.get("label"),
            "source": payload.get("source"),
            "genome": genome,
            "train_eval": payload.get("train_eval", {}),
            "metrics": metrics,
            "decision": decision,
        })
    return validated


def _run_tpe_for_anchor(anchor, bounds, tpe_trials, seed_value, store, window_plans_bytes, window_indices):
    trial_records = []

    def objective(trial):
        genome = _trial_genome(trial, anchor["genome"], bounds)
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
            f"[TPE:{anchor['label']}] trial={trial.number} fit={train_eval['fit']:.4f} "
            f"train={train_eval['mean_train']:.4f} oos={train_eval['mean_oos']:.4f}",
            flush=True,
        )
        return float(train_eval["fit"])

    sampler = optuna.samplers.TPESampler(
        seed=int(seed_value),
        n_startup_trials=min(5, max(2, int(tpe_trials) // 3)),
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(_params_from_genome(anchor["genome"], bounds))
    study.optimize(objective, n_trials=int(tpe_trials), show_progress_bar=False)
    return study, trial_records


def main():
    parser = argparse.ArgumentParser(description="Run hybrid mini-GA + Optuna/TPE search.")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--window-offset", type=int, default=23)
    parser.add_argument("--ga-pop", type=int, default=12)
    parser.add_argument("--ga-ngen", type=int, default=1)
    parser.add_argument("--ga-topk", type=int, default=4)
    parser.add_argument("--tpe-trials", type=int, default=10)
    parser.add_argument("--tpe-anchors", type=int, default=1)
    parser.add_argument("--tpe-radius-steps", type=int, default=4)
    parser.add_argument("--tpe-topk-validate", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()

    store, window_plans, seed_genome, _seed_fit, baseline_genome, _baseline_fit, baseline_info = staged_runner.open_existing_store()
    seed_genome = gr.sanitize_global_genome(seed_genome)
    cache = staged_runner.build_ticker_arrays_cache(store)
    window_indices = _compute_window_indices(len(window_plans), args.windows, args.window_offset)
    window_plans_bytes = pickle.dumps(window_plans)

    print(f"[SETUP] baseline={baseline_info.get('source')} ref={baseline_info.get('ref')}", flush=True)
    print(f"[SETUP] window_indices={window_indices}", flush=True)

    base_metrics = staged_runner.evaluate_genome_full(baseline_genome, cache, gr)
    seed_metrics = staged_runner.evaluate_genome_full(seed_genome, cache, gr)
    seed_train_eval = _evaluate_train_objective(seed_genome, store.store_path, window_plans_bytes, window_indices)

    print("[HYBRID] running mini-GA exploration...", flush=True)
    ga_result = _run_mini_ga(
        seed_genome=seed_genome,
        store=store,
        window_plans=window_plans,
        window_offset=args.window_offset,
        ga_pop=args.ga_pop,
        ga_ngen=args.ga_ngen,
        windows=args.windows,
        eval_topk=args.ga_topk,
        rng_seed=args.seed,
    )

    ga_candidate_payloads = []
    for idx, genome in enumerate(ga_result["genomes"], start=1):
        train_eval = _evaluate_train_objective(genome, store.store_path, window_plans_bytes, window_indices)
        ga_candidate_payloads.append({
            "label": f"ga_{idx}",
            "source": "ga",
            "genome": genome,
            "train_eval": train_eval,
        })

    seed_payload = {
        "label": "seed",
        "source": "seed",
        "genome": seed_genome,
        "train_eval": seed_train_eval,
    }
    anchor_pool = [seed_payload] + sorted(
        ga_candidate_payloads,
        key=lambda item: float(item["train_eval"]["fit"]),
        reverse=True,
    )
    selected_anchors = []
    seen_anchor_keys = set()
    for payload in anchor_pool:
        key = tuple(round(float(x), 12) for x in payload["genome"])
        if key in seen_anchor_keys:
            continue
        seen_anchor_keys.add(key)
        selected_anchors.append(payload)
        if len(selected_anchors) >= max(1, int(args.tpe_anchors)):
            break

    all_studies = []
    tpe_candidate_payloads = []
    for anchor_idx, anchor in enumerate(selected_anchors, start=1):
        print(
            f"[HYBRID] TPE anchor={anchor['label']} train_fit={anchor['train_eval']['fit']:.4f}",
            flush=True,
        )
        bounds = _build_local_bounds(anchor["genome"], args.tpe_radius_steps)
        study, trial_records = _run_tpe_for_anchor(
            anchor=anchor,
            bounds=bounds,
            tpe_trials=int(args.tpe_trials),
            seed_value=int(args.seed) + anchor_idx,
            store=store,
            window_plans_bytes=window_plans_bytes,
            window_indices=window_indices,
        )
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        completed_trials.sort(key=lambda t: float(t.value), reverse=True)
        all_studies.append({
            "anchor": anchor["label"],
            "best_value": float(study.best_value),
            "best_params": dict(study.best_params),
            "trial_records": trial_records,
        })
        for trial in completed_trials[: max(1, int(args.tpe_topk_validate))]:
            tpe_candidate_payloads.append({
                "label": f"{anchor['label']}_tpe_{int(trial.number)}",
                "source": "tpe",
                "genome": list(trial.user_attrs["genome"]),
                "train_eval": {
                    "fit": float(trial.value),
                    "mean_train": float(trial.user_attrs.get("mean_train", 0.0) or 0.0),
                    "mean_oos": float(trial.user_attrs.get("mean_oos", 0.0) or 0.0),
                },
            })

    validated_candidates = _validate_candidates(
        candidate_payloads=[
            seed_payload,
            *ga_candidate_payloads,
            *tpe_candidate_payloads,
        ],
        base_metrics=base_metrics,
        seed_metrics=seed_metrics,
        cache=cache,
    )

    promotable = [c for c in validated_candidates if c["decision"].get("promote")]
    best_candidate = max(
        promotable if promotable else validated_candidates,
        key=lambda item: staged_runner._priority_tuple(item["metrics"]),
    )

    result = {
        "timestamp": datetime.now().isoformat(),
        "optimizer": "hybrid_ga_tpe",
        "window_indices": list(window_indices),
        "window_offset": int(args.window_offset),
        "window_count": int(args.windows),
        "baseline_info": baseline_info,
        "seed_train_eval": seed_train_eval,
        "seed_metrics": seed_metrics,
        "base_metrics": base_metrics,
        "ga_result": ga_result,
        "ga_candidates": ga_candidate_payloads,
        "tpe_anchors": int(args.tpe_anchors),
        "tpe_trials": int(args.tpe_trials),
        "tpe_radius_steps": int(args.tpe_radius_steps),
        "tpe_anchor_results": all_studies,
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
        f"[RESULT] best_source={best_candidate['source']} "
        f"full seed fit={seed_metrics['fit']:.4f} candidate={cand['fit']:.4f} "
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
