#!/usr/bin/env python3
"""Non-GA local pattern search around the current checkpoint."""
import argparse
import json
import os
import pickle
import time
from datetime import datetime

import numpy as np

os.environ["GA_ALPHA_FOCUS"] = os.environ.get("GA_ALPHA_FOCUS", "on")

import ga_run as gr
import ga_run_modular_final as gm
import run_local_ga_staged as staged_runner


OUTPUT_JSON = "local_pattern_search_result.json"


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
    results = gm._evaluate_batch_sequential(
        [list(genome)],
        window_indices,
        store_path,
        window_plans_bytes,
    )
    fit, mean_train, mean_oos = results[0]
    return {
        "fit": float(fit),
        "mean_train": float(mean_train),
        "mean_oos": float(mean_oos),
    }


def _generate_start_genomes(seed_genome, starts, rng_seed, max_step_mult):
    rng = np.random.default_rng(int(rng_seed))
    starts = max(1, int(starts))
    out = [gr.sanitize_global_genome(list(seed_genome))]
    seen = {tuple(round(float(x), 12) for x in out[0])}
    while len(out) < starts:
        cand = list(seed_genome)
        for gene_idx, (_name, _lo, _hi, step, _is_int) in enumerate(gr.GLOBAL_PARAM_SPECS):
            step_mult = int(rng.integers(-int(max_step_mult), int(max_step_mult) + 1))
            if step_mult == 0:
                continue
            cand[gene_idx] = float(cand[gene_idx]) + float(step) * float(step_mult)
        cand = gr.sanitize_global_genome(cand)
        key = tuple(round(float(x), 12) for x in cand)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _pattern_search(seed_genome, store_path, window_plans, window_indices, max_evals, scales, rng_seed):
    window_plans_bytes = pickle.dumps(window_plans)
    rng = np.random.default_rng(int(rng_seed))
    current = gr.sanitize_global_genome(list(seed_genome))
    current_eval = _evaluate_train_objective(current, store_path, window_plans_bytes, window_indices)
    best = {
        "genome": list(current),
        "train_eval": dict(current_eval),
        "moves": [],
    }
    eval_count = 1

    print(
        f"[PATTERN] start fit={current_eval['fit']:.4f} "
        f"train={current_eval['mean_train']:.4f} oos={current_eval['mean_oos']:.4f}",
        flush=True,
    )

    for scale in scales:
        if eval_count >= max_evals:
            break
        improved_this_scale = True
        while improved_this_scale and eval_count < max_evals:
            improved_this_scale = False
            candidate_genomes = []
            candidate_meta = []
            seen = {tuple(round(float(x), 12) for x in current)}

            for gene_idx, (name, _lo, _hi, step, _is_int) in enumerate(gr.GLOBAL_PARAM_SPECS):
                delta = float(step) * float(scale)
                for direction in (-1.0, 1.0):
                    cand = list(current)
                    cand[gene_idx] = float(cand[gene_idx]) + direction * delta
                    cand = gr.sanitize_global_genome(cand)
                    key = tuple(round(float(x), 12) for x in cand)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate_genomes.append(cand)
                    candidate_meta.append({
                        "gene_idx": int(gene_idx),
                        "gene": str(name),
                        "direction": int(direction),
                        "delta": float(delta),
                    })

            remaining = max_evals - eval_count
            if not candidate_genomes or remaining <= 0:
                break
            order = list(rng.permutation(len(candidate_genomes)))
            candidate_genomes = [candidate_genomes[i] for i in order]
            candidate_meta = [candidate_meta[i] for i in order]
            candidate_genomes = candidate_genomes[:remaining]
            candidate_meta = candidate_meta[:remaining]
            batch_results = gm._evaluate_batch_sequential(
                candidate_genomes,
                window_indices,
                store_path,
                window_plans_bytes,
            )
            eval_count += len(batch_results)

            best_idx = None
            best_eval = current_eval
            for idx, result in enumerate(batch_results):
                fit, mean_train, mean_oos = result
                eval_payload = {
                    "fit": float(fit),
                    "mean_train": float(mean_train),
                    "mean_oos": float(mean_oos),
                }
                if eval_payload["fit"] > best_eval["fit"] + 1e-6:
                    best_idx = idx
                    best_eval = eval_payload
                elif abs(eval_payload["fit"] - best_eval["fit"]) <= 1e-6 and eval_payload["mean_oos"] > best_eval["mean_oos"] + 1e-6:
                    best_idx = idx
                    best_eval = eval_payload

            if best_idx is not None and best_eval["fit"] > current_eval["fit"] + 1e-6:
                current = list(candidate_genomes[best_idx])
                current_eval = dict(best_eval)
                move = dict(candidate_meta[best_idx])
                move.update({
                    "fit": float(current_eval["fit"]),
                    "mean_train": float(current_eval["mean_train"]),
                    "mean_oos": float(current_eval["mean_oos"]),
                    "scale": float(scale),
                })
                best["moves"].append(move)
                best["genome"] = list(current)
                best["train_eval"] = dict(current_eval)
                improved_this_scale = True
                print(
                    f"[PATTERN] improve scale={scale} gene={move['gene']} dir={move['direction']:+d} "
                    f"fit={current_eval['fit']:.4f} oos={current_eval['mean_oos']:.4f} evals={eval_count}",
                    flush=True,
                )
            else:
                print(f"[PATTERN] no improvement at scale={scale} evals={eval_count}", flush=True)
                break

    best["eval_count"] = int(eval_count)
    return best


def _validate_candidates(candidate_payloads, baseline_genome, base_metrics, seed_metrics, cache):
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
            "genome": genome,
            "train_eval": payload.get("train_eval", {}),
            "moves": payload.get("moves", []),
            "metrics": metrics,
            "decision": decision,
        })
    return validated


def main():
    parser = argparse.ArgumentParser(description="Run a non-GA local pattern search around the current checkpoint.")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--window-offset", type=int, default=23)
    parser.add_argument("--max-evals", type=int, default=80)
    parser.add_argument("--scales", type=str, default="2,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--start-jitter-steps", type=int, default=2)
    args = parser.parse_args()

    scales = [float(x.strip()) for x in args.scales.split(",") if x.strip()]
    t0 = time.time()

    store, window_plans, seed_genome, seed_fit, baseline_genome, _baseline_fit, baseline_info = staged_runner.open_existing_store()
    print(f"[SETUP] baseline={baseline_info.get('source')} ref={baseline_info.get('ref')}", flush=True)
    cache = staged_runner.build_ticker_arrays_cache(store)
    window_indices = _compute_window_indices(len(window_plans), args.windows, args.window_offset)
    print(f"[SETUP] window_indices={window_indices}", flush=True)

    seed_train_eval = _evaluate_train_objective(seed_genome, store.store_path, pickle.dumps(window_plans), window_indices)
    start_genomes = _generate_start_genomes(
        seed_genome=seed_genome,
        starts=int(args.starts),
        rng_seed=int(args.seed),
        max_step_mult=int(args.start_jitter_steps),
    )
    budget_per_start = max(6, int(np.ceil(int(args.max_evals) / max(len(start_genomes), 1))))
    start_results = []
    for start_idx, start_genome in enumerate(start_genomes, start=1):
        print(f"[MULTI] start {start_idx}/{len(start_genomes)} budget={budget_per_start}", flush=True)
        result = _pattern_search(
            seed_genome=start_genome,
            store_path=store.store_path,
            window_plans=window_plans,
            window_indices=window_indices,
            max_evals=budget_per_start,
            scales=scales,
            rng_seed=int(args.seed) + start_idx,
        )
        result["label"] = f"start_{start_idx}"
        start_results.append(result)

    print("[VALIDATE] evaluating full metrics for main / seed / candidate...", flush=True)
    base_metrics = staged_runner.evaluate_genome_full(baseline_genome, cache, gr)
    seed_metrics = staged_runner.evaluate_genome_full(seed_genome, cache, gr)
    validated_candidates = _validate_candidates(
        candidate_payloads=start_results,
        baseline_genome=baseline_genome,
        base_metrics=base_metrics,
        seed_metrics=seed_metrics,
        cache=cache,
    )
    promoted_candidates = [c for c in validated_candidates if c["decision"].get("promote")]
    if promoted_candidates:
        best_candidate = max(
            promoted_candidates,
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
    candidate_metrics = best_candidate["metrics"]
    candidate_vs_seed = best_candidate["decision"]
    pattern_best = {
        "label": best_candidate["label"],
        "genome": best_candidate["genome"],
        "train_eval": best_candidate["train_eval"],
        "moves": best_candidate["moves"],
        "eval_count": int(sum(int(r.get("eval_count", 0)) for r in start_results)),
    }

    result = {
        "timestamp": datetime.now().isoformat(),
        "optimizer": "pattern_search",
        "window_indices": list(window_indices),
        "window_offset": int(args.window_offset),
        "window_count": int(args.windows),
        "max_evals": int(args.max_evals),
        "scales": list(scales),
        "baseline_info": baseline_info,
        "seed_train_eval": seed_train_eval,
        "candidate_train_eval": pattern_best["train_eval"],
        "train_fit_delta_vs_seed": float(pattern_best["train_eval"]["fit"] - seed_train_eval["fit"]),
        "seed_metrics": seed_metrics,
        "candidate_metrics": candidate_metrics,
        "base_metrics": base_metrics,
        "candidate_vs_seed": candidate_vs_seed,
        "starts": int(args.starts),
        "start_jitter_steps": int(args.start_jitter_steps),
        "budget_per_start": int(budget_per_start),
        "start_results": start_results,
        "validated_candidates": validated_candidates,
        "pattern_search": pattern_best,
        "elapsed_s": round(time.time() - t0, 1),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[RESULT] train fit seed={seed_train_eval['fit']:.4f} candidate={pattern_best['train_eval']['fit']:.4f}", flush=True)
    print(
        f"[RESULT] full seed fit={seed_metrics['fit']:.4f} candidate={candidate_metrics['fit']:.4f} "
        f"| WR {seed_metrics['wr_med_all']*100:.1f}% -> {candidate_metrics['wr_med_all']*100:.1f}% "
        f"| WR_tgt {seed_metrics['wr_target_med_all']*100:.1f}% -> {candidate_metrics['wr_target_med_all']*100:.1f}% "
        f"| alpha {seed_metrics['mean_alpha_ann']*100:.2f}% -> {candidate_metrics['mean_alpha_ann']*100:.2f}% "
        f"| MDD {seed_metrics['mdd_med']*100:.2f}% -> {candidate_metrics['mdd_med']*100:.2f}%",
        flush=True,
    )
    print(f"[RESULT] candidate_beats_seed={candidate_vs_seed['promote']}", flush=True)
    print(f"[RESULT] saved to {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
