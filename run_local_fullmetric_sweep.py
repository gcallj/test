#!/usr/bin/env python3
"""
Targeted full-metric sweep around the current incumbent checkpoint.

Goal: explore small +/- step moves on parameters that are likely to affect
WR_target (take-profit hits) while respecting the shared staged-runner
guardrails (hit-rate-first, risk bounded).

This avoids relying on the GA training objective to surface candidates that are
"good enough" by fitness but fail WR_target guardrails.

Outputs:
  - local_fullmetric_sweep_result.json (or --out)
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np

os.environ["GA_ALPHA_FOCUS"] = os.environ.get("GA_ALPHA_FOCUS", "off")

import ga_run as gr
import run_local_ga_staged as staged_runner


DEFAULT_FOCUS_GENES = [
    "stop_atr_mult",
    "stop_tighten_after_bars",
    "stop_tighten_factor",
    "time_stop_bars",
    "reward_risk_ratio",
    "partial_take_pct",
    "partial_take_level",
    "partial_take_pct_2",
    "partial_take_level_2",
    "trailing_stop_mode",
]


def _spec_index_by_name(name: str) -> int:
    for idx, (spec_name, _lo, _hi, _step, _is_int) in enumerate(gr.GLOBAL_PARAM_SPECS):
        if spec_name == name:
            return int(idx)
    raise KeyError(f"Unknown gene name: {name}")


def _fmt_pct1(x: float) -> str:
    return f"{x * 100.0:.1f}%"


def _atomic_dump(path: str, payload: dict) -> None:
    path = os.path.abspath(path)
    tmp = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _diff_moves(seed_genome: list[float], cand_genome: list[float]) -> list[dict]:
    moves = []
    for idx, (name, _lo, _hi, step, is_int) in enumerate(gr.GLOBAL_PARAM_SPECS):
        a = float(seed_genome[idx])
        b = float(cand_genome[idx])
        if abs(a - b) <= 1e-12:
            continue
        moves.append(
            {
                "gene_idx": int(idx),
                "gene": str(name),
                "from": int(a) if is_int else float(a),
                "to": int(b) if is_int else float(b),
                "step": float(step),
            }
        )
    return moves


def _rank_key(metrics: dict) -> tuple:
    # Higher is better by staged runner priorities.
    return staged_runner._priority_tuple(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted full-metric sweep around the incumbent checkpoint.")
    parser.add_argument("--alpha-focus", choices=("on", "off"), default="off")
    parser.add_argument("--genes", type=str, default=",".join(DEFAULT_FOCUS_GENES))
    parser.add_argument("--combo-budget", type=int, default=12, help="How many 2-move combos to evaluate.")
    parser.add_argument(
        "--combo-anchor-labels",
        type=str,
        default="",
        help="Comma-separated single-move labels that should be force-paired in combo search even if they fail WR guardrails.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="local_fullmetric_sweep_result.json")
    args = parser.parse_args()

    os.environ["GA_ALPHA_FOCUS"] = str(args.alpha_focus).lower().strip()

    t0 = time.time()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    rng = np.random.default_rng(int(args.seed))

    store, window_plans, seed_genome, seed_fit, baseline_genome, _baseline_fit, baseline_info = (
        staged_runner.open_existing_store()
    )
    cache = staged_runner.build_ticker_arrays_cache(store)

    base_metrics = staged_runner.evaluate_genome_full(baseline_genome, cache, gr)
    seed_metrics = staged_runner.evaluate_genome_full(seed_genome, cache, gr)

    gene_names = [g.strip() for g in str(args.genes).split(",") if g.strip()]
    anchor_labels = [x.strip() for x in str(args.combo_anchor_labels).split(",") if x.strip()]
    anchor_label_set = set(anchor_labels)
    focus = []
    for name in gene_names:
        idx = _spec_index_by_name(name)
        spec = gr.GLOBAL_PARAM_SPECS[idx]
        focus.append((name, idx, float(spec[3]), bool(spec[4])))

    # Step multipliers: allow slightly larger jumps for stop/time controls.
    # Genes em boundary (v5 entry_aggressiveness=1.0 max, v7 *_gate=0.0 min)
    # precisam de mais pontos "para dentro" do range — default (-1,+1) gera
    # apenas 1 movimento util (o outro eh clipado pelo sanitize).
    multipliers_by_gene = {
        "stop_atr_mult": (-2, -1, 1, 2),
        "time_stop_bars": (-3, -2, -1, 1, 2, 3),
        "stop_tighten_after_bars": (-2, -1, 1, 2),
        "reward_risk_ratio": (-1, 1, 2),
        # v5: current=1.0 (lenient), range 0.0-1.0 step 0.1 — so moves sao negativos
        "entry_aggressiveness": (-1, -2, -3, -5, -10),
        # v7: current=0.0 (off), range 0.0-2.0 step 0.25 — so moves positivos.
        # Multipliers (1,2,4,6,8) geram 0.25, 0.50, 1.00, 1.50, 2.00 (max).
        # Com ATR-based, esses valores realmente acionam o gate.
        "resistance_overext_gate": (1, 2, 4, 6, 8),
        "support_broken_gate": (1, 2, 4, 6, 8),
    }

    candidates: list[dict] = []
    seen = set()

    seed_sanitized = gr.sanitize_global_genome(list(seed_genome))
    seed_key = tuple(round(float(x), 12) for x in seed_sanitized)
    seen.add(seed_key)

    def _add_candidate(genome: list[float], label: str, parent_label: str | None = None) -> None:
        gg = gr.sanitize_global_genome(list(genome))
        key = tuple(round(float(x), 12) for x in gg)
        if key in seen:
            return
        seen.add(key)
        candidates.append({"label": str(label), "parent": parent_label, "genome": gg})

    # Single-gene moves.
    for name, idx, step, is_int in focus:
        multipliers = multipliers_by_gene.get(name, (-1, 1))
        for mult in multipliers:
            cand = list(seed_sanitized)
            cand[idx] = float(cand[idx]) + float(step) * float(mult)
            _add_candidate(cand, label=f"{name}:{mult:+d}")

    # Evaluate single moves (and keep the best by priority among those that satisfy guardrails).
    single_results: list[dict] = []
    best_promoted = None
    best_guarded = None

    for i, cand in enumerate(candidates, start=1):
        genome = cand["genome"]
        t_eval0 = time.time()
        metrics = staged_runner.evaluate_genome_full(genome, cache, gr)
        metrics["eval_time_s"] = round(time.time() - t_eval0, 1)
        decision = staged_runner._candidate_beats_incumbent(metrics, base_metrics, seed_metrics)
        payload = {
            "label": cand["label"],
            "parent": cand.get("parent"),
            "moves": _diff_moves(seed_sanitized, genome),
            "metrics": metrics,
            "decision": decision,
        }
        single_results.append(payload)

        if decision.get("promote"):
            if best_promoted is None or _rank_key(metrics) > _rank_key(best_promoted["metrics"]):
                best_promoted = payload
        guarded_ok = bool(
            decision.get("incumbent_wr_guardrail", {}).get("all_pass", False)
            and decision.get("incumbent_safety_guardrail", {}).get("all_pass", False)
        )
        if guarded_ok:
            if best_guarded is None or _rank_key(metrics) > _rank_key(best_guarded["metrics"]):
                best_guarded = payload

        if i % 4 == 0 or decision.get("promote"):
            print(
                f"[EVAL {i:02d}/{len(candidates):02d}] {cand['label']:<26} "
                f"fit={metrics.get('fit', float('nan')):>7.3f} "
                f"WR={_fmt_pct1(float(metrics.get('wr_med_all', 0.0) or 0.0))} "
                f"WR_tgt={_fmt_pct1(float(metrics.get('wr_target_med_all', 0.0) or 0.0))} "
                f"alpha={_fmt_pct1(float(metrics.get('mean_alpha_ann', 0.0) or 0.0))} "
                f"MDD={_fmt_pct1(float(metrics.get('mdd_med', 0.0) or 0.0))} "
                f"promote={bool(decision.get('promote'))}",
                flush=True,
            )

    # 2-move combos from the best guarded single moves.
    combo_results: list[dict] = []
    if best_promoted is None and int(args.combo_budget) > 0:
        guarded_sorted = sorted(
            [r for r in single_results if r["decision"].get("incumbent_wr_guardrail", {}).get("all_pass", False)],
            key=lambda r: _rank_key(r["metrics"]),
            reverse=True,
        )
        top_for_combo = guarded_sorted[: min(8, len(guarded_sorted))]
        anchor_sorted = sorted(
            [r for r in single_results if r["label"] in anchor_label_set],
            key=lambda r: _rank_key(r["metrics"]),
            reverse=True,
        )
        anchored_pairs = []
        regular_pairs = []
        seen_pairs = set()

        def _append_pair(target: list[tuple[dict, dict]], left: dict, right: dict) -> None:
            pair_key = tuple(sorted((str(left["label"]), str(right["label"]))))
            if left["label"] == right["label"] or pair_key in seen_pairs:
                return
            seen_pairs.add(pair_key)
            target.append((left, right))

        for anchor in anchor_sorted:
            for partner in top_for_combo:
                _append_pair(anchored_pairs, anchor, partner)

        for i in range(len(top_for_combo)):
            for j in range(i + 1, len(top_for_combo)):
                _append_pair(regular_pairs, top_for_combo[i], top_for_combo[j])
        rng.shuffle(regular_pairs)
        combo_pairs = (anchored_pairs + regular_pairs)[: int(args.combo_budget)]

        for k, (a, b) in enumerate(combo_pairs, start=1):
            cand = list(seed_sanitized)
            for move in a["moves"] + b["moves"]:
                cand[int(move["gene_idx"])] = float(move["to"])
            cand = gr.sanitize_global_genome(cand)
            key = tuple(round(float(x), 12) for x in cand)
            if key in seen:
                continue
            seen.add(key)
            t_eval0 = time.time()
            metrics = staged_runner.evaluate_genome_full(cand, cache, gr)
            metrics["eval_time_s"] = round(time.time() - t_eval0, 1)
            decision = staged_runner._candidate_beats_incumbent(metrics, base_metrics, seed_metrics)
            payload = {
                "label": f"combo:{a['label']}+{b['label']}",
                "moves": _diff_moves(seed_sanitized, cand),
                "metrics": metrics,
                "decision": decision,
                "parents": [a["label"], b["label"]],
            }
            combo_results.append(payload)
            if decision.get("promote"):
                if best_promoted is None or _rank_key(metrics) > _rank_key(best_promoted["metrics"]):
                    best_promoted = payload
            print(
                f"[COMBO {k:02d}/{len(combo_pairs):02d}] "
                f"fit={metrics.get('fit', float('nan')):>7.3f} "
                f"WR={_fmt_pct1(float(metrics.get('wr_med_all', 0.0) or 0.0))} "
                f"WR_tgt={_fmt_pct1(float(metrics.get('wr_target_med_all', 0.0) or 0.0))} "
                f"alpha={_fmt_pct1(float(metrics.get('mean_alpha_ann', 0.0) or 0.0))} "
                f"MDD={_fmt_pct1(float(metrics.get('mdd_med', 0.0) or 0.0))} "
                f"promote={bool(decision.get('promote'))}",
                flush=True,
            )
            if best_promoted is not None:
                break

    payload = {
        "started_at": started_at,
        "method": "fullmetric_sweep",
        "alpha_focus": str(args.alpha_focus),
        "baseline_info": baseline_info,
        "seed_fit": float(seed_fit),
        "focus_genes": gene_names,
        "single_count": int(len(single_results)),
        "combo_count": int(len(combo_results)),
        "combo_anchor_labels": anchor_labels,
        "base_metrics": base_metrics,
        "seed_metrics": seed_metrics,
        "single_results": single_results,
        "combo_results": combo_results,
        "best_promoted": best_promoted,
        "best_guarded": best_guarded,
        "elapsed_s": round(time.time() - t0, 1),
    }
    _atomic_dump(str(args.out), payload)

    print("\n============================================================", flush=True)
    print("SWEEP SUMMARY", flush=True)
    print("============================================================", flush=True)
    print(
        f"baseline={baseline_info.get('source')} ref={baseline_info.get('ref')} "
        f"fit={base_metrics.get('fit', float('nan')):.4f} WR={_fmt_pct1(base_metrics.get('wr_med_all', 0.0))} "
        f"WR_tgt={_fmt_pct1(base_metrics.get('wr_target_med_all', 0.0))}",
        flush=True,
    )
    print(
        f"incumbent=worktree fit={seed_metrics.get('fit', float('nan')):.4f} "
        f"WR={_fmt_pct1(seed_metrics.get('wr_med_all', 0.0))} "
        f"WR_tgt={_fmt_pct1(seed_metrics.get('wr_target_med_all', 0.0))}",
        flush=True,
    )
    if best_promoted is not None:
        bm = best_promoted["metrics"]
        print(
            f"PROMOTABLE found: fit={bm.get('fit', float('nan')):.4f} "
            f"WR={_fmt_pct1(bm.get('wr_med_all', 0.0))} "
            f"WR_tgt={_fmt_pct1(bm.get('wr_target_med_all', 0.0))} "
            f"alpha={_fmt_pct1(bm.get('mean_alpha_ann', 0.0))} "
            f"MDD={_fmt_pct1(bm.get('mdd_med', 0.0))} "
            f"moves={len(best_promoted.get('moves', []))}",
            flush=True,
        )
    else:
        print("No promotable candidate found.", flush=True)
    print(f"Wrote {os.path.abspath(str(args.out))}", flush=True)


if __name__ == "__main__":
    main()

