#!/usr/bin/env python3
"""OOS-aware targeted sweep for swing Premium/HQ candidates."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

os.environ["GA_ALPHA_FOCUS"] = os.environ.get("GA_ALPHA_FOCUS", "on")

import ga_run as gr
import run_local_ga_staged as staged_runner
from analysis.holdout_operability_diagnostic import _slice_by_index, _ticker_stat, stat_view
from analysis.recent_operability import recent_summary_from_views


CACHE_SCHEMA_VERSION = 2

DEFAULT_FOCUS_GENES = [
    "min_hold_bars",
    "early_exit_block_bars",
    "profit_take_min_bars",
    "early_take_quality_gate",
    "entry_score_threshold",
    "min_signal_strength",
    "score_percentile_trigger",
    "vol_regime_mode",
    "ma_filter_mode",
    "resistance_overext_gate",
    "support_broken_gate",
]


def _spec_index_by_name(name: str) -> int:
    for idx, (spec_name, _lo, _hi, _step, _is_int) in enumerate(gr.GLOBAL_PARAM_SPECS):
        if spec_name == name:
            return int(idx)
    raise KeyError(f"Unknown gene name: {name}")


def _fmt_pct1(x: float) -> str:
    return f"{float(x) * 100.0:.1f}%"


def _atomic_dump(path: str | Path, payload: dict) -> None:
    path = Path(path).resolve()
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{int(time.time() * 1000)}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _diff_moves(seed_genome: list[float], cand_genome: list[float]) -> list[dict]:
    moves = []
    for idx, (name, _lo, _hi, step, is_int) in enumerate(gr.GLOBAL_PARAM_SPECS):
        a = float(seed_genome[idx])
        b = float(cand_genome[idx])
        if abs(a - b) <= 1e-12:
            continue
        moves.append({
            "gene_idx": int(idx),
            "gene": str(name),
            "from": int(a) if is_int else float(a),
            "to": int(b) if is_int else float(b),
            "step": float(step),
        })
    return moves


def _metric(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(metrics.get(key, default) or default)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def oos_gate(
    train_metrics: dict,
    holdout_metrics: dict,
    reference_full_metrics: dict,
    *,
    min_holdout_operable: int = 5,
    min_wr_operable: float = 0.60,
    min_wr_after_3: float = 0.60,
    min_expected_hold_score: float = 70.0,
    alpha_tolerance: float = 0.005,
) -> dict:
    """Fail-closed promotion gate for recent OOS operability."""
    ref_alpha = _metric(reference_full_metrics, "alpha_ann_mean_operable", _metric(reference_full_metrics, "mean_alpha_ann", 0.0))
    hold_n = int(_metric(holdout_metrics, "n_buy_operable", 0.0))
    hold_wr = _metric(holdout_metrics, "wr_med_operable")
    hold_wr_target = _metric(holdout_metrics, "wr_target_med_operable")
    hold_alpha = _metric(holdout_metrics, "alpha_ann_mean_operable")
    hold_expectancy = _metric(holdout_metrics, "expectancy_net_med_operable")
    hold_med = _metric(holdout_metrics, "hold_med_operable")
    hold_max_p75 = _metric(holdout_metrics, "max_hold_bars_p75_operable")
    hold_expected = _metric(holdout_metrics, "expected_hold_score_med_operable")
    hold_wr3 = _metric(holdout_metrics, "wr_after_3_bars_med_operable")
    train_n = int(_metric(train_metrics, "n_buy_operable", 0.0))

    checks = {
        "holdout_coverage_ok": hold_n >= int(min_holdout_operable),
        "train_coverage_ok": train_n >= int(min_holdout_operable),
        "holdout_wr_ok": hold_wr >= float(min_wr_operable),
        "holdout_wr_target_ok": hold_wr_target >= 0.55,
        "holdout_alpha_ok": hold_alpha >= ref_alpha - float(alpha_tolerance),
        "holdout_expectancy_ok": hold_expectancy >= 0.0,
        "holdout_swing_hold_ok": 3.0 <= hold_med <= 10.0 and hold_max_p75 <= 20.0,
        "holdout_expected_hold_ok": hold_expected >= float(min_expected_hold_score),
        "holdout_wr_after_3_ok": hold_wr3 >= float(min_wr_after_3),
    }
    checks["all_pass"] = bool(all(checks.values()))
    checks.update({
        "n_buy_operable_holdout": hold_n,
        "n_buy_operable_train": train_n,
        "wr_med_operable_holdout": hold_wr,
        "wr_target_med_operable_holdout": hold_wr_target,
        "alpha_ann_mean_operable_holdout": hold_alpha,
        "expectancy_net_med_operable_holdout": hold_expectancy,
        "hold_med_operable_holdout": hold_med,
        "max_hold_bars_p75_operable_holdout": hold_max_p75,
        "expected_hold_score_med_operable_holdout": hold_expected,
        "wr_after_3_bars_med_operable_holdout": hold_wr3,
        "reference_alpha_floor": ref_alpha - float(alpha_tolerance),
    })
    return checks


def recent_gate(
    recent_summary: dict,
    *,
    min_recent_ready: int = 1,
    min_recent_watch: int = 3,
    min_recent_score: float = 60.0,
) -> dict:
    """Soft OOS triage gate.

    This gate ranks candidates for the next sweep/debug cycle. Promotion still
    requires the hard OOS gate in ``oos_gate``.
    """
    n_ready = int(_metric(recent_summary, "n_ready", 0.0))
    n_watch = int(_metric(recent_summary, "n_watch", 0.0))
    score_p75 = _metric(recent_summary, "score_p75", 0.0)
    checks = {
        "recent_coverage_ok": bool(n_ready >= int(min_recent_ready) or n_watch >= int(min_recent_watch)),
        "recent_score_ok": bool(score_p75 >= float(min_recent_score)),
        "n_recent_ready": n_ready,
        "n_recent_watch": n_watch,
        "recent_score_p75": score_p75,
        "recent_score_median": _metric(recent_summary, "score_median", 0.0),
        "recent_score_max": _metric(recent_summary, "score_max", 0.0),
    }
    checks["all_pass"] = bool(checks["recent_coverage_ok"] and checks["recent_score_ok"])
    return checks


def _rank_oos(
    full_metrics: dict,
    train_metrics: dict,
    holdout_metrics: dict,
    gate: dict,
    recent_gate_result: dict | None = None,
) -> tuple:
    recent_gate_result = recent_gate_result or {}
    return (
        int(bool(gate.get("all_pass"))),
        int(bool(recent_gate_result.get("all_pass"))),
        int(recent_gate_result.get("n_recent_ready", 0) or 0),
        int(recent_gate_result.get("n_recent_watch", 0) or 0),
        round(_metric(recent_gate_result, "recent_score_p75"), 6),
        int(gate.get("n_buy_operable_holdout", 0) or 0),
        int(bool(gate.get("holdout_swing_hold_ok"))),
        round(_metric(holdout_metrics, "wr_med_operable"), 6),
        round(_metric(holdout_metrics, "wr_after_3_bars_med_operable"), 6),
        round(_metric(holdout_metrics, "wr_target_med_operable"), 6),
        round(_metric(holdout_metrics, "alpha_ann_mean_operable"), 6),
        round(_metric(holdout_metrics, "expectancy_net_med_operable"), 6),
        staged_runner._priority_tuple(full_metrics),
    )


def _snapshot_metrics(metrics: dict) -> dict:
    keys = (
        "fit",
        "wr_med_all",
        "wr_target_med_all",
        "wr_med_operable",
        "wr_target_med_operable",
        "alpha_ann_mean_operable",
        "expectancy_net_med_operable",
        "mdd_med_operable",
        "hold_med_operable",
        "max_hold_bars_p75_operable",
        "expected_hold_score_med_operable",
        "wr_after_3_bars_med_operable",
        "target_wr_after_3_bars_med_operable",
        "n_buy_operable",
        "trades_med",
    )
    out = {}
    for key in keys:
        value = metrics.get(key, 0)
        out[key] = int(value) if key == "n_buy_operable" else float(value or 0.0)
    return out


def _recent_holdout_summary(genome: list[float], holdout_cache: dict, args) -> dict:
    if str(getattr(args, "recent_eval", "on")).lower().strip() == "off":
        return {
            "n_evaluated": 0,
            "n_ready": 0,
            "n_watch": 0,
            "tier_counts": {},
            "score_median": 0.0,
            "score_p75": 0.0,
            "score_max": 0.0,
            "top_recent_candidates": [],
            "errors": 0,
            "recent_eval": "off",
        }
    views: list[dict] = []
    errors = 0
    max_tickers = int(getattr(args, "recent_max_tickers", 0) or 0)
    items = sorted(holdout_cache.items())
    if max_tickers > 0:
        items = items[:max_tickers]
    for ticker, payload in items:
        try:
            stat = _ticker_stat(genome, payload, gr)
            view = stat_view(stat, gr)
            view["ticker"] = ticker
            views.append(view)
        except Exception:
            errors += 1
    summary = recent_summary_from_views(views)
    summary["errors"] = int(errors)
    summary["recent_eval"] = "on"
    summary["recent_max_tickers"] = int(max_tickers)
    return summary


def _evaluate(genome: list[float], cache: dict, train_cache: dict, holdout_cache: dict, reference_metrics: dict, args) -> dict:
    t0 = time.time()
    full_metrics = staged_runner.evaluate_genome_full(genome, cache, gr)
    train_metrics = staged_runner.evaluate_genome_full(genome, train_cache, gr)
    holdout_metrics = staged_runner.evaluate_genome_full(genome, holdout_cache, gr)
    gate = oos_gate(
        train_metrics,
        holdout_metrics,
        reference_metrics,
        min_holdout_operable=int(args.min_holdout_operable),
        min_wr_operable=float(args.min_wr_operable),
        min_wr_after_3=float(args.min_wr_after_3),
        min_expected_hold_score=float(args.min_expected_hold_score),
        alpha_tolerance=float(args.alpha_tolerance),
    )
    recent_summary = _recent_holdout_summary(genome, holdout_cache, args)
    recent_gate_result = recent_gate(
        recent_summary,
        min_recent_ready=int(args.min_recent_ready),
        min_recent_watch=int(args.min_recent_watch),
        min_recent_score=float(args.min_recent_score),
    )
    staged_decision = staged_runner._candidate_beats_incumbent(full_metrics, reference_metrics, reference_metrics)
    return {
        "metrics": full_metrics,
        "metrics_train": train_metrics,
        "metrics_holdout": holdout_metrics,
        "oos_gate": gate,
        "recent_holdout": recent_summary,
        "recent_gate": recent_gate_result,
        "staged_decision": staged_decision,
        "eval_time_s": round(time.time() - t0, 1),
    }


def _candidate_moves(seed_sanitized: list[float], gene_names: list[str]) -> list[dict]:
    multipliers_by_gene = {
        "stop_atr_mult": (-2, -1, 1, 2),
        "time_stop_bars": (-3, -2, -1, 1, 2, 3),
        "min_hold_bars": (1, 2, 3, 4, 5),
        "early_exit_block_bars": (1, 2, 3, 4, 5),
        "profit_take_min_bars": (1, 2, 3, 4, 5),
        "early_take_quality_gate": (2, 3, 4, 5, 6, 7, 8),
        "early_take_score_decay_max": (-4, -2, -1, 1, 2, 4, 6),
        "reward_risk_ratio": (-1, 1, 2),
        "entry_aggressiveness": (-1, -2, -3, -5, -10),
        "resistance_overext_gate": (1, 2, 4, 6, 8),
        "support_broken_gate": (1, 2, 4, 6, 8),
    }
    candidates: list[dict] = []
    seen = {tuple(round(float(x), 12) for x in seed_sanitized)}
    for name in gene_names:
        idx = _spec_index_by_name(name)
        _spec_name, _lo, _hi, step, _is_int = gr.GLOBAL_PARAM_SPECS[idx]
        for mult in multipliers_by_gene.get(name, (-1, 1)):
            cand = list(seed_sanitized)
            cand[idx] = float(cand[idx]) + float(step) * float(mult)
            cand = gr.sanitize_global_genome(cand)
            key = tuple(round(float(x), 12) for x in cand)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"label": f"{name}:{mult:+d}", "genome": cand})
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="OOS-aware full metric sweep.")
    parser.add_argument("--alpha-focus", choices=("on", "off"), default="on")
    parser.add_argument("--genes", type=str, default=",".join(DEFAULT_FOCUS_GENES))
    parser.add_argument("--combo-budget", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--combo-anchor-labels", type=str, default="")
    parser.add_argument("--holdout-days", type=int, default=90)
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--min-holdout-operable", type=int, default=5)
    parser.add_argument("--min-wr-operable", type=float, default=0.60)
    parser.add_argument("--min-wr-after-3", type=float, default=0.60)
    parser.add_argument("--min-expected-hold-score", type=float, default=70.0)
    parser.add_argument("--alpha-tolerance", type=float, default=0.005)
    parser.add_argument("--recent-eval", choices=("on", "off"), default="on")
    parser.add_argument("--recent-max-tickers", type=int, default=0)
    parser.add_argument("--min-recent-ready", type=int, default=1)
    parser.add_argument("--min-recent-watch", type=int, default=3)
    parser.add_argument("--min-recent-score", type=float, default=60.0)
    parser.add_argument("--print-every", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="local_fullmetric_sweep_oos_result.json")
    args = parser.parse_args()

    os.environ["GA_ALPHA_FOCUS"] = str(args.alpha_focus).lower().strip()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    rng = np.random.default_rng(int(args.seed))

    store, _window_plans, seed_genome, seed_fit, baseline_genome, _baseline_fit, baseline_info = (
        staged_runner.open_existing_store()
    )
    cache = staged_runner.build_ticker_arrays_cache(store)
    train_cache, holdout_cache = _slice_by_index(cache, int(args.holdout_days), int(args.lookback_days))

    seed_sanitized = gr.sanitize_global_genome(list(seed_genome))
    base_metrics = staged_runner.evaluate_genome_full(baseline_genome, cache, gr)
    seed_metrics = staged_runner.evaluate_genome_full(seed_sanitized, cache, gr)

    # Keep the same best-ever seed switch as the classic sweep.
    try:
        from analysis.best_ever_guard import load_best_ever
        best_ever = load_best_ever()
        if best_ever and best_ever.get("last_best_genome"):
            best_genome = gr.sanitize_global_genome(list(best_ever["last_best_genome"]))
            best_metrics = staged_runner.evaluate_genome_full(best_genome, cache, gr)
            if best_metrics.get("fit", 0.0) > seed_metrics.get("fit", 0.0):
                print(
                    f"[OOS-SWEEP] using best_ever seed fit={best_metrics.get('fit', 0.0):.4f} "
                    f"over seed fit={seed_metrics.get('fit', 0.0):.4f}",
                    flush=True,
                )
                seed_sanitized = best_genome
                seed_metrics = best_metrics
    except Exception as exc:
        print(f"[OOS-SWEEP] WARN best_ever switch failed: {exc}", flush=True)

    reference_metrics = seed_metrics
    gene_names = [g.strip() for g in str(args.genes).split(",") if g.strip()]
    anchor_labels = [x.strip() for x in str(args.combo_anchor_labels).split(",") if x.strip()]

    candidates = _candidate_moves(seed_sanitized, gene_names)
    if int(args.max_candidates) > 0:
        candidates = candidates[: int(args.max_candidates)]
    single_results: list[dict] = []
    best_promoted = None
    best_oos_guarded = None
    best_recent_guarded = None

    def _maybe_update_best(payload: dict) -> None:
        nonlocal best_promoted, best_oos_guarded, best_recent_guarded
        gate = payload["oos_gate"]
        recent = payload.get("recent_gate", {})
        metrics = payload["metrics"]
        rank = _rank_oos(metrics, payload["metrics_train"], payload["metrics_holdout"], gate, recent)
        if recent.get("all_pass"):
            if best_recent_guarded is None or rank > best_recent_guarded["_rank"]:
                best_recent_guarded = dict(payload, _rank=rank)
        if gate.get("all_pass"):
            if best_oos_guarded is None or rank > best_oos_guarded["_rank"]:
                best_oos_guarded = dict(payload, _rank=rank)
            if payload["staged_decision"].get("promote"):
                if best_promoted is None or rank > best_promoted["_rank"]:
                    best_promoted = dict(payload, _rank=rank)

    for idx, cand in enumerate(candidates, start=1):
        data = _evaluate(cand["genome"], cache, train_cache, holdout_cache, reference_metrics, args)
        payload = {
            "label": cand["label"],
            "moves": _diff_moves(seed_sanitized, cand["genome"]),
            **data,
        }
        single_results.append(payload)
        _maybe_update_best(payload)
        gate = payload["oos_gate"]
        recent = payload.get("recent_gate", {})
        if idx % max(1, int(args.print_every)) == 0 or gate.get("all_pass"):
            hm = payload["metrics_holdout"]
            print(
                f"[OOS {idx:02d}/{len(candidates):02d}] {cand['label']:<28} "
                f"n_ho={gate.get('n_buy_operable_holdout', 0):>2} "
                f"ready={recent.get('n_recent_ready', 0):>2} "
                f"watch={recent.get('n_recent_watch', 0):>2} "
                f"r75={recent.get('recent_score_p75', 0.0):>5.1f} "
                f"WR_ho={_fmt_pct1(hm.get('wr_med_operable', 0.0))} "
                f"WR3_ho={_fmt_pct1(hm.get('wr_after_3_bars_med_operable', 0.0))} "
                f"hold_ho={hm.get('hold_med_operable', 0.0):.1f} "
                f"alpha_ho={_fmt_pct1(hm.get('alpha_ann_mean_operable', 0.0))} "
                f"recent={recent.get('all_pass')} oos={gate.get('all_pass')} staged={payload['staged_decision'].get('promote')}",
                flush=True,
            )

    combo_results: list[dict] = []
    if int(args.combo_budget) > 0 and best_promoted is None:
        ranked = sorted(
            single_results,
            key=lambda r: _rank_oos(
                r["metrics"],
                r["metrics_train"],
                r["metrics_holdout"],
                r["oos_gate"],
                r.get("recent_gate", {}),
            ),
            reverse=True,
        )
        anchors = [r for r in single_results if r["label"] in set(anchor_labels)]
        pool = list(dict.fromkeys([id(r) for r in anchors + ranked[:8]]))
        id_to_result = {id(r): r for r in anchors + ranked[:8]}
        pool_results = [id_to_result[i] for i in pool]
        pairs = []
        seen_pairs = set()
        for i in range(len(pool_results)):
            for j in range(i + 1, len(pool_results)):
                key = tuple(sorted((pool_results[i]["label"], pool_results[j]["label"])))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    pairs.append((pool_results[i], pool_results[j]))
        rng.shuffle(pairs)
        for combo_idx, (a, b) in enumerate(pairs[: int(args.combo_budget)], start=1):
            cand = list(seed_sanitized)
            for move in a["moves"] + b["moves"]:
                cand[int(move["gene_idx"])] = float(move["to"])
            cand = gr.sanitize_global_genome(cand)
            data = _evaluate(cand, cache, train_cache, holdout_cache, reference_metrics, args)
            payload = {
                "label": f"combo:{a['label']}+{b['label']}",
                "parents": [a["label"], b["label"]],
                "moves": _diff_moves(seed_sanitized, cand),
                **data,
            }
            combo_results.append(payload)
            _maybe_update_best(payload)
            gate = payload["oos_gate"]
            recent = payload.get("recent_gate", {})
            hm = payload["metrics_holdout"]
            print(
                f"[COMBO-OOS {combo_idx:02d}/{min(len(pairs), int(args.combo_budget)):02d}] "
                f"n_ho={gate.get('n_buy_operable_holdout', 0):>2} "
                f"ready={recent.get('n_recent_ready', 0):>2} "
                f"watch={recent.get('n_recent_watch', 0):>2} "
                f"r75={recent.get('recent_score_p75', 0.0):>5.1f} "
                f"WR_ho={_fmt_pct1(hm.get('wr_med_operable', 0.0))} "
                f"WR3_ho={_fmt_pct1(hm.get('wr_after_3_bars_med_operable', 0.0))} "
                f"hold_ho={hm.get('hold_med_operable', 0.0):.1f} "
                f"alpha_ho={_fmt_pct1(hm.get('alpha_ann_mean_operable', 0.0))} "
                f"recent={recent.get('all_pass')} oos={gate.get('all_pass')} staged={payload['staged_decision'].get('promote')}",
                flush=True,
            )
            if best_promoted is not None:
                break

    def _clean_best(best: dict | None) -> dict | None:
        if best is None:
            return None
        out = dict(best)
        out.pop("_rank", None)
        return out

    payload = {
        "started_at": started_at,
        "method": "fullmetric_sweep_oos",
        "schema_version": CACHE_SCHEMA_VERSION,
        "alpha_focus": str(args.alpha_focus),
        "holdout_days": int(args.holdout_days),
        "lookback_days": int(args.lookback_days),
        "baseline_info": baseline_info,
        "seed_fit": float(seed_fit),
        "focus_genes": gene_names,
        "combo_anchor_labels": anchor_labels,
        "single_count": len(single_results),
        "combo_count": len(combo_results),
        "base_metrics": base_metrics,
        "seed_metrics": seed_metrics,
        "single_results": single_results,
        "combo_results": combo_results,
        "best_recent_guarded": _clean_best(best_recent_guarded),
        "best_oos_guarded": _clean_best(best_oos_guarded),
        "best_promoted": _clean_best(best_promoted),
    }
    _atomic_dump(args.out, payload)
    print("\n============================================================")
    print("OOS SWEEP SUMMARY")
    print("============================================================")
    print(
        f"seed full WR_op={_fmt_pct1(seed_metrics.get('wr_med_operable', 0.0))} "
        f"hold={seed_metrics.get('hold_med_operable', 0.0):.1f} "
        f"alpha_op={_fmt_pct1(seed_metrics.get('alpha_ann_mean_operable', 0.0))}"
    )
    if best_promoted is not None:
        print(f"PROMOTABLE OOS candidate: {best_promoted['label']}")
    elif best_oos_guarded is not None:
        print(f"OOS-guarded but not staged-promotable: {best_oos_guarded['label']}")
    elif best_recent_guarded is not None:
        print(f"Recent-guarded diagnostic candidate: {best_recent_guarded['label']}")
    else:
        print("No OOS/recent guarded candidate found.")
    print(f"Wrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
