"""
Staged local GA runner with full progress visibility.

Designed to:
- Run locally single-process (no OOM)
- Bypass AutoTuner completely (explicit small config)
- Stage in chunks (default 4 generations each)
- Between chunks: compute full metrics (WR/MDD/ret/sharpe), print progress
- Save incremental progress to chunk_progress.json
- Auto-resume from last chunk if killed
- Preserve main checkpoint until improvement is confirmed
- Use v7 WR-biased fitness (already applied to ga_run.py)

Usage:
    python run_local_ga_staged.py                # run 10 chunks from best
    python run_local_ga_staged.py --chunks 5     # run 5 chunks
    python run_local_ga_staged.py --reset        # fresh start (ignore prior progress)
    python run_local_ga_staged.py --apply-best   # if best_found > current, overwrite main checkpoint

Output files (in repo root):
    local_ga_chunk_progress.json      # per-chunk metrics log
    local_ga_best_so_far.json         # best genome found (not applied yet)
    output/ga_checkpoints_local/      # per-generation checkpoints
"""
import os
import sys
import json
import time
import argparse
import shutil
import copy
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

import numpy as np

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
os.environ["GA_AUTO_TUNE"] = "0"
os.environ["GA_RUN_MODE"] = "train"
os.environ["GA_EVAL_WORKERS"] = "1"
os.environ["GA_ALPHA_FOCUS"] = os.environ.get("GA_ALPHA_FOCUS", "on")

# Force stdout to UTF-8 (Windows cp1252 fallback breaks on Δ etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Config
CHUNKS_DEFAULT = 10
NGEN_PER_CHUNK = 4
POP_SIZE = 24
WINDOWS_STAGE = 4
CHECKPOINT_DIR = "output/ga_checkpoints_local"
PROGRESS_FILE = "local_ga_chunk_progress.json"
BEST_SOFAR_FILE = "local_ga_best_so_far.json"
ROUND_STATE_FILE = "local_ga_round_state.json"
MAIN_CHECKPOINT = "global_ga_checkpoint.json"
BACKUP_MAIN = "global_ga_checkpoint_pre_staged.json.bak"


def _fmt_pct(x):
    return f"{x*100:.2f}%"


def _fmt_pct1(x):
    return f"{x*100:.1f}%"


def _atomic_json_dump(path: str, data: Any) -> None:
    """
    Write JSON atomically to avoid corrupting progress/checkpoint artifacts if the
    process is interrupted mid-write.
    """
    path = os.path.abspath(path)
    tmp_path = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _compute_acceptance(chunk_metrics, base_metrics):
    base_wr = float(base_metrics.get("wr_med_all", 0.0) or 0.0)
    base_wr_target = float(base_metrics.get("wr_target_med_all", 0.0) or 0.0)
    base_trades = float(base_metrics.get("trades_med", 0.0) or 0.0)
    trades_floor = 0.70 * base_trades if base_trades > 0 else 0.0
    base_mdd = float(base_metrics.get("mdd_med", 0.0) or 0.0)
    base_mdd_p75 = float(base_metrics.get("mdd_p75", base_mdd) or base_mdd)
    base_mdd_duration = float(base_metrics.get("mdd_duration_med", 0.0) or 0.0)
    base_alpha_pos = float(base_metrics.get("alpha_ann_pos_rate", 0.0) or 0.0)
    wr_delta = float(chunk_metrics.get("wr_med_all", 0.0) or 0.0) - base_wr
    wr_target_delta = float(chunk_metrics.get("wr_target_med_all", 0.0) or 0.0) - base_wr_target
    mean_alpha_delta = float(chunk_metrics.get("mean_alpha_ann", 0.0) or 0.0) - float(base_metrics.get("mean_alpha_ann", 0.0) or 0.0)
    alpha_pos_delta = float(chunk_metrics.get("alpha_ann_pos_rate", 0.0) or 0.0) - base_alpha_pos

    # Main-baseline guardrails:
    # - Keep absolute floors for operational safety.
    # - Avoid requiring +5pp improvements vs `main` once the baseline itself is already strong,
    #   otherwise promotions become mathematically impossible and the optimizer stalls.
    # Tolerances are absolute rate units (e.g., 0.0025 == 0.25pp).
    wr_floor = max(0.60, base_wr - 0.0025)
    wr_target_floor = max(0.55, base_wr_target - 0.005)

    strong_alpha_wr_relax = bool(
        mean_alpha_delta >= 0.008
        and wr_delta >= 0.10
        and wr_target_delta >= 0.10
    )
    alpha_pos_target = max(base_alpha_pos + (0.005 if strong_alpha_wr_relax else 0.008), 0.17)
    alpha_pos_floor = max(0.17, base_alpha_pos - 0.005)

    mdd_relaxation_bonus = 0.0
    if mean_alpha_delta >= 0.004:
        mdd_relaxation_bonus += 0.005
    elif mean_alpha_delta >= 0.002:
        mdd_relaxation_bonus += 0.002
    if wr_delta >= 0.25:
        mdd_relaxation_bonus += 0.003
    if wr_target_delta >= 0.20:
        mdd_relaxation_bonus += 0.002
    if alpha_pos_delta >= 0.015:
        mdd_relaxation_bonus += 0.002
    elif float(chunk_metrics.get("alpha_ann_pos_rate", 0.0) or 0.0) >= alpha_pos_target - 1e-9:
        mdd_relaxation_bonus += 0.001
    mdd_relaxation_bonus = min(mdd_relaxation_bonus, 0.012)

    mdd_tolerance = max(0.04, 0.20 * base_mdd)
    mdd_limit = base_mdd + mdd_tolerance + mdd_relaxation_bonus
    # Keep p75/duration limits for observability, but don't hard-gate promotions on them.
    mdd_p75_limit = base_mdd_p75 + max(0.05, 0.25 * base_mdd_p75) + min(0.010, mdd_relaxation_bonus)
    mdd_duration_limit = base_mdd_duration + max(16.0, 0.25 * base_mdd_duration)

    avg_hold = float(chunk_metrics.get("avg_hold_bars_med", 0.0) or 0.0)
    median_hold = float(chunk_metrics.get("median_hold_bars_med", 0.0) or 0.0)
    max_hold_p75 = float(chunk_metrics.get("max_hold_bars_p75", 0.0) or 0.0)
    swing_hold_ok = bool(
        avg_hold <= 25.0
        and median_hold <= 16.0
        and max_hold_p75 <= 60.0
    )

    acceptance = {
        "wr_med_all_up": bool(float(chunk_metrics.get("wr_med_all", 0.0) or 0.0) > base_wr + 1e-9),
        "wr_target_med_all_up": bool(float(chunk_metrics.get("wr_target_med_all", 0.0) or 0.0) > base_wr_target + 1e-9),
        "wr_med_all_floor": bool(float(chunk_metrics.get("wr_med_all", 0.0) or 0.0) >= wr_floor - 1e-9),
        "wr_target_med_all_floor": bool(float(chunk_metrics.get("wr_target_med_all", 0.0) or 0.0) >= wr_target_floor - 1e-9),
        "mdd_med_not_worse": bool(float(chunk_metrics.get("mdd_med", 0.0) or 0.0) <= mdd_limit + 1e-9),
        "mdd_p75_not_worse": bool(float(chunk_metrics.get("mdd_p75", 0.0) or 0.0) <= mdd_p75_limit + 1e-9),
        "mdd_duration_not_worse": bool(float(chunk_metrics.get("mdd_duration_med", 0.0) or 0.0) <= mdd_duration_limit + 1e-9),
        "mean_alpha_ann_up": bool(float(chunk_metrics.get("mean_alpha_ann", 0.0) or 0.0) > float(base_metrics.get("mean_alpha_ann", 0.0) or 0.0) + 1e-9),
        "mean_alpha_ann_floor": bool(
            float(chunk_metrics.get("mean_alpha_ann", 0.0) or 0.0)
            >= float(base_metrics.get("mean_alpha_ann", 0.0) or 0.0) - 0.002
        ),
        "alpha_ann_pos_rate_up": bool(float(chunk_metrics.get("alpha_ann_pos_rate", 0.0) or 0.0) >= alpha_pos_target - 1e-9),
        "alpha_ann_pos_rate_floor": bool(float(chunk_metrics.get("alpha_ann_pos_rate", 0.0) or 0.0) >= alpha_pos_floor - 1e-9),
        "trades_med_floor": bool(float(chunk_metrics.get("trades_med", 0.0) or 0.0) >= trades_floor - 1e-9),
        "swing_hold_ok": swing_hold_ok,
    }
    acceptance["wr_med_all_floor_target"] = wr_floor
    acceptance["wr_target_med_all_floor_target"] = wr_target_floor
    acceptance["alpha_ann_pos_rate_target"] = alpha_pos_target
    acceptance["alpha_ann_pos_rate_floor_target"] = alpha_pos_floor
    acceptance["mdd_med_limit"] = mdd_limit
    acceptance["mdd_p75_limit"] = mdd_p75_limit
    acceptance["mdd_duration_limit"] = mdd_duration_limit
    acceptance["mdd_relaxation_bonus"] = mdd_relaxation_bonus
    acceptance["strong_alpha_wr_relax"] = strong_alpha_wr_relax
    acceptance["all_pass"] = all(
        acceptance[key]
        for key in (
            "wr_med_all_floor",
            "wr_target_med_all_floor",
            "mdd_med_not_worse",
            "mean_alpha_ann_floor",
            "alpha_ann_pos_rate_floor",
            "trades_med_floor",
        )
    )
    return acceptance


def _compute_incumbent_safety_guardrail(candidate_metrics, incumbent_metrics):
    incumbent_trades = float(incumbent_metrics.get("trades_med", 0.0) or 0.0)
    trades_floor = max(0.90 * incumbent_trades, incumbent_trades - 2.0) if incumbent_trades > 0 else 0.0
    incumbent_mdd = float(incumbent_metrics.get("mdd_med", 0.0) or 0.0)
    incumbent_mdd_p75 = float(incumbent_metrics.get("mdd_p75", incumbent_mdd) or incumbent_mdd)
    incumbent_duration = float(incumbent_metrics.get("mdd_duration_med", 0.0) or 0.0)
    incumbent_duration_p75 = float(
        incumbent_metrics.get("mdd_duration_p75", incumbent_duration) or incumbent_duration
    )

    wr_delta = float(candidate_metrics.get("wr_med_all", 0.0) or 0.0) - float(incumbent_metrics.get("wr_med_all", 0.0) or 0.0)
    wr_target_delta = float(candidate_metrics.get("wr_target_med_all", 0.0) or 0.0) - float(incumbent_metrics.get("wr_target_med_all", 0.0) or 0.0)
    alpha_delta = float(candidate_metrics.get("mean_alpha_ann", 0.0) or 0.0) - float(incumbent_metrics.get("mean_alpha_ann", 0.0) or 0.0)

    risk_relax = 0.0
    if wr_delta >= 0.010 and wr_target_delta >= 0.010 and alpha_delta >= 0.003:
        risk_relax = 0.006
    elif wr_delta >= 0.005 and alpha_delta >= 0.0015:
        risk_relax = 0.003

    mdd_limit = incumbent_mdd + max(0.01, 0.10 * incumbent_mdd) + risk_relax
    mdd_p75_limit = incumbent_mdd_p75 + max(0.012, 0.10 * incumbent_mdd_p75) + min(0.005, risk_relax)
    duration_limit = incumbent_duration + max(6.0, 0.15 * incumbent_duration) + (3.0 if risk_relax > 0.0 else 0.0)
    duration_p75_limit = incumbent_duration_p75 + max(12.0, 0.20 * incumbent_duration_p75) + (
        6.0 if risk_relax > 0.0 else 0.0
    )

    avg_hold = float(candidate_metrics.get("avg_hold_bars_med", 0.0) or 0.0)
    median_hold = float(candidate_metrics.get("median_hold_bars_med", 0.0) or 0.0)
    max_hold_p75 = float(candidate_metrics.get("max_hold_bars_p75", 0.0) or 0.0)
    safety = {
        "trades_ok": bool(float(candidate_metrics.get("trades_med", 0.0) or 0.0) >= trades_floor - 1e-9),
        "mdd_ok": bool(float(candidate_metrics.get("mdd_med", 0.0) or 0.0) <= mdd_limit + 1e-9),
        "mdd_p75_ok": bool(float(candidate_metrics.get("mdd_p75", 0.0) or 0.0) <= mdd_p75_limit + 1e-9),
        "mdd_duration_ok": bool(float(candidate_metrics.get("mdd_duration_med", 0.0) or 0.0) <= duration_limit + 1e-9),
        "mdd_duration_p75_ok": bool(
            float(candidate_metrics.get("mdd_duration_p75", 0.0) or 0.0) <= duration_p75_limit + 1e-9
        ),
        "swing_hold_ok": bool(
            # Only cap extremes: holding profile is a priority tie-breaker, not a hard gate.
            avg_hold <= 25.0
            and median_hold <= 16.0
            and max_hold_p75 <= 60.0
        ),
    }
    safety["trades_floor"] = trades_floor
    safety["mdd_limit"] = mdd_limit
    safety["mdd_p75_limit"] = mdd_p75_limit
    safety["mdd_duration_limit"] = duration_limit
    safety["mdd_duration_p75_limit"] = duration_p75_limit
    safety["risk_relax"] = risk_relax
    safety["all_pass"] = all(
        safety[key]
        for key in (
            "trades_ok",
            "mdd_ok",
            "mdd_p75_ok",
            "mdd_duration_ok",
            "mdd_duration_p75_ok",
            "swing_hold_ok",
        )
    )
    return safety


def _candidate_beats_incumbent(candidate_metrics, base_metrics, incumbent_metrics):
    acceptance = _compute_acceptance(candidate_metrics, base_metrics)
    wr_guardrail = _compute_incumbent_wr_guardrail(candidate_metrics, incumbent_metrics)
    safety_guardrail = _compute_incumbent_safety_guardrail(candidate_metrics, incumbent_metrics)
    priority_better = _is_better_by_priority(candidate_metrics, incumbent_metrics)
    return {
        "acceptance_vs_main": acceptance,
        "incumbent_wr_guardrail": wr_guardrail,
        "incumbent_safety_guardrail": safety_guardrail,
        "priority_better": priority_better,
        "promote": bool(
            acceptance.get("all_pass", False)
            and wr_guardrail.get("all_pass", False)
            and safety_guardrail.get("all_pass", False)
            and priority_better
        ),
    }


def _compute_incumbent_wr_guardrail(candidate_metrics, incumbent_metrics):
    incumbent_wr = float(incumbent_metrics.get("wr_med_all", 0.0) or 0.0)
    incumbent_wr_target = float(incumbent_metrics.get("wr_target_med_all", 0.0) or 0.0)
    # Guardrail: prioritize hit-rate stability. Allow only small regressions vs incumbent.
    # Tolerances are in absolute rate units (e.g., 0.0025 == 0.25pp).
    wr_floor = max(incumbent_wr - 0.0025, 0.62 if incumbent_wr >= 0.62 else max(0.50, incumbent_wr - 0.005))
    wr_target_floor = max(
        incumbent_wr_target - 0.005,
        0.55 if incumbent_wr_target >= 0.55 else max(0.45, incumbent_wr_target - 0.010),
    )
    wr_ok = bool(float(candidate_metrics.get("wr_med_all", 0.0) or 0.0) >= wr_floor - 1e-9)
    wr_target_ok = bool(float(candidate_metrics.get("wr_target_med_all", 0.0) or 0.0) >= wr_target_floor - 1e-9)
    return {
        "wr_floor": wr_floor,
        "wr_target_floor": wr_target_floor,
        "wr_ok": wr_ok,
        "wr_target_ok": wr_target_ok,
        "all_pass": bool(wr_ok and wr_target_ok),
    }


def _priority_tuple(metrics: Dict[str, float]) -> tuple:
    """
    Lexicographic ranking aligned with swing usefulness priorities:
      1) hit rate (WR)
      2) target-hit rate (WR_target)
      3) annualized alpha
      4) manageable drawdown (lower is better)
      5) swing holding profile (median hold closer to ~5 bars)
      6) fitness (tie-breaker)

    Values are rounded to reduce churn on tiny differences.
    """
    wr = float(metrics.get("wr_med_all", 0.0) or 0.0)
    wr_target = float(metrics.get("wr_target_med_all", 0.0) or 0.0)
    alpha_ann = float(metrics.get("mean_alpha_ann", -1e9) or -1e9)
    mdd = float(metrics.get("mdd_med", 1e9) or 1e9)  # lower is better
    hold_med = float(metrics.get("median_hold_bars_med", 0.0) or 0.0)
    hold_score = -abs(hold_med - 5.0)
    fit = float(metrics.get("fit", -1e18) or -1e18)
    return (
        round(wr, 6),
        round(wr_target, 6),
        round(alpha_ann, 6),
        round(-mdd, 6),
        round(hold_score, 6),
        round(fit, 6),
    )


def _is_better_by_priority(candidate: Dict[str, float], incumbent: Dict[str, float]) -> bool:
    """
    Priority comparator with small tolerance bands to avoid promoting churn:
    - Treat WR changes within ~0.25pp as "tie", then compare WR_target (0.5pp),
      then alpha_ann (0.10pp), then MDD (0.25pp), then hold profile, then fitness.
    """
    wr_tol = 0.0025
    wr_target_tol = 0.005
    alpha_tol = 0.001
    mdd_tol = 0.0025
    hold_tol = 0.5

    c_wr = float(candidate.get("wr_med_all", 0.0) or 0.0)
    i_wr = float(incumbent.get("wr_med_all", 0.0) or 0.0)
    if c_wr > i_wr + wr_tol:
        return True
    if c_wr < i_wr - wr_tol:
        return False

    c_wrt = float(candidate.get("wr_target_med_all", 0.0) or 0.0)
    i_wrt = float(incumbent.get("wr_target_med_all", 0.0) or 0.0)
    if c_wrt > i_wrt + wr_target_tol:
        return True
    if c_wrt < i_wrt - wr_target_tol:
        return False

    c_alpha = float(candidate.get("mean_alpha_ann", -1e9) or -1e9)
    i_alpha = float(incumbent.get("mean_alpha_ann", -1e9) or -1e9)
    if c_alpha > i_alpha + alpha_tol:
        return True
    if c_alpha < i_alpha - alpha_tol:
        return False

    c_mdd = float(candidate.get("mdd_med", 1e9) or 1e9)
    i_mdd = float(incumbent.get("mdd_med", 1e9) or 1e9)
    if c_mdd < i_mdd - mdd_tol:
        return True
    if c_mdd > i_mdd + mdd_tol:
        return False

    c_hold = float(candidate.get("median_hold_bars_med", 0.0) or 0.0)
    i_hold = float(incumbent.get("median_hold_bars_med", 0.0) or 0.0)
    c_hold_dist = abs(c_hold - 5.0)
    i_hold_dist = abs(i_hold - 5.0)
    if c_hold_dist < i_hold_dist - hold_tol:
        return True
    if c_hold_dist > i_hold_dist + hold_tol:
        return False

    c_fit = float(candidate.get("fit", -1e18) or -1e18)
    i_fit = float(incumbent.get("fit", -1e18) or -1e18)
    return bool(c_fit > i_fit + 1e-4)


def _load_round_state():
    if os.path.exists(ROUND_STATE_FILE):
        try:
            with open(ROUND_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "round_index": int(data.get("round_index", 0)),
                "last_window_offset": int(data.get("last_window_offset", 0)),
            }
        except Exception:
            pass
    return {"round_index": 0, "last_window_offset": 0}


def _save_round_state(round_state):
    _atomic_json_dump(ROUND_STATE_FILE, round_state)


def _resolve_round_state(reset, requested_offset, total_windows, window_count):
    round_state = _load_round_state()
    if reset:
        round_state["round_index"] = int(round_state.get("round_index", 0)) + 1
    round_index = int(round_state.get("round_index", 0))
    if requested_offset is not None:
        offset = int(requested_offset)
    elif total_windows <= 0:
        offset = 0
    else:
        stride = max(1, total_windows // max(window_count * 2, 1) + 1)
        offset = int((round_index * stride) % total_windows)
    round_state["last_window_offset"] = int(offset)
    _save_round_state(round_state)
    return round_state, int(offset)


def _load_main():
    """
    Load the baseline checkpoint from git branch `main` when available.
    Falls back to the current worktree file if git/main cannot be read.
    """
    try:
        repo_dir = os.path.abspath(os.getcwd())
        repo_dir_posix = Path(repo_dir).as_posix()

        def _git(args):
            # Mark this worktree as safe for this invocation (avoids dubious ownership errors).
            return subprocess.run(
                ["git", "-c", f"safe.directory={repo_dir}", "-c", f"safe.directory={repo_dir_posix}", *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=repo_dir,
            )

        _git(["rev-parse", "--verify", "main"])
        blob = _git(["show", f"main:{MAIN_CHECKPOINT}"])
        ref_sha = _git(["rev-parse", "main"]).stdout.strip()
        data = json.loads(blob.stdout)
        data["_baseline_source"] = "git:main"
        data["_baseline_ref"] = ref_sha
        return data
    except Exception:
        with open(MAIN_CHECKPOINT, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        data["_baseline_source"] = "worktree"
        data["_baseline_ref"] = os.path.abspath(MAIN_CHECKPOINT)
        return data


def _load_seed_checkpoint():
    with open(MAIN_CHECKPOINT, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    data["_seed_source"] = os.path.abspath(MAIN_CHECKPOINT)
    return data


def _snapshot_metrics(metrics):
    keys = (
        "fit",
        "wr_med_all",
        "wr_target_med_all",
        "mean_alpha_ann",
        "alpha_ann_pos_rate",
        "mdd_med",
        "mdd_p75",
        "mdd_duration_med",
        "avg_hold_bars_med",
        "median_hold_bars_med",
        "max_hold_bars_p75",
        "trades_med",
        "n_ge_70",
    )
    out = {}
    for k in keys:
        if k == "n_ge_70":
            out[k] = int(metrics.get(k, 0) or 0)
        else:
            out[k] = float(metrics.get(k, 0.0) or 0.0)
    return out


def _write_checkpoint_with_metadata(
    best_so_far,
    base_metrics,
    seed_metrics,
    baseline_info,
    acceptance,
    incumbent_guardrail,
    incumbent_safety_guardrail,
    promoted_reason,
):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    day = datetime.now().strftime("%Y-%m-%d")

    backup_dir = os.path.join(".local_ga_backups", "promotions")
    os.makedirs(backup_dir, exist_ok=True)

    with open(MAIN_CHECKPOINT, "r", encoding="utf-8-sig") as f:
        prev_ckpt = json.load(f)

    prev_path = os.path.join(backup_dir, f"global_ga_checkpoint_prev_{ts}.json")
    new_path = os.path.join(backup_dir, f"global_ga_checkpoint_new_{ts}.json")

    shutil.copy(MAIN_CHECKPOINT, prev_path)

    payload = {
        "fitness": float(best_so_far["fit"]),
        "genome": list(best_so_far["genome"]),
        "accepted_from_staged_chunk": int(best_so_far.get("found_at_chunk", -1)),
        "accepted_on": day,
        "accepted_reason": str(promoted_reason),
        "baseline_main_ref": baseline_info.get("ref") if baseline_info.get("source") == "git:main" else None,
        "previous_checkpoint_fitness": float(prev_ckpt.get("fitness", 0.0) or 0.0),
        "previous_checkpoint_fit_current_formula": float(seed_metrics.get("fit", 0.0) or 0.0),
        "metrics_snapshot": {
            "main": _snapshot_metrics(base_metrics),
            "incumbent": _snapshot_metrics(seed_metrics),
            "candidate": _snapshot_metrics(best_so_far["metrics"]),
        },
        "acceptance_vs_main": dict(acceptance),
        "incumbent_wr_guardrail": dict(incumbent_guardrail),
        "incumbent_safety_guardrail": dict(incumbent_safety_guardrail),
    }

    _atomic_json_dump(MAIN_CHECKPOINT, payload)

    shutil.copy(MAIN_CHECKPOINT, new_path)
    return prev_path, new_path


def _load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"chunks": [], "best_so_far": None}


def _save_progress(data):
    _atomic_json_dump(PROGRESS_FILE, data)


def _load_best_so_far():
    if os.path.exists(BEST_SOFAR_FILE):
        try:
            with open(BEST_SOFAR_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_best_so_far(data):
    _atomic_json_dump(BEST_SOFAR_FILE, data)


def _build_seed_best_so_far(seed_genome, seed_metrics):
    return {
        "fit": float(seed_metrics["fit"]),
        "wr_med_all": float(seed_metrics["wr_med_all"]),
        "genome": list(seed_genome),
        "metrics": dict(seed_metrics),
        "found_at_chunk": 0,
    }


def _sync_best_so_far_state(progress, best_so_far):
    progress["best_so_far"] = copy.deepcopy(best_so_far)
    _save_progress(progress)
    _save_best_so_far(best_so_far)


def _same_genome(lhs, rhs, tol=1e-12):
    lhs = list(lhs or [])
    rhs = list(rhs or [])
    if len(lhs) != len(rhs):
        return False
    return all(abs(float(a) - float(b)) <= tol for a, b in zip(lhs, rhs))


def open_existing_store():
    """
    Open the existing PayloadStore built by the main ga_run.py pipeline.
    This ensures score_matrix matches production (proper hybrid feature selection).
    Returns: (store, window_plans, seed_genome, prev_fit)
    """
    import ga_run as gr
    from payload_store import PayloadStore, build_window_plans
    import pandas as pd

    gr.GA_AUTO_TUNE = 0
    gr.FAST_MODE = True

    # Try primary dir first, fallback to staged if main was cleaned
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
        raise RuntimeError(
            f"No existing PayloadStore found in {candidates}. "
            f"Run 'python ga_run.py' once first (it builds the store during Phase 1)."
        )

    print(f"[SETUP] Opening existing PayloadStore at {store_dir}", flush=True)
    store = PayloadStore(store_dir, mode="r")
    print(f"[SETUP] Store loaded: {len(store.tickers)} tickers, "
          f"{store.meta.n_features} features, {store.meta.total_rows} total rows",
          flush=True)

    dmin = store.min_date
    dmax = store.max_date
    print(f"[SETUP] Date range: {dmin.date()} -> {dmax.date()}", flush=True)

    windows = gr.build_temporal_windows(dmin, dmax, train_years=3,
                                        test_months=6, step_months=6)
    if not windows:
        windows = [(
            dmin,
            dmax - pd.Timedelta(days=180),
            dmax - pd.Timedelta(days=179),
            dmax,
        )]
    window_plans = build_window_plans(store, windows)
    print(f"[SETUP] {len(window_plans)} walk-forward windows built", flush=True)

    baseline = _load_main()
    seed_ckpt = _load_seed_checkpoint()
    seed_genome = seed_ckpt.get("genome", []) or baseline.get("genome", [])
    seed_fit = float(seed_ckpt.get("fitness", baseline.get("fitness", 0.0)))
    baseline_genome = baseline.get("genome", [])
    baseline_fit = float(baseline.get("fitness", 0.0))
    print(f"[SETUP] Seed genome fitness: {seed_fit:.4f}", flush=True)
    print(f"[SETUP] Seed source: {seed_ckpt.get('_seed_source')}", flush=True)
    print(f"[SETUP] Baseline source: {baseline.get('_baseline_source')} ({baseline.get('_baseline_ref')})", flush=True)

    return store, window_plans, seed_genome, seed_fit, baseline_genome, baseline_fit, {
        "source": baseline.get("_baseline_source"),
        "ref": baseline.get("_baseline_ref"),
        "fitness": baseline_fit,
        "seed_source": seed_ckpt.get("_seed_source"),
        "seed_fitness": seed_fit,
    }


def evaluate_genome_full(genome, ticker_arrays_cache, gr):
    """
    Run full backtest of a genome across all tickers (single process).
    Returns dict with WR_med, WR_mean, MDD_med, ret_med, sharpe_med, fit, and
    the new P0 metric medians used for staged reporting.
    """
    gp = gr.decode_global_params(genome)
    stats = []
    for tk, p in ticker_arrays_cache.items():
        st = gr.backtest_stats_global_intraday(
            p["open"], p["high"], p["low"], p["close"],
            p["score_matrix"], p["atr"], gp,
            precomputed=p,
        )
        gr._attach_benchmark_stats(st, p["close"])
        st["excess_return"] = st["total_return"] - st["buy_hold_return"]
        stats.append(st)

    # Overall fitness uses ALL stats (same as training)
    fit = gr.global_fitness_from_stats(stats)

    # Filtered metrics for reporting (n_trades > 5)
    active = [s for s in stats if s.get("n_trades", 0) > 5]
    all_stats = stats  # unfiltered
    wrs = [s["win_rate"] for s in active]
    wr_targets = [s.get("win_rate_target", 0) for s in active]  # v8
    mdds = [abs(s["mdd"]) for s in active]
    rets = [s["total_return"] for s in active]
    sharpes = [s["sharpe"] for s in active]
    sortinos = [s.get("sortino", 0.0) for s in active]
    profit_factors = [s.get("profit_factor", 0.0) for s in active]
    expectancies = [s.get("expectancy", 0.0) for s in active]
    payoff_ratios = [s.get("payoff_ratio", 0.0) for s in active]
    mdd_durations = [s.get("max_drawdown_duration", 0.0) for s in active]
    cagrs = [s.get("cagr", 0.0) for s in active]
    avg_hold_bars = [s.get("avg_hold_bars", 0.0) for s in active]
    median_hold_bars = [s.get("median_hold_bars", 0.0) for s in active]
    max_hold_bars = [s.get("max_hold_bars", 0.0) for s in active]
    trades = [s["n_trades"] for s in active]

    all_wrs = [s["win_rate"] for s in all_stats]
    all_wr_targets = [s.get("win_rate_target", 0) for s in all_stats]  # v8
    all_excess = [s.get("excess_return", 0.0) for s in all_stats]
    all_alpha_ann = [s.get("alpha_ann", 0.0) for s in all_stats]
    active_alpha_ann = [s.get("alpha_ann", 0.0) for s in active]
    active_bh_cagrs = [s.get("buy_hold_cagr", 0.0) for s in active]

    # Also unfiltered median (matches what summary shows: no n_trades > 5 filter)
    return {
        "fit": float(fit),
        "n_tickers_total": len(all_stats),
        "n_tickers_active": len(active),
        "wr_med_all": float(np.median(all_wrs)) if all_wrs else 0,
        "wr_mean_all": float(np.mean(all_wrs)) if all_wrs else 0,
        "wr_med": float(np.median(wrs)) if wrs else 0,
        "wr_mean": float(np.mean(wrs)) if wrs else 0,
        "wr_p75": float(np.percentile(wrs, 75)) if wrs else 0,
        # v8: win_rate_target metrics
        "wr_target_med_all": float(np.median(all_wr_targets)) if all_wr_targets else 0,
        "wr_target_mean_all": float(np.mean(all_wr_targets)) if all_wr_targets else 0,
        "wr_target_med_active": float(np.median(wr_targets)) if wr_targets else 0,
        "mean_excess": float(np.mean(all_excess)) if all_excess else 0,
        "med_excess": float(np.median(all_excess)) if all_excess else 0,
        "mean_alpha_ann": float(np.mean(all_alpha_ann)) if all_alpha_ann else 0,
        "med_alpha_ann": float(np.median(all_alpha_ann)) if all_alpha_ann else 0,
        "alpha_ann_pos_rate": float(np.mean(np.asarray(all_alpha_ann, dtype=np.float64) > 0.0)) if all_alpha_ann else 0,
        "mdd_med": float(np.median(mdds)) if mdds else 0,
        "mdd_p75": float(np.percentile(mdds, 75)) if mdds else 0,
        "ret_med": float(np.median(rets)) if rets else 0,
        "sharpe_med": float(np.median(sharpes)) if sharpes else 0,
        "sortino_med": float(np.median(sortinos)) if sortinos else 0,
        "profit_factor_med": float(np.median(profit_factors)) if profit_factors else 0,
        "expectancy_med": float(np.median(expectancies)) if expectancies else 0,
        "payoff_ratio_med": float(np.median(payoff_ratios)) if payoff_ratios else 0,
        "mdd_duration_med": float(np.median(mdd_durations)) if mdd_durations else 0,
        "mdd_duration_p75": float(np.percentile(mdd_durations, 75)) if mdd_durations else 0,
        "cagr_med": float(np.median(cagrs)) if cagrs else 0,
        "avg_hold_bars_med": float(np.median(avg_hold_bars)) if avg_hold_bars else 0,
        "median_hold_bars_med": float(np.median(median_hold_bars)) if median_hold_bars else 0,
        "max_hold_bars_p75": float(np.percentile(max_hold_bars, 75)) if max_hold_bars else 0,
        "alpha_ann_med_active": float(np.median(active_alpha_ann)) if active_alpha_ann else 0,
        "buy_hold_cagr_med": float(np.median(active_bh_cagrs)) if active_bh_cagrs else 0,
        "trades_med": float(np.median(trades)) if trades else 0,
        "n_ge_70": int(sum(1 for w in all_wrs if w >= 0.70)),
        "n_ge_50_target": int(sum(1 for w in all_wr_targets if w >= 0.50)),  # v8
    }


def build_ticker_arrays_cache(store):
    """
    Extract per-ticker arrays from PayloadStore into a plain-dict cache.
    Mirrors the main pipeline's Phase 3 (ga_run.py line 2777-2793).
    All numeric arrays cast to float64 (from float32 memmap).
    """
    cache = {}
    for tk in store.tickers:
        a = store.get_ticker_arrays(tk)
        p = {}
        for k in ("open", "high", "low", "close", "atr", "ma", "vol_rank",
                  "volume", "ret_fwd", "long_votes", "short_votes"):
            if k in a and a[k] is not None:
                p[k] = np.asarray(a[k], dtype=np.float64)
        if "score_matrix" in a and a["score_matrix"] is not None:
            p["score_matrix"] = np.asarray(a["score_matrix"], dtype=np.float64)
        if "dates" in a and a["dates"] is not None:
            p["dates"] = a["dates"]
        if "valid_mask" in a and a["valid_mask"] is not None:
            p["valid_mask"] = np.asarray(a["valid_mask"], dtype=bool)
        if p.get("close") is not None and len(p["close"]) >= 252:
            cache[tk] = p
    return cache


def run_chunk(
    chunk_id,
    stage,
    ngen,
    pop_size,
    window_count,
    store_path,
    window_plans,
    seed_genomes,
    resume_ckpt=None,
    window_offset=0,
):
    """Run one GA chunk (ngen generations). Returns best genome, HOF, and selected windows."""
    import pickle
    import ga_run_modular_final as gm
    from ga_run_modular_final import (
        _run_stage, _load_latest_checkpoint, GACheckpoint,
    )
    from auto_tune import RuntimeGuard

    # Local runner override: keep mutations tight so we can actually improve a strong incumbent
    # without constantly falling off a fitness cliff.
    try:
        if int(stage) == 2:
            gm.GA_MUT_SIGMA_START = 0.08
            gm.GA_MUT_SIGMA_END = 0.02
            gm.GA_GENE_MUT_PB_START = 0.18
            gm.GA_GENE_MUT_PB_END = 0.06
            gm.GA_MUT_PB = 0.35
        else:
            gm.GA_MUT_SIGMA_START = 0.10
            gm.GA_MUT_SIGMA_END = 0.03
            gm.GA_GENE_MUT_PB_START = 0.22
            gm.GA_GENE_MUT_PB_END = 0.08
            gm.GA_MUT_PB = 0.40
    except Exception:
        pass

    wp_bytes = pickle.dumps(window_plans)

    # Uniform window indices, with rotational offset for diversity across chunks
    n_win = len(window_plans)
    step = n_win / window_count
    window_indices = [min(max(0, int(round(i * step + window_offset)) % n_win), n_win - 1)
                      for i in range(window_count)]
    window_indices = sorted(set(window_indices))[:window_count]
    while len(window_indices) < window_count:
        # Pad if dedupe collapsed any
        for j in range(n_win):
            if j not in window_indices:
                window_indices.append(j)
                break

    print(f"\n{'='*60}", flush=True)
    print(f"[CHUNK {chunk_id}] windows={window_indices} pop={pop_size} ngen={ngen}",
          flush=True)
    print(f"{'='*60}", flush=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    t0 = time.time()
    pop, hof = _run_stage(
        stage=int(stage),
        pop_size=pop_size,
        ngen=ngen,
        window_indices=window_indices,
        n_workers=1,  # SINGLE PROCESS — no OOM
        store_path=store_path,
        window_plans_bytes=wp_bytes,
        checkpoint_dir=CHECKPOINT_DIR,
        batch_size=4,
        seed_genomes=seed_genomes,
        resume_from=resume_ckpt,
        guard=None,  # no runtime guard (we control chunk size)
    )
    elapsed = time.time() - t0

    best_genome = list(hof[0]) if len(hof) else None
    best_fit = float(hof[0].fitness.values[0]) if len(hof) else -1e9
    print(f"[CHUNK {chunk_id}] completed {ngen} gens in {elapsed/60:.1f}m "
          f"| hof[0].fit={best_fit:.4f}", flush=True)

    return best_genome, best_fit, hof, elapsed, window_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=CHUNKS_DEFAULT,
                        help="Number of chunks to run")
    parser.add_argument("--ngen-per-chunk", type=int, default=NGEN_PER_CHUNK,
                        help="Generations per chunk")
    parser.add_argument("--pop-size", type=int, default=POP_SIZE,
                        help="Population size")
    parser.add_argument("--windows", type=int, default=WINDOWS_STAGE,
                        help="Walk-forward windows per chunk")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate baseline/seed and exit (no GA run)")
    parser.add_argument("--eval-topk", type=int, default=4,
                        help="Evaluate top-K HOF candidates per chunk under full metrics")
    parser.add_argument("--stage", type=int, default=2, choices=(1, 2),
                        help="GA stage to run (1=explore, 2=refine)")
    parser.add_argument("--alpha-focus", type=str, default=os.environ.get("GA_ALPHA_FOCUS", "on"),
                        choices=("on", "off"),
                        help="Fitness mode: 'on' emphasizes alpha more; 'off' balances toward hit-rate")
    parser.add_argument("--reset", action="store_true",
                        help="Clear prior progress and restart fresh")
    parser.add_argument("--apply-best", action="store_true",
                        help="If best_found > main, overwrite main checkpoint at end")
    parser.add_argument("--window-offset", type=int, default=None,
                        help="Rotational offset for window indices (default: auto-rotate by round)")
    args = parser.parse_args()

    # Ensure env is set before importing GA modules (they read GA_ALPHA_FOCUS at import time).
    os.environ["GA_ALPHA_FOCUS"] = str(args.alpha_focus).lower().strip()

    if args.reset:
        for f in (PROGRESS_FILE, BEST_SOFAR_FILE):
            if os.path.exists(f):
                os.remove(f)
                print(f"[RESET] removed {f}", flush=True)
        if os.path.exists(CHECKPOINT_DIR):
            # Only remove files, not the directory (Windows permission quirks)
            for fname in os.listdir(CHECKPOINT_DIR):
                try:
                    os.remove(os.path.join(CHECKPOINT_DIR, fname))
                except Exception:
                    pass
            print(f"[RESET] cleared {CHECKPOINT_DIR}", flush=True)

    round_state, resolved_window_offset = _resolve_round_state(
        reset=args.reset,
        requested_offset=args.window_offset,
        total_windows=36,
        window_count=args.windows,
    )
    print(
        f"[ROUND] round_index={round_state['round_index']} "
        f"window_offset={resolved_window_offset}"
        + (" (explicit)" if args.window_offset is not None else " (auto)"),
        flush=True,
    )

    # Setup
    import ga_run as gr
    gr.GA_AUTO_TUNE = 0
    gr.FAST_MODE = True
    gr.GA_EVAL_WORKERS = 1

    (
        store,
        window_plans,
        seed_genome,
        seed_fit,
        baseline_genome,
        prev_main_fit,
        baseline_info,
    ) = open_existing_store()

    round_state, resolved_window_offset = _resolve_round_state(
        reset=False,
        requested_offset=args.window_offset if args.window_offset is not None else resolved_window_offset,
        total_windows=len(window_plans),
        window_count=args.windows,
    )

    # Build cache for full metric evaluation
    print("[SETUP] Building ticker array cache for metric eval...", flush=True)
    t0 = time.time()
    cache = build_ticker_arrays_cache(store)
    print(f"[SETUP] Cache ready: {len(cache)} tickers in {time.time()-t0:.0f}s",
          flush=True)

    # Initial metrics
    print("\n[BASE] Evaluating main baseline checkpoint...",
          flush=True)
    t0 = time.time()
    base_metrics = evaluate_genome_full(baseline_genome, cache, gr)
    base_metrics["eval_time_s"] = round(time.time() - t0, 1)
    print(f"[BASE] fit={base_metrics['fit']:.4f} "
          f"WR_med_all={_fmt_pct1(base_metrics['wr_med_all'])} "
          f"WR_mean_all={_fmt_pct1(base_metrics['wr_mean_all'])} "
          f"alpha_ann_mean={_fmt_pct1(base_metrics.get('mean_alpha_ann', 0.0))} "
          f"MDD={_fmt_pct1(base_metrics['mdd_med'])} "
          f"ret={base_metrics['ret_med']:.2f} "
          f"#>=70%={base_metrics['n_ge_70']}/{base_metrics['n_tickers_total']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    print("\n[SEED] Evaluating current accepted worktree checkpoint...",
          flush=True)
    t0 = time.time()
    seed_metrics = evaluate_genome_full(seed_genome, cache, gr)
    seed_metrics["eval_time_s"] = round(time.time() - t0, 1)
    print(f"[SEED] fit={seed_metrics['fit']:.4f} "
          f"WR_med_all={_fmt_pct1(seed_metrics['wr_med_all'])} "
          f"WR_mean_all={_fmt_pct1(seed_metrics['wr_mean_all'])} "
          f"alpha_ann_mean={_fmt_pct1(seed_metrics.get('mean_alpha_ann', 0.0))} "
          f"MDD={_fmt_pct1(seed_metrics['mdd_med'])} "
          f"ret={seed_metrics['ret_med']:.2f} "
          f"#>=70%={seed_metrics['n_ge_70']}/{seed_metrics['n_tickers_total']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    progress = _load_progress()
    # Always refresh baseline + seed metrics under the current fitness code,
    # even when resuming a prior chunk log.
    progress["base_metrics"] = base_metrics
    progress["seed_metrics"] = seed_metrics
    progress["baseline_info"] = baseline_info
    progress["round_info"] = {
        "round_index": round_state["round_index"],
        "window_offset": resolved_window_offset,
    }
    _save_progress(progress)

    # Best-so-far tracking
    seed_best = _build_seed_best_so_far(seed_genome, seed_metrics)
    best_so_far = seed_best

    raw_best_so_far = _load_best_so_far() or progress.get("best_so_far")
    if raw_best_so_far and raw_best_so_far.get("genome") and not _same_genome(raw_best_so_far.get("genome"), seed_genome):
        print("[SETUP] Re-evaluating persisted best-so-far under current fitness...", flush=True)
        t0 = time.time()
        raw_metrics = evaluate_genome_full(raw_best_so_far["genome"], cache, gr)
        raw_metrics["eval_time_s"] = round(time.time() - t0, 1)
        raw_best_eval = {
            "fit": float(raw_metrics["fit"]),
            "wr_med_all": float(raw_metrics["wr_med_all"]),
            "genome": list(raw_best_so_far["genome"]),
            "metrics": raw_metrics,
            "found_at_chunk": int(raw_best_so_far.get("found_at_chunk", -1)),
        }

        # Only keep a persisted candidate if it still beats the current accepted seed under
        # the shared promotion logic (hit-rate-first + safety guards).
        raw_vs_seed = _candidate_beats_incumbent(raw_best_eval["metrics"], base_metrics, seed_metrics)
        if raw_vs_seed.get("promote"):
            best_so_far = raw_best_eval
        else:
            best_so_far = seed_best
    _sync_best_so_far_state(progress, best_so_far)

    if args.eval_only:
        print("\n[EVAL-ONLY] Exiting after baseline/seed evaluation.", flush=True)
        return

    # Resume detection
    from ga_run_modular_final import _load_latest_checkpoint
    resume_ckpt = _load_latest_checkpoint(CHECKPOINT_DIR)
    start_chunk_id = len(progress["chunks"]) + 1
    if resume_ckpt:
        print(f"[RESUME] Found checkpoint: stage={resume_ckpt.stage} "
              f"gen={resume_ckpt.generation} "
              f"fit={resume_ckpt.best_fitness:.4f}", flush=True)
    print(f"[INFO] Starting at chunk {start_chunk_id} of {start_chunk_id + args.chunks - 1}",
          flush=True)
    print(f"[INFO] Using window_offset={resolved_window_offset}", flush=True)

    store_path = store.store_path

    # Chunk loop
    for i in range(args.chunks):
        chunk_id = start_chunk_id + i

        # Seed from best-so-far genome
        seed = [list(best_so_far["genome"])]
        # Plus 7 small perturbations
        import random
        from ga_run import GLOBAL_PARAM_SPECS, sanitize_global_genome
        random.seed(42 + chunk_id)
        np.random.seed(42 + chunk_id)
        for _ in range(7):
            varied = []
            for g_val, (_, lo, hi, _, _) in zip(best_so_far["genome"],
                                                 GLOBAL_PARAM_SPECS):
                v = float(np.clip(
                    g_val + random.gauss(0.0, 0.05 * (hi - lo)),
                    lo, hi))
                varied.append(v)
            seed.append(sanitize_global_genome(varied))

        # Load latest checkpoint each chunk (auto-resume within chunk)
        resume_ckpt = _load_latest_checkpoint(CHECKPOINT_DIR)

        try:
            best_genome, best_fit_train, hof, chunk_time, chunk_window_indices = run_chunk(
                chunk_id=chunk_id,
                stage=args.stage,
                ngen=args.ngen_per_chunk,
                pop_size=args.pop_size,
                window_count=args.windows,
                store_path=store_path,
                window_plans=window_plans,
                seed_genomes=seed,
                resume_ckpt=resume_ckpt,
                window_offset=resolved_window_offset,
            )
        except Exception as e:
            print(f"[CHUNK {chunk_id}] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            break

        if best_genome is None:
            print(f"[CHUNK {chunk_id}] no best genome returned", flush=True)
            break

        # Full metrics for the best candidates of this chunk (top-K from HOF).
        topk = max(1, int(args.eval_topk))
        cand_genomes = []
        seen = set()
        try:
            hof_list = list(hof) if hof is not None else []
        except Exception:
            hof_list = []

        for ind in hof_list[:topk]:
            genome = list(ind)
            key = tuple(round(float(x), 12) for x in genome)
            if key in seen:
                continue
            seen.add(key)
            cand_genomes.append(genome)

        # Ensure the reported HOF best is included even if HOF ordering differs.
        key_best = tuple(round(float(x), 12) for x in best_genome)
        if key_best not in seen:
            cand_genomes.insert(0, list(best_genome))

        print(f"[CHUNK {chunk_id}] evaluating {len(cand_genomes)} candidate(s) on all tickers...",
              flush=True)
        t_eval = time.time()
        evaluated = []
        try:
            for gi, genome in enumerate(cand_genomes, start=1):
                m = evaluate_genome_full(genome, cache, gr)
                evaluated.append((float(m.get("fit", -1e18)), genome, m))
                print(f"[CHUNK {chunk_id}] full-eval {gi}/{len(cand_genomes)} fit={m.get('fit', 0.0):.4f} "
                      f"WR={_fmt_pct1(m.get('wr_med_all', 0.0))} "
                      f"alpha_ann={_fmt_pct1(m.get('mean_alpha_ann', 0.0))}",
                      flush=True)
        except Exception as e:
            print(f"[CHUNK {chunk_id}] eval ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            break

        # Choose the chunk representative by swing-trade priority, not raw fitness.
        # Fitness can over-rank candidates that slightly regress hit-rate, which we
        # explicitly want to avoid for operational safety.
        evaluated.sort(key=lambda t: _priority_tuple(t[2]), reverse=True)
        best_full_fit, best_full_genome, best_full_metrics = evaluated[0]
        best_genome = list(best_full_genome)
        chunk_metrics = dict(best_full_metrics)
        chunk_metrics["eval_time_s"] = round(time.time() - t_eval, 1)
        chunk_metrics["_eval_candidates"] = int(len(evaluated))

        print(f"[CHUNK {chunk_id}] collecting walk-forward diagnostics...",
              flush=True)
        t_diag = time.time()
        try:
            from ga_run_modular_final import compute_walkforward_diagnostics
            oos_diagnostics = compute_walkforward_diagnostics(
                best_genome,
                store,
                window_plans,
                chunk_window_indices,
            )
            oos_diagnostics["eval_time_s"] = round(time.time() - t_diag, 1)
        except Exception as e:
            print(f"[CHUNK {chunk_id}] diagnostics WARN: {e}", flush=True)
            oos_diagnostics = {
                "selected_window_indices": list(chunk_window_indices),
                "window_count": 0,
                "train_mean": {},
                "oos_mean": {},
                "gap_summary": {},
                "window_metrics": [],
                "error": str(e),
            }

        # Evaluate promotion eligibility vs the currently accepted checkpoint (seed).
        candidate_vs_seed = _candidate_beats_incumbent(
            chunk_metrics,
            base_metrics,
            seed_metrics,
        )
        acceptance = candidate_vs_seed["acceptance_vs_main"]

        # Progress
        wr_all = chunk_metrics["wr_med_all"]
        wr_base = base_metrics["wr_med_all"]
        delta_wr = (wr_all - wr_base) * 100.0
        delta_fit = chunk_metrics["fit"] - base_metrics["fit"]

        # SAVE PROGRESS FIRST (before any print that could crash on encoding)
        chunk_entry = {
            "chunk_id": chunk_id,
            "timestamp": datetime.now().isoformat(),
            "chunk_time_s": round(chunk_time, 1),
            "metrics": chunk_metrics,
            "delta_fit_vs_main": round(delta_fit, 4),
            "delta_wr_vs_main_pp": round(delta_wr, 2),
            "acceptance_vs_main": acceptance,
            "incumbent_wr_guardrail": candidate_vs_seed["incumbent_wr_guardrail"],
            "incumbent_safety_guardrail": candidate_vs_seed["incumbent_safety_guardrail"],
            "priority_better_vs_incumbent": candidate_vs_seed["priority_better"],
            "baseline_info": baseline_info,
            "selected_window_indices": list(chunk_window_indices),
            "oos_diagnostics": oos_diagnostics,
            "best_genome": list(best_genome),
        }
        progress["chunks"].append(chunk_entry)
        _save_progress(progress)
        print(f"[CHUNK {chunk_id}] progress saved to {PROGRESS_FILE}", flush=True)

        # Print metrics (ASCII-safe)
        wr_target_all = chunk_metrics.get("wr_target_med_all", 0)
        base_wr_target = base_metrics.get("wr_target_med_all", 0)
        delta_wr_target = (wr_target_all - base_wr_target) * 100
        print(f"[CHUNK {chunk_id}] METRICS vs MAIN:", flush=True)
        print(f"  fit={chunk_metrics['fit']:.4f} (delta {delta_fit:+.4f})", flush=True)
        print(f"  WR_med_all (summary)={_fmt_pct1(wr_all)} (delta {delta_wr:+.2f}pp)",
              flush=True)
        print(f"  WR_target_med_all   ={_fmt_pct1(wr_target_all)} (delta {delta_wr_target:+.2f}pp)",
              flush=True)
        print(f"  WR_mean_all         ={_fmt_pct1(chunk_metrics['wr_mean_all'])}",
              flush=True)
        print(f"  WR_med (active)     ={_fmt_pct1(chunk_metrics['wr_med'])}", flush=True)
        print(f"  MDD_med             ={_fmt_pct1(chunk_metrics['mdd_med'])}", flush=True)
        print(f"  MDD_p75             ={_fmt_pct1(chunk_metrics.get('mdd_p75', 0.0))}", flush=True)
        print(f"  Return_med          ={chunk_metrics['ret_med']:.3f}", flush=True)
        print(f"  Sharpe_med          ={chunk_metrics['sharpe_med']:.3f}", flush=True)
        print(f"  Sortino_med         ={chunk_metrics.get('sortino_med', 0.0):.3f}", flush=True)
        print(f"  ProfitFactor_med    ={chunk_metrics.get('profit_factor_med', 0.0):.3f}", flush=True)
        print(f"  Expectancy_med      ={chunk_metrics.get('expectancy_med', 0.0):.4f}", flush=True)
        print(f"  PayoffRatio_med     ={chunk_metrics.get('payoff_ratio_med', 0.0):.3f}", flush=True)
        print(f"  MDD_duration_med    ={chunk_metrics.get('mdd_duration_med', 0.0):.0f} bars", flush=True)
        print(f"  MDD_duration_p75    ={chunk_metrics.get('mdd_duration_p75', 0.0):.0f} bars", flush=True)
        print(f"  AvgHoldBars_med     ={chunk_metrics.get('avg_hold_bars_med', 0.0):.1f}", flush=True)
        print(f"  MedHoldBars_med     ={chunk_metrics.get('median_hold_bars_med', 0.0):.1f}", flush=True)
        print(f"  MaxHoldBars_p75     ={chunk_metrics.get('max_hold_bars_p75', 0.0):.1f}", flush=True)
        print(f"  CAGR_med            ={_fmt_pct1(chunk_metrics.get('cagr_med', 0.0))}", flush=True)
        print(f"  Mean_excess         ={chunk_metrics.get('mean_excess', 0.0):+.4f}", flush=True)
        print(f"  Mean_alpha_ann      ={_fmt_pct1(chunk_metrics.get('mean_alpha_ann', 0.0))}", flush=True)
        print(f"  Med_alpha_ann       ={_fmt_pct1(chunk_metrics.get('med_alpha_ann', 0.0))}", flush=True)
        print(f"  Alpha_pos_rate      ={_fmt_pct1(chunk_metrics.get('alpha_ann_pos_rate', 0.0))}", flush=True)
        print(f"  BuyHold_CAGR_med    ={_fmt_pct1(chunk_metrics.get('buy_hold_cagr_med', 0.0))}", flush=True)
        print(f"  Trades_med          ={chunk_metrics['trades_med']:.0f}", flush=True)
        print(f"  N tickers >= 70% WR ={chunk_metrics['n_ge_70']}/{chunk_metrics['n_tickers_total']}",
              flush=True)
        print(f"  N tickers >= 50% WR_tgt={chunk_metrics.get('n_ge_50_target', 0)}/{chunk_metrics['n_tickers_total']}",
              flush=True)
        gap_summary = oos_diagnostics.get("gap_summary", {})
        if gap_summary:
            print(f"  OOS gap fitness     ={gap_summary.get('fitness_gap', 0.0):+.4f}", flush=True)
            print(f"  OOS gap WR          ={gap_summary.get('win_rate_gap', 0.0) * 100:+.2f}pp", flush=True)
            print(f"  OOS gap WR_tgt      ={gap_summary.get('win_rate_target_gap', 0.0) * 100:+.2f}pp", flush=True)
            print(f"  OOS gap excess      ={gap_summary.get('excess_return_gap', 0.0):+.4f}", flush=True)
            print(f"  OOS gap alpha_ann   ={gap_summary.get('alpha_ann_gap', 0.0) * 100:+.2f}pp", flush=True)
        print(f"  Acceptance vs main  ={acceptance['all_pass']}", flush=True)

        # Clean up per-chunk GA checkpoints AFTER progress is saved
        if os.path.exists(CHECKPOINT_DIR):
            for f in os.listdir(CHECKPOINT_DIR):
                if f.startswith("stage"):
                    try:
                        os.remove(os.path.join(CHECKPOINT_DIR, f))
                    except Exception:
                        pass

        # Update best-so-far if improved
        # Prioritize swing usefulness lexicographically:
        # hit-rate > target-hit > alpha_ann > drawdown > hold profile > fitness.
        incumbent_guardrail = candidate_vs_seed["incumbent_wr_guardrail"]
        incumbent_safety = candidate_vs_seed["incumbent_safety_guardrail"]
        # Best-so-far should be the best *promotable* candidate found so far:
        # - must pass shared promotion logic vs the seed checkpoint, and
        # - must beat the previously tracked best by the same priority comparator.
        improved = bool(candidate_vs_seed["promote"]) and _is_better_by_priority(chunk_metrics, best_so_far["metrics"])
        if improved:
            reason = (
                f"prio WR {_fmt_pct1(best_so_far['metrics'].get('wr_med_all', 0.0))}"
                f" -> {_fmt_pct1(chunk_metrics.get('wr_med_all', 0.0))}"
                f" | WR_tgt {_fmt_pct1(best_so_far['metrics'].get('wr_target_med_all', 0.0))}"
                f" -> {_fmt_pct1(chunk_metrics.get('wr_target_med_all', 0.0))}"
                f" | alpha_ann {_fmt_pct1(best_so_far['metrics'].get('mean_alpha_ann', 0.0))}"
                f" -> {_fmt_pct1(chunk_metrics.get('mean_alpha_ann', 0.0))}"
                f" | MDD {_fmt_pct1(best_so_far['metrics'].get('mdd_med', 0.0))}"
                f" -> {_fmt_pct1(chunk_metrics.get('mdd_med', 0.0))}"
            )

        if improved:
            best_so_far = {
                "fit": chunk_metrics["fit"],
                "wr_med_all": chunk_metrics["wr_med_all"],
                "genome": list(best_genome),
                "metrics": chunk_metrics,
                "found_at_chunk": chunk_id,
            }
            _sync_best_so_far_state(progress, best_so_far)
            print(f"[CHUNK {chunk_id}] *** NEW BEST: {reason} ***", flush=True)
        else:
            _sync_best_so_far_state(progress, best_so_far)
            print(f"[CHUNK {chunk_id}] no improvement vs best so far "
                  f"(fit {best_so_far['fit']:.4f} "
                  f"WR {_fmt_pct1(best_so_far['wr_med_all'])} "
                  f"alpha_ann {_fmt_pct1(best_so_far['metrics'].get('mean_alpha_ann', 0.0))} "
                  f"| wr_guard={incumbent_guardrail.get('all_pass', False)} "
                  f"| safety_guard={incumbent_safety.get('all_pass', False)} "
                  f"| priority={candidate_vs_seed.get('priority_better', False)})", flush=True)

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Main baseline:", flush=True)
    print(f"  source={baseline_info.get('source')} ref={baseline_info.get('ref')}", flush=True)
    print(f"  fit={base_metrics['fit']:.4f} "
          f"WR_med_all={_fmt_pct1(base_metrics['wr_med_all'])} "
          f"#>=70%={base_metrics['n_ge_70']}", flush=True)
    print(f"Best found:", flush=True)
    print(f"  fit={best_so_far['fit']:.4f} "
          f"WR_med_all={_fmt_pct1(best_so_far['wr_med_all'])} "
          f"#>=70%={best_so_far['metrics']['n_ge_70']} "
          f"(chunk {best_so_far['found_at_chunk']})", flush=True)

    delta_wr_final = (best_so_far["wr_med_all"] - base_metrics["wr_med_all"]) * 100.0
    delta_fit_final = best_so_far["fit"] - base_metrics["fit"]
    print(f"\ndelta WR_med_all vs main: {delta_wr_final:+.2f}pp", flush=True)
    print(f"delta fit vs main: {delta_fit_final:+.4f}", flush=True)

    # Apply best if requested and improved
    if args.apply_best:
        candidate = best_so_far["metrics"]
        apply_eval = _candidate_beats_incumbent(candidate, base_metrics, seed_metrics)
        acceptance = apply_eval["acceptance_vs_main"]
        incumbent_guardrail = apply_eval["incumbent_wr_guardrail"]
        incumbent_safety_guardrail = apply_eval["incumbent_safety_guardrail"]
        eligible = bool(apply_eval["promote"])

        if eligible:
            promoted_reason = (
                f"Promoted staged GA candidate (chunk {best_so_far.get('found_at_chunk', -1)}) "
                f"under current fitness. "
                f"ΔWR={((candidate.get('wr_med_all', 0.0) - base_metrics.get('wr_med_all', 0.0)) * 100.0):+.2f}pp "
                f"ΔWR_tgt={((candidate.get('wr_target_med_all', 0.0) - base_metrics.get('wr_target_med_all', 0.0)) * 100.0):+.2f}pp "
                f"Δalpha_ann={((candidate.get('mean_alpha_ann', 0.0) - base_metrics.get('mean_alpha_ann', 0.0)) * 100.0):+.2f}pp "
                f"ΔMDD={((candidate.get('mdd_med', 0.0) - base_metrics.get('mdd_med', 0.0)) * 100.0):+.2f}pp "
                f"ΔAvgHoldBars={candidate.get('avg_hold_bars_med', 0.0) - base_metrics.get('avg_hold_bars_med', 0.0):+.1f}."
            )

            # Legacy one-off backup for compatibility with older workflows.
            if not os.path.exists(BACKUP_MAIN):
                shutil.copy(MAIN_CHECKPOINT, BACKUP_MAIN)
                print(f"[APPLY] Backed up checkpoint -> {BACKUP_MAIN}", flush=True)

            prev_path, new_path = _write_checkpoint_with_metadata(
                best_so_far=best_so_far,
                base_metrics=base_metrics,
                seed_metrics=seed_metrics,
                baseline_info=baseline_info,
                acceptance=acceptance,
                incumbent_guardrail=incumbent_guardrail,
                incumbent_safety_guardrail=incumbent_safety_guardrail,
                promoted_reason=promoted_reason,
            )
            print(f"[APPLY] Updated {MAIN_CHECKPOINT} with promoted candidate", flush=True)
            print(f"[APPLY] Saved backups: prev={prev_path} new={new_path}", flush=True)

            progress["promotion"] = {
                "timestamp": datetime.now().isoformat(),
                "prev_backup": prev_path,
                "new_backup": new_path,
                "candidate": _snapshot_metrics(candidate),
                "main": _snapshot_metrics(base_metrics),
                "incumbent": _snapshot_metrics(seed_metrics),
                "acceptance_vs_main": dict(acceptance),
                "incumbent_wr_guardrail": dict(incumbent_guardrail),
                "incumbent_safety_guardrail": dict(incumbent_safety_guardrail),
            }
            _save_progress(progress)
        else:
            print("[APPLY] Candidate not eligible for promotion; checkpoint unchanged.", flush=True)
            print(f"[APPLY] acceptance_vs_main={acceptance.get('all_pass', False)} "
                  f"incumbent_guardrail={incumbent_guardrail.get('all_pass', False)} "
                  f"incumbent_safety={incumbent_safety_guardrail.get('all_pass', False)} "
                  f"priority_better={apply_eval.get('priority_better', False)}", flush=True)


if __name__ == "__main__":
    main()
