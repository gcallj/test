import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import run_local_ga_staged as staged_runner


ROOT = Path(__file__).resolve().parent
PROGRESS_FILE = ROOT / "local_ga_chunk_progress.json"
CHECKPOINT_FILE = ROOT / "global_ga_checkpoint.json"
STATUS_FILE = ROOT / "overnight_alpha_until_0600_status.json"


def _now_local():
    return datetime.now().astimezone()


def _default_deadline():
    now = _now_local()
    deadline = now.replace(hour=6, minute=0, second=0, microsecond=0)
    if now >= deadline:
        deadline += timedelta(days=1)
    return deadline


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _log(message):
    stamp = _now_local().strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"[{stamp}] {message}", flush=True)


def _summarize_candidate(candidate):
    metrics = candidate["metrics"]
    return (
        f"chunk {candidate['chunk_id']} "
        f"fit={metrics.get('fit', 0.0):.4f} "
        f"WR={metrics.get('wr_med_all', 0.0) * 100:.1f}% "
        f"WR_tgt={metrics.get('wr_target_med_all', 0.0) * 100:.1f}% "
        f"alpha={metrics.get('mean_alpha_ann', 0.0) * 100:.2f}% "
        f"MDD={metrics.get('mdd_med', 0.0) * 100:.2f}%"
    )


def _find_best_accepted(progress):
    best = progress.get("best_so_far") or {}
    metrics = best.get("metrics") or {}
    genome = best.get("genome") or []
    found_at_chunk = int(best.get("found_at_chunk", 0) or 0)
    if not genome or found_at_chunk <= 0 or not metrics:
        return None
    acceptance = staged_runner._compute_acceptance(metrics, progress.get("base_metrics", {}))
    if not acceptance.get("all_pass", False):
        return None
    return {
        "chunk_id": found_at_chunk,
        "metrics": metrics,
        "best_genome": list(genome),
        "acceptance_vs_main": acceptance,
    }


def _should_promote(candidate, progress):
    decision = staged_runner._candidate_beats_incumbent(
        candidate.get("metrics", {}),
        progress.get("base_metrics", {}),
        progress.get("seed_metrics", {}),
    )
    return bool(decision.get("promote", False)), decision


def _promote_candidate(candidate, progress, round_index):
    current_checkpoint = _load_json(CHECKPOINT_FILE)
    backup_name = (
        f"global_ga_checkpoint_pre_overnight_round{round_index}_"
        f"{_now_local().strftime('%Y%m%d_%H%M%S')}.json.bak"
    )
    backup_path = CHECKPOINT_FILE.with_name(backup_name)
    shutil.copyfile(CHECKPOINT_FILE, backup_path)

    candidate_metrics = candidate.get("metrics", {})
    seed_metrics = progress.get("seed_metrics", {})
    baseline_info = progress.get("baseline_info", {})
    promoted = {
        "fitness": float(candidate_metrics.get("fit", 0.0) or 0.0),
        "accepted_from_staged_chunk": int(candidate.get("chunk_id", -1)),
        "accepted_on": _now_local().date().isoformat(),
        "accepted_reason": (
            f"Promoted overnight accepted candidate from round {round_index}: "
            f"cleared the swing-trade safety gates versus main and the incumbent "
            f"while improving the prioritized staged-runner objective."
        ),
        "baseline_main_ref": baseline_info.get("ref"),
        "previous_checkpoint_fitness": current_checkpoint.get("fitness"),
        "previous_checkpoint_fit_current_formula": float(seed_metrics.get("fit", current_checkpoint.get("fitness", 0.0)) or 0.0),
        "genome": list(candidate.get("best_genome", [])),
    }
    _save_json(CHECKPOINT_FILE, promoted)
    return backup_path


def _update_status(data):
    _save_json(STATUS_FILE, data)


def main():
    parser = argparse.ArgumentParser(description="Run staged alpha-focused GA rounds until the next 06:00 local time.")
    parser.add_argument("--deadline", type=str, default=None, help="ISO datetime with offset, e.g. 2026-04-15T06:00:00+02:00")
    parser.add_argument("--chunks-per-round", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--sleep-between", type=float, default=5.0)
    args = parser.parse_args()

    deadline = datetime.fromisoformat(args.deadline) if args.deadline else _default_deadline()
    if deadline.tzinfo is None:
        deadline = deadline.astimezone()
    deadline = deadline.astimezone()

    _log(f"Overnight run starting; deadline is {deadline.isoformat()}")
    round_index = 0
    promotions = []

    while _now_local() < deadline:
        if args.max_rounds is not None and round_index >= args.max_rounds:
            break
        round_index += 1
        remaining = deadline - _now_local()
        _log(f"Starting overnight round {round_index}; remaining time {remaining}")
        _update_status({
            "status": "running",
            "round_index": round_index,
            "deadline": deadline.isoformat(),
            "started_at": _now_local().isoformat(),
            "promotions": promotions,
        })

        cmd = [
            sys.executable,
            "run_local_ga_staged.py",
            "--reset",
            "--chunks",
            str(args.chunks_per_round),
        ]
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        if result.returncode != 0:
            _log(f"Round {round_index} failed with exit code {result.returncode}")
            _update_status({
                "status": "failed",
                "round_index": round_index,
                "deadline": deadline.isoformat(),
                "finished_at": _now_local().isoformat(),
                "returncode": result.returncode,
                "promotions": promotions,
            })
            return result.returncode

        progress = _load_json(PROGRESS_FILE)
        seed_metrics = progress.get("seed_metrics", {})
        candidate = _find_best_accepted(progress)
        if candidate is None:
            _log(f"Round {round_index} ended with no accepted candidate against main")
        else:
            _log(f"Best accepted candidate in round {round_index}: {_summarize_candidate(candidate)}")
            should_promote, decision = _should_promote(candidate, progress)
            if should_promote:
                backup_path = _promote_candidate(candidate, progress, round_index)
                promotions.append({
                    "round_index": round_index,
                    "backup_path": str(backup_path),
                    "candidate": {
                        "chunk_id": int(candidate.get("chunk_id", -1)),
                        "fit": float(candidate["metrics"].get("fit", 0.0) or 0.0),
                        "wr_med_all": float(candidate["metrics"].get("wr_med_all", 0.0) or 0.0),
                        "mean_alpha_ann": float(candidate["metrics"].get("mean_alpha_ann", 0.0) or 0.0),
                        "mdd_med": float(candidate["metrics"].get("mdd_med", 0.0) or 0.0),
                    },
                })
                _log(f"Promoted round {round_index} candidate and backed up checkpoint to {backup_path.name}")
            else:
                wr_guard = decision.get("incumbent_wr_guardrail", {}).get("all_pass", False)
                safety_guard = decision.get("incumbent_safety_guardrail", {}).get("all_pass", False)
                priority = decision.get("priority_better", False)
                _log(
                    f"Round {round_index} accepted candidate did not clear incumbent promotion guardrails "
                    f"(wr_guard={wr_guard} safety_guard={safety_guard} priority={priority})"
                )

        _update_status({
            "status": "running",
            "round_index": round_index,
            "deadline": deadline.isoformat(),
            "last_finished_at": _now_local().isoformat(),
            "promotions": promotions,
        })
        if _now_local() >= deadline:
            break
        time.sleep(max(args.sleep_between, 0.0))

    _update_status({
        "status": "completed",
        "round_index": round_index,
        "deadline": deadline.isoformat(),
        "finished_at": _now_local().isoformat(),
        "promotions": promotions,
    })
    _log(f"Overnight run finished after {round_index} round(s); promotions={len(promotions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
