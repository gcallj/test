"""
Autonomous loop runner: keep running run_local_ga_staged.py in chunks until target time.

Usage:
    python autonomous_loop.py --until 06:00
    python autonomous_loop.py --until 06:00 --chunks-per-run 3

After each batch:
- Calls safe_backup.py to create timestamped backup
- If WR improved vs main checkpoint, applies it to main + commits to git
- Logs progress to autonomous_loop.log
"""
import os
import sys
import json
import time
import argparse
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

PYTHON = sys.executable
PROGRESS_FILE = "local_ga_chunk_progress.json"
BEST_FILE = "local_ga_best_so_far.json"
MAIN_CHECKPOINT = "global_ga_checkpoint.json"
BACKUP_DIR = ".local_ga_backups"
LOG_FILE = "autonomous_loop.log"


def log(msg):
    """Log to both stdout and log file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_main():
    if not os.path.exists(MAIN_CHECKPOINT):
        return None
    with open(MAIN_CHECKPOINT, "r") as f:
        return json.load(f)


def get_main_wr():
    """Read current main checkpoint's WR_med_all from progress file's base_metrics."""
    if not os.path.exists(PROGRESS_FILE):
        return None
    with open(PROGRESS_FILE, "r") as f:
        prog = json.load(f)
    base = prog.get("base_metrics", {})
    return base.get("wr_med_all", None)


def get_best_so_far():
    """Read best_so_far from local_ga_best_so_far.json."""
    if not os.path.exists(BEST_FILE):
        return None
    with open(BEST_FILE, "r") as f:
        return json.load(f)


def parse_until(s):
    """Parse 'HH:MM' or 'HH:MM:SS' as today (or tomorrow if past)."""
    parts = s.split(":")
    h = int(parts[0])
    m = int(parts[1])
    sec = int(parts[2]) if len(parts) > 2 else 0
    now = datetime.now()
    target = now.replace(hour=h, minute=m, second=sec, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def run_safe_backup(reason=""):
    cmd = [PYTHON, "safe_backup.py", reason]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            log(f"[BACKUP OK] {reason}")
        else:
            log(f"[BACKUP FAIL] {result.stderr[:200]}")
    except Exception as e:
        log(f"[BACKUP ERROR] {e}")


def commit_to_git(message):
    """Stage best files and commit (don't push automatically to avoid conflicts)."""
    try:
        files = [
            MAIN_CHECKPOINT,
            BEST_FILE,
            PROGRESS_FILE,
            BACKUP_DIR + "/",
        ]
        for f in files:
            if os.path.exists(f) or os.path.exists(f.rstrip("/")):
                subprocess.run(["git", "add", f], capture_output=True)
        result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True, text=True, timeout=30,
        )
        if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
            log("[GIT] nothing to commit")
            return False
        if result.returncode == 0:
            log(f"[GIT COMMIT] {message[:80]}")
            push = subprocess.run(
                ["git", "push", "origin", "main"],
                capture_output=True, text=True, timeout=120,
            )
            if push.returncode == 0:
                log(f"[GIT PUSH] success")
            else:
                log(f"[GIT PUSH] failed: {push.stderr[:200]}")
            return True
        else:
            log(f"[GIT COMMIT] failed: {result.stderr[:200]}")
            return False
    except Exception as e:
        log(f"[GIT ERROR] {e}")
        return False


def run_chunks(n_chunks, window_offset=0, pop_size=24, ngen=4, windows=4):
    """Run n_chunks chunks of run_local_ga_staged.py."""
    cmd = [
        PYTHON, "-u", "run_local_ga_staged.py",
        "--chunks", str(n_chunks),
        "--window-offset", str(window_offset),
        "--pop-size", str(pop_size),
        "--ngen-per-chunk", str(ngen),
        "--windows", str(windows),
    ]
    log(f"[RUN] starting {n_chunks} chunks (offset={window_offset}, pop={pop_size}, ngen={ngen}, win={windows})...")
    t0 = time.time()
    try:
        # Stream output to log
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                # Only log key lines to avoid clutter
                if any(k in line for k in [
                    "[CHUNK", "[RUN", "[BASE", "*** NEW BEST", "delta",
                    "ERROR", "Traceback", "S2 gen", "completed"
                ]):
                    log(f"  {line}")
        proc.wait(timeout=14400)  # 4h max per batch
        return proc.returncode == 0, time.time() - t0
    except subprocess.TimeoutExpired:
        proc.kill()
        log(f"[RUN TIMEOUT after 4h]")
        return False, time.time() - t0
    except Exception as e:
        log(f"[RUN ERROR] {e}")
        return False, time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", type=str, default="06:00",
                        help="Stop time HH:MM (default 06:00)")
    parser.add_argument("--chunks-per-run", type=int, default=3,
                        help="Chunks per inner run (default 3)")
    parser.add_argument("--max-batches", type=int, default=20,
                        help="Max batches (safety limit)")
    args = parser.parse_args()

    target_time = parse_until(args.until)
    log(f"=" * 60)
    log(f"AUTONOMOUS LOOP STARTED")
    log(f"Target stop time: {target_time}")
    log(f"Chunks per batch: {args.chunks_per_run}")
    log(f"Max batches: {args.max_batches}")
    log(f"=" * 60)

    # Initial state
    main_ck = load_main()
    main_fit = float(main_ck.get('fitness', 0)) if main_ck else 0
    log(f"[INIT] main checkpoint fitness: {main_fit}")

    # CRITICAL: initial_wr should reflect the CURRENT main checkpoint's WR,
    # not the original base_metrics (which may be stale).
    # We compute it from best_so_far (which tracks the chunk that was applied).
    initial_wr = None
    best = get_best_so_far()
    if best:
        log(f"[INIT] best so far: WR={best.get('wr_med_all', 0)*100:.2f}% "
            f"fit={best.get('fit', 0):.4f} chunk={best.get('found_at_chunk', 0)}")
        # If best_so_far's fit matches main_fit, use it as current WR baseline
        if abs(best.get("fit", 0) - main_fit) < 1e-3:
            initial_wr = best.get("wr_med_all", None)
            log(f"[INIT] best_so_far matches main -> initial_wr={initial_wr*100:.2f}%")

    # Fallback: read from progress base_metrics
    if initial_wr is None and os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            prog = json.load(f)
        # First try to find chunk in progress that matches main_fit
        for ch in prog.get("chunks", []):
            ch_fit = ch.get("metrics", {}).get("fit", 0)
            if abs(ch_fit - main_fit) < 1e-3:
                initial_wr = ch.get("metrics", {}).get("wr_med_all", None)
                log(f"[INIT] matched chunk {ch.get('chunk_id')} fit={ch_fit:.4f} -> initial_wr={initial_wr*100:.2f}%")
                break
        # Fallback to base_metrics
        if initial_wr is None:
            base = prog.get("base_metrics", {})
            initial_wr = base.get("wr_med_all", None)
            log(f"[INIT] using base_metrics WR={initial_wr*100:.2f}%" if initial_wr else "[INIT] no base WR")

    # SAFETY: never let initial_wr be lower than the actually applied main checkpoint
    # We use best_so_far as the absolute floor since we know it's better than main
    if best and best.get("wr_med_all"):
        bsf_wr = best["wr_med_all"]
        if initial_wr is None or bsf_wr > initial_wr:
            initial_wr = bsf_wr
            log(f"[INIT] SAFETY: raising initial_wr to best_so_far={initial_wr*100:.2f}%")

    # Initial backup
    run_safe_backup("autonomous_start")

    # Diversification strategy: rotate window offset across batches
    # This breaks the GA out of local optima by exploring different time periods
    BATCH_CONFIGS = [
        # (offset, pop, ngen, win) - rotate through different exploration profiles
        (0, 24, 4, 4),     # default
        (3, 24, 4, 4),     # shifted windows
        (6, 30, 4, 5),     # bigger pop, more windows
        (9, 24, 6, 4),     # more gens
        (1, 28, 4, 4),     # tiny shift
        (5, 24, 4, 5),     # +1 window
        (2, 30, 5, 4),     # bigger pop + gens
        (7, 24, 4, 4),     # back to small
    ]

    batch_id = 0
    while batch_id < args.max_batches:
        batch_id += 1
        now = datetime.now()
        remaining = (target_time - now).total_seconds()
        if remaining < 600:  # less than 10 min remaining
            log(f"[STOP] only {remaining:.0f}s remaining, stopping (need >=600s for chunk)")
            break

        # Pick config based on batch number
        cfg = BATCH_CONFIGS[(batch_id - 1) % len(BATCH_CONFIGS)]
        offset, pop, ngen, win = cfg

        log(f"")
        log(f"=" * 60)
        log(f"BATCH {batch_id} | remaining: {remaining/3600:.1f}h | "
            f"config: offset={offset} pop={pop} ngen={ngen} windows={win}")
        log(f"=" * 60)

        ok, dt = run_chunks(args.chunks_per_run, window_offset=offset,
                            pop_size=pop, ngen=ngen, windows=win)
        log(f"[BATCH {batch_id}] done in {dt/60:.1f}m, ok={ok}")

        # After each batch: backup + check for improvement
        new_best = get_best_so_far()
        if new_best:
            new_wr = new_best.get("wr_med_all", 0)
            new_fit = new_best.get("fit", 0)
            new_n_ge_70 = new_best.get("metrics", {}).get("n_ge_70", 0)
            new_sharpe = new_best.get("metrics", {}).get("sharpe_med", 0)
            chunk_id_found = new_best.get("found_at_chunk", 0)

            log(f"[BATCH {batch_id}] best_so_far: WR={new_wr*100:.2f}% "
                f"fit={new_fit:.4f} n>=70={new_n_ge_70} sharpe={new_sharpe:.3f} "
                f"(chunk {chunk_id_found})")

            # Apply only if STRICTLY better than current initial_wr
            # AND not catastrophically worse on other metrics
            if initial_wr is not None and new_wr > initial_wr + 1e-6:
                # Get current main metrics for safety comparison
                cur_n_ge_70 = 0
                cur_sharpe = 0
                if os.path.exists(PROGRESS_FILE):
                    with open(PROGRESS_FILE, "r") as f:
                        prog = json.load(f)
                    main_fit = load_main().get("fitness", 0) if load_main() else 0
                    for ch in prog.get("chunks", []):
                        if abs(ch.get("metrics", {}).get("fit", 0) - main_fit) < 1e-3:
                            cur_n_ge_70 = ch.get("metrics", {}).get("n_ge_70", 0)
                            cur_sharpe = ch.get("metrics", {}).get("sharpe_med", 0)
                            break

                # Safety check: don't degrade other key metrics too much
                degrade_n70 = (cur_n_ge_70 - new_n_ge_70) if cur_n_ge_70 else 0
                degrade_sharpe_pct = ((cur_sharpe - new_sharpe) / max(cur_sharpe, 1e-6)) if cur_sharpe else 0

                if degrade_n70 > 5 or degrade_sharpe_pct > 0.15:
                    log(f"[BATCH {batch_id}] SKIP APPLY: WR up but n>=70 degraded by {degrade_n70} "
                        f"or sharpe degraded by {degrade_sharpe_pct*100:.0f}%")
                else:
                    # Save backup before overwriting
                    run_safe_backup(f"batch{batch_id}_before_apply")
                    # Apply
                    main_ck_now = load_main()
                    old_fit = main_ck_now.get("fitness", 0) if main_ck_now else 0
                    with open(MAIN_CHECKPOINT, "w") as f:
                        json.dump({
                            "fitness": new_fit,
                            "genome": new_best["genome"],
                        }, f, indent=2)
                    log(f"[APPLY] Updated {MAIN_CHECKPOINT} "
                        f"WR {(initial_wr*100):.2f}% -> {new_wr*100:.2f}% "
                        f"fit {old_fit:.4f} -> {new_fit:.4f}")
                    # Commit
                    commit_to_git(
                        f"Autonomous batch {batch_id}: WR {(initial_wr*100):.2f}% "
                        f"-> {(new_wr*100):.2f}%, fit {old_fit:.4f} -> {new_fit:.4f}"
                    )
                    initial_wr = new_wr  # update tracking
            else:
                log(f"[BATCH {batch_id}] no improvement vs current best ({initial_wr*100:.2f}%)" if initial_wr else "[BATCH] no initial wr")

        run_safe_backup(f"batch{batch_id}_done")

    log(f"")
    log(f"=" * 60)
    log(f"AUTONOMOUS LOOP FINISHED at {datetime.now()}")
    log(f"=" * 60)

    # Final state
    final_best = get_best_so_far()
    if final_best:
        log(f"[FINAL BEST] WR={final_best.get('wr_med_all', 0)*100:.2f}% "
            f"fit={final_best.get('fit', 0):.4f} "
            f"chunk={final_best.get('found_at_chunk', 0)}")

    main_ck = load_main()
    if main_ck:
        log(f"[FINAL MAIN] fitness={main_ck.get('fitness', 0):.4f}")

    run_safe_backup("autonomous_end")


if __name__ == "__main__":
    main()
