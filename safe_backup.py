"""
Safe backup utility for local GA results.
Creates timestamped + redundant backups of best_so_far.
"""
import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

BACKUP_DIR = ".local_ga_backups"
BEST_FILE = "local_ga_best_so_far.json"
PROGRESS_FILE = "local_ga_chunk_progress.json"


def safe_backup(reason=""):
    """Create a timestamped backup of local_ga_best_so_far.json + summary."""
    if not os.path.exists(BEST_FILE):
        print(f"[BACKUP] {BEST_FILE} not found, nothing to backup")
        return None

    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Read best
    with open(BEST_FILE, "r") as f:
        best = json.load(f)

    wr = best.get("wr_med_all", 0) * 100
    fit = best.get("fit", 0)
    chunk = best.get("found_at_chunk", 0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"best_wr{wr:.2f}pct_fit{fit:.2f}_ch{chunk}_{ts}.json"
    if reason:
        fname = f"best_{reason}_wr{wr:.2f}pct_{ts}.json"
    target = os.path.join(BACKUP_DIR, fname)

    shutil.copy(BEST_FILE, target)
    print(f"[BACKUP] saved {target}")

    # Also save progress snapshot
    if os.path.exists(PROGRESS_FILE):
        prog_target = os.path.join(BACKUP_DIR, f"progress_{ts}.json")
        shutil.copy(PROGRESS_FILE, prog_target)
        print(f"[BACKUP] saved {prog_target}")

    # Clean up old failed backups (zero-length etc)
    for f in os.listdir(BACKUP_DIR):
        full = os.path.join(BACKUP_DIR, f)
        if os.path.getsize(full) < 100 or f.endswith(".json") and "_." in f:
            try:
                os.remove(full)
                print(f"[BACKUP] removed corrupt {f}")
            except Exception:
                pass

    return target


def list_backups():
    if not os.path.exists(BACKUP_DIR):
        print("No backups dir")
        return
    files = sorted(os.listdir(BACKUP_DIR), reverse=True)
    print(f"\nBackups in {BACKUP_DIR}:")
    for f in files:
        full = os.path.join(BACKUP_DIR, f)
        size = os.path.getsize(full)
        mtime = datetime.fromtimestamp(os.path.getmtime(full)).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {mtime}  {size:>8} bytes  {f}")


if __name__ == "__main__":
    reason = sys.argv[1] if len(sys.argv) > 1 else ""
    safe_backup(reason=reason)
    list_backups()
