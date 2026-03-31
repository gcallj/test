#!/usr/bin/env python3
"""Run GA in stages as separate subprocesses (clean memory between stages)."""
import subprocess, json, time, sys

TOTAL_STAGES = 10
t_total = time.time()

for stage in range(1, TOTAL_STAGES + 1):
    ck = json.load(open('global_ga_checkpoint.json'))
    print(f"\n{'='*60}", flush=True)
    print(f"STAGE {stage}/{TOTAL_STAGES} | Starting fitness: {ck['fitness']:.4f}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-c", """
import os
os.environ['GA_RUN_MODE'] = 'train'
os.environ['GA_FAST_MODE'] = 'false'
os.environ['GA_EVAL_WORKERS'] = '1'
import ga_run
ga_run.GA_STAGE1_POP_SIZE = 30
ga_run.GA_STAGE1_NGEN = 10
ga_run.GA_MAX_WINDOWS = 4
ga_run.GA_MAX_WINDOWS_STAGE2 = 4
ga_run.GA_EVAL_WORKERS = 1
ga_run.run()
"""],
        cwd="/home/user/test",
        timeout=3600,
    )

    elapsed = time.time() - t0
    total_elapsed = (time.time() - t_total) / 60

    if result.returncode != 0:
        print(f"STAGE {stage} CRASHED (exit {result.returncode}) after {elapsed:.0f}s — continuing from checkpoint", flush=True)
        continue

    ck = json.load(open('global_ga_checkpoint.json'))
    print(f"STAGE {stage} DONE: fitness={ck['fitness']:.4f} ({elapsed:.0f}s) | Total: {total_elapsed:.0f}min", flush=True)

ck = json.load(open('global_ga_checkpoint.json'))
print(f"\n{'='*60}", flush=True)
print(f"ALL DONE: fitness={ck['fitness']:.4f} | {(time.time()-t_total)/60:.0f}min total", flush=True)
print(f"{'='*60}", flush=True)
