#!/usr/bin/env python3
"""
Full GA retrain script optimized for GitHub Codespaces (32GB RAM, 8 cores).
Uses larger population, more windows, and more workers than local machine.

Usage (from Codespace terminal):
    python retrain_codespace.py              # full retrain (pop=60, 40 gens, 10 windows)
    python retrain_codespace.py --stages 3   # run in 3 stages of ~13 gens each
"""
import os
import sys
import json
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Full GA retrain for Codespaces")
    parser.add_argument("--stages", type=int, default=1, help="Number of stages (1=all at once, 3=staged)")
    parser.add_argument("--pop", type=int, default=60, help="Population size")
    parser.add_argument("--gens", type=int, default=40, help="Total generations")
    parser.add_argument("--windows", type=int, default=10, help="Walk-forward windows")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    # Environment setup for Codespaces
    os.environ['GA_RUN_MODE'] = 'train'
    os.environ['GA_FAST_MODE'] = 'false'
    os.environ['GA_EVAL_WORKERS'] = str(args.workers)

    print(f"=== CODESPACE RETRAIN ===")
    print(f"Pop: {args.pop} | Gens: {args.gens} | Windows: {args.windows} | Workers: {args.workers}")
    print(f"Stages: {args.stages}")
    print(f"RAM: {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3):.1f} GB")
    print(f"CPUs: {os.cpu_count()}")
    print()

    # Step 1: ETL (fresh data)
    print("=" * 60)
    print("STEP 1: ETL (fresh market data)")
    print("=" * 60)
    t0 = time.time()
    os.system("python -u run_pipeline.py --steps 1")
    print(f"ETL done in {time.time()-t0:.0f}s")

    # Step 2: Rebuild history
    print("\n" + "=" * 60)
    print("STEP 2: Rebuild history_consolidated")
    print("=" * 60)
    t0 = time.time()
    os.system("python -u run_pipeline.py --steps 4")
    print(f"Step 4 done in {time.time()-t0:.0f}s")

    # Step 3: GA retrain
    # Temporarily patch ga_run.py config for Codespace hardware
    import ga_run
    ga_run.GA_STAGE1_POP_SIZE = args.pop
    ga_run.GA_MAX_WINDOWS = args.windows
    ga_run.GA_MAX_WINDOWS_STAGE2 = min(args.windows, 8)
    ga_run.GA_EVAL_WORKERS = args.workers

    gens_per_stage = args.gens // args.stages
    total_t0 = time.time()

    for stage in range(1, args.stages + 1):
        print(f"\n{'=' * 60}")
        print(f"STEP 3.{stage}: GA RETRAIN (stage {stage}/{args.stages}, {gens_per_stage} gens)")
        print(f"{'=' * 60}")

        ga_run.GA_STAGE1_NGEN = gens_per_stage

        # Read current checkpoint
        try:
            with open('global_ga_checkpoint.json') as f:
                ck = json.load(f)
            print(f"Starting from fitness: {ck['fitness']:.4f}")
        except:
            print("No checkpoint found, starting fresh")

        t0 = time.time()
        try:
            ga_run.run()
        except Exception as e:
            print(f"ERROR in stage {stage}: {e}")
            break
        elapsed = time.time() - t0

        # Read result
        with open('global_ga_checkpoint.json') as f:
            ck = json.load(f)
        print(f"Stage {stage} done: fitness={ck['fitness']:.4f} ({elapsed:.0f}s)")

    total_time = time.time() - total_t0
    print(f"\n{'=' * 60}")
    print(f"RETRAIN COMPLETE: {total_time/60:.1f} min total")
    with open('global_ga_checkpoint.json') as f:
        ck = json.load(f)
    print(f"Final fitness: {ck['fitness']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
