#!/usr/bin/env python3
"""
Retrain GA in stages with subprocess isolation.
Auto-detects RAM and adjusts pop/windows accordingly.
Works on Codespaces (8GB), local (16GB), or any machine.

Usage:
    python retrain_codespace.py                    # auto-detect config
    python retrain_codespace.py --stages 10        # 10 stages of 10 gens
    python retrain_codespace.py --pop 60 --windows 10  # force config
"""
import os, sys, json, time, subprocess, argparse


def get_ram_gb():
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
    except:
        return 8.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=int, default=10)
    parser.add_argument("--gens-per-stage", type=int, default=10)
    parser.add_argument("--pop", type=int, default=0, help="0=auto-detect from RAM")
    parser.add_argument("--windows", type=int, default=0, help="0=auto-detect from RAM")
    parser.add_argument("--workers", type=int, default=0, help="0=auto-detect from RAM")
    parser.add_argument("--skip-etl", action="store_true", help="Skip ETL step")
    args = parser.parse_args()

    ram_gb = get_ram_gb()
    cpus = os.cpu_count() or 2

    # Auto-detect config based on RAM
    if args.pop == 0:
        if ram_gb >= 28:
            args.pop, args.windows, args.workers = 60, 10, 4
        elif ram_gb >= 14:
            args.pop, args.windows, args.workers = 30, 4, 1
        else:
            args.pop, args.windows, args.workers = 25, 3, 1

    if args.windows == 0:
        args.windows = 4
    if args.workers == 0:
        args.workers = 1

    total_gens = args.stages * args.gens_per_stage
    print(f"=== RETRAIN CONFIG ===")
    print(f"RAM: {ram_gb:.1f} GB | CPUs: {cpus}")
    print(f"Pop: {args.pop} | Windows: {args.windows} | Workers: {args.workers}")
    print(f"Stages: {args.stages} x {args.gens_per_stage} gens = {total_gens} total")
    print()

    # Step 1: ETL
    if not args.skip_etl:
        print("=" * 60)
        print("STEP 1: ETL")
        print("=" * 60, flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, "-u", "run_pipeline.py", "--steps", "1"],
                       cwd=os.path.dirname(os.path.abspath(__file__)) or ".")
        print(f"ETL done in {time.time()-t0:.0f}s\n", flush=True)

    # Step 2: GA stages (each in separate subprocess for clean memory)
    t_total = time.time()
    for stage in range(1, args.stages + 1):
        try:
            ck = json.load(open('global_ga_checkpoint.json'))
            fit = ck['fitness']
        except:
            fit = 0.0
        print(f"\n{'='*60}", flush=True)
        print(f"STAGE {stage}/{args.stages} | fitness: {fit:.4f}", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, "-u", "-c", f"""
import os
os.environ['GA_RUN_MODE'] = 'train'
os.environ['GA_FAST_MODE'] = 'false'
os.environ['GA_EVAL_WORKERS'] = '{args.workers}'
import ga_run
ga_run.GA_STAGE1_POP_SIZE = {args.pop}
ga_run.GA_STAGE1_NGEN = {args.gens_per_stage}
ga_run.GA_MAX_WINDOWS = {args.windows}
ga_run.GA_MAX_WINDOWS_STAGE2 = min({args.windows}, 4)
ga_run.GA_EVAL_WORKERS = {args.workers}
ga_run.run()
"""],
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
            timeout=7200,
        )
        elapsed = time.time() - t0
        total_min = (time.time() - t_total) / 60

        try:
            ck = json.load(open('global_ga_checkpoint.json'))
            new_fit = ck['fitness']
        except:
            new_fit = fit

        if result.returncode != 0:
            print(f"Stage {stage} crashed ({elapsed:.0f}s) — fitness: {new_fit:.4f} — continuing", flush=True)
        else:
            print(f"Stage {stage} done ({elapsed:.0f}s) — fitness: {new_fit:.4f} | total: {total_min:.0f}min", flush=True)

    # Final
    ck = json.load(open('global_ga_checkpoint.json'))
    print(f"\n{'='*60}", flush=True)
    print(f"ALL DONE: fitness={ck['fitness']:.4f} | {(time.time()-t_total)/60:.0f}min", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
