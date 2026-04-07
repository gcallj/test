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
from pathlib import Path
from datetime import datetime

import numpy as np

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
os.environ["GA_AUTO_TUNE"] = "0"
os.environ["GA_RUN_MODE"] = "train"
os.environ["GA_EVAL_WORKERS"] = "1"

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
MAIN_CHECKPOINT = "global_ga_checkpoint.json"
BACKUP_MAIN = "global_ga_checkpoint_pre_staged.json.bak"


def _fmt_pct(x):
    return f"{x*100:.2f}%"


def _fmt_pct1(x):
    return f"{x*100:.1f}%"


def _load_main():
    with open(MAIN_CHECKPOINT, "r") as f:
        return json.load(f)


def _load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"chunks": [], "best_so_far": None}


def _save_progress(data):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _load_best_so_far():
    if os.path.exists(BEST_SOFAR_FILE):
        try:
            with open(BEST_SOFAR_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_best_so_far(data):
    with open(BEST_SOFAR_FILE, "w") as f:
        json.dump(data, f, indent=2)


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

    store_dir = os.path.abspath(gr.GA_MEMMAP_DIR)
    if not os.path.exists(store_dir):
        raise RuntimeError(
            f"No existing PayloadStore at {store_dir}. "
            f"Run 'python ga_run.py' once first (it builds the store during Phase 1)."
        )
    if not os.path.exists(os.path.join(store_dir, "meta.json")):
        raise RuntimeError(
            f"meta.json missing in {store_dir}. Store is corrupted or incomplete."
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

    main = _load_main()
    seed_genome = main.get("genome", [])
    prev_fit = float(main.get("fitness", 0.0))
    print(f"[SETUP] Seed genome fitness: {prev_fit:.4f}", flush=True)

    return store, window_plans, seed_genome, prev_fit


def evaluate_genome_full(genome, ticker_arrays_cache, gr):
    """
    Run full backtest of a genome across all tickers (single process).
    Returns dict with WR_med, WR_mean, MDD_med, ret_med, sharpe_med, n_trades_med, fit.
    """
    gp = gr.decode_global_params(genome)
    stats = []
    for tk, p in ticker_arrays_cache.items():
        st = gr.backtest_stats_global_intraday(
            p["open"], p["high"], p["low"], p["close"],
            p["score_matrix"], p["atr"], gp,
            precomputed=p,
        )
        stats.append(st)

    # Overall fitness uses ALL stats (same as training)
    fit = gr.global_fitness_from_stats(stats)

    # Filtered metrics for reporting (n_trades > 5)
    active = [s for s in stats if s.get("n_trades", 0) > 5]
    all_stats = stats  # unfiltered
    wrs = [s["win_rate"] for s in active]
    mdds = [abs(s["mdd"]) for s in active]
    rets = [s["total_return"] for s in active]
    sharpes = [s["sharpe"] for s in active]
    trades = [s["n_trades"] for s in active]

    all_wrs = [s["win_rate"] for s in all_stats]

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
        "mdd_med": float(np.median(mdds)) if mdds else 0,
        "ret_med": float(np.median(rets)) if rets else 0,
        "sharpe_med": float(np.median(sharpes)) if sharpes else 0,
        "trades_med": float(np.median(trades)) if trades else 0,
        "n_ge_70": int(sum(1 for w in all_wrs if w >= 0.70)),
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
    ngen,
    pop_size,
    window_count,
    store_path,
    window_plans,
    seed_genomes,
    resume_ckpt=None,
):
    """Run one GA chunk (ngen generations). Returns (best_genome, best_fit, hof)."""
    import pickle
    from ga_run_modular_final import (
        _run_stage, _load_latest_checkpoint, GACheckpoint,
    )
    from auto_tune import RuntimeGuard

    wp_bytes = pickle.dumps(window_plans)

    # Uniform window indices for this chunk
    n_win = len(window_plans)
    step = n_win / window_count
    window_indices = [min(max(0, int(round(i * step))), n_win - 1)
                      for i in range(window_count)]

    print(f"\n{'='*60}", flush=True)
    print(f"[CHUNK {chunk_id}] windows={window_indices} pop={pop_size} ngen={ngen}",
          flush=True)
    print(f"{'='*60}", flush=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    t0 = time.time()
    pop, hof = _run_stage(
        stage=2,  # refinement mode (narrower search)
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

    return best_genome, best_fit, hof, elapsed


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
    parser.add_argument("--reset", action="store_true",
                        help="Clear prior progress and restart fresh")
    parser.add_argument("--apply-best", action="store_true",
                        help="If best_found > main, overwrite main checkpoint at end")
    args = parser.parse_args()

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

    # Setup
    import ga_run as gr
    gr.GA_AUTO_TUNE = 0
    gr.FAST_MODE = True
    gr.GA_EVAL_WORKERS = 1

    store, window_plans, seed_genome, prev_main_fit = open_existing_store()

    # Build cache for full metric evaluation
    print("[SETUP] Building ticker array cache for metric eval...", flush=True)
    t0 = time.time()
    cache = build_ticker_arrays_cache(store)
    print(f"[SETUP] Cache ready: {len(cache)} tickers in {time.time()-t0:.0f}s",
          flush=True)

    # Initial metrics
    print("\n[BASE] Evaluating seed genome (current main checkpoint)...",
          flush=True)
    t0 = time.time()
    base_metrics = evaluate_genome_full(seed_genome, cache, gr)
    base_metrics["eval_time_s"] = round(time.time() - t0, 1)
    print(f"[BASE] fit={base_metrics['fit']:.4f} "
          f"WR_med_all={_fmt_pct1(base_metrics['wr_med_all'])} "
          f"WR_mean_all={_fmt_pct1(base_metrics['wr_mean_all'])} "
          f"MDD={_fmt_pct1(base_metrics['mdd_med'])} "
          f"ret={base_metrics['ret_med']:.2f} "
          f"#>=70%={base_metrics['n_ge_70']}/{base_metrics['n_tickers_total']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    progress = _load_progress()
    if not progress["chunks"]:
        progress["base_metrics"] = base_metrics
        _save_progress(progress)

    # Best-so-far tracking
    best_so_far = _load_best_so_far() or {
        "fit": base_metrics["fit"],
        "wr_med_all": base_metrics["wr_med_all"],
        "genome": list(seed_genome),
        "metrics": base_metrics,
        "found_at_chunk": 0,
    }

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
            best_genome, best_fit_train, hof, chunk_time = run_chunk(
                chunk_id=chunk_id,
                ngen=args.ngen_per_chunk,
                pop_size=args.pop_size,
                window_count=args.windows,
                store_path=store_path,
                window_plans=window_plans,
                seed_genomes=seed,
                resume_ckpt=resume_ckpt,
            )
        except Exception as e:
            print(f"[CHUNK {chunk_id}] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            break

        if best_genome is None:
            print(f"[CHUNK {chunk_id}] no best genome returned", flush=True)
            break

        # Full metrics for the best genome of this chunk
        print(f"[CHUNK {chunk_id}] evaluating best genome on all tickers...",
              flush=True)
        t_eval = time.time()
        try:
            chunk_metrics = evaluate_genome_full(best_genome, cache, gr)
            chunk_metrics["eval_time_s"] = round(time.time() - t_eval, 1)
        except Exception as e:
            print(f"[CHUNK {chunk_id}] eval ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            break

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
            "delta_fit_vs_base": round(delta_fit, 4),
            "delta_wr_vs_base_pp": round(delta_wr, 2),
            "best_genome": list(best_genome),
        }
        progress["chunks"].append(chunk_entry)
        _save_progress(progress)
        print(f"[CHUNK {chunk_id}] progress saved to {PROGRESS_FILE}", flush=True)

        # Print metrics (ASCII-safe)
        print(f"[CHUNK {chunk_id}] METRICS vs BASE:", flush=True)
        print(f"  fit={chunk_metrics['fit']:.4f} (delta {delta_fit:+.4f})", flush=True)
        print(f"  WR_med_all (summary)={_fmt_pct1(wr_all)} (delta {delta_wr:+.2f}pp)",
              flush=True)
        print(f"  WR_mean_all         ={_fmt_pct1(chunk_metrics['wr_mean_all'])}",
              flush=True)
        print(f"  WR_med (active)     ={_fmt_pct1(chunk_metrics['wr_med'])}", flush=True)
        print(f"  MDD_med             ={_fmt_pct1(chunk_metrics['mdd_med'])}", flush=True)
        print(f"  Return_med          ={chunk_metrics['ret_med']:.3f}", flush=True)
        print(f"  Sharpe_med          ={chunk_metrics['sharpe_med']:.3f}", flush=True)
        print(f"  Trades_med          ={chunk_metrics['trades_med']:.0f}", flush=True)
        print(f"  N tickers >= 70% WR ={chunk_metrics['n_ge_70']}/{chunk_metrics['n_tickers_total']}",
              flush=True)

        # Clean up per-chunk GA checkpoints AFTER progress is saved
        if os.path.exists(CHECKPOINT_DIR):
            for f in os.listdir(CHECKPOINT_DIR):
                if f.startswith("stage"):
                    try:
                        os.remove(os.path.join(CHECKPOINT_DIR, f))
                    except Exception:
                        pass

        # Update best-so-far if improved
        # Primary: WR_med_all (the metric we care about)
        # Tiebreaker: fitness
        improved = False
        if chunk_metrics["wr_med_all"] > best_so_far["wr_med_all"] + 1e-6:
            improved = True
            reason = f"WR_med_all {_fmt_pct1(best_so_far['wr_med_all'])} -> {_fmt_pct1(chunk_metrics['wr_med_all'])}"
        elif (
            abs(chunk_metrics["wr_med_all"] - best_so_far["wr_med_all"]) < 1e-6
            and chunk_metrics["fit"] > best_so_far["fit"] + 1e-4
        ):
            improved = True
            reason = f"same WR, fit {best_so_far['fit']:.4f} -> {chunk_metrics['fit']:.4f}"

        if improved:
            best_so_far = {
                "fit": chunk_metrics["fit"],
                "wr_med_all": chunk_metrics["wr_med_all"],
                "genome": list(best_genome),
                "metrics": chunk_metrics,
                "found_at_chunk": chunk_id,
            }
            _save_best_so_far(best_so_far)
            print(f"[CHUNK {chunk_id}] *** NEW BEST: {reason} ***", flush=True)
        else:
            print(f"[CHUNK {chunk_id}] no improvement vs best so far "
                  f"(WR {_fmt_pct1(best_so_far['wr_med_all'])} "
                  f"fit {best_so_far['fit']:.4f})", flush=True)

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Base (main checkpoint):", flush=True)
    print(f"  fit={base_metrics['fit']:.4f} "
          f"WR_med_all={_fmt_pct1(base_metrics['wr_med_all'])} "
          f"#>=70%={base_metrics['n_ge_70']}", flush=True)
    print(f"Best found:", flush=True)
    print(f"  fit={best_so_far['fit']:.4f} "
          f"WR_med_all={_fmt_pct1(best_so_far['wr_med_all'])} "
          f"#>=70%={best_so_far['metrics']['n_ge_70']} "
          f"(chunk {best_so_far['found_at_chunk']})", flush=True)

    delta_wr_final = (best_so_far["wr_med_all"] - base_metrics["wr_med_all"]) * 100.0
    print(f"\ndelta WR_med_all: {delta_wr_final:+.2f}pp", flush=True)

    # Apply best if requested and improved
    if args.apply_best:
        if best_so_far["wr_med_all"] > base_metrics["wr_med_all"] + 1e-6:
            if not os.path.exists(BACKUP_MAIN):
                shutil.copy(MAIN_CHECKPOINT, BACKUP_MAIN)
                print(f"[APPLY] Backed up main -> {BACKUP_MAIN}", flush=True)
            with open(MAIN_CHECKPOINT, "w") as f:
                json.dump({
                    "fitness": best_so_far["fit"],
                    "genome": best_so_far["genome"],
                }, f, indent=2)
            print(f"[APPLY] Updated {MAIN_CHECKPOINT} with new best", flush=True)
        else:
            print(f"[APPLY] No improvement -> main checkpoint unchanged", flush=True)


if __name__ == "__main__":
    main()
