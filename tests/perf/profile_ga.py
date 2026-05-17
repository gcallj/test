"""Synthetic-workload profile entry point (plan item R2 scaffold).

Wraps `cProfile` over a small fabricated workload that exercises the per-genome
hot path (`_finalize_backtest_stats` + `global_fitness_from_stats`). Designed to
run without real PayloadStore data so it works in any environment.

For the *real* profile (the one that decides whether Numba is worth it),
swap in a real fold/genome pair from a recent sweep — the structure of this
script is the template. Use `--top N` to limit the printed report.

Usage:

    python tests/perf/profile_ga.py
    python tests/perf/profile_ga.py --top 30 --reps 200
    python tests/perf/profile_ga.py --stats-out profile.pstats

Then inspect with `snakeviz profile.pstats` (install separately) or:

    python -c "import pstats; pstats.Stats('profile.pstats').sort_stats('cumulative').print_stats(30)"
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import types
from pathlib import Path

import numpy as np

# Stub heavy deps so the script runs even when GA env isn't set up.
if "matplotlib" not in sys.modules:
    _mpl = types.ModuleType("matplotlib")
    _pyplot = types.ModuleType("matplotlib.pyplot")
    _mpl.pyplot = _pyplot
    sys.modules["matplotlib"] = _mpl
    sys.modules["matplotlib.pyplot"] = _pyplot

if "deap" not in sys.modules:
    _deap = types.ModuleType("deap")
    _base = types.ModuleType("deap.base")
    _creator = types.ModuleType("deap.creator")
    _tools = types.ModuleType("deap.tools")
    _algorithms = types.ModuleType("deap.algorithms")

    def _creator_create(name, *args, **kwargs):
        setattr(_creator, name, type(name, (), {}))

    _creator.create = _creator_create
    _deap.base = _base
    _deap.creator = _creator
    _deap.tools = _tools
    _deap.algorithms = _algorithms
    sys.modules["deap"] = _deap
    sys.modules["deap.base"] = _base
    sys.modules["deap.creator"] = _creator
    sys.modules["deap.tools"] = _tools
    sys.modules["deap.algorithms"] = _algorithms

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import ga_run  # noqa: E402


def _fabricate_ticker_stat(rng: np.random.Generator) -> dict:
    """Mint one per-ticker stats dict by running _finalize_backtest_stats on
    a realistic synthetic trade-return distribution."""
    n_trades = int(rng.integers(20, 60))
    returns = rng.normal(loc=0.005, scale=0.03, size=n_trades).astype(float)
    exit_types = rng.choice(["take", "stop", "time", "trailing", "hard_loss"], size=n_trades).tolist()
    hold_bars = rng.integers(2, 12, size=n_trades).astype(float).tolist()
    return ga_run._finalize_backtest_stats(
        trade_rets=returns.tolist(),
        trade_exit_types=exit_types,
        trade_hold_bars=hold_bars,
        exposure=float(rng.uniform(0.4, 0.8)),
        total_return=float(returns.sum()),
        mdd=float(-rng.uniform(0.05, 0.25)),
        n_bars=252,
        max_drawdown_duration=int(rng.integers(10, 100)),
    )


def _workload(reps: int, n_tickers: int, seed: int = 42) -> float:
    """Approximation of the per-genome cost: aggregate fitness over N tickers, R times."""
    rng = np.random.default_rng(seed)
    stats = [_fabricate_ticker_stat(rng) for _ in range(n_tickers)]
    last_fit = 0.0
    for _ in range(reps):
        last_fit = ga_run.global_fitness_from_stats(stats)
    return last_fit


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Profile the per-genome hot path with synthetic data.")
    p.add_argument("--reps", type=int, default=200, help="How many fitness evaluations to repeat (default 200).")
    p.add_argument("--tickers", type=int, default=50, help="Synthetic per-ticker stats per fold (default 50).")
    p.add_argument("--top", type=int, default=20, help="Top-N pstats rows to print.")
    p.add_argument("--stats-out", default=None, help="Optional path to dump raw pstats binary for snakeviz.")
    p.add_argument("--sort", default="cumulative", help="pstats sort key (cumulative|tottime|...).")
    args = p.parse_args(argv)

    pr = cProfile.Profile()
    pr.enable()
    fitness = _workload(reps=args.reps, n_tickers=args.tickers)
    pr.disable()

    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).strip_dirs().sort_stats(args.sort).print_stats(args.top)
    print(buf.getvalue())
    print(f"[profile_ga] reps={args.reps} tickers={args.tickers} last_fitness={fitness:.4f}")

    if args.stats_out:
        out = Path(args.stats_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pstats.Stats(pr).dump_stats(out)
        print(f"[profile_ga] wrote raw stats to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
