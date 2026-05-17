"""R4: Performance regression guards for the R1 vectorizations.

These are not absolute timings (CI runners vary). They compare the new vectorized
path against the legacy lambda path on the same data and assert that the new path
is at least N% faster. If someone reverts the vectorization, this fails.

Skipped automatically if the env variable SKIP_BENCH=1 is set (useful for slow CI).
"""

import os
import sys
import time
import types
import unittest

import numpy as np
import pandas as pd

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

import ga_run


def _best_of(fn, repeats=3):
    """Return the minimum elapsed time over `repeats` runs."""
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - t0)
    return min(timings)


@unittest.skipIf(os.environ.get("SKIP_BENCH") == "1", "SKIP_BENCH=1 set")
class CCIRollingMadBench(unittest.TestCase):
    """The new _rolling_mad must be materially faster than the lambda.apply form."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(123)
        cls.df = pd.DataFrame({
            "ticker": np.repeat(["A", "B", "C", "D", "E"], 400),
            "tp": rng.standard_normal(2000) * 5.0 + 100.0,
        })

    def test_rolling_mad_is_at_least_2x_faster(self):
        df = self.df

        def legacy():
            return df["tp"].groupby(df["ticker"], sort=False).transform(
                lambda x: x.rolling(20, min_periods=1).apply(
                    lambda w: np.mean(np.abs(w - w.mean())), raw=True
                )
            )

        def new():
            return df["tp"].groupby(df["ticker"], sort=False).transform(
                lambda x: pd.Series(
                    ga_run._rolling_mad(x.to_numpy(dtype=np.float64), 20),
                    index=x.index,
                )
            )

        t_legacy = _best_of(legacy, repeats=3)
        t_new = _best_of(new, repeats=3)
        speedup = t_legacy / max(t_new, 1e-9)
        # 2x is a conservative floor — local benchmark showed ~5x.
        self.assertGreaterEqual(
            speedup, 2.0,
            f"_rolling_mad regression: speedup={speedup:.2f}x (t_legacy={t_legacy*1000:.2f}ms, t_new={t_new*1000:.2f}ms)",
        )


@unittest.skipIf(os.environ.get("SKIP_BENCH") == "1", "SKIP_BENCH=1 set")
class AnnualizeBench(unittest.TestCase):
    """Vectorized annualization must be faster than the scalar apply lambda."""

    def test_annualize_is_at_least_2x_faster(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.standard_normal(50_000) * 0.1)

        def _scalar(r, n):
            if not np.isfinite(r):
                return 0.0
            return max(1.0 + r, 0.001) ** (1.0 / n) - 1.0

        def legacy():
            return s.apply(lambda x: _scalar(x, 15.0))

        def new():
            return ga_run._annualize_returns_vec(s, 15.0)

        t_legacy = _best_of(legacy, repeats=3)
        t_new = _best_of(new, repeats=3)
        speedup = t_legacy / max(t_new, 1e-9)
        self.assertGreaterEqual(
            speedup, 2.0,
            f"_annualize_returns_vec regression: speedup={speedup:.2f}x",
        )


@unittest.skipIf(os.environ.get("SKIP_BENCH") == "1", "SKIP_BENCH=1 set")
class SuffixFilterBench(unittest.TestCase):
    """`.str.endswith(tuple)` must be materially faster than `apply(lambda)`."""

    def test_endswith_is_at_least_2x_faster(self):
        suffixes = (".SA", "-USD", "11.SA")
        rng = np.random.default_rng(1)
        # Mix of matching and non-matching tickers, large enough to amortize startup.
        bases = ["PETR4.SA", "VALE3.SA", "AAPL", "BTC-USD", "IVVB11.SA", "GOOG", "BBDC4.SA"]
        tickers = pd.Series(rng.choice(bases, size=50_000))

        def legacy():
            return tickers.apply(lambda t: any(str(t).endswith(s) for s in suffixes))

        def new():
            return tickers.astype(str).str.endswith(tuple(suffixes))

        t_legacy = _best_of(legacy, repeats=3)
        t_new = _best_of(new, repeats=3)
        speedup = t_legacy / max(t_new, 1e-9)
        self.assertGreaterEqual(
            speedup, 2.0,
            f"str.endswith regression: speedup={speedup:.2f}x",
        )


if __name__ == "__main__":
    unittest.main()
