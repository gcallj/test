import sys
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


class RollingMadParity(unittest.TestCase):
    """_rolling_mad must match Series.rolling(n, min_periods=1).apply(lambda)."""

    def _legacy_mad(self, arr: np.ndarray, n: int) -> np.ndarray:
        return (
            pd.Series(arr)
            .rolling(n, min_periods=1)
            .apply(lambda w: np.mean(np.abs(w - w.mean())), raw=True)
            .to_numpy()
        )

    def test_short_series_partial_windows(self):
        arr = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
        np.testing.assert_allclose(
            ga_run._rolling_mad(arr, 20),
            self._legacy_mad(arr, 20),
            rtol=1e-12, atol=1e-12,
        )

    def test_full_window_random(self):
        rng = np.random.default_rng(42)
        arr = rng.standard_normal(200) * 5.0 + 100.0
        np.testing.assert_allclose(
            ga_run._rolling_mad(arr, 20),
            self._legacy_mad(arr, 20),
            rtol=1e-12, atol=1e-12,
        )

    def test_with_groupby_transform(self):
        # End-to-end: same shape and values as the original CCI MAD path.
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "ticker": ["A"] * 60 + ["B"] * 40,
            "tp": rng.standard_normal(100) * 2.0 + 50.0,
        })
        legacy = df["tp"].groupby(df["ticker"], sort=False).transform(
            lambda x: x.rolling(20, min_periods=1).apply(
                lambda w: np.mean(np.abs(w - w.mean())), raw=True
            )
        )
        new = df["tp"].groupby(df["ticker"], sort=False).transform(
            lambda x: pd.Series(ga_run._rolling_mad(x.to_numpy(dtype=np.float64), 20), index=x.index)
        )
        np.testing.assert_allclose(legacy.to_numpy(), new.to_numpy(), rtol=1e-12, atol=1e-12)


class AnnualizeParity(unittest.TestCase):
    """_annualize_returns_vec must match the scalar _annualize_scalar."""

    @staticmethod
    def _scalar(r, n):
        if not np.isfinite(r):
            return 0.0
        ratio = max(1.0 + r, 0.001)
        return ratio ** (1.0 / n) - 1.0

    def test_matches_scalar_for_typical_returns(self):
        returns = pd.Series([0.30, 0.05, -0.10, -0.95, 0.0, 2.5, np.inf, -np.inf, np.nan])
        n_years = 15.0
        legacy = returns.apply(lambda x: self._scalar(x, n_years))
        new = ga_run._annualize_returns_vec(returns, n_years)
        np.testing.assert_allclose(new.to_numpy(), legacy.to_numpy(), rtol=1e-12, atol=1e-12)

    def test_non_numeric_coerced(self):
        returns = pd.Series([0.10, "0.20", "bad", None])
        new = ga_run._annualize_returns_vec(returns, 15.0)
        # "bad" and None coerce to NaN -> 0.0 per scalar contract
        expected_finite_mask = [True, True, False, False]
        for i, ok in enumerate(expected_finite_mask):
            if ok:
                self.assertAlmostEqual(
                    new.iloc[i],
                    self._scalar(float(returns.iloc[i]) if returns.iloc[i] is not None else float("nan"), 15.0),
                    places=12,
                )
            else:
                self.assertEqual(new.iloc[i], 0.0)


class SuffixFilterParity(unittest.TestCase):
    """`.str.endswith(tuple(...))` must match `.apply(lambda ...endswith)`."""

    def test_matches_apply_form(self):
        suffixes = (".SA", "-USD")
        tickers = pd.Series(["PETR4.SA", "VALE3.SA", "BTC-USD", "AAPL", "IVVB11.SA", ""])
        legacy = tickers.apply(lambda t: any(str(t).endswith(s) for s in suffixes))
        new = tickers.astype(str).str.endswith(tuple(suffixes))
        np.testing.assert_array_equal(new.to_numpy(), legacy.to_numpy())


class MaxConsecutiveParity(unittest.TestCase):
    """_max_consecutive must give the longest run of True."""

    def test_basic(self):
        self.assertEqual(ga_run._max_consecutive(np.array([True, True, False, True])), 2)

    def test_all_false(self):
        self.assertEqual(ga_run._max_consecutive(np.array([False, False, False])), 0)

    def test_all_true(self):
        self.assertEqual(ga_run._max_consecutive(np.array([True] * 5)), 5)

    def test_empty(self):
        self.assertEqual(ga_run._max_consecutive(np.array([], dtype=bool)), 0)

    def test_plan_fixture_3_losses(self):
        # Plan reference: [+1,+1,-1,-1,-1,+1] -> max_consec_losses = 3
        tr = np.array([1.0, 1.0, -1.0, -1.0, -1.0, 1.0])
        self.assertEqual(ga_run._max_consecutive(tr < 0), 3)
        self.assertEqual(ga_run._max_consecutive(tr > 0), 2)


if __name__ == "__main__":
    unittest.main()
