"""Tests for T2 (consistency penalty) and T3 (break-even penalty) — both opt-in."""

import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

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


def _finalize(trade_rets, exit_types=None, hold_bars=None):
    exit_types = exit_types or ["time"] * len(trade_rets)
    hold_bars = hold_bars or [5.0] * len(trade_rets)
    return ga_run._finalize_backtest_stats(
        trade_rets=trade_rets,
        trade_exit_types=exit_types,
        trade_hold_bars=hold_bars,
        exposure=0.5,
        total_return=float(np.sum(trade_rets)),
        mdd=-0.1,
        n_bars=252,
        max_drawdown_duration=10,
    )


class BreakEvenDiagnostic(unittest.TestCase):
    """T3: pct_trades_below_breakeven is emitted regardless of penalty weight."""

    def test_emitted_in_empty_stats(self):
        stats = _finalize([])
        self.assertIn("pct_trades_below_breakeven", stats)
        self.assertIn("breakeven_magnitude", stats)
        self.assertEqual(stats["pct_trades_below_breakeven"], 0.0)

    def test_all_trades_below_breakeven(self):
        # All magnitudes below default 0.011 -> pct=1.0
        stats = _finalize([0.005, -0.003, 0.008, -0.001])
        self.assertEqual(stats["pct_trades_below_breakeven"], 1.0)

    def test_all_trades_above_breakeven(self):
        stats = _finalize([0.05, -0.04, 0.03, -0.02])
        self.assertEqual(stats["pct_trades_below_breakeven"], 0.0)

    def test_mixed(self):
        stats = _finalize([0.05, 0.005, -0.04, -0.003])  # 2 below, 2 above
        self.assertEqual(stats["pct_trades_below_breakeven"], 0.5)

    def test_breakeven_magnitude_reflects_constant(self):
        stats = _finalize([0.05])
        self.assertAlmostEqual(stats["breakeven_magnitude"], ga_run._BREAKEVEN_MAGNITUDE, places=12)


class GlobalFitnessBreakevenPenalty(unittest.TestCase):
    """T3: penalty is OFF by default; opt-in via GA_BREAKEVEN_W."""

    def _per_ticker_stats(self, pct_below):
        # Minimal stats dict sufficient for global_fitness_from_stats inputs.
        return [{
            "sharpe": 0.5, "sortino": 0.5, "total_return": 0.1,
            "buy_hold_return": 0.05, "excess_return": 0.05,
            "alpha_ann": 0.0, "expectancy": 0.005, "payoff_ratio": 1.2,
            "profit_factor": 1.4, "cagr": 0.07, "buy_hold_cagr": 0.05,
            "avg_hold_bars": 5.0, "median_hold_bars": 5.0, "max_hold_bars": 10.0,
            "max_drawdown_duration": 30.0, "mdd": -0.10, "n_trades": 20.0,
            "exposure": 0.5, "win_rate": 0.6, "win_rate_target": 0.5,
            "dist_quality": 0.5, "big_wins_pct": 0.10, "big_losses_pct": 0.05,
            "pct_trades_below_breakeven": pct_below,
        }]

    def test_default_off_no_effect(self):
        stats_high = self._per_ticker_stats(pct_below=0.8)
        stats_low = self._per_ticker_stats(pct_below=0.0)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GA_BREAKEVEN_W", None)
            fit_high = ga_run.global_fitness_from_stats(stats_high)
            fit_low = ga_run.global_fitness_from_stats(stats_low)
        # Default off — pct_below doesn't affect fitness.
        self.assertEqual(fit_high, fit_low)

    def test_optin_penalizes_high_pct_below(self):
        stats_high = self._per_ticker_stats(pct_below=0.8)
        stats_low = self._per_ticker_stats(pct_below=0.0)
        with mock.patch.dict(os.environ, {"GA_BREAKEVEN_W": "5.0"}, clear=False):
            fit_high = ga_run.global_fitness_from_stats(stats_high)
            fit_low = ga_run.global_fitness_from_stats(stats_low)
        # Penalty kicks in -> high pct_below yields lower fitness.
        self.assertLess(fit_high, fit_low)
        self.assertAlmostEqual(fit_low - fit_high, 5.0 * (0.8 - 0.0), places=6)


class WalkForwardConsistencyPenalty(unittest.TestCase):
    """T2: consistency penalty is OFF by default."""

    def _stub_walkforward(self, val_fits):
        """Drive evaluate_global_walkforward by stubbing the inner fitness fn.

        We can't run a full walk-forward in a unit test, so we monkey-patch
        backtest_stats_global_intraday + buyhold_capped + global_fitness_from_stats
        to inject deterministic per-fold fitnesses.
        """
        # Build fake payloads: one window per val_fit value.
        windows = []
        for _ in val_fits:
            payload = {
                "open": np.linspace(100, 101, 30),
                "high": np.linspace(100, 102, 30),
                "low": np.linspace(99, 100, 30),
                "close": np.linspace(100, 101, 30),
                "score_matrix": np.zeros((30, 2)),
                "atr": np.full(30, 1.0),
                "vol_rank": np.full(30, 0.5),
                "volume": np.full(30, 1_000_000.0),
            }
            windows.append(({"A": payload}, {"A": payload}))

        # Two-call pattern: train + test fitness for each window.
        # Inject same value for train+test (no overfit penalty), then verify
        # that consistency penalty alone changes the final fitness.
        seq = []
        for v in val_fits:
            seq.append(v)  # train
            seq.append(v)  # test
        call_state = {"i": 0}

        def fake_fitness(_stats):
            i = call_state["i"]
            call_state["i"] = i + 1
            return seq[i]

        return windows, fake_fitness

    def test_default_off_consistency_does_not_affect_fitness(self):
        windows, fake = self._stub_walkforward([10.0, 10.0, 10.0])
        with mock.patch.object(ga_run, "global_fitness_from_stats", side_effect=fake), \
             mock.patch.object(ga_run, "buyhold_capped", return_value=0.0):
            os.environ.pop("GA_CONSISTENCY_W", None)
            os.environ.pop("GA_CONSISTENCY_MAX_BAD_FOLDS", None)
            fitness, _, _ = ga_run.evaluate_global_walkforward([0.0], windows)
        self.assertAlmostEqual(fitness, 10.0, places=6)

    def test_optin_high_variance_lowers_fitness(self):
        # Mean = 10, std ~ 4.0 across folds.
        windows_a, fake_a = self._stub_walkforward([10.0, 10.0, 10.0])
        windows_b, fake_b = self._stub_walkforward([5.0, 10.0, 15.0])
        with mock.patch.dict(os.environ, {"GA_CONSISTENCY_W": "1.0"}, clear=False), \
             mock.patch.object(ga_run, "buyhold_capped", return_value=0.0):
            with mock.patch.object(ga_run, "global_fitness_from_stats", side_effect=fake_a):
                fit_low_var, _, _ = ga_run.evaluate_global_walkforward([0.0], windows_a)
            with mock.patch.object(ga_run, "global_fitness_from_stats", side_effect=fake_b):
                fit_high_var, _, _ = ga_run.evaluate_global_walkforward([0.0], windows_b)
        # Both have the same mean — only variance differs. Penalty lowers high-var.
        self.assertGreater(fit_low_var, fit_high_var)
        # std([5,10,15]) ≈ 4.082; penalty = W * std ≈ 4.082
        self.assertAlmostEqual(fit_low_var - fit_high_var, float(np.std([5.0, 10.0, 15.0])), places=4)

    def test_max_bad_folds_triggers_haircut(self):
        # 3 negative folds out of 3 — far above the threshold of 2.
        windows, fake = self._stub_walkforward([-1.0, -1.0, -1.0])
        with mock.patch.dict(os.environ, {
            "GA_CONSISTENCY_W": "0.0",
            "GA_CONSISTENCY_MAX_BAD_FOLDS": "2",
        }, clear=False), \
             mock.patch.object(ga_run, "global_fitness_from_stats", side_effect=fake), \
             mock.patch.object(ga_run, "buyhold_capped", return_value=0.0):
            fitness, _, _ = ga_run.evaluate_global_walkforward([0.0], windows)
        # Without haircut: mean_te = -1.0. With haircut: -1.0 * 0.5 = -0.5.
        self.assertAlmostEqual(fitness, -0.5, places=6)


class FitnessVersionBumped(unittest.TestCase):
    def test_version_is_v6_after_t2_t3_landed(self):
        from analysis.fitness_versioning import FITNESS_VERSION
        self.assertEqual(FITNESS_VERSION, "v6.0")


if __name__ == "__main__":
    unittest.main()
