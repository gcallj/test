"""v6.0 aggregation must surface T3/D1 per-ticker fields into the operable snapshot."""

import sys
import types
import unittest

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
from run_local_ga_staged import _snapshot_metrics


def _operable_ticker_stat(**overrides):
    """Return a ticker stat dict that passes is_operational_stat (PREMIUM tier)."""
    base = {
        "win_rate": 0.78, "win_rate_target": 0.74,
        "alpha_ann": 0.0, "mdd": -0.10, "n_trades": 50.0,
        "expectancy": 0.005, "median_hold_bars": 5.0, "max_hold_bars": 10.0,
        "swing_survival_3d": 0.7, "wr_after_3_bars": 0.7,
        "target_wr_after_3_bars": 0.6, "early_take_dependency": 0.2,
        "expected_hold_score": 80.0,
        "pct_trades_below_breakeven": 0.0,
        "max_consec_losses": 0.0,
        "max_consec_wins": 0.0,
        "breakeven_magnitude": 0.011,
    }
    base.update(overrides)
    return base


class OperationalAggregatesT3D1(unittest.TestCase):
    def test_empty_returns_zero_aggregates(self):
        out = ga_run.operational_metrics_from_stats([])
        self.assertEqual(out["pct_trades_below_breakeven_med_operable"], 0.0)
        self.assertEqual(out["max_consec_losses_p75_operable"], 0.0)
        self.assertEqual(out["max_consec_wins_med_operable"], 0.0)

    def test_aggregates_match_median_p75(self):
        stats = [
            _operable_ticker_stat(pct_trades_below_breakeven=0.10, max_consec_losses=2, max_consec_wins=5),
            _operable_ticker_stat(pct_trades_below_breakeven=0.20, max_consec_losses=3, max_consec_wins=6),
            _operable_ticker_stat(pct_trades_below_breakeven=0.30, max_consec_losses=4, max_consec_wins=7),
            _operable_ticker_stat(pct_trades_below_breakeven=0.40, max_consec_losses=5, max_consec_wins=8),
        ]
        out = ga_run.operational_metrics_from_stats(stats)
        self.assertAlmostEqual(out["pct_trades_below_breakeven_med_operable"], 0.25, places=10)
        # p75 of [2,3,4,5] = 4.25
        self.assertAlmostEqual(out["max_consec_losses_p75_operable"], 4.25, places=10)
        self.assertAlmostEqual(out["max_consec_wins_med_operable"], 6.5, places=10)
        self.assertEqual(out["n_buy_operable"], 4)

    def test_non_operable_tickers_excluded(self):
        # First ticker is operable (wr>=74, wr_tgt>=72); second is not (low wr).
        stats = [
            _operable_ticker_stat(pct_trades_below_breakeven=0.5, max_consec_losses=10),
            _operable_ticker_stat(win_rate=0.50, win_rate_target=0.40, pct_trades_below_breakeven=0.0, max_consec_losses=0),
        ]
        out = ga_run.operational_metrics_from_stats(stats)
        self.assertEqual(out["n_buy_operable"], 1)
        # Aggregation looks at the single operable ticker only.
        self.assertAlmostEqual(out["pct_trades_below_breakeven_med_operable"], 0.5, places=10)


class SnapshotMetricsIncludeV6(unittest.TestCase):
    def test_new_keys_round_trip(self):
        metrics = {
            "fit": 18.45,
            "wr_med_all": 0.65,
            "wr_target_med_all": 0.60,
            "mean_alpha_ann": -0.05,
            "alpha_ann_pos_rate": 0.30,
            "mdd_med": 0.12,
            "mdd_p75": 0.18,
            "mdd_duration_med": 60.0,
            "avg_hold_bars_med": 5.5,
            "median_hold_bars_med": 5.0,
            "max_hold_bars_p75": 10.0,
            "trades_med": 40.0,
            "n_ge_70": 7,
            "pct_trades_below_breakeven_med_operable": 0.22,
            "max_consec_losses_p75_operable": 4.0,
            "max_consec_wins_med_operable": 6.0,
        }
        snap = _snapshot_metrics(metrics)
        for k in ("pct_trades_below_breakeven_med_operable",
                  "max_consec_losses_p75_operable",
                  "max_consec_wins_med_operable"):
            self.assertIn(k, snap, f"snapshot must surface {k}")
        self.assertAlmostEqual(snap["pct_trades_below_breakeven_med_operable"], 0.22, places=10)
        self.assertAlmostEqual(snap["max_consec_losses_p75_operable"], 4.0, places=10)
        self.assertAlmostEqual(snap["max_consec_wins_med_operable"], 6.0, places=10)

    def test_missing_v6_keys_default_to_zero(self):
        # Legacy metric dicts (pre-v6.0) shouldn't crash; defaults to 0.
        snap = _snapshot_metrics({"fit": 18.45, "n_ge_70": 5})
        self.assertEqual(snap["pct_trades_below_breakeven_med_operable"], 0.0)
        self.assertEqual(snap["max_consec_losses_p75_operable"], 0.0)
        self.assertEqual(snap["max_consec_wins_med_operable"], 0.0)


if __name__ == "__main__":
    unittest.main()
