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


class ConsecutiveLossesEndToEnd(unittest.TestCase):
    """D1: max_consec_losses / max_consec_wins / current_streak surface via _finalize_backtest_stats."""

    def _finalize(self, trade_rets):
        return ga_run._finalize_backtest_stats(
            trade_rets=trade_rets,
            trade_exit_types=["time"] * len(trade_rets),
            trade_hold_bars=[5.0] * len(trade_rets),
            exposure=0.5,
            total_return=float(np.sum(trade_rets)),
            mdd=-0.1,
            n_bars=252,
            max_drawdown_duration=10,
        )

    def test_empty_trades_returns_zero_streaks(self):
        stats = self._finalize([])
        self.assertEqual(stats["max_consec_losses"], 0.0)
        self.assertEqual(stats["max_consec_wins"], 0.0)
        self.assertEqual(stats["current_streak"], 0.0)

    def test_plan_fixture(self):
        # Plan reference: [+1,+1,-1,-1,-1,+1] -> max_consec_losses=3, max_consec_wins=2, current_streak=+1
        stats = self._finalize([1.0, 1.0, -1.0, -1.0, -1.0, 1.0])
        self.assertEqual(stats["max_consec_losses"], 3.0)
        self.assertEqual(stats["max_consec_wins"], 2.0)
        self.assertEqual(stats["current_streak"], 1.0)

    def test_trailing_losing_streak_is_negative(self):
        # 2 wins then 4 losses at the tail
        stats = self._finalize([1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
        self.assertEqual(stats["max_consec_losses"], 4.0)
        self.assertEqual(stats["max_consec_wins"], 2.0)
        self.assertEqual(stats["current_streak"], -4.0)

    def test_all_wins(self):
        stats = self._finalize([0.01, 0.02, 0.03])
        self.assertEqual(stats["max_consec_losses"], 0.0)
        self.assertEqual(stats["max_consec_wins"], 3.0)
        self.assertEqual(stats["current_streak"], 3.0)

    def test_all_losses(self):
        stats = self._finalize([-0.01, -0.02, -0.03])
        self.assertEqual(stats["max_consec_losses"], 3.0)
        self.assertEqual(stats["max_consec_wins"], 0.0)
        self.assertEqual(stats["current_streak"], -3.0)


if __name__ == "__main__":
    unittest.main()
