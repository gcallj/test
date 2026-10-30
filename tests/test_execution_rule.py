"""Execution rule of the backtest (docs/adr/0001).

A buy signal at the close of bar s becomes a limit order live for
ORDER_VALIDITY_DAYS bars; it fills at the open when the market opens through
the limit, at the limit when the bar trades through it, and otherwise expires
as a missed entry. Decisions never use data from the bar being decided.
"""

import dataclasses
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ga_run as gr  # noqa: E402
import ga_run_modular_final as gm  # noqa: E402


def _market(seed: int, n: int = 1200):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.016, n)))
    o = c * (1 + rng.normal(0, 0.006, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
    prev = np.roll(c, 1)
    prev[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()
    x = rng.normal(size=(n, 10))
    return o, h, l, c, x, atr


def _params(**overrides):
    genome = json.loads((ROOT / "global_ga_checkpoint.json").read_text(encoding="utf-8"))["genome"]
    gp = gr.decode_global_params(gr.sanitize_global_genome(genome))
    return dataclasses.replace(gp, **overrides) if overrides else gp


class ExecutionRuleTest(unittest.TestCase):
    def setUp(self):
        self.o, self.h, self.l, self.c, self.x, self.atr = _market(seed=7)
        self.gp = _params(momentum_confirm_days=0.0, regime_threshold=0.20)

    def _run(self, gp=None, n=None):
        trades = []
        sl = slice(0, n)
        stats = gr.backtest_stats_global_intraday(
            self.o[sl], self.h[sl], self.l[sl], self.c[sl], self.x[sl], self.atr[sl],
            gp or self.gp, trades_out=trades,
        )
        return stats, trades

    def test_fills_follow_the_limit_order_rule(self):
        stats, trades = self._run()
        filled = [t for t in trades if t["status"] == "filled"]
        self.assertGreater(len(filled), 5)
        for t in filled:
            s, f, limit = t["signal_idx"], t["fill_idx"], t["limit"]
            self.assertTrue(1 <= f - s <= gm.ORDER_VALIDITY_DAYS, t)
            # Every bar before the fill left the limit untouched.
            for k in range(s + 1, f):
                self.assertGreater(self.l[k], limit)
            expected = self.o[f] if self.o[f] <= limit else limit
            self.assertAlmostEqual(t["entry_px"], expected, places=9)
            self.assertLessEqual(self.l[f], limit + 1e-9)

    def test_missed_entries_never_fill_at_another_price(self):
        stats, trades = self._run()
        missed = [t for t in trades if t["status"] == "missed"]
        self.assertGreater(len(missed), 0)
        for t in missed:
            s = t["signal_idx"]
            for k in range(s + 1, s + 1 + gm.ORDER_VALIDITY_DAYS):
                self.assertGreater(self.l[k], t["limit"])
            self.assertEqual(t["exit_idx"], s + gm.ORDER_VALIDITY_DAYS)
        n_orders = stats["n_orders"]
        self.assertAlmostEqual(stats["fill_rate"], (n_orders - stats["n_missed"] - stats["n_cancelled"]) / n_orders)

    def test_cost_is_charged_per_side(self):
        gp = dataclasses.replace(self.gp, partial_take_pct=0.0, partial_take_pct_2=0.0)
        _, trades = self._run(gp)
        closed = [t for t in trades if t["status"] == "filled" and "exit_px" in t]
        self.assertGreater(len(closed), 5)
        for t in closed:
            gross = t["exit_px"] / t["entry_px"] - 1.0
            self.assertAlmostEqual(t["ret"], gross - gm.ROUND_TRIP_COST_PCT, places=9)

    def test_decisions_do_not_use_future_bars(self):
        cut = 900
        _, full = self._run()
        _, prefix = self._run(n=cut)
        # Every order resolved strictly before the cut must be identical.
        done_full = [t for t in full if t.get("exit_idx", 10**9) < cut - 1]
        done_prefix = [t for t in prefix if t.get("exit_idx", 10**9) < cut - 1]
        self.assertGreater(len(done_full), 5)
        self.assertEqual(len(done_full), len(done_prefix))
        for a, b in zip(done_full, done_prefix):
            self.assertEqual(a["signal_idx"], b["signal_idx"])
            self.assertEqual(a["status"], b["status"])
            self.assertAlmostEqual(a["limit"], b["limit"], places=9)
            self.assertAlmostEqual(a.get("ret", 0.0), b.get("ret", 0.0), places=9)

    def test_trace_does_not_change_the_result(self):
        trace = []
        with_trace = gr.backtest_stats_global_intraday(
            self.o, self.h, self.l, self.c, self.x, self.atr, self.gp, trace=trace)
        without = gr.backtest_stats_global_intraday(
            self.o, self.h, self.l, self.c, self.x, self.atr, self.gp)
        self.assertEqual(with_trace["n_trades"], without["n_trades"])
        self.assertAlmostEqual(with_trace["total_return"], without["total_return"], places=12)
        self.assertEqual(len(trace), len(self.c) - 1)

    def test_ga_trainer_and_evaluator_share_one_engine(self):
        a = gr.backtest_stats_global_intraday(self.o, self.h, self.l, self.c, self.x, self.atr, self.gp,
                                              stage_gate="off")
        b = gm._backtest_stats_global_intraday(self.o, self.h, self.l, self.c, self.x, self.atr, self.gp)
        self.assertEqual(a["n_trades"], b["n_trades"])
        self.assertAlmostEqual(a["total_return"], b["total_return"], places=12)


if __name__ == "__main__":
    unittest.main()
