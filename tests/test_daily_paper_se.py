"""Daily orders (daily_se) and paper-trading stop rules (paper_se)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import daily_se  # noqa: E402
import market_se  # noqa: E402
import paper_se  # noqa: E402


class TomorrowBarTest(unittest.TestCase):
    def test_adds_one_empty_business_day(self):
        dates = pd.bdate_range("2026-10-26", periods=5).to_numpy()  # Mon..Fri
        p = {"dates": dates, "close": np.arange(5.0), "score_matrix": np.ones((5, 3)),
             "valid_mask": np.ones(5, dtype=bool), "feat_cols": ["a", "b", "c"]}
        out = daily_se.with_tomorrow(p)
        self.assertEqual(len(out["close"]), 6)
        self.assertTrue(np.isnan(out["close"][-1]))
        self.assertTrue(np.isnan(out["score_matrix"][-1]).all())
        self.assertEqual(pd.Timestamp(out["dates"][-1]).day_name(), "Monday")
        self.assertEqual(out["feat_cols"], ["a", "b", "c"])


def _log(rows):
    df = pd.DataFrame(rows)
    df["recorded_at"] = [f"2026-11-{i + 1:02d}T00:00:00" for i in range(len(df))]
    return df


ACCT = {"stage": "real1", "capital": 10_000.0, "size": 5_000.0}


class StopRulesTest(unittest.TestCase):
    def test_low_fill_rate_stops_trading(self):
        log = _log([{"kind": "missed", "ticker": f"T{i}.ST", "date": "2026-11-02", "price": None,
                     "planned": None, "reason": None, "stage": "real1"} for i in range(5)]
                   + [{"kind": "fill", "ticker": "A.ST", "date": "2026-11-03", "price": 100.0,
                       "planned": 100.0, "reason": None, "stage": "real1"}])
        out = paper_se.check(log, ACCT, {"replay_fill_rate": 0.75, "worst_fold_mdd": -0.10}, sessions=6)
        rules = {name: ok for name, ok, _ in out["rules"]}
        self.assertFalse(rules["fill rate"])
        self.assertTrue(out["advice"].startswith("PARAR"))

    def test_slippage_is_measured_against_the_planned_price(self):
        fill = pd.Series({"kind": "fill", "price": 101.0, "planned": 100.0})
        stop_exit = pd.Series({"kind": "exit", "price": 94.0, "planned": 95.0})
        self.assertAlmostEqual(paper_se.slippage(fill), 0.01)
        self.assertAlmostEqual(paper_se.slippage(stop_exit), 1.0 / 95.0)

    def test_clean_record_advances_after_enough_trades(self):
        rows = []
        for i in range(paper_se.REAL1_MIN_TRADES):
            rows.append({"kind": "fill", "ticker": "A.ST", "date": "d", "price": 100.0, "planned": 100.0,
                         "reason": None, "stage": "real1"})
            rows.append({"kind": "exit", "ticker": "A.ST", "date": "d", "price": 103.0, "planned": 103.0,
                         "reason": "alvo", "stage": "real1"})
        out = paper_se.check(_log(rows), ACCT, {"replay_fill_rate": 0.75, "worst_fold_mdd": -0.10}, sessions=30)
        self.assertTrue(all(ok for _, ok, _ in out["rules"]))
        self.assertIn("real2", out["advice"])
        self.assertEqual(out["trades"], paper_se.REAL1_MIN_TRADES)

    def test_realized_trades_pay_round_trip_costs(self):
        rows = [{"kind": "fill", "ticker": "A.ST", "date": "d1", "price": 100.0, "planned": 100.0, "reason": None, "stage": "paper"},
                {"kind": "exit", "ticker": "A.ST", "date": "d2", "price": 110.0, "planned": 110.0, "reason": "alvo", "stage": "paper"}]
        t = paper_se.realized_trades(_log(rows))
        self.assertAlmostEqual(t.loc[0, "resultado_liq"], 0.10 - market_se.ROUND_TRIP_COST_PCT)


if __name__ == "__main__":
    unittest.main()
