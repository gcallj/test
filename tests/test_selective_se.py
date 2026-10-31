"""Strategy v3 (selective_se): causality of features, labels and walk-forward training."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import market_se  # noqa: E402
import selective_se as v3  # noqa: E402

BENCH = market_se.BENCHMARK_SYMBOL


def _px(seed=1, n=900, k=14):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2012-01-02", periods=n)
    cols = {f"S{i}.ST": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n))) for i in range(k)}
    cols[BENCH] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    adj = pd.DataFrame(cols, index=days)
    opn = adj.shift(1).fillna(adj) * (1 + rng.normal(0, 0.003, adj.shape))
    vol = pd.DataFrame(1e6, index=days, columns=adj.columns)
    return {"adj": adj, "close": adj, "open": opn, "adj_open": opn, "volume": vol, "turnover": adj * vol}


class FeatureCausalityTest(unittest.TestCase):
    def test_features_ignore_future_prices(self):
        px = _px()
        cut = px["adj"].index[600]
        fut = {k: v.copy() for k, v in px.items()}
        for k in ("adj", "close", "open", "adj_open", "turnover"):
            fut[k].loc[fut[k].index > cut] *= 3.0
        a, b = v3.build(px), v3.build(fut)
        for name in v3.STOCK_FEATURES:
            pd.testing.assert_frame_equal(a.stock[name].loc[:cut], b.stock[name].loc[:cut], check_exact=False)
        pd.testing.assert_frame_equal(a.market.loc[:cut], b.market.loc[:cut], check_exact=False)

    def test_label_starts_at_the_next_open(self):
        px = _px()
        d = v3.build(px)
        fwd = v3.forward_return(d, 10)
        t, i = "S3.ST", 500
        expected = d.adj[t].iloc[i + 10] / d.adj_open[t].iloc[i + 1] - 1
        self.assertAlmostEqual(fwd[t].iloc[i], expected, places=12)

    def test_purge_leaves_no_label_overlapping_the_retrain_date(self):
        days = pd.bdate_range("2015-01-01", periods=400)
        retrain = days[300]
        cut = v3.purged_cutoff(days, retrain, 20)
        i_cut = days.get_loc(cut)
        # the last sample's label ends at the close of i_cut + 20, before the retrain date
        self.assertLess(days[i_cut + 20], retrain)
        self.assertLess(days[i_cut + 1], retrain)

    def test_weekly_dates_are_last_trading_day_of_each_week(self):
        days = pd.bdate_range("2020-01-01", "2020-03-31")
        wd = v3.weekly_dates(days, "2020-01-01", "2020-03-31")
        self.assertTrue(all(x.dayofweek == 4 for x in wd[:-1]))


class StatisticsTest(unittest.TestCase):
    def test_bootstrap_detects_a_real_mean_and_not_noise(self):
        rng = np.random.default_rng(0)
        self.assertLess(v3.stationary_bootstrap_p(pd.Series(rng.normal(0.003, 0.01, 1500)), reps=800), 0.01)
        self.assertGreater(v3.stationary_bootstrap_p(pd.Series(rng.normal(0.0, 0.01, 1500)), reps=800), 0.05)

    def test_newey_west_t_close_to_plain_t_for_iid(self):
        rng = np.random.default_rng(1)
        x = pd.Series(rng.normal(0.05, 1.0, 4000))
        plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
        self.assertAlmostEqual(v3.newey_west_t(x, 3), plain, delta=0.4)

    def test_deflated_sharpe_penalises_many_trials(self):
        rng = np.random.default_rng(2)
        r = pd.Series(rng.normal(0.0006, 0.01, 1500))
        few = v3.deflated_sharpe(r, [0.0, 0.01])["dsr"]
        many = v3.deflated_sharpe(r, list(rng.normal(0, 0.03, 200)))["dsr"]
        self.assertGreater(few, many)


class SimulationTest(unittest.TestCase):
    def test_minimum_hold_and_capacity(self):
        px = _px(seed=4)
        d = v3.build(px)
        wd = v3.weekly_dates(d.days, d.days[300], d.days[-1])
        rng = np.random.default_rng(3)
        idx = pd.MultiIndex.from_product([wd, [t for t in d.tickers]], names=["date", "ticker"])
        pred = pd.Series(rng.normal(size=len(idx)), index=idx)
        mpred = pd.Series(0.01, index=wd)
        spec = v3.Spec3(horizon=10, entry_q=0.2, k=3)
        res = v3.simulate(d, pred, mpred, spec, d.days[300], d.days[-1])
        self.assertLessEqual(res.slots_used.max(), 3)
        closed = res.trades[res.trades["status"] == "fechada"]
        self.assertGreater(len(closed), 5)
        self.assertTrue((closed["pregoes"] >= spec.horizon).all())

    def test_timing_goes_to_cash(self):
        px = _px(seed=5)
        d = v3.build(px)
        wd = v3.weekly_dates(d.days, d.days[300], d.days[-1])
        idx = pd.MultiIndex.from_product([wd, d.tickers], names=["date", "ticker"])
        pred = pd.Series(np.random.default_rng(1).normal(size=len(idx)), index=idx)
        mpred = pd.Series(-0.01, index=wd)
        res = v3.simulate(d, pred, mpred, v3.Spec3(horizon=10, timing=True), d.days[300], d.days[-1])
        self.assertEqual(res.invested.max(), 0.0)


if __name__ == "__main__":
    unittest.main()
