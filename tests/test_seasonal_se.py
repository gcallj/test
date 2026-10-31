"""Strategy v7 (seasonal_se): ex-dividend detection, dividend-month selection, season switches, cash and costs."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import seasonal_se as v7  # noqa: E402


def _data(seed=1, n=800, k=8):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2019-01-02", periods=n)
    tk = [f"S{i}.ST" for i in range(k)]
    adj = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, (n, k)), axis=0)), index=days, columns=tk)
    opx = adj.shift(1).fillna(adj) * (1 + rng.normal(0, 0.002, (n, k)))
    dy = pd.DataFrame(0.0, index=days, columns=tk)
    elig = pd.DataFrame(True, index=days, columns=tk)
    quint = pd.DataFrame(rng.integers(0, 5, (n, k)).astype(float), index=days, columns=tk)
    nav = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n))), index=days)
    rate = pd.Series(2.0, index=days)
    return v7.Data(days, tk, adj, opx, elig, quint, dy, dy.rolling(252, min_periods=1).sum(), nav, nav.copy(),
                   pd.Series(0.0, index=days), rate)


class ExDividendTest(unittest.TestCase):
    def test_yield_from_the_yahoo_adjustment_factor(self):
        days = pd.bdate_range("2020-01-01", periods=10)
        close = pd.DataFrame({"A.ST": 100.0}, index=days)
        adj = close.copy()
        adj.iloc[:5] *= 1 - 0.03          # 3% dividend goes ex on day 5
        dy = v7.ex_dividend_yield(close, adj)
        self.assertAlmostEqual(dy["A.ST"].iloc[5], 0.03, places=10)
        self.assertEqual(dy["A.ST"].drop(days[5]).abs().sum(), 0.0)
        adj2 = close.copy()
        adj2.iloc[:5] *= 1 - 0.001        # 0.1%: below the floor, ignored
        self.assertEqual(v7.ex_dividend_yield(close, adj2).abs().sum().sum(), 0.0)


class SelectionTest(unittest.TestCase):
    def test_predicted_payers_use_last_years_month_and_rank_by_yield(self):
        d = _data()
        d.dy.loc["2019-04-10", "S1.ST"] = 0.05
        d.dy.loc["2019-04-20", "S2.ST"] = 0.02
        d.dy.loc["2019-05-05", "S3.ST"] = 0.08          # May, not April
        d.dy.loc["2020-04-03", "S4.ST"] = 0.09          # this year: unknown at the decision
        d.trailing_yield = d.dy.rolling(252, min_periods=1).sum()
        april = pd.Timestamp("2020-04-01")
        dec = d.days[d.days.get_loc(april) - 1]
        self.assertEqual(v7.predicted_payers(d, dec, april, 6), ["S1.ST", "S2.ST"])
        d.eligible.loc[dec, "S1.ST"] = False
        self.assertEqual(v7.predicted_payers(d, dec, april, 6), ["S2.ST"])

    def test_events_ignore_prices_after_the_exit(self):
        d = _data(seed=2)
        for m in (3, 4, 5):
            d.dy.loc[f"2019-{m:02d}-10", f"S{m}.ST"] = 0.04
        d.trailing_yield = d.dy.rolling(252, min_periods=1).sum()
        a = v7.dividend_month_events(d, "2020-01-01", "2020-12-31", 6)
        self.assertEqual(set(a["ticker"]), {"S3.ST", "S4.ST", "S5.ST"})
        d2 = _data(seed=2)
        d2.dy, d2.trailing_yield = d.dy, d.trailing_yield
        d2.opx.loc["2020-07-01":] *= 3.0
        d2.adj.loc["2020-07-01":] *= 3.0
        b = v7.dividend_month_events(d2, "2020-01-01", "2020-12-31", 6)
        pd.testing.assert_frame_equal(a, b)


class SimulationTest(unittest.TestCase):
    def test_t1_switches_on_the_first_session_of_may_and_november(self):
        d = _data()
        r = v7.simulate(d, "2019-02-01", "2020-12-31", t1=True, t2=False)
        dates = [s["date"] for s in r.switches]
        self.assertEqual(dates[:4], ["2019-02-01", "2019-05-01", "2019-11-01", "2020-05-01"])
        self.assertEqual([s["to"] for s in r.switches][:4], ["fundo", "caixa", "fundo", "caixa"])
        self.assertAlmostEqual(r.cash_weight.loc["2019-07-15"], 1.0)
        self.assertAlmostEqual(r.fund_weight.loc["2019-12-16"], 1.0)

    def test_cash_earns_the_policy_rate_and_the_fund_tracks_the_index(self):
        d = _data()
        r = v7.simulate(d, "2019-05-01", "2019-10-31", t1=True, t2=False)
        n = len(r.equity)
        self.assertAlmostEqual(r.equity.iloc[-1] / r.equity.iloc[0], (1 + 0.02 / 252) ** (n - 1), places=9)
        r2 = v7.simulate(d, "2019-11-01", "2020-04-30", t1=True, t2=False)
        nav = d.nav.loc[r2.equity.index]
        np.testing.assert_allclose(r2.equity / r2.equity.iloc[0], nav / nav.iloc[0], rtol=1e-9)

    def test_t2_trades_span_one_calendar_month_and_pay_both_sides(self):
        d = _data(seed=3)
        d.dy.loc["2019-04-10", "S1.ST"] = 0.05
        d.trailing_yield = d.dy.rolling(252, min_periods=1).sum()
        r = v7.simulate(d, "2020-01-02", "2020-12-31", t1=False, t2=True)
        t = r.trades
        self.assertEqual(len(t), 1)
        self.assertEqual((t["data_compra"].iloc[0], t["data_venda"].iloc[0]), ("2020-04-01", "2020-05-01"))
        gross = t["preco_venda"].iloc[0] / t["preco_compra"].iloc[0]
        b, s = 0.0025, 0.0010
        self.assertAlmostEqual(t["resultado_liq"].iloc[0], gross * (1 - b) ** 2 * (1 - s) / (1 + s) - 1, places=9)
        self.assertLessEqual(r.slots_used.max(), 6)


if __name__ == "__main__":
    unittest.main()
