"""Monthly momentum rotation (rotation_se, docs/adr/0005)."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import market_se  # noqa: E402
import rotation_se as rot  # noqa: E402

BENCH = market_se.BENCHMARK_SYMBOL


def _px(seed=3, n=700, drifts=None, bench_drift=0.0004):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-01-01", periods=n)
    drifts = drifts if drifts is not None else np.linspace(-0.0006, 0.0012, 10)
    cols = {}
    for i, mu in enumerate(drifts):
        cols[f"T{i}.ST"] = 100 * np.exp(np.cumsum(rng.normal(mu, 0.015, n)))
    cols[BENCH] = 100 * np.exp(np.cumsum(rng.normal(bench_drift, 0.01, n)))
    adj = pd.DataFrame(cols, index=days)
    opn = adj * (1 + rng.normal(0, 0.003, adj.shape))
    vol = pd.DataFrame(1e6, index=days, columns=adj.columns)
    return {"adj": adj, "close": adj, "open": opn, "adj_open": opn, "volume": vol, "turnover": adj * vol}


class RotationTest(unittest.TestCase):
    def test_decisions_do_not_use_future_prices(self):
        px = _px()
        cut = px["adj"].index[500]
        fut = {k: v.copy() for k, v in px.items()}
        for k in ("adj", "close", "open", "adj_open"):
            fut[k].loc[fut[k].index > cut] *= np.linspace(0.5, 2.0, fut[k].shape[1])
        start, end = px["adj"].index[260], px["adj"].index[-1]
        a = rot.simulate(rot.Market(px), rot.Spec(), start, end)
        b = rot.simulate(rot.Market(fut), rot.Spec(), start, end)
        da = [r for r in a.rebalances if pd.Timestamp(r["date"]) <= cut]
        db = [r for r in b.rebalances if pd.Timestamp(r["date"]) <= cut]
        self.assertGreater(len(da), 5)
        self.assertEqual([(r.get("keep"), r.get("buy")) for r in da], [(r.get("keep"), r.get("buy")) for r in db])
        self.assertAlmostEqual(a.equity.loc[:cut].iloc[-1], b.equity.loc[:cut].iloc[-1], places=6)

    def test_closed_trades_pay_both_sides(self):
        px = _px()
        c = rot.Costs()
        res = rot.simulate(rot.Market(px), rot.Spec(market_regime=False), px["adj"].index[260], px["adj"].index[-1], costs=c)
        closed = res.trades[res.trades["status"] == "fechada"]
        self.assertGreater(len(closed), 3)
        for r in closed.itertuples():
            gross = r.preco_venda / r.preco_compra
            expected = gross * (1 - c.slippage) * (1 - c.brokerage) * (1 - c.brokerage) / (1 + c.slippage) - 1
            self.assertAlmostEqual(r.resultado_liq, expected, places=9)

    def test_risk_off_goes_to_cash(self):
        px = _px(bench_drift=-0.003)  # benchmark below its 200-day average
        res = rot.simulate(rot.Market(px), rot.Spec(market_regime=True), px["adj"].index[400], px["adj"].index[-1])
        tail = slice(res.equity.index[30], None)
        self.assertEqual(res.invested.loc[tail].max(), 0.0)
        self.assertEqual(res.fund_weight.loc[tail].max(), 0.0)

    def test_empty_slots_go_to_the_index_fund(self):
        px = _px(drifts=np.full(10, -0.002), bench_drift=0.001)  # no stock passes the trend filter
        res = rot.simulate(rot.Market(px), rot.Spec(fill="index"), px["adj"].index[400], px["adj"].index[-1])
        self.assertGreater(res.fund_weight.iloc[-1], 0.95)
        res_cash = rot.simulate(rot.Market(px), rot.Spec(fill="cash"), px["adj"].index[400], px["adj"].index[-1])
        self.assertEqual(res_cash.fund_weight.max(), 0.0)

    def test_buffer_lowers_turnover(self):
        px = _px(seed=9)
        m = rot.Market(px)
        s, e = px["adj"].index[260], px["adj"].index[-1]
        tight = rot.simulate(m, rot.Spec(k=3, buffer=3, market_regime=False, stock_trend=False), s, e)
        loose = rot.simulate(m, rot.Spec(k=3, buffer=8, market_regime=False, stock_trend=False), s, e)
        self.assertLessEqual(len(loose.trades), len(tight.trades))

    def test_vol_target_cuts_exposure_in_turbulent_markets(self):
        rng = np.random.default_rng(11)
        px = _px(seed=11)
        for k in ("adj", "close", "open", "adj_open"):  # ~60% annual volatility
            noise = np.exp(np.cumsum(rng.normal(0, 0.035, px[k].shape), axis=0))
            px[k] = px[k] * noise
        m = rot.Market(px)
        s, e = px["adj"].index[300], px["adj"].index[-1]
        calm = rot.simulate(m, rot.Spec(market_regime=False, stock_trend=False), s, e)
        vt = rot.simulate(m, rot.Spec(market_regime=False, stock_trend=False, vol_target=0.20), s, e)
        self.assertGreater(calm.invested.iloc[40:].mean(), 0.9)
        self.assertLess(vt.invested.iloc[40:].mean(), 0.6)

    def test_never_more_than_k_positions(self):
        px = _px(seed=5)
        res = rot.simulate(rot.Market(px), rot.Spec(k=4, market_regime=False), px["adj"].index[260], px["adj"].index[-1])
        self.assertLessEqual(res.slots_used.max(), 4)


if __name__ == "__main__":
    unittest.main()
