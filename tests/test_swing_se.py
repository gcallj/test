"""Strategy v6 (swing_se): v1's execution rule on one account, causality and the robust fitness."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import swing_se as v6  # noqa: E402

F = len(v6.FEATURES)


def _panel(seed=0, T=320, N=8):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-01-01", periods=T)
    C = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (T, N)), axis=0))
    O = np.vstack([C[:1], C[:-1]]) * (1 + rng.normal(0, 0.003, (T, N)))
    H = np.maximum(O, C) * (1 + np.abs(rng.normal(0, 0.006, (T, N))))
    L = np.minimum(O, C) * (1 - np.abs(rng.normal(0, 0.006, (T, N))))
    ATR = np.full((T, N), 2.0)
    NAV = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, T)))
    X = rng.uniform(-0.5, 0.5, (F, T, N))
    return v6.Panel(days, [f"S{i}.ST" for i in range(N)], O, H, L, C, ATR, NAV, np.ones((T, N), bool), X)


def _genome(**kw):
    w = np.zeros(F)
    w[0] = 1.0
    d = dict(entry_q=0.9, exit_q=0.3, discount_atr=0.5, stop_atr=2.0, reward_risk=3.0, time_stop=20)
    d.update(kw)
    return v6.Genome(w, d["entry_q"], d["exit_q"], d["discount_atr"], d["stop_atr"], d["reward_risk"], int(d["time_stop"]))


def _one_signal(p, i, j=0):
    """Only ticker j is eligible, only on day i: the single possible entry signal."""
    p.elig[:] = False
    p.elig[i, j] = True


class ExecutionTest(unittest.TestCase):
    def _flat(self, p, i):
        p.C[:] = 100.0
        p.O[:] = 100.0
        p.H[:] = 100.5
        p.L[:] = 99.5
        _one_signal(p, i)

    def test_limit_fills_at_open_when_the_market_opens_below(self):
        p = _panel()
        i = 250
        self._flat(p, i)
        p.O[i + 1, 0], p.L[i + 1, 0] = 98.0, 97.8        # limit = 100 - 0.5*2 = 99
        r = v6.run(p, [(p.days[0], _genome())], p.days[200], p.days[-1], k=1)
        self.assertAlmostEqual(r.trades["preco_compra"].iloc[0], 98.0)
        self.assertEqual(r.trades["data_compra"].iloc[0], str(p.days[i + 1].date()))

    def test_limit_fills_at_the_limit_when_touched_and_expires_after_three_sessions(self):
        p = _panel()
        i = 250
        self._flat(p, i)
        p.L[i + 2, 0] = 98.9
        r = v6.run(p, [(p.days[0], _genome())], p.days[200], p.days[-1], k=1)
        self.assertAlmostEqual(r.trades["preco_compra"].iloc[0], 99.0)
        p2 = _panel()
        self._flat(p2, i)
        p2.L[i + 4, 0] = 98.0                              # day 4: the order already expired
        r2 = v6.run(p2, [(p2.days[0], _genome())], p2.days[200], p2.days[-1], k=1)
        self.assertEqual(len(r2.trades), 0)

    def test_stop_gap_exits_at_the_open_and_pays_both_sides(self):
        p = _panel()
        i = 250
        self._flat(p, i)
        p.L[i + 1, 0] = 98.9                               # fill at 99; stop = 99 - 2*2 = 95
        p.O[i + 3, 0], p.L[i + 3, 0], p.C[i + 3, 0] = 90.0, 89.0, 90.0
        r = v6.run(p, [(p.days[0], _genome())], p.days[200], p.days[-1], k=1)
        t = r.trades.iloc[0]
        self.assertEqual(t["motivo"], "stop")
        self.assertAlmostEqual(t["preco_venda"], 90.0)
        b, s = 0.0025, 0.0010
        self.assertAlmostEqual(t["resultado_liq"], (90.0 / 99.0) * (1 - b) ** 2 * (1 - s) / (1 + s) - 1, places=9)

    def test_time_stop(self):
        p = _panel()
        i = 250
        self._flat(p, i)
        p.L[i + 1, 0] = 98.9
        r = v6.run(p, [(p.days[0], _genome(time_stop=12, exit_q=0.0))], p.days[200], p.days[-1], k=1)
        self.assertEqual(r.trades["motivo"].iloc[0], "time stop")
        self.assertEqual(int(r.trades["pregoes"].iloc[0]), 12)


class PortfolioTest(unittest.TestCase):
    def test_no_signal_means_the_index_fund(self):
        p = _panel()
        r = v6.run(p, [(p.days[0], _genome(entry_q=1.01))], p.days[100], p.days[-1], k=6)
        self.assertEqual(len(r.trades), 0)
        nav = pd.Series(p.NAV, index=p.days).loc[r.equity.index]
        np.testing.assert_allclose(r.equity / r.equity.iloc[0], nav / nav.iloc[0], rtol=1e-9)

    def test_capacity_and_causality(self):
        p = _panel(seed=3)
        g = _genome(entry_q=0.8, exit_q=0.5)
        a = v6.run(p, [(p.days[0], g)], p.days[100], p.days[-1], k=3)
        self.assertLessEqual(int(a.slots_used.max()), 3)
        self.assertGreater(len(a.trades), 10)
        cut = 220
        q = _panel(seed=3)
        for arr in (q.O, q.H, q.L, q.C):
            arr[cut:] *= 1.8
        q.X[:, cut:] = -q.X[:, cut:]
        b = v6.run(q, [(q.days[0], g)], q.days[100], q.days[-1], k=3)
        ta = a.trades[pd.to_datetime(a.trades["data_venda"]) < q.days[cut]]
        tb = b.trades[pd.to_datetime(b.trades["data_venda"]) < q.days[cut]]
        pd.testing.assert_frame_equal(ta.reset_index(drop=True), tb.reset_index(drop=True))
        self.assertAlmostEqual(a.equity.loc[:q.days[cut - 1]].iloc[-1], b.equity.loc[:q.days[cut - 1]].iloc[-1], places=6)

    def test_genome_switch_takes_effect_at_the_fold_start(self):
        p = _panel(seed=4)
        g1, g2 = _genome(entry_q=0.8), _genome(entry_q=1.01)
        sw = p.days[200]
        r = v6.run(p, [(p.days[0], g1), (sw, g2)], p.days[100], p.days[-1], k=3)
        entries = pd.to_datetime(r.trades["data_compra"])
        self.assertGreater(len(entries), 3)
        self.assertTrue((entries <= p.days[203]).all())  # orders placed before the switch may still fill (3 sessions)


class FitnessTest(unittest.TestCase):
    def test_block_scores_penalise_unstable_excess(self):
        idx = pd.bdate_range("2020-01-01", periods=400)
        nav = pd.Series(np.linspace(100, 120, 400), index=idx)
        steady = nav * np.linspace(1.0, 1.05, 400)
        erratic = nav.copy()
        erratic.iloc[:100] *= np.linspace(1.0, 1.2, 100)
        erratic.iloc[100:] *= 1.2 * np.linspace(1.0, 0.875, 300)
        s1 = v6.block_scores(steady, nav, 4, 0.5)
        s2 = v6.block_scores(erratic, nav, 4, 0.5)
        self.assertGreater(s1.mean() - s1.std(), s2.mean() - s2.std())


if __name__ == "__main__":
    unittest.main()
