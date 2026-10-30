"""Portfolio replay accounting (portfolio_se.simulate)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import market_se  # noqa: E402
import portfolio_se as ps  # noqa: E402

DAYS = pd.bdate_range("2025-01-06", periods=12)


def _d(i):
    return DAYS[i].date().isoformat()


def _order(ticker, signal, status, strength=0.5, fill=None, exit_=None, end=None, ret=None, entry=100.0):
    return {
        "ticker": ticker, "fold": 1, "data_sinal": _d(signal), "status": status,
        "limite": entry, "stop": entry * 0.95, "alvo": entry * 1.1, "forca": strength,
        "data_compra": _d(fill) if fill is not None else "",
        "preco_compra": entry if fill is not None else np.nan,
        "data_venda": _d(exit_) if exit_ is not None else "",
        "data_fim_ordem": _d(end) if end is not None else "",
        "preco_venda": np.nan, "saida": "", "resultado_liq": ret, "pregoes": None,
    }


def _closes(tickers, price=100.0):
    return pd.DataFrame({t: np.full(len(DAYS), price) for t in tickers}, index=DAYS)


class SimulateTest(unittest.TestCase):
    def test_round_trip_changes_cash_by_the_net_result(self):
        orders = pd.DataFrame([_order("A", 1, "executada", fill=2, exit_=4, ret=0.10)])
        res = ps.simulate(orders, _closes(["A"]), DAYS)
        self.assertAlmostEqual(res.equity.iloc[-1], market_se.CAPITAL_SEK + 0.10 * market_se.POSITION_SIZE_SEK)
        self.assertEqual(len(res.taken), 1)

    def test_missed_order_reserves_then_releases_cash(self):
        orders = pd.DataFrame([_order("A", 1, "perdida", end=4)])
        res = ps.simulate(orders, _closes(["A"]), DAYS)
        self.assertEqual(res.slots_used.loc[DAYS[2]], 1)
        self.assertEqual(res.slots_used.loc[DAYS[5]], 0)
        self.assertAlmostEqual(res.equity.iloc[-1], market_se.CAPITAL_SEK)

    def test_slots_are_capped_and_strongest_orders_win(self):
        rows = [_order(f"T{i}", 1, "executada", strength=i / 10, fill=2, exit_=8, ret=0.01) for i in range(8)]
        res = ps.simulate(pd.DataFrame(rows), _closes([f"T{i}" for i in range(8)]), DAYS)
        self.assertEqual(res.skipped, 8 - market_se.MAX_POSITIONS)
        taken = set(res.taken["ticker"])
        self.assertEqual(taken, {f"T{i}" for i in range(8 - market_se.MAX_POSITIONS, 8)})

    def test_open_position_is_marked_to_market(self):
        orders = pd.DataFrame([_order("A", 1, "aberta", fill=2, entry=100.0)])
        closes = _closes(["A"])
        closes.loc[DAYS[5]:, "A"] = 110.0
        res = ps.simulate(orders, closes, DAYS)
        self.assertAlmostEqual(res.equity.iloc[-1], market_se.CAPITAL_SEK + 0.10 * market_se.POSITION_SIZE_SEK)

    def test_brokerage_scenario_shifts_each_trade_by_both_sides(self):
        orders = pd.DataFrame([_order("A", 1, "executada", fill=2, exit_=4, ret=0.01)])
        adj = ps.adjust_for_brokerage(orders, 0.0)
        self.assertAlmostEqual(adj.loc[0, "resultado_liq"], 0.01 + 2 * market_se.BROKERAGE_PCT_PER_ORDER)


if __name__ == "__main__":
    unittest.main()
