"""Strategy v4 (events_se, altdata_se): public-information timing, event rules and the portfolio."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altdata_se as alt  # noqa: E402
import events_se as v4  # noqa: E402
import market_se  # noqa: E402

BENCH = market_se.BENCHMARK_SYMBOL


def _px(seed=1, n=700, k=20):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-04", periods=n)
    cols = {f"S{i}.ST": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n))) for i in range(k)}
    cols[BENCH] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    adj = pd.DataFrame(cols, index=days)
    opn = adj.shift(1).fillna(adj) * (1 + rng.normal(0, 0.003, adj.shape))
    vol = pd.DataFrame(1e6, index=days, columns=adj.columns)
    return {"adj": adj, "close": adj, "open": opn, "adj_open": opn, "volume": vol, "turnover": adj * vol}


def _spec():
    s = {"version": 4, "period": ["2016-06-01", "2018-12-31"], "tests": {"alpha": 0.05, "bootstrap_reps": 300,
         "nw_t_min": 2.0, "min_mean_ar_long": 0.007},
         "strategy": {"k": 3, "priority": ["H1", "H4", "H3"]},
         "signals": {
             "H1": {"nature": ["Förvärv"], "instrument_type": ["Aktie"], "status": ["Aktuell"], "volume_unit": ["Antal"],
                    "currency": ["SEK"], "exclude_venue": ["Utanför handelsplats"], "exclude_share_program": True,
                    "min_value_sek": 250000, "cooldown_sessions": 60, "horizon": 60, "direction": "+"},
             "H2": {"covered_isin_prefix": ["SE"], "top_fraction": 0.1, "min_pct": 0.5, "horizon": 20, "direction": "-"},
             "H3": {"categories": ["Inside information"], "tercile_window_days": 365, "min_history_events": 5,
                    "top_fraction": 1 / 3, "cooldown_sessions": 20, "horizon": 20, "direction": "+"},
             "H4": {"categories": ["Interim report (Q1 and Q3)"], "close_time": "17:30", "tercile_window_days": 365,
                    "min_history_events": 5, "top_fraction": 1 / 3, "cooldown_sessions": 20, "horizon": 60,
                    "direction": "+"}}}
    return s


def _mkt(seed=1):
    m = v4.market_from_prices(_px(seed))
    m.isin = {t: f"SE{i:010d}" for i, t in enumerate(m.tickers)}
    return m


class TimingTest(unittest.TestCase):
    def test_publication_is_tradable_only_from_the_next_session(self):
        days = pd.bdate_range("2024-01-01", "2024-01-31")
        stamps = pd.Series(pd.to_datetime(["2024-01-05 07:30", "2024-01-05 18:40", "2024-01-06 12:00"]))
        got = alt.next_session_after(days, stamps, 1)
        self.assertTrue((got == pd.Timestamp("2024-01-08")).all())  # Friday morning, Friday evening, Saturday
        self.assertEqual(alt.next_session_after(days, stamps.iloc[:1], 2).iloc[0], pd.Timestamp("2024-01-09"))

    def test_short_position_enters_two_sessions_later_and_below_threshold_resets(self):
        days = pd.bdate_range("2024-03-01", periods=15)
        df = pd.DataFrame({"holder": ["A", "B", "A"], "ticker": ["X.ST"] * 3, "pct": [0.8, 0.6, 0.0],
                           "position_date": pd.to_datetime(["2024-03-04", "2024-03-05", "2024-03-11"])})
        panel = alt.short_interest_panel(df, days, ["X.ST"])["X.ST"]
        self.assertEqual(panel.loc["2024-03-05"], 0.0)          # T+1: not public yet in the morning
        self.assertAlmostEqual(panel.loc["2024-03-06"], 0.8)    # T+2 open
        self.assertAlmostEqual(panel.loc["2024-03-07"], 1.4)
        self.assertAlmostEqual(panel.loc["2024-03-13"], 0.6)    # A fell below 0.5% on 03-11

    def test_reaction_session_after_the_close_moves_to_the_next_day(self):
        days = pd.bdate_range("2024-04-01", periods=10)
        rel = pd.Series(pd.to_datetime(["2024-04-02 07:00", "2024-04-02 17:45", "2024-04-06 09:00"]))
        r = v4.reaction_session(days, rel, "17:30")
        self.assertEqual(list(r.dt.strftime("%m-%d")), ["04-02", "04-03", "04-08"])

    def test_ods_reader_handles_repeated_cells(self):
        import io, zipfile
        xml = ('<?xml version="1.0"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
               'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
               '<office:body><office:spreadsheet><table:table table:name="t"><table:table-row>'
               '<table:table-cell><text:p>a</text:p></table:table-cell><table:table-cell table:number-columns-repeated="2"/>'
               '<table:table-cell office:value="1.5"><text:p>1,5</text:p></table:table-cell></table:table-row>'
               '</table:table></office:spreadsheet></office:body></office:document-content>')
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("content.xml", xml)
        self.assertEqual(alt.read_ods(buf.getvalue()), [["a", None, None, "1.5"]])


class IssuerMapTest(unittest.TestCase):
    def test_two_of_isin_lei_and_name_must_agree(self):
        volvo = next(t for t in alt.universe() if t["yf_symbol"] == "VOLV-B.ST")
        core = pd.DataFrame({"issuer": ["Aktiebolaget Volvo"] * 6, "lei": ["L1"] * 6, "isin": [volvo["isin"]] * 6})
        m = alt.IssuerMap(core)
        self.assertEqual(m.ticker(isin=volvo["isin"], lei="L1", name="Aktiebolaget Volvo"), "VOLV-B.ST")
        self.assertEqual(m.ticker(isin="SE0000000001", lei="L1", name="Aktiebolaget Volvo"), "VOLV-B.ST")  # old ISIN
        self.assertEqual(m.ticker(isin=volvo["isin"], lei="L1", name="Jens Holmberg"), "VOLV-B.ST")      # LEI agrees
        self.assertIsNone(m.ticker(isin="SE0000000001", lei="L2", name="Aktiebolaget Volvo"))           # LEI differs
        self.assertIsNone(m.ticker(isin=volvo["isin"], lei="L9", name="Systemair AB"))                 # another issuer
        self.assertIsNone(m.ticker(isin="SE0000000002", lei="L1", name="Midsona AB"))                  # wrong LEI typed
        # short register (no LEI): ISIN plus a consistent issuer name
        self.assertEqual(m.ticker(isin=volvo["isin"], name="AB Volvo (publ)"), "VOLV-B.ST")
        self.assertIsNone(m.ticker(isin=volvo["isin"], name="ICA Gruppen AB"))


class EventRuleTest(unittest.TestCase):
    def _ins(self, m, **over):
        base = {"ticker": m.tickers[0], "published": pd.Timestamp(m.days[300]) + pd.Timedelta(hours=18),
                "nature": "Förvärv", "instrument_type": "Aktie", "status": "Aktuell", "volume_unit": "Antal",
                "currency": "SEK", "venue": "NASDAQ STOCKHOLM AB", "share_program": False, "volume": 10000.0,
                "price": 50.0, "person": "P1"}
        base.update(over)
        return base

    def test_insider_filters_threshold_and_cooldown(self):
        m = _mkt()
        rows = [self._ins(m),                                            # 500k SEK: event
                self._ins(m, nature="Avyttring"),                         # sale
                self._ins(m, share_program=True, ticker=m.tickers[1]),    # share program
                self._ins(m, venue="Utanför handelsplats", ticker=m.tickers[2]),
                self._ins(m, volume=1000.0, ticker=m.tickers[3]),         # 50k SEK: too small
                self._ins(m, published=pd.Timestamp(m.days[330]))]        # inside cooldown of the first
        ev = v4.insider_events(pd.DataFrame(rows), m, _spec())
        self.assertEqual(list(ev["ticker"]), [m.tickers[0]])
        self.assertEqual(ev["entry"].iloc[0], m.days[301])

    def test_earnings_events_use_closes_up_to_r_plus_1_and_enter_at_r_plus_2(self):
        m = _mkt()
        s = _spec()
        rows = [{"ticker": m.tickers[i % 10], "released": pd.Timestamp(m.days[100 + 7 * i]) + pd.Timedelta(hours=8),
                 "category": "Interim report (Q1 and Q3)"} for i in range(60)]
        news = pd.DataFrame(rows)
        a = v4.earnings_events(news, m, s)
        self.assertGreater(len(a), 3)
        r_pos = m.days.get_indexer(pd.DatetimeIndex(a["entry"])) - 2
        # changing prices from r+2 onwards cannot change which events are selected
        m2 = _mkt()
        for p in r_pos[:1]:
            m2.adj.iloc[p + 2:] *= 1.7
        b = v4.earnings_events(news, m2, s)
        self.assertEqual(a["entry"].iloc[0], b["entry"].iloc[0])
        self.assertAlmostEqual(a["magnitude"].iloc[0], b["magnitude"].iloc[0], places=12)

    def test_trailing_percentile_only_uses_earlier_dates(self):
        dates = pd.Series(pd.date_range("2020-01-01", periods=10, freq="7D"))
        vals = pd.Series(np.arange(10, dtype=float))
        pct = v4._trailing_tercile(vals, dates, 365, 3)
        self.assertTrue(np.isnan(pct.iloc[2]))
        self.assertEqual(pct.iloc[5], 1.0)
        vals2 = vals.copy(); vals2.iloc[9] = -100.0
        self.assertEqual(v4._trailing_tercile(vals2, dates, 365, 3).iloc[5], 1.0)

    def test_short_groups_pick_the_top_of_covered_stocks(self):
        m = _mkt()
        s = _spec()
        panel = pd.DataFrame(0.0, index=m.days, columns=m.tickers)
        panel[m.tickers[4]] = 3.0
        panel[m.tickers[5]] = 0.3   # below the 0.5% floor
        g = v4.short_groups(panel, m, s)
        self.assertEqual(set(g["ticker"]), {m.tickers[4]})
        self.assertTrue((pd.DatetimeIndex(g["entry"]).dayofweek == 0).mean() > 0.9)


class AbnormalReturnTest(unittest.TestCase):
    def test_matched_control_excludes_the_stock_and_ignores_later_prices(self):
        m = _mkt()
        t, p, H = m.tickers[2], 400, 20
        ev = pd.DataFrame({"family": ["H1"], "ticker": [t], "entry": [m.days[p]], "magnitude": [1.0], "info": [""]})
        a = v4.abnormal_returns(m, ev, H)
        exp_ret = m.opx[t].iloc[p + H] / m.opx[t].iloc[p] - 1
        self.assertAlmostEqual(a["ret"].iloc[0], exp_ret, places=12)
        q = m.quint.iloc[p - 1]
        peers = [c for c in m.tickers if c != t and q[c] == q[t]]
        ctrl = np.mean([m.opx[c].iloc[p + H] / m.opx[c].iloc[p] - 1 for c in peers])
        self.assertAlmostEqual(a["ctrl"].iloc[0], ctrl, places=12)
        m.opx.iloc[p + H + 1:] *= 5.0
        self.assertAlmostEqual(v4.abnormal_returns(m, ev, H)["ar"].iloc[0], a["ar"].iloc[0], places=12)

    def test_holm(self):
        # thresholds 0.05/4, 0.05/3, 0.05/2, 0.05: stop at the first failure
        self.assertEqual(v4.holm({"a": 0.001, "b": 0.015, "c": 0.06, "d": 0.001}, 0.05),
                         {"a": True, "b": True, "c": False, "d": True})
        self.assertEqual(v4.holm({"a": 0.001, "b": 0.02, "c": 0.021, "d": 0.5}, 0.05),
                         {"a": True, "b": False, "c": False, "d": False})
        self.assertEqual(v4.holm({"a": float("nan"), "b": 0.001}, 0.05), {"a": False, "b": True})


class PortfolioTest(unittest.TestCase):
    def _events(self, m, n=80, fam="H1"):
        rng = np.random.default_rng(5)
        return pd.DataFrame({"family": fam, "ticker": rng.choice(m.tickers, n),
                             "entry": m.days[np.sort(rng.integers(300, 650, n))], "magnitude": rng.random(n), "info": ""})

    def test_capacity_holding_period_and_fund_fill(self):
        m = _mkt()
        ev = self._events(m)
        res = v4.simulate(m, ev, {"H1": 20, "H3": 20, "H4": 60}, m.days[280], m.days[-1], k=3)
        self.assertLessEqual(res.slots_used.max(), 3)
        closed = res.trades[res.trades["status"] == "fechada"]
        self.assertGreater(len(closed), 10)
        self.assertTrue((closed["pregoes"] == 20).all())
        self.assertTrue(((res.invested + res.fund_weight) > 0.999).all())

    def test_no_events_means_the_index_fund(self):
        m = _mkt()
        res = v4.simulate(m, self._events(m).iloc[:0], {"H1": 20}, m.days[280], m.days[-1], priority=("H1",))
        b = m.bench.loc[res.equity.index]
        np.testing.assert_allclose(res.equity / res.equity.iloc[0], b / b.iloc[0], rtol=1e-9)

    def test_blocked_names_are_not_bought(self):
        m = _mkt()
        ev = self._events(m)
        weeks = pd.DatetimeIndex(sorted(set(m.days[280:]) & set(m.days[m.days.dayofweek == 0])))
        blocked = pd.DataFrame({"family": "H2", "ticker": ev["ticker"].iloc[0], "entry": weeks, "magnitude": 1.0, "info": ""})
        res = v4.simulate(m, ev, {"H1": 20}, m.days[280], m.days[-1], k=3, priority=("H1",), blocked=blocked)
        self.assertNotIn(ev["ticker"].iloc[0], set(res.trades["ticker"]))

    def test_entries_ignore_future_prices(self):
        m, m2 = _mkt(), _mkt()
        cut = 500
        m2.adj.iloc[cut:] *= 2.0
        m2.opx.iloc[cut:] *= 2.0
        m2.bench.iloc[cut:] *= 2.0
        ev = self._events(m)
        a = v4.simulate(m, ev, {"H1": 20}, m.days[280], m.days[-1], k=3, priority=("H1",))
        b = v4.simulate(m2, ev, {"H1": 20}, m.days[280], m.days[-1], k=3, priority=("H1",))
        ta = a.trades[pd.to_datetime(a.trades["data_compra"]) < m.days[cut]]
        tb = b.trades[pd.to_datetime(b.trades["data_compra"]) < m.days[cut]]
        self.assertEqual(list(ta["ticker"]), list(tb["ticker"]))
        self.assertAlmostEqual(a.equity.iloc[cut - 281], b.equity.iloc[cut - 281], places=6)


if __name__ == "__main__":
    unittest.main()
