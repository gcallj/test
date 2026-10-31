"""Strategy v5 (analyst_parse, revisions_se): parsing Swedish analyst articles, timing and the downgrade filter."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyst_parse as P  # noqa: E402
import events_se as v4  # noqa: E402
import revisions_se as v5  # noqa: E402
from test_events_se import _mkt  # noqa: E402

P.ALIASES = P.build_aliases()


def _parse(headline, lead="", body=""):
    return {r["ticker"]: (P._kind(r), r) for r in
            P.parse_article({"url": "u", "published": "x", "headline": headline, "lead": lead, "body": body}, P.ALIASES)}


class ParseTest(unittest.TestCase):
    def test_target_cut_with_from_wording(self):
        out = _parse("Nordea sänker riktkursen för HMS Networks – upprepar köp",
                     "Nordea upprepar köp för HMS Networks men sänker riktkursen till 626 kronor från 630.")
        self.assertEqual(out["HMS.ST"][0], "target_down")

    def test_roundup_with_bank_brokers_is_about_the_company(self):
        out = _parse("Höjda riktkurser för Getinge efter rapporten - SHB och DNB höjer till köp",
                     "Handelsbanken och DNB har även höjt sina rekommendationer för aktien till köp från behåll.",
                     "Vidare höjer SEB riktkursen till 270 kronor (232) och upprepar köp, Pareto höjer riktkursen till 215 kronor (205).")
        self.assertEqual(set(out), {"GETI-B.ST"})
        kind, r = out["GETI-B.ST"]
        self.assertEqual(kind, "upgrade")
        self.assertEqual(r["n_t_up"], 2)

    def test_bank_as_the_analysed_company(self):
        self.assertEqual(_parse("Pareto sänker Swedbank till sälj")["SWED-A.ST"][0], "downgrade")
        self.assertEqual(_parse("Danske Bank höjer riktkursen för SEB till 180 kronor (170)")["SEB-A.ST"][0], "target_up")

    def test_several_companies_in_one_sentence(self):
        out = _parse("Citi höjer riktkurser för verkstadsjättarna",
                     "Citi höjer riktkursen för Atlas Copco till 200 kronor (190), för Sandvik till 280 kronor (260) "
                     "och sänker Alfa Laval till neutral.")
        self.assertEqual(out["ATCO-A.ST"][0], "target_up")
        self.assertEqual(out["SAND.ST"][0], "target_up")
        self.assertEqual(out["ALFA.ST"][0], "downgrade")

    def test_initiation_and_sources(self):
        self.assertEqual(_parse("Jefferies inleder bevakning av Saab med köp")["SAAB-B.ST"][0], "upgrade")
        self.assertEqual(_parse("Affärsvärlden ger köpråd för Autoliv")["ALIV-SDB.ST"][0], "media_buy")
        self.assertEqual(_parse("Penser höjer motiverat värde för Lagercrantz", "Erik Penser höjer motiverat värde till 300 kronor (280).")
                         ["LAGR-B.ST"][0], "commissioned")
        self.assertEqual(_parse("Nordea höjer riktkursen för Avanza till 465 kronor, upprepar köp")["AZA.ST"][0], "target_up")


class EventTest(unittest.TestCase):
    def _acts(self, m, rows):
        base = {"url": "u", "headline": "", "source": "broker", "n_up": 0, "n_down": 0, "n_t_up": 0, "n_t_down": 0,
                "n_init_pos": 0, "n_init_neg": 0, "media_buy": 0, "media_sell": 0}
        return pd.DataFrame([dict(base, **r) for r in rows])

    def _spec(self):
        return {"period": ["2016-01-01", "2020-12-31"], "signals": {
            "R1": {"min_rating_net": 1, "cooldown_sessions": 20, "horizon": 20},
            "R2": {"min_rating_net": 1, "cooldown_sessions": 60, "horizon": 60},
            "R3": {"min_target_net": 1, "cooldown_sessions": 20, "horizon": 20}}}

    def test_kinds_timing_and_sources(self):
        m = _mkt()
        t = m.tickers
        fri = m.days[m.days.dayofweek == 4][60]
        rows = [{"ticker": t[0], "published": fri + pd.Timedelta(hours=15), "n_up": 1, "url": "a"},
                {"ticker": t[1], "published": fri + pd.Timedelta(hours=9), "n_down": 1, "url": "b"},
                {"ticker": t[2], "published": fri + pd.Timedelta(hours=8), "n_t_up": 2, "url": "c"},
                {"ticker": t[3], "published": fri + pd.Timedelta(hours=8), "n_t_up": 2, "n_down": 1, "url": "d"},
                {"ticker": t[4], "published": fri + pd.Timedelta(hours=8), "n_up": 1, "source": "commissioned", "url": "e"},
                {"ticker": t[0], "published": fri + pd.Timedelta(days=3, hours=8), "n_up": 1, "url": "f"}]
        ev = v5.build_events(m, self._acts(m, rows), self._spec())
        monday = m.days[m.days.get_loc(fri) + 1]
        self.assertEqual(list(ev["R1"]["ticker"]), [t[0]])           # second upgrade inside the cooldown
        self.assertEqual(ev["R1"]["entry"].iloc[0], monday)          # published Friday 15:00 -> Monday open
        self.assertEqual(set(ev["R2"]["ticker"]), {t[1], t[3]})
        self.assertEqual(list(ev["R3"]["ticker"]), [t[2]])           # target raises with a downgrade do not count

    def test_downgrade_blocks_purchases_for_the_horizon(self):
        m = _mkt()
        t = m.tickers[0]
        p = 400
        ev = pd.DataFrame({"family": "R1", "ticker": [t, t], "entry": [m.days[p + 10], m.days[p + 70]],
                           "magnitude": 1.0, "info": ""})
        down = pd.DataFrame({"family": "R2", "ticker": [t], "entry": [m.days[p]], "magnitude": 1.0, "info": ""})
        blocks = v5.block_windows(m, down, 60)
        res = v4.simulate(m, ev, {"R1": 20}, m.days[300], m.days[-1], k=3, priority=("R1",), block_windows=blocks)
        bought = list(pd.to_datetime(res.trades["data_compra"]))
        self.assertEqual(bought, [m.days[p + 70]])


if __name__ == "__main__":
    unittest.main()
