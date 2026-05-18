import unittest

import numpy as np
import pandas as pd

from analysis.sector_breakdown import breakdown_by_segment, format_markdown


def _rows(**cols):
    return pd.DataFrame(cols)


class BreakdownBySegment(unittest.TestCase):
    def test_basic_aggregation_with_known_tickers(self):
        # VALE3.SA and PETR4.SA are both commodity_largecap.
        # ITUB4.SA is financials.
        df = _rows(
            ticker=["VALE3.SA", "PETR4.SA", "ITUB4.SA"],
            wr=[0.70, 0.60, 0.55],
            n_trades=[10, 20, 30],
            ret_ann=[0.05, 0.03, 0.02],
            bh_ann=[0.04, 0.01, 0.03],
            alpha_ann_pp=[1.0, 2.0, -1.0],
            mdd=[-0.10, -0.15, -0.08],
        )
        out = breakdown_by_segment(df)
        self.assertIn("commodity_largecap", out.index)
        self.assertIn("financials", out.index)
        self.assertIn("ALL", out.index)
        # Commodity row: median alpha=1.5pp, sum trades=30
        self.assertAlmostEqual(out.loc["commodity_largecap", "alpha_ann_pp"], 1.5)
        self.assertEqual(out.loc["commodity_largecap", "n_trades"], 30)
        self.assertAlmostEqual(out.loc["commodity_largecap", "wr"], 0.65)
        # Financials row
        self.assertAlmostEqual(out.loc["financials", "alpha_ann_pp"], -1.0)
        # ALL row aggregates everything
        self.assertEqual(out.loc["ALL", "n_tickers"], 3)
        self.assertEqual(out.loc["ALL", "n_trades"], 60)
        # Beat-BH: VALE3 (0.05>0.04) and ITUB4 (0.02<0.03 = False), PETR4 (0.03>0.01).
        # So 2/3 = 66.66% global; commodity_largecap 2/2 = 100%; financials 0/1 = 0%.
        self.assertAlmostEqual(out.loc["commodity_largecap", "beat_bh_pct"], 100.0)
        self.assertAlmostEqual(out.loc["financials", "beat_bh_pct"], 0.0)
        self.assertAlmostEqual(out.loc["ALL", "beat_bh_pct"], 200.0 / 3.0)

    def test_unknown_ticker_routed_to_unknown_segment(self):
        df = _rows(
            ticker=["TOTALLY_FAKE.SA"],
            wr=[0.5], n_trades=[1], alpha_ann_pp=[0.0], mdd=[-0.05],
        )
        out = breakdown_by_segment(df)
        # The ticker isn't in SECTOR_MAP and doesn't match the .SA/.11/-USD heuristics
        # for crypto/FII. get_ticker_meta returns segment_key="other_equity".
        self.assertTrue(("unknown" in out.index) or ("other_equity" in out.index))

    def test_segments_override_takes_precedence(self):
        df = _rows(
            ticker=["VALE3.SA", "PETR4.SA"],
            wr=[0.7, 0.6], n_trades=[5, 5], alpha_ann_pp=[1.0, 2.0], mdd=[-0.1, -0.2],
        )
        out = breakdown_by_segment(df, segments_override={"VALE3.SA": "X", "PETR4.SA": "Y"})
        self.assertIn("X", out.index)
        self.assertIn("Y", out.index)
        self.assertNotIn("commodity_largecap", out.index)

    def test_missing_optional_columns_does_not_crash(self):
        # Only `ticker` and `wr` — should still produce a frame.
        df = _rows(ticker=["VALE3.SA"], wr=[0.7])
        out = breakdown_by_segment(df)
        self.assertIn("ALL", out.index)
        self.assertEqual(out.loc["ALL", "n_tickers"], 1)
        self.assertAlmostEqual(out.loc["ALL", "wr"], 0.7)
        # beat_bh_pct should NOT appear when ret_ann/bh_ann absent.
        self.assertNotIn("beat_bh_pct", out.columns)

    def test_all_row_uses_pooled_median_not_average_of_segments(self):
        # Pooled median of [1, 2, 3, 4] = 2.5; averaging segment medians (2 and 3.5) gives 2.75.
        df = _rows(
            ticker=["VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA"],
            wr=[0.5, 0.5, 0.5, 0.5], n_trades=[1, 1, 1, 1],
            ret_ann=[0.0, 0.0, 0.0, 0.0], bh_ann=[0.0, 0.0, 0.0, 0.0],
            alpha_ann_pp=[1.0, 2.0, 3.0, 4.0], mdd=[0, 0, 0, 0],
        )
        out = breakdown_by_segment(df)
        self.assertAlmostEqual(out.loc["ALL", "alpha_ann_pp"], 2.5)

    def test_requires_ticker_column(self):
        with self.assertRaises(ValueError):
            breakdown_by_segment(_rows(wr=[0.5]))


class FormatMarkdown(unittest.TestCase):
    def test_render_smoke(self):
        df = _rows(
            ticker=["VALE3.SA", "ITUB4.SA"],
            wr=[0.7, 0.55], n_trades=[10, 20], alpha_ann_pp=[1.5, -1.0], mdd=[-0.1, -0.05],
        )
        md = format_markdown(breakdown_by_segment(df))
        # Header
        self.assertIn("### Per-segment breakdown", md)
        # Both segments rendered as rows
        self.assertIn("commodity_largecap", md)
        self.assertIn("financials", md)
        # ALL row present
        self.assertIn("ALL", md)
        # Numbers formatted with two decimals (the alpha_ann_pp 1.50 must appear)
        self.assertIn("1.50", md)


if __name__ == "__main__":
    unittest.main()
