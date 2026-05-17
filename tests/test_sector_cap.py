import os
import unittest

import pandas as pd

from analysis.sector_cap import ENV_VAR, apply_sector_cap, cap_summary, get_active_cap


def _df_with_4_commodity_buys():
    # 4 commodity_largecap buys, 2 financials buys, 1 hold mixed in.
    return pd.DataFrame({
        "ticker": ["VALE3.SA", "PETR4.SA", "CSNA3.SA", "GGBR4.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA"],
        "signal": ["buy", "buy", "buy", "buy", "buy", "buy", "hold"],
        "rank": [10.0, 8.0, 7.5, 6.0, 9.0, 7.0, 5.0],
    })


class GetActiveCap(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.pop(ENV_VAR, None)

    def tearDown(self):
        if self._old is not None:
            os.environ[ENV_VAR] = self._old
        else:
            os.environ.pop(ENV_VAR, None)

    def test_default_is_none(self):
        self.assertIsNone(get_active_cap())

    def test_env_var_picked_up(self):
        os.environ[ENV_VAR] = "3"
        self.assertEqual(get_active_cap(), 3)

    def test_explicit_override_wins(self):
        os.environ[ENV_VAR] = "5"
        self.assertEqual(get_active_cap(2), 2)

    def test_off_sentinel_disables(self):
        for val in ("off", "OFF", "none", "null"):
            self.assertIsNone(get_active_cap(val))

    def test_zero_or_negative_disables(self):
        self.assertIsNone(get_active_cap(0))
        self.assertIsNone(get_active_cap(-5))

    def test_non_numeric_is_disabled(self):
        self.assertIsNone(get_active_cap("not_a_number"))


class ApplySectorCap(unittest.TestCase):
    def test_no_cap_is_noop(self):
        df = _df_with_4_commodity_buys()
        out = apply_sector_cap(df)  # default: no cap
        pd.testing.assert_frame_equal(out, df)

    def test_cap_keeps_top_n_by_rank_per_segment(self):
        df = _df_with_4_commodity_buys()
        out = apply_sector_cap(df, max_per_segment=2)
        # Commodity buys had ranks [10.0, 8.0, 7.5, 6.0] -> keep VALE3 (10) and PETR4 (8).
        kept = out.loc[out["signal"] == "buy", "ticker"].tolist()
        self.assertIn("VALE3.SA", kept)
        self.assertIn("PETR4.SA", kept)
        self.assertNotIn("CSNA3.SA", kept)
        self.assertNotIn("GGBR4.SA", kept)
        # Financials: 2 buys [ITUB4=9.0, BBDC4=7.0] both <= cap=2 -> both stay.
        self.assertIn("ITUB4.SA", kept)
        self.assertIn("BBDC4.SA", kept)

    def test_demoted_signals_get_pre_and_reason_columns(self):
        df = _df_with_4_commodity_buys()
        out = apply_sector_cap(df, max_per_segment=2)
        demoted = out[out["sector_cap_reason"].astype(bool)]
        # CSNA3 and GGBR4 were demoted
        self.assertEqual(len(demoted), 2)
        self.assertTrue((demoted["signal"] == "hold").all())
        self.assertTrue((demoted["signal_pre_sector_cap"] == "buy").all())

    def test_pre_existing_hold_unchanged(self):
        df = _df_with_4_commodity_buys()
        out = apply_sector_cap(df, max_per_segment=2)
        # BBAS3 was already hold
        bbas_row = out.loc[out["ticker"] == "BBAS3.SA"].iloc[0]
        self.assertEqual(bbas_row["signal"], "hold")
        self.assertFalse(bool(bbas_row.get("sector_cap_reason", "") or ""))

    def test_cap_high_enough_demotes_nothing(self):
        df = _df_with_4_commodity_buys()
        out = apply_sector_cap(df, max_per_segment=10)
        self.assertEqual((out["signal"] == "buy").sum(), 6)
        self.assertNotIn("sector_cap_reason", out.columns)

    def test_no_buy_rows_is_safe_noop(self):
        df = pd.DataFrame({
            "ticker": ["VALE3.SA"], "signal": ["hold"], "rank": [10.0],
        })
        out = apply_sector_cap(df, max_per_segment=1)
        pd.testing.assert_frame_equal(out, df)

    def test_missing_columns_is_safe_noop(self):
        df = pd.DataFrame({"ticker": ["VALE3.SA"], "rank": [10.0]})
        out = apply_sector_cap(df, max_per_segment=1)
        pd.testing.assert_frame_equal(out, df)

    def test_segments_override_takes_precedence(self):
        df = _df_with_4_commodity_buys()
        # Force everything into a single segment "X" -> cap=2 keeps top-2 globally.
        override = {t: "X" for t in df["ticker"]}
        out = apply_sector_cap(df, max_per_segment=2, segments_override=override)
        kept_buys = out.loc[out["signal"] == "buy", "ticker"].tolist()
        self.assertEqual(len(kept_buys), 2)
        # Top by rank: ITUB4=9.0 then VALE3=10.0 — actual top-2 are VALE3 (10) and ITUB4 (9).
        self.assertIn("VALE3.SA", kept_buys)
        self.assertIn("ITUB4.SA", kept_buys)

    def test_missing_rank_column_falls_back_to_input_order(self):
        df = pd.DataFrame({
            "ticker": ["VALE3.SA", "PETR4.SA", "CSNA3.SA"],
            "signal": ["buy", "buy", "buy"],
        })
        out = apply_sector_cap(df, max_per_segment=2)
        # All 3 are commodity_largecap. Cap=2 -> keep first 2 by input order.
        kept = out.loc[out["signal"] == "buy", "ticker"].tolist()
        self.assertEqual(kept, ["VALE3.SA", "PETR4.SA"])

    def test_idempotent(self):
        df = _df_with_4_commodity_buys()
        once = apply_sector_cap(df, max_per_segment=2)
        twice = apply_sector_cap(once, max_per_segment=2)
        pd.testing.assert_frame_equal(once, twice)


class CapSummary(unittest.TestCase):
    def test_summary_after_cap(self):
        df = _df_with_4_commodity_buys()
        out = apply_sector_cap(df, max_per_segment=2)
        summary = cap_summary(out)
        self.assertEqual(summary["n_pre_buy"], 6)
        self.assertEqual(summary["n_post_buy"], 4)  # commodity 2 + financials 2
        self.assertEqual(summary["n_capped"], 2)
        self.assertAlmostEqual(summary["pct_capped"], (100.0 * 2 / 6))

    def test_summary_without_pre_column(self):
        df = pd.DataFrame({"signal": ["buy", "buy", "hold"]})
        out = cap_summary(df)
        self.assertEqual(out["n_pre_buy"], 2)
        self.assertEqual(out["n_capped"], 0)
        self.assertEqual(out["pct_capped"], 0.0)


if __name__ == "__main__":
    unittest.main()
