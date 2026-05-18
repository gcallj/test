import os
import unittest

import pandas as pd

from trading_universe import (
    ALLOWED_TIERS,
    DEFAULT_UNIVERSE,
    ENV_VAR,
    UNIVERSE_ALL,
    UNIVERSE_HIGH_QUAL,
    UNIVERSE_PREMIUM,
    filter_apply_df,
    get_active_universe,
    passes_filter,
)


def _sample_df():
    return pd.DataFrame({
        "ticker": ["VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA"],
        "signal": ["buy", "buy", "hold", "buy"],
        "universe": ["PREMIUM", "HIGH_QUAL", "REGULAR", "REGULAR"],
        "rank": [10.0, 8.0, 5.0, 3.0],
    })


class GetActiveUniverse(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.pop(ENV_VAR, None)

    def tearDown(self):
        if self._old is not None:
            os.environ[ENV_VAR] = self._old
        else:
            os.environ.pop(ENV_VAR, None)

    def test_default_is_all(self):
        self.assertEqual(get_active_universe(), UNIVERSE_ALL)

    def test_explicit_override_wins(self):
        os.environ[ENV_VAR] = UNIVERSE_PREMIUM
        self.assertEqual(get_active_universe(UNIVERSE_HIGH_QUAL), UNIVERSE_HIGH_QUAL)

    def test_env_var_picked_up_when_no_override(self):
        os.environ[ENV_VAR] = UNIVERSE_HIGH_QUAL
        self.assertEqual(get_active_universe(), UNIVERSE_HIGH_QUAL)

    def test_invalid_value_falls_back_to_default(self):
        os.environ[ENV_VAR] = "junk_value"
        self.assertEqual(get_active_universe(), DEFAULT_UNIVERSE)

    def test_case_insensitive(self):
        os.environ[ENV_VAR] = "PREMIUM"
        self.assertEqual(get_active_universe(), UNIVERSE_PREMIUM)
        os.environ[ENV_VAR] = "  High_Qual  "
        self.assertEqual(get_active_universe(), UNIVERSE_HIGH_QUAL)


class PassesFilter(unittest.TestCase):
    def test_all_universe_lets_everyone_through(self):
        self.assertTrue(passes_filter("PREMIUM", UNIVERSE_ALL))
        self.assertTrue(passes_filter("HIGH_QUAL", UNIVERSE_ALL))
        self.assertTrue(passes_filter("REGULAR", UNIVERSE_ALL))

    def test_high_qual_keeps_premium_and_highqual(self):
        self.assertTrue(passes_filter("PREMIUM", UNIVERSE_HIGH_QUAL))
        self.assertTrue(passes_filter("HIGH_QUAL", UNIVERSE_HIGH_QUAL))
        self.assertFalse(passes_filter("REGULAR", UNIVERSE_HIGH_QUAL))

    def test_premium_keeps_only_premium(self):
        self.assertTrue(passes_filter("PREMIUM", UNIVERSE_PREMIUM))
        self.assertFalse(passes_filter("HIGH_QUAL", UNIVERSE_PREMIUM))
        self.assertFalse(passes_filter("REGULAR", UNIVERSE_PREMIUM))

    def test_lowercase_tier_works(self):
        self.assertTrue(passes_filter("premium", UNIVERSE_PREMIUM))
        self.assertTrue(passes_filter("high_qual", UNIVERSE_HIGH_QUAL))


class FilterApplyDf(unittest.TestCase):
    def test_all_universe_is_a_no_op(self):
        df = _sample_df()
        out = filter_apply_df(df, universe=UNIVERSE_ALL)
        self.assertTrue((out["signal"] == df["signal"]).all())
        self.assertNotIn("signal_pre_universe", out.columns)
        self.assertNotIn("universe_filter_reason", out.columns)

    def test_high_qual_demotes_regular_buys(self):
        df = _sample_df()
        out = filter_apply_df(df, universe=UNIVERSE_HIGH_QUAL)
        # PREMIUM (VALE3) and HIGH_QUAL (PETR4) keep their buy.
        self.assertEqual(out.loc[out["ticker"] == "VALE3.SA", "signal"].iloc[0], "buy")
        self.assertEqual(out.loc[out["ticker"] == "PETR4.SA", "signal"].iloc[0], "buy")
        # ITUB4 was already 'hold' — must stay hold.
        self.assertEqual(out.loc[out["ticker"] == "ITUB4.SA", "signal"].iloc[0], "hold")
        # BBDC4 is REGULAR buy -> demoted.
        self.assertEqual(out.loc[out["ticker"] == "BBDC4.SA", "signal"].iloc[0], "hold")
        self.assertEqual(out.loc[out["ticker"] == "BBDC4.SA", "signal_pre_universe"].iloc[0], "buy")
        self.assertIn("filtered_by_universe=high_qual", out.loc[out["ticker"] == "BBDC4.SA", "universe_filter_reason"].iloc[0])

    def test_premium_demotes_high_qual_and_regular(self):
        df = _sample_df()
        out = filter_apply_df(df, universe=UNIVERSE_PREMIUM)
        # Only PREMIUM keeps its buy.
        self.assertEqual(out.loc[out["ticker"] == "VALE3.SA", "signal"].iloc[0], "buy")
        # HIGH_QUAL (PETR4) -> demoted.
        self.assertEqual(out.loc[out["ticker"] == "PETR4.SA", "signal"].iloc[0], "hold")
        # REGULAR (BBDC4) -> demoted.
        self.assertEqual(out.loc[out["ticker"] == "BBDC4.SA", "signal"].iloc[0], "hold")

    def test_non_buy_signals_untouched(self):
        df = pd.DataFrame({
            "signal": ["sell", "hold", "sell"],
            "universe": ["REGULAR", "REGULAR", "REGULAR"],
        })
        out = filter_apply_df(df, universe=UNIVERSE_PREMIUM)
        self.assertTrue((out["signal"] == df["signal"]).all())

    def test_missing_columns_is_safe_no_op(self):
        df = pd.DataFrame({"ticker": ["VALE3.SA"], "rank": [10]})
        out = filter_apply_df(df, universe=UNIVERSE_PREMIUM)
        pd.testing.assert_frame_equal(out, df)

    def test_idempotent(self):
        df = _sample_df()
        once = filter_apply_df(df, universe=UNIVERSE_HIGH_QUAL)
        twice = filter_apply_df(once, universe=UNIVERSE_HIGH_QUAL)
        pd.testing.assert_frame_equal(once, twice)


class AllowedTiersInvariant(unittest.TestCase):
    def test_premium_subset_of_high_qual_subset_of_all(self):
        self.assertTrue(ALLOWED_TIERS[UNIVERSE_PREMIUM] <= ALLOWED_TIERS[UNIVERSE_HIGH_QUAL])
        self.assertTrue(ALLOWED_TIERS[UNIVERSE_HIGH_QUAL] <= ALLOWED_TIERS[UNIVERSE_ALL])


if __name__ == "__main__":
    unittest.main()
