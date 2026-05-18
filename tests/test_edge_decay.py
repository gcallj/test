import unittest

import numpy as np
import pandas as pd

from analysis.edge_decay import (
    DECAY_LABEL,
    DEFAULT_DECAY_THRESHOLD,
    MISSING_LABEL,
    OK_LABEL,
    attach_edge_decay_to_view,
    edge_decay_ratio,
)
from analysis.recent_operability import recent_operability_blockers


def _make_trades(start, n_days, daily_ret):
    """Build a trade log with one trade per day starting at `start`."""
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame({"entry_date": dates, "ret": [daily_ret] * n_days})


class EdgeDecayRatio(unittest.TestCase):
    def test_empty_log_returns_missing(self):
        df = pd.DataFrame({"entry_date": [], "ret": []})
        info = edge_decay_ratio(df)
        self.assertEqual(info["label"], MISSING_LABEL)
        self.assertTrue(np.isnan(info["ratio"]))

    def test_no_decay_when_constant_returns(self):
        df = _make_trades("2026-01-01", 200, daily_ret=0.01)
        info = edge_decay_ratio(df, asof="2026-07-01")
        self.assertEqual(info["label"], OK_LABEL)
        self.assertAlmostEqual(info["ratio"], 1.0, places=10)

    def test_decay_flagged_when_recent_falls(self):
        prior = _make_trades("2026-01-01", 90, daily_ret=0.02)
        recent = _make_trades("2026-04-01", 30, daily_ret=0.001)  # 5% of prior
        df = pd.concat([prior, recent], ignore_index=True)
        info = edge_decay_ratio(df, asof="2026-04-30")
        self.assertEqual(info["label"], DECAY_LABEL)
        self.assertLess(info["ratio"], DEFAULT_DECAY_THRESHOLD)
        self.assertEqual(info["n_recent"], 30)

    def test_prior_already_negative_signals_decay_when_recent_worse(self):
        prior = _make_trades("2026-01-01", 90, daily_ret=-0.005)
        recent = _make_trades("2026-04-01", 30, daily_ret=-0.02)
        df = pd.concat([prior, recent], ignore_index=True)
        info = edge_decay_ratio(df, asof="2026-04-30")
        self.assertEqual(info["label"], DECAY_LABEL)
        self.assertEqual(info["ratio"], 0.0)

    def test_prior_already_negative_recent_improved_is_missing(self):
        prior = _make_trades("2026-01-01", 90, daily_ret=-0.01)
        recent = _make_trades("2026-04-01", 30, daily_ret=0.02)
        df = pd.concat([prior, recent], ignore_index=True)
        info = edge_decay_ratio(df, asof="2026-04-30")
        self.assertEqual(info["label"], MISSING_LABEL)

    def test_threshold_argument_overrides_default(self):
        prior = _make_trades("2026-01-01", 90, daily_ret=0.02)
        recent = _make_trades("2026-04-01", 30, daily_ret=0.012)  # ratio = 0.6
        df = pd.concat([prior, recent], ignore_index=True)
        # Tight 0.7 threshold -> flagged as decay
        info_strict = edge_decay_ratio(df, asof="2026-04-30", threshold=0.7)
        self.assertEqual(info_strict["label"], DECAY_LABEL)
        # Default 0.5 threshold -> still OK
        info_default = edge_decay_ratio(df, asof="2026-04-30")
        self.assertEqual(info_default["label"], OK_LABEL)

    def test_rejects_non_positive_window(self):
        df = _make_trades("2026-01-01", 30, 0.01)
        with self.assertRaises(ValueError):
            edge_decay_ratio(df, recent=0, prior=10)
        with self.assertRaises(ValueError):
            edge_decay_ratio(df, recent=10, prior=-1)


class RecentOperabilityIntegration(unittest.TestCase):
    BASE_VIEW = {
        "n_trades": 20,
        "median_hold_bars": 6.0,
        "expected_hold_score": 80.0,
        "wr_after_3_bars": 0.65,
        "win_rate": 0.65,
        "alpha_ann": 0.0,
        "expectancy": 0.005,
        "swing_survival_3d": 0.6,
        "early_take_dependency": 0.3,
    }

    def test_no_edge_decay_field_means_no_blocker(self):
        # Missing field must not add the blocker.
        blockers = recent_operability_blockers(self.BASE_VIEW)
        self.assertNotIn("recent_edge_decay_lt_50", blockers)

    def test_edge_decay_above_threshold_means_no_blocker(self):
        view = attach_edge_decay_to_view(
            self.BASE_VIEW,
            {"ratio": 0.8, "label": OK_LABEL, "recent_mean": 0.01, "prior_mean": 0.0125},
        )
        self.assertNotIn("recent_edge_decay_lt_50", recent_operability_blockers(view))

    def test_edge_decay_below_threshold_adds_soft_blocker(self):
        view = attach_edge_decay_to_view(
            self.BASE_VIEW,
            {"ratio": 0.3, "label": DECAY_LABEL, "recent_mean": 0.003, "prior_mean": 0.01},
        )
        self.assertIn("recent_edge_decay_lt_50", recent_operability_blockers(view))

    def test_nan_ratio_does_not_block(self):
        view = attach_edge_decay_to_view(
            self.BASE_VIEW,
            {"ratio": float("nan"), "label": MISSING_LABEL, "recent_mean": float("nan"), "prior_mean": float("nan")},
        )
        self.assertNotIn("recent_edge_decay_lt_50", recent_operability_blockers(view))


if __name__ == "__main__":
    unittest.main()
