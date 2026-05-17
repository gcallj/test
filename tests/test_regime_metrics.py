import unittest

import numpy as np
import pandas as pd

from analysis.regime_metrics import (
    CHOP_LABEL,
    TREND_LABEL,
    breakdown_by_regime,
    classify_regime_from_close,
    classify_regime_from_ohlc,
    compute_adx,
    regime_dependency_index,
)


def _strong_trend_series(n=200, start=100.0, slope=0.5):
    base = start + np.arange(n) * slope
    return pd.Series(base, index=pd.date_range("2024-01-01", periods=n, freq="B"))


def _chop_series(n=200, start=100.0, amp=2.0, period=10):
    base = start + amp * np.sin(np.arange(n) * 2 * np.pi / period)
    return pd.Series(base, index=pd.date_range("2024-01-01", periods=n, freq="B"))


class ADXShape(unittest.TestCase):
    def test_finite_after_warmup_on_trend(self):
        close = _strong_trend_series(n=120)
        adx = compute_adx(close * 1.01, close * 0.99, close, window=14)
        # First ~14 bars are NaN; rest should be finite and elevated.
        self.assertTrue(np.isnan(adx.iloc[:13]).all())
        tail = adx.dropna().iloc[-30:]
        self.assertTrue((tail > 25).mean() > 0.5, f"expected mostly trending ADX, got median={tail.median()}")

    def test_zero_or_low_on_flat(self):
        close = pd.Series([100.0] * 80, index=pd.date_range("2024-01-01", periods=80, freq="B"))
        adx = compute_adx(close, close, close, window=14)
        tail = adx.dropna()
        # Flat input -> ADX is NaN or very low (no directional movement).
        if len(tail) > 0:
            self.assertLess(tail.median(), 25.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compute_adx(pd.Series([1.0, 2.0]), pd.Series([1.0]), pd.Series([1.0, 2.0]))


class ClassifyRegime(unittest.TestCase):
    def test_strong_uptrend_labelled_trend(self):
        close = _strong_trend_series(n=120)
        labels = classify_regime_from_ohlc(close * 1.01, close * 0.99, close)
        tail_labels = labels.dropna().iloc[-30:]
        self.assertGreater((tail_labels == TREND_LABEL).mean(), 0.5)

    def test_choppy_mostly_chop(self):
        close = _chop_series(n=200, amp=1.0, period=8)
        labels = classify_regime_from_ohlc(close * 1.001, close * 0.999, close)
        tail = labels.dropna().iloc[-50:]
        self.assertGreater((tail == CHOP_LABEL).mean(), 0.5)

    def test_close_only_fallback_returns_labels(self):
        close = _strong_trend_series(n=120)
        labels = classify_regime_from_close(close, window=50, slope_threshold=1e-5)
        tail = labels.dropna().iloc[-30:]
        self.assertGreater((tail == TREND_LABEL).mean(), 0.5)


class BreakdownByRegime(unittest.TestCase):
    def _make_trades(self):
        dates = pd.to_datetime([
            "2024-02-01",  # will map to trend
            "2024-02-05",  # trend
            "2024-02-12",  # trend
            "2024-03-04",  # chop
            "2024-03-08",  # chop
            "2025-01-01",  # missing — outside regime index
        ])
        return pd.DataFrame({
            "entry_date": dates,
            "ret": [0.05, 0.03, -0.01, 0.00, -0.02, 0.04],
            "alpha": [0.04, 0.02, -0.02, -0.005, -0.03, 0.03],
        })

    def _make_regime(self):
        idx = pd.date_range("2024-02-01", "2024-03-15", freq="B")
        labels = [TREND_LABEL if d < pd.Timestamp("2024-03-01") else CHOP_LABEL for d in idx]
        return pd.Series(labels, index=idx, name="regime")

    def test_aggregates_by_entry_regime(self):
        trades = self._make_trades()
        regime = self._make_regime()
        out = breakdown_by_regime(trades, regime)
        self.assertIn(TREND_LABEL, out.index)
        self.assertIn(CHOP_LABEL, out.index)
        self.assertIn("missing", out.index)
        self.assertIn("ALL", out.index)

        # 3 trend trades [0.05, 0.03, -0.01] -> mean 0.0233..., wr 2/3
        self.assertEqual(out.loc[TREND_LABEL, "n_trades"], 3)
        self.assertAlmostEqual(out.loc[TREND_LABEL, "mean_ret"], (0.05 + 0.03 - 0.01) / 3.0, places=10)
        self.assertAlmostEqual(out.loc[TREND_LABEL, "wr"], 2.0 / 3.0, places=10)

        # 2 chop trades [0.00, -0.02] -> wr=0 (only strict >0 counts), mean -0.01
        self.assertEqual(out.loc[CHOP_LABEL, "n_trades"], 2)
        self.assertAlmostEqual(out.loc[CHOP_LABEL, "wr"], 0.0, places=10)
        self.assertAlmostEqual(out.loc[CHOP_LABEL, "mean_ret"], -0.01, places=10)

        # ALL pooled
        self.assertEqual(out.loc["ALL", "n_trades"], 6)

    def test_supports_alternate_return_column_names(self):
        regime = self._make_regime()
        trades = self._make_trades().rename(columns={"ret": "trade_return"})
        out = breakdown_by_regime(trades, regime)
        self.assertIn(TREND_LABEL, out.index)

    def test_raises_when_no_return_column(self):
        regime = self._make_regime()
        bad = pd.DataFrame({"entry_date": [pd.Timestamp("2024-02-01")]})
        with self.assertRaises(ValueError):
            breakdown_by_regime(bad, regime)

    def test_regime_dependency_index_bounds(self):
        # Perfectly symmetric (both at 0) -> 0
        df = pd.DataFrame(
            {"mean_ret": [0.0, 0.0]}, index=[TREND_LABEL, CHOP_LABEL],
        )
        self.assertEqual(regime_dependency_index(df), 0.0)
        # All edge in trend -> 1
        df = pd.DataFrame(
            {"mean_ret": [0.05, 0.0]}, index=[TREND_LABEL, CHOP_LABEL],
        )
        self.assertAlmostEqual(regime_dependency_index(df), 1.0)
        # Equal magnitude opposite signs -> 1
        df = pd.DataFrame(
            {"mean_ret": [0.05, -0.05]}, index=[TREND_LABEL, CHOP_LABEL],
        )
        self.assertAlmostEqual(regime_dependency_index(df), 1.0)
        # Same magnitude same sign -> 0
        df = pd.DataFrame(
            {"mean_ret": [0.05, 0.05]}, index=[TREND_LABEL, CHOP_LABEL],
        )
        self.assertAlmostEqual(regime_dependency_index(df), 0.0)


if __name__ == "__main__":
    unittest.main()
