"""Walk-forward calendar (docs/adr/0002)."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import walkforward as wf  # noqa: E402


class FoldCalendarTest(unittest.TestCase):
    def test_six_contiguous_folds_ending_at_the_anchor(self):
        folds = wf.folds()
        self.assertEqual(len(folds), wf.N_FOLDS)
        self.assertEqual(pd.Timestamp(folds[-1].trade_end), wf.ANCHOR_END)
        for prev, cur in zip(folds, folds[1:]):
            self.assertEqual(pd.Timestamp(prev.trade_end) + pd.Timedelta(days=1), pd.Timestamp(cur.trade_start))

    def test_training_ends_the_day_before_trading_starts(self):
        for f in wf.folds():
            self.assertEqual(pd.Timestamp(f.train_end) + pd.Timedelta(days=1), pd.Timestamp(f.trade_start))

    def test_three_years_of_trading(self):
        folds = wf.folds()
        span = pd.Timestamp(folds[-1].trade_end) - pd.Timestamp(folds[0].trade_start)
        self.assertTrue(pd.Timedelta(days=3 * 365 - 3) <= span <= pd.Timedelta(days=3 * 366))

    def test_fold_environment_hides_the_future(self):
        f = wf.get_fold(3)
        env = wf.fold_env(f, base={})
        self.assertEqual(env["GA_TRAIN_END"], f.train_end)
        self.assertEqual(env["GA_NO_SEED"], "1")
        self.assertEqual(env["GA_HOLDOUT_DAYS"], "0")
        self.assertIn("ga_memmap_fold3", env["GA_MEMMAP_DIR"])
        self.assertIn("ga_checkpoints_fold3", env["GA_CHECKPOINT_DIR"])


if __name__ == "__main__":
    unittest.main()
