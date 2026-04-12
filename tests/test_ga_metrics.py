import json
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace

import numpy as np

if "matplotlib" not in sys.modules:
    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub

if "deap" not in sys.modules:
    deap_stub = types.ModuleType("deap")
    base_stub = types.ModuleType("deap.base")
    creator_stub = types.ModuleType("deap.creator")
    tools_stub = types.ModuleType("deap.tools")
    algorithms_stub = types.ModuleType("deap.algorithms")

    def _creator_create(name, *args, **kwargs):
        setattr(creator_stub, name, type(name, (), {}))

    creator_stub.create = _creator_create
    deap_stub.base = base_stub
    deap_stub.creator = creator_stub
    deap_stub.tools = tools_stub
    deap_stub.algorithms = algorithms_stub
    sys.modules["deap"] = deap_stub
    sys.modules["deap.base"] = base_stub
    sys.modules["deap.creator"] = creator_stub
    sys.modules["deap.tools"] = tools_stub
    sys.modules["deap.algorithms"] = algorithms_stub

import ga_run
import ga_run_modular_final as ga_run_modular
import payload_store
import run_local_ga_staged as staged_runner


def _build_gp():
    return SimpleNamespace(
        vote_threshold_long=0.10,
        vote_threshold_short=0.90,
        z_threshold=0.10,
        signal_ema_span=2,
        entry_confirmation_days=1,
        score_percentile_trigger=0.50,
        stop_atr_mult=0.50,
        stop_tighten_after_bars=50,
        stop_tighten_factor=1.0,
        max_loss_per_trade_pct=0.50,
        reward_risk_ratio=0.50,
        partial_take_pct=0.0,
        partial_take_level=1.0,
        time_stop_bars=10,
        entry_discount_atr_frac=0.0,
        volatility_filter_percentile=0.0,
        score_strength_scaling=0.0,
        ma_filter_period=100,
        ma_filter_mode=0,
        consecutive_loss_cooldown=0,
        equity_drawdown_stop_pct=0.0,
        vol_regime_mode=0,
        partial_take_pct_2=0.0,
        partial_take_level_2=1.5,
        min_signal_strength=0.0,
        trailing_stop_mode=0,
        volume_confirm_mode=0,
        momentum_confirm_days=0,
        entry_score_threshold=0.0,
        regime_threshold=0.0,
    )


def _sample_stat(**overrides):
    stat = {
        "sharpe": 0.8,
        "sortino": 1.0,
        "total_return": 0.12,
        "excess_return": 0.04,
        "expectancy": 0.010,
        "payoff_ratio": 1.2,
        "profit_factor": 1.4,
        "cagr": 0.10,
        "max_drawdown_duration": 40.0,
        "mdd": -0.10,
        "n_trades": 24.0,
        "exposure": 0.65,
        "win_rate": 0.66,
        "win_rate_target": 0.48,
        "dist_quality": 0.62,
        "big_wins_pct": 0.18,
        "big_losses_pct": 0.07,
        "pct_gt15": 0.05,
        "pct_10_15": 0.06,
    }
    stat.update(overrides)
    return stat


class TestBacktestMetricHelpers(unittest.TestCase):
    def test_no_trade_metrics_are_zero_and_finite(self):
        stats = ga_run._finalize_backtest_stats([], [], 0.25, 0.0, 0.0, 252, 0)
        self.assertEqual(stats["sortino"], 0.0)
        self.assertEqual(stats["profit_factor"], 0.0)
        self.assertEqual(stats["expectancy"], 0.0)
        self.assertEqual(stats["payoff_ratio"], 0.0)
        self.assertEqual(stats["max_drawdown_duration"], 0.0)
        self.assertEqual(stats["cagr"], 0.0)
        for value in stats.values():
            if isinstance(value, float):
                self.assertTrue(np.isfinite(value))

    def test_all_win_metrics_are_bounded_and_positive(self):
        stats = ga_run._finalize_backtest_stats(
            [0.02, 0.03, 0.01],
            ["take", "time", "take"],
            0.60,
            0.061,
            -0.01,
            252,
            3,
        )
        self.assertGreater(stats["sortino"], 0.0)
        self.assertLessEqual(stats["sortino"], 10.0)
        self.assertEqual(stats["profit_factor"], 10.0)
        self.assertEqual(stats["payoff_ratio"], 10.0)
        self.assertGreater(stats["expectancy"], 0.0)
        self.assertGreater(stats["cagr"], 0.0)

    def test_all_loss_metrics_are_bounded_and_negative(self):
        stats = ga_run._finalize_backtest_stats(
            [-0.02, -0.03, -0.01],
            ["stop", "hard_loss", "time"],
            0.60,
            -0.059,
            -0.08,
            252,
            7,
        )
        self.assertLess(stats["sortino"], 0.0)
        self.assertEqual(stats["profit_factor"], 0.0)
        self.assertEqual(stats["payoff_ratio"], 0.0)
        self.assertLess(stats["expectancy"], 0.0)
        self.assertEqual(stats["max_drawdown_duration"], 7.0)

    def test_mixed_trade_metrics_match_expected_ratios(self):
        stats = ga_run._finalize_backtest_stats(
            [0.02, -0.01, 0.03, -0.02],
            ["take", "stop", "time", "hard_loss"],
            0.55,
            0.018,
            -0.06,
            252,
            5,
        )
        self.assertAlmostEqual(stats["profit_factor"], 0.05 / 0.03, places=6)
        self.assertAlmostEqual(stats["payoff_ratio"], 0.025 / 0.015, places=6)
        self.assertAlmostEqual(stats["expectancy"], 0.005, places=6)
        self.assertEqual(stats["max_drawdown_duration"], 5.0)

    def test_recovered_and_unrecovered_drawdown_durations_are_preserved(self):
        recovered = ga_run._finalize_backtest_stats([0.01, -0.02, 0.03], ["time"] * 3, 0.4, 0.018, -0.05, 252, 4)
        unrecovered = ga_run._finalize_backtest_stats([0.01, -0.02, -0.01], ["time"] * 3, 0.4, -0.020, -0.08, 252, 9)
        self.assertEqual(recovered["max_drawdown_duration"], 4.0)
        self.assertEqual(unrecovered["max_drawdown_duration"], 9.0)


class TestParityAndFitness(unittest.TestCase):
    def test_metric_helper_parity_between_ga_modules(self):
        trade_rets = [0.02, -0.01, 0.03, -0.02]
        exit_types = ["take", "stop", "time", "hard_loss"]
        main_stats = ga_run._finalize_backtest_stats(trade_rets, exit_types, 0.55, 0.018, -0.06, 252, 5)
        modular_stats = ga_run_modular._finalize_backtest_stats(trade_rets, exit_types, 0.55, 0.018, -0.06, 252, 5)
        self.assertEqual(set(main_stats.keys()), set(modular_stats.keys()))
        for key in main_stats:
            if isinstance(main_stats[key], float):
                self.assertAlmostEqual(main_stats[key], modular_stats[key], places=9, msg=key)
            else:
                self.assertEqual(main_stats[key], modular_stats[key], key)

    def test_backtest_parity_between_ga_modules(self):
        gp = _build_gp()
        n = 80
        close = np.linspace(100.0, 126.0, n)
        open_ = close * 0.999
        high = close * 1.01
        low = close * 0.99
        atr = np.full(n, 1.0)
        score_matrix = np.ones((n, 2), dtype=np.float64)
        precomputed = {
            "vol_rank": np.full(n, 0.5),
            "volume": np.full(n, 1_000_000.0),
        }

        main_stats = ga_run.backtest_stats_global_intraday(
            open_, high, low, close, score_matrix, atr, gp, precomputed=precomputed, stage_gate="off"
        )
        modular_stats = ga_run_modular._backtest_stats_global_intraday(
            open_, high, low, close, score_matrix, atr, gp, precomputed=precomputed
        )

        self.assertGreater(main_stats["n_trades"], 0.0)
        self.assertEqual(set(main_stats.keys()), set(modular_stats.keys()))
        for key in ("total_return", "mdd", "sharpe", "sortino", "profit_factor", "expectancy", "payoff_ratio", "cagr"):
            self.assertAlmostEqual(main_stats[key], modular_stats[key], places=9, msg=key)

    def test_fitness_rewards_quality_with_positive_alpha(self):
        baseline = [_sample_stat(), _sample_stat(win_rate=0.64, win_rate_target=0.46)]
        improved = [
            _sample_stat(sortino=1.8, profit_factor=1.9, expectancy=0.014, payoff_ratio=1.5),
            _sample_stat(sortino=1.6, profit_factor=1.8, expectancy=0.013, payoff_ratio=1.4, win_rate=0.64, win_rate_target=0.46),
        ]
        self.assertGreater(
            ga_run.global_fitness_from_stats(improved),
            ga_run.global_fitness_from_stats(baseline),
        )
        self.assertGreater(
            ga_run_modular.global_fitness_from_stats(improved),
            ga_run_modular.global_fitness_from_stats(baseline),
        )

    def test_fitness_guardrails_block_negative_alpha_from_beating_baseline(self):
        baseline = [_sample_stat(), _sample_stat(win_rate=0.64, win_rate_target=0.46)]
        negative_alpha = [
            _sample_stat(sortino=3.5, profit_factor=3.0, expectancy=0.020, payoff_ratio=2.0, excess_return=-0.05, total_return=0.01),
            _sample_stat(sortino=3.2, profit_factor=2.8, expectancy=0.018, payoff_ratio=1.8, excess_return=-0.04, total_return=0.01, win_rate=0.64, win_rate_target=0.46),
        ]
        self.assertLessEqual(
            ga_run.global_fitness_from_stats(negative_alpha),
            ga_run.global_fitness_from_stats(baseline),
        )
        self.assertLessEqual(
            ga_run_modular.global_fitness_from_stats(negative_alpha),
            ga_run_modular.global_fitness_from_stats(baseline),
        )


class TestPayloadStoreLifecycle(unittest.TestCase):
    def test_close_preserves_store_files_for_reuse(self):
        ticker_payloads = {
            "AAA3": {
                "close": np.array([10.0, 10.5, 11.0], dtype=np.float64),
                "atr": np.array([0.5, 0.5, 0.5], dtype=np.float64),
                "score_matrix": np.ones((3, 2), dtype=np.float64),
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            store = payload_store.PayloadStore.build_from_ticker_payloads(
                ticker_payloads=ticker_payloads,
                feature_cols=["feat_a", "feat_b"],
                store_dir=tmpdir,
            )
            meta_path = os.path.join(tmpdir, "meta.json")
            self.assertTrue(os.path.exists(meta_path))

            store.close()
            self.assertTrue(os.path.exists(meta_path))

            reopened = payload_store.PayloadStore(tmpdir, mode="r")
            self.assertEqual(reopened.tickers, ["AAA3"])
            reopened.cleanup()
            self.assertFalse(os.path.exists(meta_path))


class TestStagedRunnerProgress(unittest.TestCase):
    def test_progress_file_serializes_new_metrics_and_oos_diagnostics(self):
        payload = {
            "chunks": [{
                "chunk_id": 1,
                "metrics": {
                    "sortino_med": 1.2,
                    "profit_factor_med": 1.5,
                    "expectancy_med": 0.01,
                    "payoff_ratio_med": 1.3,
                    "mdd_duration_med": 12.0,
                    "cagr_med": 0.14,
                    "mean_excess": 0.02,
                },
                "oos_diagnostics": {
                    "selected_window_indices": [0, 3],
                    "gap_summary": {
                        "fitness_gap": 0.1,
                        "win_rate_gap": 0.02,
                        "win_rate_target_gap": 0.01,
                        "mdd_gap": -0.01,
                        "excess_return_gap": 0.005,
                    },
                    "window_metrics": [{
                        "window_index": 0,
                        "train": {"fitness": 1.0},
                        "oos": {"fitness": 0.9},
                    }],
                },
            }],
        }

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                staged_runner._save_progress(payload)
                with open(staged_runner.PROGRESS_FILE, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
            finally:
                os.chdir(old_cwd)

        metrics = loaded["chunks"][0]["metrics"]
        self.assertIn("sortino_med", metrics)
        self.assertIn("profit_factor_med", metrics)
        self.assertIn("mdd_duration_med", metrics)
        self.assertIn("oos_diagnostics", loaded["chunks"][0])
        self.assertEqual(loaded["chunks"][0]["oos_diagnostics"]["selected_window_indices"], [0, 3])


if __name__ == "__main__":
    unittest.main()
