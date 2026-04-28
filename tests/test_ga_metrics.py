import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from openpyxl import Workbook

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

import ensure_daily_telegram_file as telegram_daily_file
import ga_run
import ga_run_modular_final as ga_run_modular
import overnight_alpha_until_0600 as overnight_runner
import payload_store
import run_local_ga_staged as staged_runner
from analysis import overfitting_stage


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
        entry_aggressiveness=1.0,
        resistance_overext_gate=0.0,
        support_broken_gate=0.0,
        min_hold_bars=3,
        early_exit_block_bars=3,
        profit_take_min_bars=3,
        early_take_quality_gate=0.0,
        early_take_score_decay_max=0.35,
    )


def _sample_stat(**overrides):
    stat = {
        "sharpe": 0.8,
        "sortino": 1.0,
        "total_return": 0.12,
        "excess_return": 0.04,
        "alpha_ann": 0.03,
        "expectancy": 0.010,
        "payoff_ratio": 1.2,
        "profit_factor": 1.4,
        "cagr": 0.10,
        "buy_hold_cagr": 0.07,
        "avg_hold_bars": 7.0,
        "median_hold_bars": 6.0,
        "max_hold_bars": 18.0,
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
    def test_attach_benchmark_stats_computes_annualized_alpha(self):
        stats = {"total_return": 0.21, "cagr": ga_run._compute_cagr(0.21, 252)}
        close = np.array([100.0, 110.0, 121.0], dtype=np.float64)
        enriched = ga_run._attach_benchmark_stats(stats, close)
        self.assertIn("buy_hold_cagr", enriched)
        self.assertIn("alpha_ann", enriched)
        self.assertAlmostEqual(
            enriched["alpha_ann"],
            enriched["cagr"] - enriched["buy_hold_cagr"],
            places=9,
        )

    def test_no_trade_metrics_are_zero_and_finite(self):
        stats = ga_run._finalize_backtest_stats([], [], [], 0.25, 0.0, 0.0, 252, 0)
        self.assertEqual(stats["sortino"], 0.0)
        self.assertEqual(stats["profit_factor"], 0.0)
        self.assertEqual(stats["expectancy"], 0.0)
        self.assertEqual(stats["payoff_ratio"], 0.0)
        self.assertEqual(stats["max_drawdown_duration"], 0.0)
        self.assertEqual(stats["cagr"], 0.0)
        self.assertEqual(stats["avg_hold_bars"], 0.0)
        self.assertEqual(stats["median_hold_bars"], 0.0)
        for value in stats.values():
            if isinstance(value, float):
                self.assertTrue(np.isfinite(value))

    def test_all_win_metrics_are_bounded_and_positive(self):
        stats = ga_run._finalize_backtest_stats(
            [0.02, 0.03, 0.01],
            ["take", "time", "take"],
            [4, 6, 3],
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
        self.assertAlmostEqual(stats["avg_hold_bars"], 13.0 / 3.0, places=6)

    def test_all_loss_metrics_are_bounded_and_negative(self):
        stats = ga_run._finalize_backtest_stats(
            [-0.02, -0.03, -0.01],
            ["stop", "hard_loss", "time"],
            [8, 5, 4],
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
        self.assertEqual(stats["max_hold_bars"], 8.0)

    def test_mixed_trade_metrics_match_expected_ratios(self):
        stats = ga_run._finalize_backtest_stats(
            [0.02, -0.01, 0.03, -0.02],
            ["take", "stop", "time", "hard_loss"],
            [5, 3, 9, 4],
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
        self.assertEqual(stats["median_hold_bars"], 4.5)

    def test_recovered_and_unrecovered_drawdown_durations_are_preserved(self):
        recovered = ga_run._finalize_backtest_stats([0.01, -0.02, 0.03], ["time"] * 3, [3, 6, 4], 0.4, 0.018, -0.05, 252, 4)
        unrecovered = ga_run._finalize_backtest_stats([0.01, -0.02, -0.01], ["time"] * 3, [2, 8, 9], 0.4, -0.020, -0.08, 252, 9)
        self.assertEqual(recovered["max_drawdown_duration"], 4.0)
        self.assertEqual(unrecovered["max_drawdown_duration"], 9.0)

    def test_holdability_metrics_are_reported(self):
        stats = ga_run._finalize_backtest_stats(
            [0.03, 0.02, -0.01, 0.01],
            ["take", "take", "stop", "time"],
            [2, 5, 3, 8],
            0.55,
            0.05,
            -0.03,
            252,
            4,
            trade_rets_after_3=[-0.01, 0.02, -0.02, 0.01],
            trade_target_after_3=[False, True, False, False],
            early_take_events=1,
            early_take_dependency_events=1,
        )

        self.assertAlmostEqual(stats["wr_after_3_bars"], 0.5)
        self.assertAlmostEqual(stats["target_wr_after_3_bars"], 0.25)
        self.assertAlmostEqual(stats["early_take_dependency"], 1.0)
        self.assertAlmostEqual(stats["swing_survival_3d"], 0.75)
        self.assertIn("expected_hold_score", stats)


class TestOperationalSwingPolicy(unittest.TestCase):
    def test_operational_metrics_filter_premium_and_high_quality_only(self):
        premium = _sample_stat(
            win_rate=0.76,
            win_rate_target=0.73,
            alpha_ann=0.02,
            mdd=-0.08,
            n_trades=35,
            expectancy=0.012,
            median_hold_bars=6.0,
            max_hold_bars=14.0,
        )
        high_quality = _sample_stat(
            win_rate=0.68,
            win_rate_target=0.66,
            alpha_ann=-0.05,
            mdd=-0.12,
            n_trades=25,
            expectancy=0.008,
            median_hold_bars=8.0,
            max_hold_bars=18.0,
        )
        regular = _sample_stat(
            win_rate=0.80,
            win_rate_target=0.75,
            alpha_ann=-0.25,
            mdd=-0.20,
            n_trades=50,
        )

        metrics = ga_run.operational_metrics_from_stats([premium, high_quality, regular])

        self.assertEqual(metrics["n_buy_operable"], 2)
        self.assertAlmostEqual(metrics["wr_med_operable"], 0.72, places=9)
        self.assertAlmostEqual(metrics["wr_target_med_operable"], 0.695, places=9)
        self.assertAlmostEqual(metrics["alpha_ann_mean_operable"], -0.015, places=9)
        self.assertAlmostEqual(metrics["expectancy_net_med_operable"], 0.010, places=9)
        self.assertAlmostEqual(metrics["mdd_med_operable"], 0.10, places=9)
        self.assertAlmostEqual(metrics["hold_med_operable"], 7.0, places=9)

    def test_regular_buy_is_demoted_to_hold_observar(self):
        sig, reason = ga_run.enforce_operational_buy_gate(
            "buy",
            operable=False,
            stage_num=2,
            timing_sub="ideal",
            estagio="Stage 2 Alta ideal",
            entry_score=90.0,
            confidence=90.0,
            rank_score=90.0,
            net_potential_pct=0.08,
            close_val=105.0,
            pivot=100.0,
            s1=95.0,
            r1=110.0,
        )

        self.assertEqual(sig, "hold")
        self.assertIn("REGULAR", reason)

    def test_low_expected_hold_score_demotes_buy(self):
        sig, reason = ga_run.enforce_operational_buy_gate(
            "buy",
            operable=True,
            stage_num=2,
            timing_sub="ideal",
            estagio="Stage 2 Alta ideal",
            entry_score=90.0,
            confidence=90.0,
            rank_score=90.0,
            net_potential_pct=0.08,
            close_val=105.0,
            pivot=100.0,
            s1=95.0,
            r1=110.0,
            expected_hold_score=55.0,
            early_take_dependency=0.10,
            alpha_ann=0.01,
        )

        self.assertEqual(sig, "hold")
        self.assertIn("Expected hold score", reason)


class TestParityAndFitness(unittest.TestCase):
    def test_metric_helper_parity_between_ga_modules(self):
        trade_rets = [0.02, -0.01, 0.03, -0.02]
        exit_types = ["take", "stop", "time", "hard_loss"]
        hold_bars = [5, 3, 9, 4]
        main_stats = ga_run._finalize_backtest_stats(trade_rets, exit_types, hold_bars, 0.55, 0.018, -0.06, 252, 5)
        modular_stats = ga_run_modular._finalize_backtest_stats(trade_rets, exit_types, hold_bars, 0.55, 0.018, -0.06, 252, 5)
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

    def test_legacy_genomes_keep_neutral_swing_controls_until_promoted(self):
        legacy_len = len(ga_run.GLOBAL_PARAM_SPECS) - 5
        legacy = [spec[1] for spec in ga_run.GLOBAL_PARAM_SPECS[:legacy_len]]

        gp = ga_run.decode_global_params(legacy)

        self.assertEqual(gp.min_hold_bars, 0)
        self.assertEqual(gp.early_exit_block_bars, 0)
        self.assertEqual(gp.profit_take_min_bars, 0)
        self.assertEqual(gp.early_take_quality_gate, 0.0)
        self.assertAlmostEqual(gp.early_take_score_decay_max, 0.35, places=9)

    def test_profit_take_is_blocked_before_minimum_swing_hold(self):
        n = 90
        open_ = np.full(n, 100.0, dtype=np.float64)
        close = np.full(n, 100.0, dtype=np.float64)
        high = np.full(n, 101.0, dtype=np.float64)
        low = np.full(n, 99.8, dtype=np.float64)
        atr = np.full(n, 1.0, dtype=np.float64)
        score_matrix = np.ones((n, 2), dtype=np.float64)
        precomputed = {
            "vol_rank": np.full(n, 0.5),
            "volume": np.full(n, 1_000_000.0),
        }

        fast_gp = _build_gp()
        fast_gp.min_hold_bars = 0
        fast_gp.early_exit_block_bars = 0
        fast_gp.profit_take_min_bars = 0
        swing_gp = _build_gp()

        fast_stats = ga_run.backtest_stats_global_intraday(
            open_, high, low, close, score_matrix, atr, fast_gp, precomputed=precomputed, stage_gate="off"
        )
        swing_stats = ga_run.backtest_stats_global_intraday(
            open_, high, low, close, score_matrix, atr, swing_gp, precomputed=precomputed, stage_gate="off"
        )

        self.assertGreater(fast_stats["n_trades"], 0.0)
        self.assertGreater(swing_stats["n_trades"], 0.0)
        self.assertLess(fast_stats["median_hold_bars"], 3.0)
        self.assertGreaterEqual(swing_stats["median_hold_bars"], 3.0)
        self.assertGreater(fast_stats["pct_exit_early"], 0.0)
        self.assertEqual(swing_stats["pct_early_exit_take"], 0.0)

    def test_partial_take_before_three_bars_does_not_close_swing_trade(self):
        n = 90
        open_ = np.full(n, 100.0, dtype=np.float64)
        close = np.full(n, 100.0, dtype=np.float64)
        high = np.full(n, 101.0, dtype=np.float64)
        low = np.full(n, 99.8, dtype=np.float64)
        atr = np.full(n, 1.0, dtype=np.float64)
        score_matrix = np.ones((n, 2), dtype=np.float64)
        precomputed = {
            "vol_rank": np.full(n, 0.5),
            "volume": np.full(n, 1_000_000.0),
        }
        gp = _build_gp()
        gp.partial_take_pct = 0.50
        gp.partial_take_level = 0.50
        gp.profit_take_min_bars = 3

        stats = ga_run.backtest_stats_global_intraday(
            open_, high, low, close, score_matrix, atr, gp, precomputed=precomputed, stage_gate="off"
        )

        self.assertGreater(stats["n_trades"], 0.0)
        self.assertGreater(stats["early_take_deferred_events"], 0.0)
        self.assertGreaterEqual(stats["median_hold_bars"], 3.0)
        self.assertEqual(stats["pct_early_exit_take"], 0.0)

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
            _sample_stat(sortino=3.5, profit_factor=3.0, expectancy=0.020, payoff_ratio=2.0, excess_return=-0.05, alpha_ann=-0.03, total_return=0.01),
            _sample_stat(sortino=3.2, profit_factor=2.8, expectancy=0.018, payoff_ratio=1.8, excess_return=-0.04, alpha_ann=-0.02, total_return=0.01, win_rate=0.64, win_rate_target=0.46),
        ]
        self.assertLessEqual(
            ga_run.global_fitness_from_stats(negative_alpha),
            ga_run.global_fitness_from_stats(baseline),
        )
        self.assertLessEqual(
            ga_run_modular.global_fitness_from_stats(negative_alpha),
            ga_run_modular.global_fitness_from_stats(baseline),
        )

    def test_alpha_focus_prefers_alpha_recovery_over_small_wr_edge(self):
        high_wr_bad_alpha = [
            _sample_stat(
                win_rate=0.69,
                win_rate_target=0.61,
                sortino=1.8,
                profit_factor=1.9,
                expectancy=0.018,
                payoff_ratio=1.5,
                alpha_ann=-0.12,
                cagr=0.02,
                buy_hold_cagr=0.14,
                total_return=0.05,
                mdd=-0.11,
            ),
            _sample_stat(
                win_rate=0.67,
                win_rate_target=0.58,
                sortino=1.7,
                profit_factor=1.8,
                expectancy=0.016,
                payoff_ratio=1.4,
                alpha_ann=-0.11,
                cagr=0.03,
                buy_hold_cagr=0.14,
                total_return=0.06,
                mdd=-0.11,
            ),
        ]
        alpha_recovery = [
            _sample_stat(
                win_rate=0.64,
                win_rate_target=0.56,
                sortino=1.2,
                profit_factor=1.35,
                expectancy=0.009,
                payoff_ratio=1.15,
                alpha_ann=-0.04,
                cagr=0.09,
                buy_hold_cagr=0.13,
                total_return=0.09,
                mdd=-0.10,
            ),
            _sample_stat(
                win_rate=0.63,
                win_rate_target=0.55,
                sortino=1.1,
                profit_factor=1.30,
                expectancy=0.008,
                payoff_ratio=1.10,
                alpha_ann=-0.03,
                cagr=0.10,
                buy_hold_cagr=0.13,
                total_return=0.10,
                mdd=-0.10,
            ),
        ]

        prev_alpha_focus_main = ga_run.GA_ALPHA_FOCUS
        prev_alpha_focus_mod = ga_run_modular.GA_ALPHA_FOCUS
        ga_run.GA_ALPHA_FOCUS = True
        ga_run_modular.GA_ALPHA_FOCUS = True
        try:
            self.assertGreater(
                ga_run.global_fitness_from_stats(alpha_recovery),
                ga_run.global_fitness_from_stats(high_wr_bad_alpha),
            )
            self.assertGreater(
                ga_run_modular.global_fitness_from_stats(alpha_recovery),
                ga_run_modular.global_fitness_from_stats(high_wr_bad_alpha),
            )
        finally:
            ga_run.GA_ALPHA_FOCUS = prev_alpha_focus_main
            ga_run_modular.GA_ALPHA_FOCUS = prev_alpha_focus_mod

    def test_fitness_prefers_swing_holding_profile_when_edge_is_similar(self):
        swing_profile = [
            _sample_stat(avg_hold_bars=8.0, median_hold_bars=7.0, max_hold_bars=20.0),
            _sample_stat(avg_hold_bars=9.0, median_hold_bars=8.0, max_hold_bars=22.0, win_rate=0.65, win_rate_target=0.47),
        ]
        too_long_profile = [
            _sample_stat(avg_hold_bars=42.0, median_hold_bars=30.0, max_hold_bars=75.0),
            _sample_stat(avg_hold_bars=38.0, median_hold_bars=28.0, max_hold_bars=70.0, win_rate=0.65, win_rate_target=0.47),
        ]
        self.assertGreater(
            ga_run.global_fitness_from_stats(swing_profile),
            ga_run.global_fitness_from_stats(too_long_profile),
        )
        self.assertGreater(
            ga_run_modular.global_fitness_from_stats(swing_profile),
            ga_run_modular.global_fitness_from_stats(too_long_profile),
        )

    def test_fitness_penalizes_operable_two_day_holding_profile(self):
        fast_operable = [
            _sample_stat(
                win_rate=0.76,
                win_rate_target=0.73,
                alpha_ann=-0.02,
                mdd=-0.09,
                n_trades=35,
                avg_hold_bars=2.2,
                median_hold_bars=2.0,
                max_hold_bars=6.0,
            ),
            _sample_stat(
                win_rate=0.70,
                win_rate_target=0.67,
                alpha_ann=-0.03,
                mdd=-0.10,
                n_trades=25,
                avg_hold_bars=2.3,
                median_hold_bars=2.0,
                max_hold_bars=7.0,
            ),
        ]
        swing_operable = [
            dict(s, avg_hold_bars=6.0, median_hold_bars=6.0, max_hold_bars=14.0)
            for s in fast_operable
        ]

        self.assertGreater(
            ga_run.global_fitness_from_stats(swing_operable),
            ga_run.global_fitness_from_stats(fast_operable),
        )
        self.assertGreater(
            ga_run_modular.global_fitness_from_stats(swing_operable),
            ga_run_modular.global_fitness_from_stats(fast_operable),
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


class TestStagedAcceptance(unittest.TestCase):
    def test_acceptance_uses_alpha_ann_improvement_instead_of_raw_excess_floor(self):
        base = {
            "wr_med_all": 0.60,
            "wr_target_med_all": 0.50,
            "mdd_med": 0.10,
            "mdd_p75": 0.12,
            "mdd_duration_med": 35.0,
            "mean_alpha_ann": -0.08,
            "alpha_ann_pos_rate": 0.20,
            "trades_med": 10.0,
        }
        improved = {
            "wr_med_all": 0.72,
            "wr_target_med_all": 0.62,
            "mdd_med": 0.11,
            "mdd_p75": 0.13,
            "mdd_duration_med": 39.0,
            "mean_alpha_ann": -0.05,
            "alpha_ann_pos_rate": 0.28,
            "trades_med": 12.0,
            "avg_hold_bars_med": 8.0,
            "median_hold_bars_med": 6.0,
            "max_hold_bars_p75": 20.0,
            "mean_excess": -5.0,
        }
        acceptance = staged_runner._compute_acceptance(improved, base)
        self.assertTrue(acceptance["mean_alpha_ann_up"])
        self.assertTrue(acceptance["alpha_ann_pos_rate_up"])
        self.assertTrue(acceptance["wr_med_all_floor"])
        self.assertTrue(acceptance["wr_target_med_all_floor"])
        self.assertTrue(acceptance["all_pass"])

    def test_acceptance_allows_small_mdd_increase_when_other_metrics_improve(self):
        base = {
            "wr_med_all": 0.33,
            "wr_target_med_all": 0.00,
            "mdd_med": 0.08,
            "mdd_p75": 0.10,
            "mdd_duration_med": 32.0,
            "mean_alpha_ann": -0.095,
            "alpha_ann_pos_rate": 0.16,
            "trades_med": 11.0,
        }
        improved = {
            "wr_med_all": 0.65,
            "wr_target_med_all": 0.58,
            "mdd_med": 0.117,
            "mdd_p75": 0.115,
            "mdd_duration_med": 40.0,
            "mean_alpha_ann": -0.085,
            "alpha_ann_pos_rate": 0.18,
            "trades_med": 20.0,
            "avg_hold_bars_med": 9.0,
            "median_hold_bars_med": 7.0,
            "max_hold_bars_p75": 20.0,
        }
        acceptance = staged_runner._compute_acceptance(improved, base)
        self.assertTrue(acceptance["strong_alpha_wr_relax"])
        self.assertGreater(acceptance["mdd_relaxation_bonus"], 0.0)
        self.assertAlmostEqual(acceptance["alpha_ann_pos_rate_target"], 0.17, places=9)
        self.assertTrue(acceptance["mdd_med_not_worse"])
        self.assertTrue(acceptance["all_pass"])

    def test_acceptance_requires_meaningful_alpha_pos_rate_improvement(self):
        base = {
            "wr_med_all": 0.33,
            "wr_target_med_all": 0.00,
            "mdd_med": 0.08,
            "mdd_p75": 0.10,
            "mdd_duration_med": 32.0,
            "mean_alpha_ann": -0.095,
            "alpha_ann_pos_rate": 0.16,
            "trades_med": 11.0,
        }
        improved = {
            "wr_med_all": 0.65,
            "wr_target_med_all": 0.58,
            "mdd_med": 0.117,
            "mdd_p75": 0.115,
            "mdd_duration_med": 40.0,
            "mean_alpha_ann": -0.085,
            "alpha_ann_pos_rate": 0.169,
            "trades_med": 20.0,
            "avg_hold_bars_med": 9.0,
            "median_hold_bars_med": 7.0,
            "max_hold_bars_p75": 20.0,
        }
        acceptance = staged_runner._compute_acceptance(improved, base)
        self.assertTrue(acceptance["strong_alpha_wr_relax"])
        self.assertFalse(acceptance["alpha_ann_pos_rate_up"])
        self.assertAlmostEqual(acceptance["alpha_ann_pos_rate_target"], 0.17, places=9)
        self.assertFalse(acceptance["all_pass"])

    def test_acceptance_keeps_tighter_mdd_gate_when_alpha_gain_is_small(self):
        base = {
            "wr_med_all": 0.33,
            "wr_target_med_all": 0.00,
            "mdd_med": 0.08,
            "mdd_p75": 0.10,
            "mdd_duration_med": 32.0,
            "mean_alpha_ann": -0.095,
            "alpha_ann_pos_rate": 0.16,
            "trades_med": 11.0,
        }
        improved = {
            "wr_med_all": 0.58,
            "wr_target_med_all": 0.50,
            "mdd_med": 0.131,
            "mdd_p75": 0.125,
            "mdd_duration_med": 40.0,
            "mean_alpha_ann": -0.091,
            "alpha_ann_pos_rate": 0.19,
            "trades_med": 20.0,
            "avg_hold_bars_med": 9.0,
            "median_hold_bars_med": 7.0,
            "max_hold_bars_p75": 20.0,
        }
        acceptance = staged_runner._compute_acceptance(improved, base)
        self.assertFalse(acceptance["strong_alpha_wr_relax"])
        self.assertLess(acceptance["mdd_relaxation_bonus"], 0.0121)
        self.assertFalse(acceptance["mdd_med_not_worse"])
        self.assertFalse(acceptance["all_pass"])

    def test_acceptance_blocks_tail_risk_and_non_swing_holding_profile(self):
        base = {
            "wr_med_all": 0.50,
            "wr_target_med_all": 0.40,
            "mdd_med": 0.09,
            "mdd_p75": 0.11,
            "mdd_duration_med": 30.0,
            "mean_alpha_ann": -0.07,
            "alpha_ann_pos_rate": 0.18,
            "trades_med": 12.0,
        }
        improved = {
            "wr_med_all": 0.64,
            "wr_target_med_all": 0.55,
            "mdd_med": 0.11,
            "mdd_p75": 0.18,
            "mdd_duration_med": 55.0,
            "mean_alpha_ann": -0.05,
            "alpha_ann_pos_rate": 0.22,
            "trades_med": 13.0,
            "avg_hold_bars_med": 26.0,
            "median_hold_bars_med": 16.0,
            "max_hold_bars_p75": 70.0,
        }
        acceptance = staged_runner._compute_acceptance(improved, base)
        self.assertFalse(acceptance["mdd_p75_not_worse"])
        self.assertFalse(acceptance["mdd_duration_not_worse"])
        self.assertFalse(acceptance["swing_hold_ok"])

    def test_acceptance_blocks_fast_two_day_hold_for_swing(self):
        base = {
            "wr_med_operable": 0.67,
            "wr_target_med_operable": 0.60,
            "mdd_med_operable": 0.10,
            "mdd_p75": 0.12,
            "mdd_duration_med": 30.0,
            "alpha_ann_mean_operable": -0.02,
            "expectancy_net_med_operable": 0.006,
            "alpha_ann_pos_rate": 0.18,
            "trades_med": 18.0,
        }
        fast_scalp = {
            "wr_med_operable": 0.76,
            "wr_target_med_operable": 0.70,
            "mdd_med_operable": 0.09,
            "mdd_p75": 0.11,
            "mdd_duration_med": 28.0,
            "alpha_ann_mean_operable": 0.01,
            "expectancy_net_med_operable": 0.012,
            "alpha_ann_pos_rate": 0.24,
            "trades_med": 22.0,
            "hold_med_operable": 2.0,
            "max_hold_bars_p75_operable": 6.0,
        }
        acceptance = staged_runner._compute_acceptance(fast_scalp, base)
        self.assertFalse(acceptance["swing_hold_ok"])
        self.assertFalse(acceptance["all_pass"])

    def test_candidate_with_higher_wr_but_alpha_expectancy_regression_is_rejected(self):
        base = {
            "wr_med_operable": 0.62,
            "wr_target_med_operable": 0.55,
            "mdd_med_operable": 0.10,
            "mdd_p75": 0.12,
            "mdd_duration_med": 30.0,
            "alpha_ann_mean_operable": -0.02,
            "expectancy_net_med_operable": 0.005,
            "alpha_ann_pos_rate": 0.18,
            "trades_med": 18.0,
        }
        incumbent = {
            "wr_med_operable": 0.70,
            "wr_target_med_operable": 0.66,
            "mdd_med_operable": 0.10,
            "mdd_p75": 0.12,
            "mdd_duration_med": 30.0,
            "alpha_ann_mean_operable": 0.020,
            "expectancy_net_med_operable": 0.012,
            "alpha_ann_pos_rate": 0.22,
            "trades_med": 22.0,
            "hold_med_operable": 6.0,
            "max_hold_bars_p75_operable": 18.0,
            "fit": 1.0,
        }
        higher_wr_bad_edge = {
            "wr_med_operable": 0.75,
            "wr_target_med_operable": 0.70,
            "mdd_med_operable": 0.10,
            "mdd_p75": 0.12,
            "mdd_duration_med": 30.0,
            "alpha_ann_mean_operable": 0.010,
            "expectancy_net_med_operable": 0.008,
            "alpha_ann_pos_rate": 0.25,
            "trades_med": 22.0,
            "hold_med_operable": 6.0,
            "max_hold_bars_p75_operable": 18.0,
            "fit": 1.2,
        }
        decision = staged_runner._candidate_beats_incumbent(higher_wr_bad_edge, base, incumbent)
        safety = decision["incumbent_safety_guardrail"]
        self.assertFalse(safety["alpha_not_worse"])
        self.assertFalse(safety["expectancy_not_worse"])
        self.assertFalse(decision["promote"])

    def test_incumbent_wr_guardrail_blocks_meaningful_wr_regression(self):
        incumbent = {
            "wr_med_all": 0.65,
            "wr_target_med_all": 0.64,
        }
        weaker = {
            "wr_med_all": 0.635,
            "wr_target_med_all": 0.620,
        }
        stronger = {
            "wr_med_all": 0.648,
            "wr_target_med_all": 0.636,
        }
        blocked = staged_runner._compute_incumbent_wr_guardrail(weaker, incumbent)
        allowed = staged_runner._compute_incumbent_wr_guardrail(stronger, incumbent)
        self.assertFalse(blocked["wr_ok"])
        self.assertFalse(blocked["all_pass"])
        self.assertTrue(allowed["wr_ok"])
        self.assertTrue(allowed["wr_target_ok"])
        self.assertTrue(allowed["all_pass"])

    def test_incumbent_safety_guardrail_blocks_tail_drawdown_regression(self):
        incumbent = {
            "wr_med_all": 0.65,
            "wr_target_med_all": 0.64,
            "mdd_med": 0.11,
            "mdd_p75": 0.13,
            "mdd_duration_med": 34.0,
            "trades_med": 20.0,
            "avg_hold_bars_med": 8.0,
            "median_hold_bars_med": 6.0,
            "max_hold_bars_p75": 20.0,
            "mean_alpha_ann": -0.08,
        }
        weaker = {
            "wr_med_all": 0.652,
            "wr_target_med_all": 0.642,
            "mdd_med": 0.128,
            "mdd_p75": 0.151,
            "mdd_duration_med": 50.0,
            "trades_med": 19.0,
            "avg_hold_bars_med": 8.0,
            "median_hold_bars_med": 6.0,
            "max_hold_bars_p75": 20.0,
            "mean_alpha_ann": -0.078,
        }
        stronger = {
            "wr_med_all": 0.661,
            "wr_target_med_all": 0.651,
            "mdd_med": 0.121,
            "mdd_p75": 0.140,
            "mdd_duration_med": 39.0,
            "trades_med": 19.0,
            "avg_hold_bars_med": 8.0,
            "median_hold_bars_med": 6.0,
            "max_hold_bars_p75": 20.0,
            "mean_alpha_ann": -0.074,
        }
        blocked = staged_runner._compute_incumbent_safety_guardrail(weaker, incumbent)
        allowed = staged_runner._compute_incumbent_safety_guardrail(stronger, incumbent)
        self.assertFalse(blocked["mdd_ok"])
        self.assertFalse(blocked["all_pass"])
        self.assertTrue(allowed["mdd_ok"])
        self.assertTrue(allowed["mdd_p75_ok"])
        self.assertTrue(allowed["all_pass"])

    def test_candidate_beats_incumbent_requires_shared_guardrails(self):
        base = {
            "wr_med_all": 0.33,
            "wr_target_med_all": 0.00,
            "mdd_med": 0.08,
            "mdd_p75": 0.10,
            "mdd_duration_med": 30.0,
            "mean_alpha_ann": -0.095,
            "alpha_ann_pos_rate": 0.16,
            "trades_med": 11.0,
        }
        incumbent = {
            "wr_med_all": 0.65,
            "wr_target_med_all": 0.63,
            "mdd_med": 0.11,
            "mdd_p75": 0.13,
            "mdd_duration_med": 34.0,
            "mean_alpha_ann": -0.08,
            "alpha_ann_pos_rate": 0.20,
            "trades_med": 20.0,
            "avg_hold_bars_med": 8.0,
            "median_hold_bars_med": 6.0,
            "max_hold_bars_p75": 20.0,
            "fit": 1.0,
        }
        candidate = {
            "wr_med_all": 0.645,
            "wr_target_med_all": 0.620,
            "mdd_med": 0.115,
            "mdd_p75": 0.138,
            "mdd_duration_med": 38.0,
            "mean_alpha_ann": -0.078,
            "alpha_ann_pos_rate": 0.21,
            "trades_med": 19.0,
            "avg_hold_bars_med": 8.0,
            "median_hold_bars_med": 6.0,
            "max_hold_bars_p75": 20.0,
            "fit": 1.2,
        }
        decision = staged_runner._candidate_beats_incumbent(candidate, base, incumbent)
        self.assertFalse(decision["incumbent_wr_guardrail"]["all_pass"])
        self.assertFalse(decision["promote"])

    def test_overnight_best_candidate_comes_from_best_so_far(self):
        progress = {
            "base_metrics": {
                "wr_med_all": 0.40,
                "wr_target_med_all": 0.30,
                "mdd_med": 0.10,
                "mdd_p75": 0.12,
                "mdd_duration_med": 30.0,
                "mean_alpha_ann": -0.09,
                "alpha_ann_pos_rate": 0.16,
                "trades_med": 12.0,
            },
            "best_so_far": {
                "found_at_chunk": 3,
                "genome": [0.1, 0.2],
                "metrics": {
                    "wr_med_all": 0.60,
                    "wr_target_med_all": 0.56,
                    "mdd_med": 0.12,
                    "mdd_p75": 0.14,
                    "mdd_duration_med": 36.0,
                    "mean_alpha_ann": -0.07,
                    "alpha_ann_pos_rate": 0.19,
                    "trades_med": 13.0,
                    "avg_hold_bars_med": 8.0,
                    "median_hold_bars_med": 6.0,
                    "max_hold_bars_p75": 20.0,
                    "fit": 1.0,
                },
            },
        }
        candidate = overnight_runner._find_best_accepted(progress)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["chunk_id"], 3)
        self.assertEqual(candidate["best_genome"], [0.1, 0.2])


class TestStagedRunnerProgress(unittest.TestCase):
    def test_round_state_auto_rotates_window_offset(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                round_state_1, offset_1 = staged_runner._resolve_round_state(
                    reset=True,
                    requested_offset=None,
                    total_windows=36,
                    window_count=4,
                )
                round_state_2, offset_2 = staged_runner._resolve_round_state(
                    reset=True,
                    requested_offset=None,
                    total_windows=36,
                    window_count=4,
                )
            finally:
                os.chdir(old_cwd)

        self.assertEqual(round_state_1["round_index"], 1)
        self.assertEqual(round_state_2["round_index"], 2)
        self.assertNotEqual(offset_1, offset_2)

    def test_seed_best_so_far_is_persisted_as_incumbent(self):
        seed_genome = [0.1, 0.2, 0.3]
        seed_metrics = {
            "fit": -4.75,
            "wr_med_all": 0.625,
            "mean_alpha_ann": -0.088,
            "n_ge_70": 41,
        }
        best = staged_runner._build_seed_best_so_far(seed_genome, seed_metrics)

        self.assertEqual(best["found_at_chunk"], 0)
        self.assertEqual(best["genome"], seed_genome)
        self.assertEqual(best["metrics"]["fit"], seed_metrics["fit"])
        self.assertEqual(best["wr_med_all"], seed_metrics["wr_med_all"])

    def test_progress_file_serializes_new_metrics_and_oos_diagnostics(self):
        payload = {
            "chunks": [{
                "chunk_id": 1,
                "metrics": {
                    "sortino_med": 1.2,
                    "profit_factor_med": 1.5,
                    "expectancy_med": 0.01,
                    "payoff_ratio_med": 1.3,
                    "mdd_p75": 0.14,
                    "mdd_duration_med": 12.0,
                    "mdd_duration_p75": 16.0,
                    "cagr_med": 0.14,
                    "mean_excess": 0.02,
                    "mean_alpha_ann": 0.03,
                    "avg_hold_bars_med": 8.0,
                    "median_hold_bars_med": 6.0,
                    "max_hold_bars_p75": 22.0,
                },
                "oos_diagnostics": {
                    "selected_window_indices": [0, 3],
                    "gap_summary": {
                        "fitness_gap": 0.1,
                        "win_rate_gap": 0.02,
                        "win_rate_target_gap": 0.01,
                        "mdd_gap": -0.01,
                        "excess_return_gap": 0.005,
                        "alpha_ann_gap": 0.002,
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
        self.assertIn("mdd_p75", metrics)
        self.assertIn("mdd_duration_med", metrics)
        self.assertIn("max_hold_bars_p75", metrics)
        self.assertIn("mean_alpha_ann", metrics)
        self.assertIn("oos_diagnostics", loaded["chunks"][0])
        self.assertEqual(loaded["chunks"][0]["oos_diagnostics"]["selected_window_indices"], [0, 3])

    def test_sync_best_so_far_updates_progress_and_file(self):
        best = {
            "fit": -4.75,
            "wr_med_all": 0.625,
            "genome": [0.1, 0.2, 0.3],
            "metrics": {"fit": -4.75, "wr_med_all": 0.625},
            "found_at_chunk": 0,
        }
        progress = {"chunks": [], "best_so_far": None}

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                staged_runner._sync_best_so_far_state(progress, best)
                with open(staged_runner.PROGRESS_FILE, "r", encoding="utf-8") as fh:
                    saved_progress = json.load(fh)
                with open(staged_runner.BEST_SOFAR_FILE, "r", encoding="utf-8") as fh:
                    saved_best = json.load(fh)
            finally:
                os.chdir(old_cwd)

        self.assertEqual(saved_progress["best_so_far"]["fit"], best["fit"])


class TestOverfittingGuards(unittest.TestCase):
    def test_operable_bootstrap_ci_and_wilson_fail_closed(self):
        holdout = {
            "delta_holdout_vs_train": {
                "wr_med_all": 0.0,
                "mean_alpha_ann": 0.0,
                "wr_med_operable": 0.0,
                "alpha_ann_mean_operable": 0.0,
            }
        }
        bootstrap = {
            "operable_only": True,
            "n_tickers_in_bootstrap": 5,
            "wr_ci_width": 0.16,
            "wr_wilson_lo": 0.54,
        }
        rolling = {
            "median_delta_oos_vs_train": {
                "wr_med_operable": 0.0,
                "alpha_ann_mean_operable": 0.0,
            }
        }

        verdict, reasons = overfitting_stage.classify(
            holdout,
            bootstrap,
            {"median_abs_delta": 0.0},
            rolling,
        )

        self.assertEqual(verdict, "OVERFIT")
        self.assertTrue(any("operable_wr_wilson_lo" in r for r in reasons))
        self.assertTrue(any("operable_wr_ci_width" in r for r in reasons))


class TestTelegramDailyFile(unittest.TestCase):
    def test_parse_excel_latest_date_reads_latest_apply_day(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path = os.path.join(tmpdir, "summary_latest.xlsx")
            wb = Workbook()
            ws = wb.active
            ws.title = "Apply"
            ws.append(["Date", "ticker", "signal"])
            ws.append(["2026-04-14", "AAA3", "buy"])
            ws.append(["2026-04-15", "BBB4", "hold"])
            wb.save(xlsx_path)
            wb.close()

            latest = telegram_daily_file._parse_excel_latest_date(Path(xlsx_path))

        self.assertEqual(str(latest), "2026-04-15")

    def test_has_current_day_data_detects_stale_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path = os.path.join(tmpdir, "summary_latest.xlsx")
            wb = Workbook()
            ws = wb.active
            ws.title = "Apply"
            ws.append(["Date", "ticker", "signal"])
            ws.append(["2026-04-14", "AAA3", "buy"])
            wb.save(xlsx_path)
            wb.close()

            fresh, latest = telegram_daily_file._has_current_day_data(
                Path(xlsx_path),
                telegram_daily_file._target_date("2026-04-15"),
            )

        self.assertFalse(fresh)
        self.assertEqual(str(latest), "2026-04-14")


if __name__ == "__main__":
    unittest.main()
