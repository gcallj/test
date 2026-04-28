from analysis.recent_operability import (
    READY_TIER,
    WATCH_TIER,
    recent_operability_blockers,
    recent_operability_score,
    recent_operability_tier,
    recent_summary_from_views,
)


def test_recent_operability_ready_candidate_scores_high():
    view = {
        "universe": "REGULAR",
        "n_trades": 6,
        "win_rate": 0.70,
        "win_rate_target": 0.66,
        "alpha_ann": -0.05,
        "expectancy": 0.01,
        "median_hold_bars": 4.0,
        "expected_hold_score": 78.0,
        "wr_after_3_bars": 0.66,
        "swing_survival_3d": 0.72,
        "early_take_dependency": 0.15,
    }

    score = recent_operability_score(view)

    assert score >= 75.0
    assert recent_operability_tier(view, score) == READY_TIER


def test_recent_operability_blocks_fast_edge():
    view = {
        "universe": "HIGH_QUAL",
        "n_trades": 2,
        "win_rate": 0.78,
        "win_rate_target": 0.70,
        "alpha_ann": 0.02,
        "expectancy": 0.02,
        "median_hold_bars": 1.5,
        "expected_hold_score": 58.0,
        "wr_after_3_bars": 0.45,
        "swing_survival_3d": 0.30,
        "early_take_dependency": 0.70,
    }

    blockers = recent_operability_blockers(view)

    assert "recent_low_trades_lt_3" in blockers
    assert "recent_hold_lt_2_5" in blockers
    assert "recent_wr_after_3_lt_58" in blockers
    assert recent_operability_tier(view) != READY_TIER


def test_recent_summary_counts_ready_and_watch():
    ready = {
        "ticker": "AAA",
        "universe": "REGULAR",
        "n_trades": 6,
        "win_rate": 0.70,
        "win_rate_target": 0.66,
        "alpha_ann": -0.05,
        "expectancy": 0.01,
        "median_hold_bars": 4.0,
        "expected_hold_score": 78.0,
        "wr_after_3_bars": 0.66,
        "swing_survival_3d": 0.72,
        "early_take_dependency": 0.15,
    }
    watch = {
        "ticker": "BBB",
        "universe": "REGULAR",
        "n_trades": 4,
        "win_rate": 0.66,
        "win_rate_target": 0.62,
        "alpha_ann": -0.08,
        "expectancy": 0.004,
        "median_hold_bars": 3.0,
        "expected_hold_score": 75.0,
        "wr_after_3_bars": 0.62,
        "swing_survival_3d": 0.62,
        "early_take_dependency": 0.25,
    }

    summary = recent_summary_from_views([ready, watch])

    assert summary["n_ready"] == 1
    assert summary["n_watch"] >= 1
    assert summary["top_recent_candidates"][0]["ticker"] == "AAA"
