from analysis.holdout_operability_diagnostic import failure_reasons
from run_local_fullmetric_sweep_oos import oos_gate, recent_gate


def test_failure_reasons_explain_non_swing_holdout():
    view = {
        "universe": "REGULAR",
        "n_trades": 4,
        "win_rate": 0.55,
        "win_rate_target": 0.50,
        "alpha_ann": -0.25,
        "expectancy": -0.01,
        "mdd_abs": 0.20,
        "median_hold_bars": 2.0,
        "expected_hold_score": 55.0,
        "wr_after_3_bars": 0.52,
        "swing_survival_3d": 0.30,
        "early_take_dependency": 0.50,
    }

    reasons = failure_reasons(view)

    assert "low_trades_lt_20" in reasons
    assert "hold_lt_3" in reasons
    assert "expected_hold_lt_70" in reasons
    assert "universe_regular" in reasons


def test_oos_gate_requires_holdout_coverage():
    train = {"n_buy_operable": 20}
    holdout = {
        "n_buy_operable": 0,
        "wr_med_operable": 0.80,
        "wr_target_med_operable": 0.75,
        "alpha_ann_mean_operable": 0.02,
        "expectancy_net_med_operable": 0.01,
        "hold_med_operable": 4.0,
        "max_hold_bars_p75_operable": 8.0,
        "expected_hold_score_med_operable": 80.0,
        "wr_after_3_bars_med_operable": 0.70,
    }
    ref = {"alpha_ann_mean_operable": -0.05}

    gate = oos_gate(train, holdout, ref, min_holdout_operable=5)

    assert gate["holdout_coverage_ok"] is False
    assert gate["all_pass"] is False


def test_oos_gate_accepts_stable_swing_candidate():
    train = {"n_buy_operable": 24}
    holdout = {
        "n_buy_operable": 8,
        "wr_med_operable": 0.68,
        "wr_target_med_operable": 0.62,
        "alpha_ann_mean_operable": -0.03,
        "expectancy_net_med_operable": 0.006,
        "hold_med_operable": 4.0,
        "max_hold_bars_p75_operable": 9.0,
        "expected_hold_score_med_operable": 74.0,
        "wr_after_3_bars_med_operable": 0.64,
    }
    ref = {"alpha_ann_mean_operable": -0.05}

    gate = oos_gate(train, holdout, ref, min_holdout_operable=5)

    assert gate["all_pass"] is True
    assert gate["n_buy_operable_holdout"] == 8


def test_recent_gate_ranks_diagnostic_candidates_without_hard_oos_coverage():
    summary = {
        "n_ready": 1,
        "n_watch": 2,
        "score_p75": 72.0,
        "score_median": 64.0,
        "score_max": 81.0,
    }

    gate = recent_gate(summary, min_recent_ready=1, min_recent_watch=3, min_recent_score=60.0)

    assert gate["all_pass"] is True
    assert gate["n_recent_ready"] == 1
