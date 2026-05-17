import json
import tempfile
import unittest
from pathlib import Path

from analysis.adaptive_oos_floors import (
    DEFAULT_BUFFERS,
    STATIC_FLOORS,
    adaptive_floors,
    describe_floors,
    load_baseline,
    write_baseline,
)
from run_local_fullmetric_sweep_oos import oos_gate


def _good_holdout():
    return {
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


class AdaptiveFloors(unittest.TestCase):
    def test_no_baseline_returns_static_floors(self):
        floors = adaptive_floors({})
        self.assertEqual(floors["min_wr_operable"], STATIC_FLOORS["min_wr_operable"])
        self.assertEqual(floors["min_wr_after_3"], STATIC_FLOORS["min_wr_after_3"])
        self.assertEqual(floors["min_expected_hold_score"], STATIC_FLOORS["min_expected_hold_score"])

    def test_baseline_below_static_keeps_static_floor(self):
        # B&H WR of 50%: 0.50 + buffer 0.05 = 0.55, less than static 0.60 -> keep 0.60.
        floors = adaptive_floors({"bh_wr_median": 0.50})
        self.assertEqual(floors["min_wr_operable"], 0.60)

    def test_baseline_above_static_lifts_floor(self):
        # B&H WR of 62%: 0.62 + buffer 0.05 = 0.67, lifts above static 0.60.
        floors = adaptive_floors({"bh_wr_median": 0.62})
        self.assertAlmostEqual(floors["min_wr_operable"], 0.67, places=10)

    def test_custom_buffer_respected(self):
        floors = adaptive_floors(
            {"bh_wr_median": 0.62},
            buffers={"min_wr_operable": 0.10},
        )
        self.assertAlmostEqual(floors["min_wr_operable"], 0.72, places=10)

    def test_invalid_baseline_value_falls_back_to_static(self):
        floors = adaptive_floors({"bh_wr_median": "not_a_number"})
        self.assertEqual(floors["min_wr_operable"], 0.60)

    def test_partial_baseline_uses_static_for_absent_metrics(self):
        # Only WR baseline provided — others fall back to static.
        floors = adaptive_floors({"bh_wr_median": 0.65})
        self.assertGreater(floors["min_wr_operable"], 0.60)
        self.assertEqual(floors["min_wr_after_3"], STATIC_FLOORS["min_wr_after_3"])
        self.assertEqual(floors["min_expected_hold_score"], STATIC_FLOORS["min_expected_hold_score"])

    def test_default_buffers_match_documented_values(self):
        self.assertEqual(DEFAULT_BUFFERS["min_wr_operable"], 0.05)
        self.assertEqual(DEFAULT_BUFFERS["min_wr_after_3"], 0.05)


class BaselineIO(unittest.TestCase):
    def test_load_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            absent = Path(tmp) / "missing.json"
            self.assertEqual(load_baseline(absent), {})

    def test_write_then_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "_baselines.json"
            payload = {"asof": "2026-04-15", "bh_wr_median": 0.62, "bh_wr_after_3_median": 0.61}
            write_baseline(payload, path=path)
            loaded = load_baseline(path)
            self.assertEqual(loaded, payload)

    def test_load_malformed_json_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text("not valid json at all", encoding="utf-8")
            self.assertEqual(load_baseline(path), {})


class OOSGateIntegration(unittest.TestCase):
    def test_default_call_unchanged_when_no_floors_supplied(self):
        # Sanity: existing static gate still accepts the stable candidate.
        gate = oos_gate(
            train_metrics={"n_buy_operable": 24},
            holdout_metrics=_good_holdout(),
            reference_full_metrics={"alpha_ann_mean_operable": -0.05},
            min_holdout_operable=5,
        )
        self.assertTrue(gate["all_pass"])

    def test_adaptive_floors_can_tighten_to_block_a_weak_candidate(self):
        # Baseline that lifts WR floor to 0.72 — candidate has 0.68 -> blocks.
        floors = adaptive_floors({"bh_wr_median": 0.67})  # 0.67 + 0.05 buffer = 0.72
        gate = oos_gate(
            train_metrics={"n_buy_operable": 24},
            holdout_metrics=_good_holdout(),
            reference_full_metrics={"alpha_ann_mean_operable": -0.05},
            min_holdout_operable=5,
            **floors,
        )
        self.assertFalse(gate["holdout_wr_ok"])
        self.assertFalse(gate["all_pass"])

    def test_adaptive_floors_passthrough_when_baseline_weak(self):
        # Baseline at 0.50 -> floor stays 0.60 -> stable candidate still passes.
        floors = adaptive_floors({"bh_wr_median": 0.50, "bh_wr_after_3_median": 0.48})
        gate = oos_gate(
            train_metrics={"n_buy_operable": 24},
            holdout_metrics=_good_holdout(),
            reference_full_metrics={"alpha_ann_mean_operable": -0.05},
            min_holdout_operable=5,
            **floors,
        )
        self.assertTrue(gate["all_pass"])


class DescribeFloors(unittest.TestCase):
    def test_renders_one_line_summary(self):
        s = describe_floors({"min_wr_operable": 0.65, "min_wr_after_3": 0.62})
        self.assertIn("min_wr_operable=0.6500", s)
        self.assertIn("min_wr_after_3=0.6200", s)


if __name__ == "__main__":
    unittest.main()
