import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from analysis.exec_log import record_fill, record_signal
from analysis.slippage_audit import (
    DEFAULT_MATERIAL_BIAS_BPS,
    DEFAULT_MIN_OBSERVATIONS,
    audit_slippage,
    collect_fills,
    main,
)


class _TmpLog:
    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "exec_log.jsonl"
        return self.path

    def __exit__(self, *exc):
        self._tmp.cleanup()


def _seed(path: Path, n: int, *, desired_entry: float, fill_offset_pct: float) -> None:
    """Create `n` (PENDING, FILLED) pairs at given offset.

    `fill_offset_pct = 0.001` means fill 10bps above desired (positive slippage).
    """
    for i in range(n):
        date = f"2026-01-{(i % 28) + 1:02d}"
        ticker = f"T{i:03d}.SA"
        record_signal(date, ticker, desired_entry=desired_entry, log_path=path)
        record_fill(date, ticker, fill_price=desired_entry * (1.0 + fill_offset_pct), log_path=path)


class CollectFills(unittest.TestCase):
    def test_only_filled_with_finite_slippage_returned(self):
        with _TmpLog() as path:
            _seed(path, 5, desired_entry=10.0, fill_offset_pct=0.001)
            # Add a PENDING that has no slippage to confirm it's excluded.
            record_signal("2026-02-01", "X.SA", desired_entry=10.0, log_path=path)
            fills = collect_fills(path)
            self.assertEqual(len(fills), 5)
            self.assertTrue(all(f["status"] == "FILLED" for f in fills))


class AuditSlippage(unittest.TestCase):
    def test_returns_missing_when_below_min_observations(self):
        with _TmpLog() as path:
            _seed(path, 3, desired_entry=10.0, fill_offset_pct=0.001)
            report = audit_slippage(log_path=path, min_observations=10)
            self.assertEqual(report["bias_label"], "MISSING")
            self.assertEqual(report["n_observations"], 3)

    def test_ok_label_when_within_material_threshold(self):
        with _TmpLog() as path:
            # Fill at exactly modeled = 10 bps -> bias 0, OK
            _seed(path, 12, desired_entry=10.0, fill_offset_pct=0.001)
            report = audit_slippage(log_path=path, modeled_bps=10.0, min_observations=10)
            self.assertEqual(report["bias_label"], "OK")
            self.assertEqual(report["n_observations"], 12)
            self.assertAlmostEqual(report["slippage_p50_bps"], 10.0, places=4)
            self.assertAlmostEqual(report["bias_vs_model_bps"], 0.0, places=4)

    def test_material_flag_when_realized_above_model(self):
        with _TmpLog() as path:
            # Fill at +25 bps consistently -> 15 bps bias vs modeled 10 -> MATERIAL
            _seed(path, 12, desired_entry=10.0, fill_offset_pct=0.0025)
            report = audit_slippage(log_path=path, modeled_bps=10.0, min_observations=10)
            self.assertEqual(report["bias_label"], "MATERIAL")
            self.assertGreater(report["bias_vs_model_bps"], DEFAULT_MATERIAL_BIAS_BPS)
            self.assertIn("FITNESS_VERSION", report["recommendation"])

    def test_material_flag_when_realized_below_model(self):
        with _TmpLog() as path:
            # Fill at +1 bps consistently -> bias -9 vs modeled 10 -> MATERIAL
            _seed(path, 12, desired_entry=10.0, fill_offset_pct=0.0001)
            report = audit_slippage(log_path=path, modeled_bps=10.0, min_observations=10)
            self.assertEqual(report["bias_label"], "MATERIAL")
            self.assertLess(report["bias_vs_model_bps"], -DEFAULT_MATERIAL_BIAS_BPS)
            self.assertIn("lower", report["recommendation"])

    def test_custom_bias_threshold_overrides_default(self):
        with _TmpLog() as path:
            _seed(path, 12, desired_entry=10.0, fill_offset_pct=0.0012)
            # Realized ~12 bps, modeled 10 -> 2 bps bias. Default 5 -> OK.
            # Tighter 1 bps threshold -> MATERIAL.
            ok = audit_slippage(log_path=path, modeled_bps=10.0, min_observations=10)
            tight = audit_slippage(log_path=path, modeled_bps=10.0, min_observations=10, material_bias_bps=1.0)
            self.assertEqual(ok["bias_label"], "OK")
            self.assertEqual(tight["bias_label"], "MATERIAL")

    def test_stats_have_expected_shape_when_enough_data(self):
        with _TmpLog() as path:
            _seed(path, 20, desired_entry=10.0, fill_offset_pct=0.0008)
            report = audit_slippage(log_path=path, modeled_bps=10.0, min_observations=10)
            for key in (
                "modeled_bps", "n_observations", "slippage_p10_bps",
                "slippage_p50_bps", "slippage_p90_bps", "slippage_mean_bps",
                "slippage_std_bps", "bias_vs_model_bps", "bias_label", "recommendation",
            ):
                self.assertIn(key, report, f"missing key {key}")
            self.assertEqual(report["n_observations"], 20)

    def test_missing_log_file_returns_missing(self):
        report = audit_slippage(log_path="/nonexistent/path.jsonl", min_observations=10)
        self.assertEqual(report["bias_label"], "MISSING")
        self.assertEqual(report["n_observations"], 0)


class CLI(unittest.TestCase):
    def test_main_prints_valid_json(self):
        with _TmpLog() as path:
            _seed(path, 12, desired_entry=10.0, fill_offset_pct=0.001)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["--log", str(path), "--modeled", "10.0", "--min", "10"])
            self.assertEqual(rc, 0)
            report = json.loads(buf.getvalue())
            self.assertEqual(report["bias_label"], "OK")

    def test_main_writes_json_file_when_requested(self):
        with _TmpLog() as path:
            _seed(path, 12, desired_entry=10.0, fill_offset_pct=0.001)
            with tempfile.TemporaryDirectory() as outdir:
                out_json = Path(outdir) / "report.json"
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--log", str(path), "--modeled", "10.0", "--min", "10", "--json", str(out_json)])
                self.assertEqual(rc, 0)
                self.assertTrue(out_json.exists())
                report = json.loads(out_json.read_text(encoding="utf-8"))
                self.assertEqual(report["bias_label"], "OK")


if __name__ == "__main__":
    unittest.main()
